"""
train_v2.py - Curriculum Training Pipeline (3-Phase)
════════════════════════════════════════════════════
3-Phase Curriculum Training:

  Phase 1 (epochs 1–15):  CE Loss thuần, backbone + FC head, NO prototype
                          → Đạt baseline performance, embeddings stable

  Phase 2 (epochs 16–25): Init prototypes từ Phase-1 embeddings (class-mean)
                          Freeze backbone, train Projection + Prototype head
                          → Prototype head học từ anchors có ý nghĩa semantic

  Phase 3 (epochs 26–60): Unfreeze backbone, Joint SupCon + CE loss (α tăng dần)
                          → Fine-tune toàn bộ model

Key improvements vs V1:
  • Prototypes init từ Phase-1 embeddings (thay vì random)
  • 3-phase curriculum (thay vì warmup đơn giản)
  • SupCon alpha tăng dần từ 0.05 → 0.20 (gradually introduce contrastive)
  • Prototype-based confidence scorer tích hợp sẵn
  • DualHeadCLF option để compare prototype vs FC

Usage:
    from train_v2 import TrainerV2Curriculum, TrainConfigV2

    cfg = TrainConfigV2()
    trainer = TrainerV2Curriculum(cfg)
    best_state, history = trainer.run()
"""

import copy
import os
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore")

from calibration import TemperatureScaling
from dataset import CLASS_NAMES, NUM_CLASSES, MalariaDataset, get_transforms
from losses import SupConLoss, DynamicFocalLoss, CombinedLoss
from model import (
    MalariaProtoCLFv2,
    build_model,
    compute_class_prototypes,
)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
class TrainConfigV2:
    # ── Paths ──
    BASE_DIR   = "/kaggle/input/datasets/khanhtq2101/malaria-parasite/final_malaria_full_class_classification_cropped/5 classes - May 2025"
    IMG_BASE  = "/kaggle/input/datasets/khanhtq2101/malaria-parasite/final_malaria_full_class_classification_cropped"
    TRAIN_ANN = os.path.join(BASE_DIR, "train_annotation_5classes.txt")
    VAL_ANN   = os.path.join(BASE_DIR, "val_annotation_5classes.txt")
    TEST_ANN  = os.path.join(BASE_DIR, "test_annotation_5classes.txt")
    OUTPUT_DIR = "/kaggle/working/malaria_proto_v2"

    # ── Model ──
    BACKBONE       = "convnext_tiny.in22k_ft_in1k"
    NUM_CLASSES    = 5
    PROJ_DIM       = 128
    USE_PROTOTYPE  = True
    USE_DUAL_HEAD  = False    # True → hybrid Proto + FC head
    BLEND_ALPHA    = 0.5      # chỉ dùng khi USE_DUAL_HEAD=True
    IMG_SIZE       = 224
    DROPOUT        = 0.1

    # ── Training phases ──
    EPOCHS_P1      = 15       # Phase 1: CE only (backbone + FC head)
    EPOCHS_P2      = 10       # Phase 2: train prototype (backbone frozen)
    EPOCHS_P3      = 35       # Phase 3: joint + unfreeze
    TOTAL_EPOCHS   = 60       # EPOCHS_P1 + EPOCHS_P2 + EPOCHS_P3

    BATCH_SIZE     = 32
    LR_P1          = 3e-4     # Phase 1 LR
    LR_P2          = 2e-4     # Phase 2 LR (prototype head)
    LR_P3_BACKBONE = 3e-5     # Phase 3 LR for backbone
    LR_P3_HEAD     = 2e-4     # Phase 3 LR for heads
    WEIGHT_DECAY   = 1e-4

    # ── Loss ──
    SUPCON_TEMP    = 0.07
    ALPHA_START    = 0.05     # SupCon weight start (Phase 3)
    ALPHA_END      = 0.20     # SupCon weight end (Phase 3)
    CLF_LOSS_P1    = "ce"     # Phase 1: standard CE
    CLF_LOSS_P3    = "focal"  # Phase 3: focal (class imbalance)
    MAJORITY_CLASS = 4        # Unparasitized

    # ── Calibration ──
    DO_CALIBRATION = True

    # ── Misc ──
    SEED           = 42
    NUM_WORKERS    = 4
    PIN_MEMORY     = True
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_BEST_METRIC = "macro_f1"


# ─────────────────────────────────────────────
# Weighted sampler
# ─────────────────────────────────────────────
def make_weighted_sampler(dataset):
    labels  = [lbl for _, lbl in dataset.samples]
    counts  = Counter(labels)
    n_total = len(labels)
    weights = [n_total / counts[lbl] for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=n_total, replacement=True)


# ─────────────────────────────────────────────
# Trainer V2 Curriculum
# ─────────────────────────────────────────────
class TrainerV2Curriculum:
    """
    3-Phase Curriculum Trainer.
    Phase 1: CE Loss → backbone + FC → stable embeddings
    Phase 2: Prototype init từ P1 embeddings → train prototype head
    Phase 3: Joint SupCon + CE → full fine-tune
    """

    def __init__(self, cfg: TrainConfigV2):
        self.cfg = cfg
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        self.device = torch.device(cfg.DEVICE)

        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self.scaler     = GradScaler()
        self.best_metric = 0.0
        self.best_state  = None
        self.history = {
            "phase": [], "epoch": [],
            "train_loss": [], "val_loss": [], "val_macro_f1": [],
            "alpha": [], "phase_desc": [],
        }

    # ── Data ──────────────────────────────────
    def _setup_data(self):
        cfg = self.cfg
        train_tf = get_transforms("train", cfg.IMG_SIZE)
        val_tf   = get_transforms("val",   cfg.IMG_SIZE)

        self.train_ds = MalariaDataset(cfg.TRAIN_ANN, cfg.IMG_BASE, transform=train_tf)
        self.val_ds   = MalariaDataset(cfg.VAL_ANN,   cfg.IMG_BASE, transform=val_tf)

        labels = [lbl for _, lbl in self.train_ds.samples]
        counts = Counter(labels)
        self.class_counts = [counts.get(i, 1) for i in range(cfg.NUM_CLASSES)]

        sampler = make_weighted_sampler(self.train_ds)
        self.train_loader = DataLoader(
            self.train_ds, batch_size=cfg.BATCH_SIZE,
            sampler=sampler, num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=cfg.BATCH_SIZE * 2,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
        )

    # ── Model ─────────────────────────────────
    def _setup_model(self, proto_init=None):
        cfg = self.cfg
        self.model = MalariaProtoCLFv2(
            backbone_name  = cfg.BACKBONE,
            num_classes    = cfg.NUM_CLASSES,
            proj_dim       = cfg.PROJ_DIM,
            use_prototype  = cfg.USE_PROTOTYPE,
            use_dual_head  = cfg.USE_DUAL_HEAD,
            pretrained     = True,
            proto_init     = proto_init,
            blend_alpha    = cfg.BLEND_ALPHA,
            dropout        = cfg.DROPOUT,
        ).to(self.device)

    # ── Loss ──────────────────────────────────
    def _setup_loss(self, classification_loss="ce"):
        cfg = self.cfg
        self.supcon_loss = SupConLoss(temperature=cfg.SUPCON_TEMP).to(self.device)
        if classification_loss == "ce":
            self.clf_loss = nn.CrossEntropyLoss().to(self.device)
        elif classification_loss == "focal":
            self.clf_loss = DynamicFocalLoss(
                cfg.NUM_CLASSES, self.class_counts,
            ).to(self.device)

    # ── Phase 1: CE only (backbone + FC) ──────
    def _run_phase1(self):
        """
        Phase 1: CE Loss, backbone trainable, prototype head DISABLED.
        Cuối phase: trích embeddings → compute class prototypes.
        """
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"PHASE 1: CE Loss only | {cfg.EPOCHS_P1} epochs | backbone TRAINABLE")
        print(f"{'='*60}")

        # Rebuild model: prototype disabled, FC head enabled
        self.model = MalariaProtoCLFv2(
            backbone_name  = cfg.BACKBONE,
            num_classes    = cfg.NUM_CLASSES,
            use_prototype  = False,      # dùng FC head thuần
            pretrained     = True,
            dropout        = cfg.DROPOUT,
        ).to(self.device)
        self._setup_loss(classification_loss="ce")
        self.ce_criterion = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.LR_P1, weight_decay=cfg.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.EPOCHS_P1, eta_min=1e-6,
        )

        for epoch in range(1, cfg.EPOCHS_P1 + 1):
            t_loss = self._train_epoch_phase1(epoch)
            v_loss, macro_f1 = self._val_epoch_phase1()
            self.scheduler.step()

            self._log_epoch("P1", epoch, cfg.TOTAL_EPOCHS, t_loss, v_loss, macro_f1, alpha=None)

            if macro_f1 > self.best_metric:
                self.best_metric = macro_f1
                self.best_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state,
                           os.path.join(cfg.OUTPUT_DIR, "phase1_best.pth"))

        # Phase 1 done → compute prototypes from embeddings
        print("\n[Phase 1] Computing class prototypes from embeddings...")
        self.model.load_state_dict(self.best_state)
        proto_init = compute_class_prototypes(
            self.model, self.train_loader, self.device,
            max_samples_per_class=2000,
        )
        print(f"[Phase 1] Computed prototypes: shape={proto_init.shape}")
        print(f"  Per-class norms: {proto_init.norm(dim=1).tolist()}")
        return proto_init

    def _train_epoch_phase1(self, epoch):
        self.model.train()
        total_loss, n = 0.0, 0
        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                proj_feats, logits = self.model(imgs)
                loss = self.ce_criterion(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _val_epoch_phase1(self):
        from sklearn.metrics import f1_score
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            _, logits = self.model(imgs)
            loss = self.ce_criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return total_loss / len(self.val_loader), f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # ── Phase 2: Prototype training ───────────
    def _run_phase2(self, proto_init):
        """
        Phase 2: Prototype head, backbone FROZEN.
        Init prototypes = class-mean từ Phase 1.
        """
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"PHASE 2: Prototype Training | {cfg.EPOCHS_P2} epochs | backbone FROZEN")
        print(f"         Prototype init: class-mean from Phase-1 embeddings")
        print(f"{'='*60}")

        # Rebuild model: prototype enabled, init với pretrained prototypes
        self.model = MalariaProtoCLFv2(
            backbone_name  = cfg.BACKBONE,
            num_classes    = cfg.NUM_CLASSES,
            proj_dim       = cfg.PROJ_DIM,
            use_prototype  = True,
            pretrained     = True,
            proto_init     = proto_init,    # ← pretrained prototypes
            dropout        = cfg.DROPOUT,
        ).to(self.device)

        # Freeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        self._setup_loss(classification_loss="focal")

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.LR_P2, weight_decay=cfg.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.EPOCHS_P2, eta_min=1e-6,
        )

        for epoch in range(1, cfg.EPOCHS_P2 + 1):
            t_loss, t_clf = self._train_epoch_phase2(epoch)
            v_loss, macro_f1 = self._val_epoch_phase2()

            if epoch > 1:
                self.scheduler.step()

            self._log_epoch("P2", epoch, cfg.TOTAL_EPOCHS, t_loss, v_loss, macro_f1, alpha=None,
                            extra=f"clf={t_clf:.4f}")

            if macro_f1 > self.best_metric:
                self.best_metric = macro_f1
                self.best_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state,
                           os.path.join(cfg.OUTPUT_DIR, "phase2_best.pth"))

        return self.model

    def _train_epoch_phase2(self, epoch):
        self.model.train()
        total_loss, total_clf, n = 0.0, 0.0, 0
        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                proj_feats, logits = self.model(imgs)
                loss = self.clf_loss(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            total_clf += loss.item()
            n += 1
        return total_loss / max(n, 1), total_clf / max(n, 1)

    @torch.no_grad()
    def _val_epoch_phase2(self):
        from sklearn.metrics import f1_score
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            _, logits = self.model(imgs)
            loss = self.clf_loss(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return total_loss / len(self.val_loader), f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # ── Phase 3: Joint SupCon + CE ───────────
    def _run_phase3(self, proto_init):
        """
        Phase 3: Joint SupCon + CE, backbone UNFROZEN.
        α (SupCon weight) tăng dần từ ALPHA_START → ALPHA_END.
        """
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"PHASE 3: Joint SupCon + CE | {cfg.EPOCHS_P3} epochs | backbone UNFROZEN")
        print(f"         α ramps from {cfg.ALPHA_START} → {cfg.ALPHA_END}")
        print(f"{'='*60}")

        # Rebuild model: prototype + FC (dual head) hoặc prototype only
        if cfg.USE_DUAL_HEAD:
            self.model = MalariaProtoCLFv2(
                backbone_name  = cfg.BACKBONE,
                num_classes    = cfg.NUM_CLASSES,
                proj_dim       = cfg.PROJ_DIM,
                use_prototype  = True,
                use_dual_head  = True,
                pretrained     = False,
                proto_init     = proto_init,
                blend_alpha    = cfg.BLEND_ALPHA,
                dropout        = cfg.DROPOUT,
            ).to(self.device)
            self.model.load_state_dict(self.best_state, strict=False)
        else:
            self.model = MalariaProtoCLFv2(
                backbone_name  = cfg.BACKBONE,
                num_classes    = cfg.NUM_CLASSES,
                proj_dim       = cfg.PROJ_DIM,
                use_prototype  = True,
                pretrained     = False,
                proto_init     = proto_init,
                dropout        = cfg.DROPOUT,
            ).to(self.device)
            # Load Phase 2 weights (backbone + proj_head + clf_head)
            sd = self.best_state
            # strip FC-only params nếu có
            self.model.load_state_dict(sd, strict=False)

        # Unfreeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = True

        self._setup_loss(classification_loss="focal")

        self.optimizer = torch.optim.AdamW([
            {"params": self.model.backbone.parameters(),  "lr": cfg.LR_P3_BACKBONE},
            {"params": self.model.proj_head.parameters(), "lr": cfg.LR_P3_HEAD},
            {"params": self.model.clf_head.parameters(),  "lr": cfg.LR_P3_HEAD},
        ], weight_decay=cfg.WEIGHT_DECAY)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.EPOCHS_P3, eta_min=1e-6,
        )

        for epoch in range(1, cfg.EPOCHS_P3 + 1):
            # Alpha ramp-up
            alpha = cfg.ALPHA_START + (cfg.ALPHA_END - cfg.ALPHA_START) * (epoch - 1) / max(cfg.EPOCHS_P3 - 1, 1)
            t_loss, t_sc, t_clf = self._train_epoch_phase3(epoch, alpha)
            v_loss, macro_f1 = self._val_epoch_phase3()

            if epoch > 1:
                self.scheduler.step()

            self._log_epoch("P3", epoch, cfg.TOTAL_EPOCHS, t_loss, v_loss, macro_f1, alpha=alpha,
                            extra=f"SC:{t_sc:.4f} CLF:{t_clf:.4f}")

            if macro_f1 > self.best_metric:
                self.best_metric = macro_f1
                self.best_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state,
                           os.path.join(cfg.OUTPUT_DIR, "best_model_v2.pth"))

        return self.model

    def _train_epoch_phase3(self, epoch, alpha):
        self.model.train()
        total_loss, total_sc, total_clf, n = 0.0, 0.0, 0.0, 0
        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                proj_feats, logits = self.model(imgs)
                l_sc  = self.supcon_loss(proj_feats, labels)
                l_clf = self.clf_loss(logits, labels)
                loss  = alpha * l_sc + (1.0 - alpha) * l_clf
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            total_sc   += l_sc.item()
            total_clf  += l_clf.item()
            n += 1
        return total_loss/n, total_sc/n, total_clf/n

    @torch.no_grad()
    def _val_epoch_phase3(self):
        from sklearn.metrics import f1_score
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            _, logits = self.model(imgs)
            loss = self.clf_loss(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return total_loss / len(self.val_loader), f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # ── Logging ────────────────────────────────
    def _log_epoch(self, phase, epoch, total_epochs, t_loss, v_loss, macro_f1, alpha=None, extra=""):
        self.history["phase"].append(phase)
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(t_loss)
        self.history["val_loss"].append(v_loss)
        self.history["val_macro_f1"].append(macro_f1)
        self.history["alpha"].append(alpha if alpha is not None else 0.0)
        self.history["phase_desc"].append(f"{phase} ep{epoch}")

        tag = "✓ BEST" if macro_f1 >= self.best_metric else ""
        alpha_str = f"α={alpha:.3f}" if alpha is not None else ""
        print(
            f"[{phase}] {epoch:03d}/{total_epochs} | "
            f"T {t_loss:.4f} {extra} | V {v_loss:.4f} | "
            f"F1 {macro_f1:.4f} {alpha_str} {tag}"
        )

    # ── Main run ─────────────────────────────
    def run(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"Curriculum Training V2 | {cfg.BACKBONE}")
        print(f"Phase 1: {cfg.EPOCHS_P1}ep CE only | Phase 2: {cfg.EPOCHS_P2}ep Proto | Phase 3: {cfg.EPOCHS_P3}ep Joint")
        print(f"{'='*60}")

        # Phase 1: CE only
        proto_init = self._run_phase1()
        p1_best_f1 = self.best_metric
        print(f"\n[Phase 1] Best macro-F1: {p1_best_f1:.4f}")

        # Phase 2: Prototype training
        self._run_phase2(proto_init)
        p2_best_f1 = self.best_metric
        print(f"\n[Phase 2] Best macro-F1: {p2_best_f1:.4f}")

        # Phase 3: Joint training
        self._run_phase3(proto_init)
        p3_best_f1 = self.best_metric
        print(f"\n[Phase 3] Best macro-F1: {p3_best_f1:.4f}")

        # Load best
        self.model.load_state_dict(self.best_state)

        # Calibration
        if cfg.DO_CALIBRATION:
            print("\n[Calibration] Fitting temperature scaling...")
            ts = TemperatureScaling(self.model)
            T  = ts.fit(self.val_loader, self.device)
            torch.save({
                "model_state": self.best_state,
                "temperature": T,
                "cfg": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
                "proto_init_summary": {
                    "method": "class_mean_from_phase1_embeddings",
                    "shape": list(proto_init.shape),
                },
            }, os.path.join(cfg.OUTPUT_DIR, "calibrated_model_v2.pth"))
            print(f"[Done] Temperature T={T:.4f}")

        print(f"\n{'='*60}")
        print(f"BEST MACRO-F1: {self.best_metric:.4f}")
        print(f"  Phase 1 best: {p1_best_f1:.4f}")
        print(f"  Phase 2 best: {p2_best_f1:.4f}")
        print(f"  Phase 3 best: {p3_best_f1:.4f}")
        print(f"{'='*60}")

        return self.model, self.history


# ─────────────────────────────────────────────
# Convenience: build model từ checkpoint với proto_init
# ─────────────────────────────────────────────
def load_model_v2(checkpoint_path, device, use_dual_head=False, blend_alpha=0.5):
    """
    Load model với checkpoint, tự động đọc config.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "cfg" in ckpt:
        cfg_dict = ckpt["cfg"]
        model_cfg = {
            "backbone":      cfg_dict.get("BACKBONE", "convnext_tiny.in22k_ft_in1k"),
            "num_classes":   cfg_dict.get("NUM_CLASSES", 5),
            "proj_dim":      cfg_dict.get("PROJ_DIM", 128),
            "use_prototype": cfg_dict.get("USE_PROTOTYPE", True),
            "use_dual_head":  use_dual_head,
            "blend_alpha":    blend_alpha,
            "pretrained":    False,
        }
        temperature = ckpt.get("temperature", 1.0)
        state_dict  = ckpt["model_state"]
    else:
        model_cfg  = {
            "backbone": "convnext_tiny.in22k_ft_in1k",
            "num_classes": 5, "proj_dim": 128,
            "use_prototype": True, "use_dual_head": False,
            "pretrained": False,
        }
        temperature = 1.0
        state_dict  = ckpt

    model = build_model(model_cfg)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, temperature


if __name__ == "__main__":
    cfg = TrainConfigV2()
    trainer = TrainerV2Curriculum(cfg)
    trainer.run()
