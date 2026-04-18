"""
train_v2.py - Curriculum Training Pipeline v2 (3-Phase)
═══════════════════════════════════════════════════════
3-Phase Curriculum Training:

  Phase 1 (epochs 1–15):  CE Loss thuần, backbone + FC head, NO prototype
                          → Đạt baseline performance, embeddings stable

  Phase 2a (epochs 1–N):   SupCon + PrototypePushLoss, backbone FROZEN
                          → Học embedding space có tính phân biệt cao
                          → SupCon: intra-class cohesion
                          → PrototypePushLoss: inter-class separation
  Phase 2b (epochs N+1–10): CE/Focal fine-tune, backbone vẫn FROZEN
                          → Tinh chỉnh decision boundary trên embedding space

  Phase 3 (epochs 1–35):  Joint SupCon (normalized) + CE, backbone UNFROZEN
                          → Fine-tune toàn bộ model với α = 0.30 → 0.80
                          → SmoothedEarlyStopping (patience=8, smoothing=5)

Key improvements over v1 (based on training log analysis):
  • SC loss was stuck at ~1.75 because α was too small (0.05–0.20).
    Fixed: α range raised to 0.30–0.80 so contrastive gradient is meaningful.
  • SC loss magnitude (~1.75) dwarfed CE loss (~0.001), collapsing gradients.
    Fixed: online loss normalization keeps SC gradient on par with CE gradient.
  • Phase 2 did NOT improve F1 (still 0.8761 after CE-only training).
    Fixed: P2a = SupCon-only to learn embedding space; P2b = CE fine-tune.
  • Phase 3 F1 oscillated (0.80–0.87) with no convergence signal.
    Fixed: SmoothedEarlyStopping + P2 embedding improvements.
  • Prototype class-norm ≈ 1.0 for all classes → no discriminative init.
    Fixed: P2a + PrototypePushLoss actively separates prototypes.

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

from calibration import TemperatureScaling
from dataset import MalariaDataset, get_transforms
from losses import (
    DynamicFocalLoss,
    PrototypePushLoss,
    SmoothedEarlyStopping,
    SupConLoss,
)
from model import MalariaProtoCLFv2, build_model, compute_class_prototypes

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
class TrainConfigV2:
    # ── Paths ──
    BASE_DIR = "/kaggle/input/datasets/khanhtq2101/malaria-parasite/final_malaria_full_class_classification_cropped/5 classes - May 2025"
    IMG_BASE = "/kaggle/input/datasets/khanhtq2101/malaria-parasite/final_malaria_full_class_classification_cropped"
    TRAIN_ANN = os.path.join(BASE_DIR, "train_annotation_5classes.txt")
    VAL_ANN = os.path.join(BASE_DIR, "val_annotation_5classes.txt")
    TEST_ANN = os.path.join(BASE_DIR, "test_annotation_5classes.txt")
    OUTPUT_DIR = "/kaggle/working/malaria_proto_v2"

    # ── Model ──
    BACKBONE = "convnext_tiny.in22k_ft_in1k"
    NUM_CLASSES = 5
    PROJ_DIM = 128
    USE_PROTOTYPE = True
    USE_DUAL_HEAD = False  # True → hybrid Proto + FC head
    BLEND_ALPHA = 0.5  # chỉ dùng khi USE_DUAL_HEAD=True
    IMG_SIZE = 224
    DROPOUT = 0.1

    # ── Training phases ──
    EPOCHS_P1 = 15  # Phase 1: CE only (backbone + FC head)
    EPOCHS_P2 = 15  # Phase 2: train prototype (backbone very-low LR)
    EPOCHS_P3 = 25  # Phase 3: joint + unfreeze
    TOTAL_EPOCHS = 55  # EPOCHS_P1 + EPOCHS_P2 + EPOCHS_P3

    BATCH_SIZE = 32
    LR_P1 = 3e-4  # Phase 1 LR
    LR_P2_HEAD = 5e-4  # Phase 2 LR (prototype head)
    LR_P2_BACKBONE = 1e-6  # Phase 2: backbone very-low LR (nearly frozen)
    LR_P3_BACKBONE = 3e-5  # Phase 3 LR for backbone
    LR_P3_HEAD = 2e-4  # Phase 3 LR for heads
    WEIGHT_DECAY = 1e-4

    # ── Phase 3 warmup ──
    P3_WARMUP_EPOCHS = 3  # Linear warmup for backbone LR in Phase 3

    # ── Loss ──
    SUPCON_TEMP = 0.07
    ALPHA_START = 0.05  # SupCon weight start (Phase 3) — lightweight, fine-tune focus
    ALPHA_END   = 0.15  # SupCon weight end (Phase 3)   — stays gentle to not overpower CE

    # ── Phase 3 loss balance ──
    USE_LOSS_NORMALIZATION = True  # Normalize SC loss by its running mean to balance gradient magnitude
    SUPCON_TARGET = 0.50  # Target SupCon loss value — used for loss normalization

    # ── Phase 2 structure ──
    # P2a: SupCon + PushLoss epochs = P2_CLF_EPOCHS + 2 (min 2)
    # P2b: CE/Focal fine-tune remaining epochs
    P2_CLF_EPOCHS    = 5      # Brief CE fine-tune epochs within Phase 2

    # ── Early stopping ──
    EARLY_STOP_PATIENCE = 10  # increased from 8 for more tolerance
    EARLY_STOP_MIN_DELTA = 0.002  # improvement threshold
    EARLY_STOP_SMOOTH    = 5   # smoothing window for F1 tracking

    # ── Prototype push loss (used in both Phase 2 and Phase 3) ──
    PUSH_LOSS_WEIGHT = 0.1  # increased from 0.05 for stronger inter-class separation

    CLF_LOSS_P1 = "ce"  # Phase 1: standard CE
    CLF_LOSS_P3 = "focal"  # Phase 3: focal (class imbalance)
    MAJORITY_CLASS = 4  # Unparasitized

    # ── Calibration ──
    DO_CALIBRATION = True

    # ── Misc ──
    SEED = 42
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_BEST_METRIC = "macro_f1"  # accuracy, macro_f1, etc.


# ─────────────────────────────────────────────
# Weighted sampler
# ─────────────────────────────────────────────
def make_weighted_sampler(dataset):
    labels = [lbl for _, lbl in dataset.samples]
    counts = Counter(labels)
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
        self.scaler = GradScaler()
        self.best_metric = 0.0
        self.best_state = None
        self.history = {
            "phase": [],
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_macro_f1": [],
            "alpha": [],
            "phase_desc": [],
        }

    # ── Data ──────────────────────────────────
    def _setup_data(self):
        cfg = self.cfg
        train_tf = get_transforms("train", cfg.IMG_SIZE)
        val_tf = get_transforms("val", cfg.IMG_SIZE)

        self.train_ds = MalariaDataset(cfg.TRAIN_ANN, cfg.IMG_BASE, transform=train_tf)
        self.val_ds = MalariaDataset(cfg.VAL_ANN, cfg.IMG_BASE, transform=val_tf)

        labels = [lbl for _, lbl in self.train_ds.samples]
        counts = Counter(labels)
        self.class_counts = [counts.get(i, 1) for i in range(cfg.NUM_CLASSES)]

        sampler = make_weighted_sampler(self.train_ds)
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.BATCH_SIZE,
            sampler=sampler,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
        )

    # ── Model ─────────────────────────────────
    def _setup_model(self, proto_init=None):
        cfg = self.cfg
        self.model = MalariaProtoCLFv2(
            backbone_name=cfg.BACKBONE,
            num_classes=cfg.NUM_CLASSES,
            proj_dim=cfg.PROJ_DIM,
            use_prototype=cfg.USE_PROTOTYPE,
            use_dual_head=cfg.USE_DUAL_HEAD,
            pretrained=True,
            proto_init=proto_init,
            blend_alpha=cfg.BLEND_ALPHA,
            dropout=cfg.DROPOUT,
        ).to(self.device)

    # ── Loss ──────────────────────────────────
    def _setup_loss(self, classification_loss="ce", normalize_supcon=False):
        cfg = self.cfg
        self.supcon_loss = SupConLoss(
            temperature=cfg.SUPCON_TEMP,
            normalize_loss=normalize_supcon,
            target_loss=cfg.SUPCON_TARGET,
        ).to(self.device)
        if classification_loss == "ce":
            self.clf_loss = nn.CrossEntropyLoss().to(self.device)
        elif classification_loss == "focal":
            self.clf_loss = DynamicFocalLoss(
                cfg.NUM_CLASSES,
                self.class_counts,
            ).to(self.device)

        # Prototype push-away loss (used in Phase 2 and 3)
        self.push_loss = PrototypePushLoss(weight=cfg.PUSH_LOSS_WEIGHT)

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

        # Phase 1: FC head only (use_prototype=False).
        # Backbone + proj_head learn strong representations via CE loss.
        # Prototypes are computed AFTER training from trained embeddings.
        self.model = MalariaProtoCLFv2(
            backbone_name=cfg.BACKBONE,
            num_classes=cfg.NUM_CLASSES,
            proj_dim=cfg.PROJ_DIM,
            use_prototype=False,  # FC head — clean baseline, no prototype interference
            use_dual_head=False,
            pretrained=True,
            proto_init=None,
            dropout=cfg.DROPOUT,
        ).to(self.device)
        self._setup_loss(classification_loss="ce")
        self.ce_criterion = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.LR_P1,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.EPOCHS_P1,
            eta_min=1e-6,
        )

        for epoch in range(1, cfg.EPOCHS_P1 + 1):
            t_loss = self._train_epoch_phase1(epoch)
            v_loss, macro_f1 = self._val_epoch_phase1()
            self.scheduler.step()

            self._log_epoch("P1", epoch, cfg.TOTAL_EPOCHS, t_loss, v_loss, macro_f1, alpha=None)

            if macro_f1 > self.best_metric:
                self.best_metric = macro_f1
                self.best_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state, os.path.join(cfg.OUTPUT_DIR, "phase1_best.pth"))

        # Phase 1 done → compute prototypes from embeddings
        print("\n[Phase 1] Computing class prototypes from embeddings...")
        self.model.load_state_dict(self.best_state)
        proto_init = compute_class_prototypes(
            self.model,
            self.train_loader,
            self.device,
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
        Phase 2: Split into two sub-phases (backbone FROZEN throughout):
          P2a – SupCon + Push Loss  (epochs 1 → P2_SUPCON_EPOCHS):
                  Learn a well-separated embedding space using the class-mean
                  prototypes as anchors. SupCon maximises intra-class cohesion,
                  PrototypePushLoss maximises inter-class separation.
          P2b – CE / Focal fine-tune (epochs P2_SUPCON_EPOCHS+1 → EPOCHS_P2):
                  Standard classification head training on the now-improved
                  embedding space to set the decision boundary.
        """
        cfg = self.cfg
        p2_supcon_epochs = min(cfg.EPOCHS_P2, cfg.P2_CLF_EPOCHS + 2)  # at least 2 SupCon epochs
        p2_clf_epochs    = cfg.EPOCHS_P2

        print(f"\n{'='*60}")
        print(f"PHASE 2: Prototype Training | {cfg.EPOCHS_P2} epochs | backbone VERY-LOW LR")
        print(f"         Prototype init: class-mean from Phase-1 embeddings")
        print(f"         P2a: SupCon+PushLoss  ({p2_supcon_epochs} ep)")
        print(f"         P2b: CE/Focal fine-tune ({p2_clf_epochs - p2_supcon_epochs} ep)")
        print(f"{'='*60}")

        # Rebuild model: prototype enabled, init với pretrained prototypes
        self.model = MalariaProtoCLFv2(
            backbone_name=cfg.BACKBONE,
            num_classes=cfg.NUM_CLASSES,
            proj_dim=cfg.PROJ_DIM,
            use_prototype=True,
            pretrained=False,  # weights loaded below
            proto_init=proto_init,
            dropout=cfg.DROPOUT,
        ).to(self.device)

        # Load Phase 1 best weights (backbone + proj_head), then load prototypes
        p1_state = self.best_state
        p2_state = self.model.state_dict()
        loaded_keys, skipped_keys = [], []
        for key in list(p1_state.keys()):
            if key in p2_state:
                p2_state[key] = p1_state[key]
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
        self.model.load_state_dict(p2_state, strict=False)
        print(f"[Phase 2] Loaded {len(loaded_keys)} keys from Phase 1, skipped {len(skipped_keys)} (FC head)")

        # Freeze backbone throughout Phase 2; only train prototype + proj_head
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.LR_P2_HEAD,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        p2_global_epoch = 0

        # ── P2a: SupCon + Push (embedding space) ──────────────────────
        if p2_supcon_epochs > 0:
            print(f"\n[Phase 2a] SupCon + Push Loss | {p2_supcon_epochs} ep | backbone FROZEN")
            self._setup_loss(classification_loss="ce", normalize_supcon=False)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=p2_supcon_epochs,
                eta_min=1e-6,
            )

            for epoch in range(1, p2_supcon_epochs + 1):
                p2_global_epoch += 1
                t_loss, t_sc, t_push = self._train_epoch_phase2a(epoch)
                v_loss, macro_f1 = self._val_epoch_phase2(loss_fn=self.ce_criterion)

                if epoch > 1:
                    self.scheduler.step()

                self._log_epoch(
                    "P2a", p2_global_epoch, cfg.TOTAL_EPOCHS,
                    t_loss, v_loss, macro_f1,
                    alpha=None, extra=f"SC:{t_sc:.4f} push:{t_push:.4f}",
                )

                if macro_f1 > self.best_metric:
                    self.best_metric = macro_f1
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    torch.save(self.best_state, os.path.join(cfg.OUTPUT_DIR, "phase2a_best.pth"))

        # ── P2b: CE / Focal fine-tune ──────────────────────────────────
        p2b_epochs = cfg.EPOCHS_P2 - p2_supcon_epochs
        if p2b_epochs > 0:
            print(f"\n[Phase 2b] CE/Focal fine-tune | {p2b_epochs} ep | backbone FROZEN")
            self._setup_loss(classification_loss="focal", normalize_supcon=False)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=p2b_epochs,
                eta_min=1e-6,
            )

            for epoch in range(1, p2b_epochs + 1):
                p2_global_epoch += 1
                t_loss, t_clf = self._train_epoch_phase2b(epoch)
                v_loss, macro_f1 = self._val_epoch_phase2(loss_fn=self.clf_loss)

                if epoch > 1:
                    self.scheduler.step()

                self._log_epoch(
                    "P2b", p2_global_epoch, cfg.TOTAL_EPOCHS,
                    t_loss, v_loss, macro_f1,
                    alpha=None, extra=f"clf={t_clf:.4f}",
                )

                if macro_f1 > self.best_metric:
                    self.best_metric = macro_f1
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    torch.save(self.best_state, os.path.join(cfg.OUTPUT_DIR, "phase2_best.pth"))

        return self.model

    def _train_epoch_phase2a(self, epoch):
        """P2a: SupCon loss + prototype push-away loss. No CE/Focal."""
        self.model.train()
        total_loss, total_sc, total_push, n = 0.0, 0.0, 0.0, 0
        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                proj_feats, _ = self.model(imgs)
                l_sc   = self.supcon_loss(proj_feats, labels)
                l_push = self.push_loss(self.model.clf_head.prototypes)
                loss   = l_sc + l_push
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss  += loss.item()
            total_sc   += l_sc.item()
            total_push += l_push.item()
            n += 1
        return total_loss / max(n, 1), total_sc / max(n, 1), total_push / max(n, 1)

    def _train_epoch_phase2b(self, epoch):
        """P2b: CE / Focal classification loss only."""
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
    def _val_epoch_phase2(self, loss_fn=None):
        """Pass loss_fn so validation metric matches the training objective."""
        from sklearn.metrics import f1_score

        if loss_fn is None:
            loss_fn = self.clf_loss

        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            _, logits = self.model(imgs)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return total_loss / len(self.val_loader), f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # ── Phase 3: Joint SupCon + CE ───────────
    def _run_phase3(self, proto_init):
        """
        Phase 3: Joint SupCon + CE, backbone UNFROZEN.
        α (SupCon weight) tăng dần từ ALPHA_START → ALPHA_END.

        Key changes vs v1:
          • SupCon loss is loss-normalized so its gradient magnitude is comparable
            to the CE/focal gradient (fixes SC loss stalling at ~1.75).
          • PrototypePushLoss keeps inter-class separation while backbone fine-tunes.
          • SmoothedEarlyStopping prevents noisy validation from triggering false
            best-model saves or wasting epochs.
        """
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"PHASE 3: Joint SupCon + CE | {cfg.EPOCHS_P3} epochs | backbone UNFROZEN")
        print(f"         alpha ramps from {cfg.ALPHA_START} -> {cfg.ALPHA_END}")
        print(f"         SupCon loss normalization: {cfg.USE_LOSS_NORMALIZATION}")
        print(f"         SmoothedEarlyStopping: patience={cfg.EARLY_STOP_PATIENCE}, "
              f"min_delta={cfg.EARLY_STOP_MIN_DELTA}, smoothing={cfg.EARLY_STOP_SMOOTH}")
        print(f"{'='*60}")

        # Build Phase 3 model: MUST match Phase 2 structure (PrototypeHead).
        self.model = MalariaProtoCLFv2(
            backbone_name=cfg.BACKBONE,
            num_classes=cfg.NUM_CLASSES,
            proj_dim=cfg.PROJ_DIM,
            use_prototype=True,
            use_dual_head=cfg.USE_DUAL_HEAD,
            pretrained=False,   # weights loaded below
            proto_init=proto_init,
            blend_alpha=cfg.BLEND_ALPHA,
            dropout=cfg.DROPOUT,
        ).to(self.device)

        # Safe weight loading from best Phase-2 checkpoint
        p2_state = self.best_state
        p3_state = self.model.state_dict()
        loaded_keys, skipped_keys = [], []
        for key in list(p2_state.keys()):
            if key in p3_state:
                p3_state[key] = p2_state[key]
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)

        self.model.load_state_dict(p3_state, strict=False)  # strict=False: handle FC→Prototype mismatch
        print(f"[Phase 3] Loaded {len(loaded_keys)} compatible keys from Phase 2")
        if skipped_keys:
            print(f"[Phase 3] Skipped {len(skipped_keys)} keys: {skipped_keys[:3]}...")

        # Unfreeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        print(f"[Phase 3] Backbone unfrozen, prototype weights preserved")

        # Loss: SupCon + Focal + prototype push.
        # IMPORTANT: normalize_supcon=False in Phase 3 because:
        #   1. alpha (0.05→0.15) is the primary SC weight control.
        #   2. normalize_loss=True interacts badly with external alpha scaling
        #      (alpha * normalized_l_sc, where normalize changes l_sc magnitude).
        #   3. Phase 2 already learned a good embedding space; Phase 3 just fine-tunes.
        self._setup_loss(
            classification_loss="focal",
            normalize_supcon=False,  # alpha controls SC weight directly
        )

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.backbone.parameters(), "lr": cfg.LR_P3_BACKBONE},
                {"params": self.model.proj_head.parameters(), "lr": cfg.LR_P3_HEAD},
                {"params": self.model.clf_head.parameters(), "lr": cfg.LR_P3_HEAD},
            ],
            weight_decay=cfg.WEIGHT_DECAY,
        )

        # Warmup + Cosine schedule for backbone LR
        warmup_epochs = cfg.P3_WARMUP_EPOCHS
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,  # start at 10% of target LR
                total_iters=warmup_epochs,
                # attaches to the backbone param group (index 0)
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.EPOCHS_P3 - warmup_epochs,
                eta_min=1e-6,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.EPOCHS_P3,
                eta_min=1e-6,
            )

        # Smoothed early stopping
        es = SmoothedEarlyStopping(
            patience=cfg.EARLY_STOP_PATIENCE,
            min_delta=cfg.EARLY_STOP_MIN_DELTA,
            smoothing=cfg.EARLY_STOP_SMOOTH,
        )

        for epoch in range(1, cfg.EPOCHS_P3 + 1):
            alpha = cfg.ALPHA_START + (cfg.ALPHA_END - cfg.ALPHA_START) * (epoch - 1) / max(cfg.EPOCHS_P3 - 1, 1)
            t_loss, t_sc, t_clf, t_push = self._train_epoch_phase3(alpha)
            v_loss, macro_f1 = self._val_epoch_phase3()

            if epoch > 1:
                self.scheduler.step()

            self._log_epoch(
                "P3",
                epoch,
                cfg.TOTAL_EPOCHS,
                t_loss,
                v_loss,
                macro_f1,
                alpha=alpha,
                extra=f"SC:{t_sc:.4f} CLF:{t_clf:.4f} push:{t_push:.4f}",
            )

            if macro_f1 > self.best_metric:
                self.best_metric = macro_f1
                self.best_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state, os.path.join(cfg.OUTPUT_DIR, "best_model_v2.pth"))
                es.history.clear()          # reset early-stop counter on genuine improvement
                es._wait = 0

            if es(macro_f1):
                print(f"[Phase 3] Early stopping triggered at epoch {epoch} "
                      f"(smoothed best F1: {es.best_smoothed:.4f})")
                break

        return self.model

    def _train_epoch_phase3(self, alpha):
        self.model.train()
        total_loss, total_sc, total_clf, total_push, n = 0.0, 0.0, 0.0, 0.0, 0
        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                proj_feats, logits = self.model(imgs)
                l_sc   = self.supcon_loss(proj_feats, labels)
                l_clf  = self.clf_loss(logits, labels)
                l_push = self.push_loss(self.model.clf_head.prototypes)
                loss   = alpha * l_sc + (1.0 - alpha) * l_clf + l_push
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss  += loss.item()
            total_sc   += l_sc.item()
            total_clf  += l_clf.item()
            total_push += l_push.item()
            n += 1
        return total_loss / n, total_sc / n, total_clf / n, total_push / n

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

    # ── Restore best state (FC → Prototype compat) ──
    def _restore_best_state(self, proto_init: torch.Tensor):
        """
        Finalise self.model (PrototypeHead) with the best checkpoint.

        Problem:
          Phase 1 model has FC head  → keys: "clf_head.0.weight"
          Phase 2/3 model has PrototypeHead → keys: "clf_head.prototypes"
          run() finishes with a PrototypeHead model but self.best_state may
          contain FC keys (from Phase 1 best) OR prototype keys (from Phase 3 best).

        Solution:
          1. Load best backbone/proj_head weights (strict=False drops FC keys).
          2. Overwrite clf_head.prototypes with the prototypes that were trained
             during Phase 2/3 — they are in the current self.model.state_dict()
             because P3 trains the prototype head to convergence.
        """
        self.model.load_state_dict(self.best_state, strict=False)
        final_state = self.model.state_dict()
        trained_protos = final_state.get("clf_head.prototypes")
        if trained_protos is not None:
            final_state["clf_head.prototypes"] = trained_protos
        self.model.load_state_dict(final_state)

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
        print(
            f"Phase 1: {cfg.EPOCHS_P1}ep CE only | Phase 2: {cfg.EPOCHS_P2}ep Proto | Phase 3: {cfg.EPOCHS_P3}ep Joint"
        )
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

        # Restore best state into a PrototypeHead model.
        # self.model is a MalariaProtoCLFv2 with PrototypeHead (P2/P3 structure).
        # best_state may be from Phase 1 (FC keys) or Phase 2/3 (prototype keys).
        # strict=False drops FC keys silently; prototype keys are loaded normally.
        self._restore_best_state(proto_init)

        # Calibration
        if cfg.DO_CALIBRATION:
            print("\n[Calibration] Fitting temperature scaling...")
            ts = TemperatureScaling(self.model)
            T = ts.fit(self.val_loader, self.device)
            torch.save(
                {
                    "model_state": self.best_state,
                    "temperature": T,
                    "cfg": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
                    "proto_init_summary": {
                        "method": "class_mean_from_phase1_embeddings",
                        "shape": list(proto_init.shape),
                    },
                },
                os.path.join(cfg.OUTPUT_DIR, "calibrated_model_v2.pth"),
            )
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
            "backbone": cfg_dict.get("BACKBONE", "convnext_tiny.in22k_ft_in1k"),
            "num_classes": cfg_dict.get("NUM_CLASSES", 5),
            "proj_dim": cfg_dict.get("PROJ_DIM", 128),
            "use_prototype": cfg_dict.get("USE_PROTOTYPE", True),
            "use_dual_head": use_dual_head,
            "blend_alpha": blend_alpha,
            "pretrained": False,
        }
        temperature = ckpt.get("temperature", 1.0)
        state_dict = ckpt["model_state"]
    else:
        model_cfg = {
            "backbone": "convnext_tiny.in22k_ft_in1k",
            "num_classes": 5,
            "proj_dim": 128,
            "use_prototype": True,
            "use_dual_head": False,
            "pretrained": False,
        }
        temperature = 1.0
        state_dict = ckpt

    model = build_model(model_cfg)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, temperature


if __name__ == "__main__":
    cfg = TrainConfigV2()
    trainer = TrainerV2Curriculum(cfg)
    trainer.run()
