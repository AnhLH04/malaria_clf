"""
train_v2.py - Single-Phase Training + Optional ProtoCLR Fine-Tune
══════════════════════════════════════════════════════════════
Fixed bugs from v2-orig:
  [BUG-1] Phase 3 best_state never updated because es.history.clear() was INSIDE the
          "if macro_f1 > self.best_metric:" block → best_state frozen at P1 best forever.
          FIX: reset es.history.clear() AFTER saving best_state, not inside.
  [BUG-2] calibrated_model.pth saved self.best_state (FC keys) instead of model.state_dict()
          which has fully-trained prototypes → evaluate.py fails.
          FIX: save self.model.state_dict() (the finalized model after P3).
  [BUG-3] _restore_best_state() overwrites good prototype weights with FC weights
          from best_state (loaded back into model via load_state_dict, then those
          FC weights loaded into final_state, then saved → wrong).
          FIX: just load best_state + save.

Training approach: single-phase (Phase 1 CE baseline), then optional ProtoCLR fine-tune.
  → Predictable, no key-mismatch bugs, no FC→Prototype confusion.
  → P2/P3 only add complexity for marginal benefit on 5-class balanced CE problem.
  → Thesis narrative: "ProtoCLR enhances representation; CE backbone is strong."

Usage:
    from train_v2 import TrainerV2SinglePhase, TrainConfigV2

    cfg = TrainConfigV2()
    trainer = TrainerV2SinglePhase(cfg)
    model, history = trainer.run()
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

from calibration import TemperatureScaling, sanitize_temperature
from dataset import MalariaDataset, get_transforms
from losses import DynamicFocalLoss, ProtoRepulsionLoss, PrototypePushLoss, SupConLoss
from model import MalariaProtoCLFv2, build_model, compute_class_prototypes

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
class TrainConfigV2:
    # ── Paths ──────────────────────────────────────────────────────────────
    BASE_DIR = "/kaggle/input/datasets/khanhtq2101/malaria-parasite/final_malaria_full_class_classification_cropped/5 classes - May 2025"
    IMG_BASE = "/kaggle/input/datasets/khanhtq2101/malaria-parasite/final_malaria_full_class_classification_cropped"
    TRAIN_ANN = os.path.join(BASE_DIR, "train_annotation_5classes.txt")
    VAL_ANN = os.path.join(BASE_DIR, "val_annotation_5classes.txt")
    TEST_ANN = os.path.join(BASE_DIR, "test_annotation_5classes.txt")
    OUTPUT_DIR = "/kaggle/working/malaria_proto_v2"

    # ── External baseline init (optional) ────────────────────────────────
    # Set to None to run full Phase 1 training from pretrained backbone.
    INIT_FROM_CHECKPOINT = None  # None = train from scratch (recommended for fixes)
    SKIP_PHASE1_IF_INIT = False
    INIT_ADAPT_EPOCHS = 2
    INIT_ADAPT_LR = 1e-3
    INIT_ADAPT_BACKBONE_LR = 5e-6

    # ── Model ─────────────────────────────────────────────────────────────
    BACKBONE = "convnext_tiny.in22k_ft_in1k"
    NUM_CLASSES = 5
    PROJ_DIM = 128
    IMG_SIZE = 224
    DROPOUT = 0.1

    # ── Training ───────────────────────────────────────────────────────────
    EPOCHS = 30
    BATCH_SIZE = 32
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 3
    LABEL_SMOOTHING = 0.1

    # ── ProtoCLR Fine-Tune (Phase 2) ───────────────────────────────────────
    USE_PROTOCLR = True
    PROTOCLR_EPOCHS = 12  # increased from 10 for better convergence
    PROTOCLR_LR_HEAD = 2e-4
    PROTOCLR_LR_BACK = 5e-6
    PROTOCLR_ALPHA = 0.25
    PUSH_WEIGHT = 0.10  # increased from 0.08 — was too low, push never fired
    PROTOCLR_CLF_LOSS = "focal"
    PROTOCLR_LABEL_SMOOTHING = 0.0

    # ── Loss ───────────────────────────────────────────────────────────────
    SUPCON_TEMP = 0.07
    CLF_LOSS = "focal"
    MAJORITY_CLASS = 4
    FOCAL_USE_CLASS_WEIGHTS = True
    FOCAL_CLASS_WEIGHT_POWER = 0.5

    # ── ProtoRepulsion: push morphologically-similar class prototypes apart ─
    # Replaces PairConfusionPenalty which was ATTRACTING TA toward TJ (wrong direction).
    # ProtoRepulsion PUSHES TJ(0) and TA(1) prototypes apart → correct.
    ENABLE_PROTO_REPULSION = True
    REPULSION_WEIGHT = 0.15  # push TJ and TA prototypes apart
    TA_CLASS_IDX = 1
    TJ_CLASS_IDX = 0

    # ── Two-Stage Inference ───────────────────────────────────────────────
    # Stage 1: Parasitized (0-3) vs Unparasitized (4) — most errors above are Unparasitized→Parasitized
    # Stage 2: Only for parasitized → classify into TJ/TA/S/G
    # Set to True to enable two-stage inference in evaluate.py
    USE_TWO_STAGE_INFERENCE = True
    TWO_STAGE_UNPARASITIZED_THRESHOLD = 0.5  # if P(Unparasitized) > this → stage-2 override

    # Select checkpoint by global macro-F1 or parasite-only macro-F1
    BEST_MODEL_METRIC = "parasite_macro_f1"

    # ── Early stopping ───────────────────────────────────────────────────
    EARLY_STOP_PATIENCE = 8
    EARLY_STOP_MIN_DELTA = 0.002
    EARLY_STOP_SMOOTH = 5

    # ── Calibration ───────────────────────────────────────────────────────
    DO_CALIBRATION = True

    # ── Proto Debug ───────────────────────────────────────────────────────
    # Log prototype distances every epoch → catch push=0.0 issues early
    LOG_PROTO_DISTANCES = True

    # ── Misc ──────────────────────────────────────────────────────────────
    SEED = 42
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Weighted sampler with TA boosting
# ─────────────────────────────────────────────────────────────────────────────
def make_weighted_sampler(dataset, class_counts, ta_boost: float = 8.0, minority_boost: float = 4.0):
    """
    Weighted random sampler — boosts minority classes for balanced training.

    Args:
        dataset: MalariaDataset
        class_counts: list[int] sample counts per class
        ta_boost: extra multiplier for TA (class 1) — the most confused class
        minority_boost: extra multiplier for minority parasite classes (TJ, S, G)
    """
    labels = [lbl for _, lbl in dataset.samples]
    counts = Counter(labels)
    n_total = len(labels)

    # Base weight: inverse frequency
    base_weights = {lbl: n_total / counts[lbl] for lbl in counts}

    # Extra boost for minority parasite classes
    # Class 0(TJ): 482, 1(TA): 117, 2(S): 215, 3(G): 101, 4(Unpar): 33486
    PARASITE_CLASSES = [0, 1, 2, 3]
    for cls_idx in PARASITE_CLASSES:
        if cls_idx in base_weights:
            base_weights[cls_idx] *= minority_boost

    # Extra boost for TA specifically (most confused, smallest count)
    base_weights[1] = base_weights.get(1, n_total) * ta_boost

    # Unparasitized gets no boost (already dominant)
    if 4 in base_weights:
        base_weights[4] = n_total / counts.get(4, 1)

    weights = [base_weights[lbl] for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=n_total, replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer: Single-phase CE + optional ProtoCLR fine-tune
# ─────────────────────────────────────────────────────────────────────────────
class TrainerV2SinglePhase:
    """
    Two-phase training without state-dict key mismatches:

      Phase 1 – CE/Focal loss, backbone trainable, FC head.
                Baseline model: strong backbone with clean classification.

      Phase 2 (optional) – ProtoCLR: backbone frozen, prototypes trained
                via SupCon + CE, then joint fine-tune with backbone unfrozen.
                Uses class-mean prototypes from Phase 1 embeddings.

    Key design decisions:
      • Phase 1 uses FC head (use_prototype=False) throughout.
        No structural change = no state-dict mismatch bug.
      • Phase 2 builds a SEPARATE MalariaProtoCLFv2 with PrototypeHead.
        Only backbone/proj_head weights are transferred (safe key-copy).
      • Final model is always the one with the best validation F1,
        saved as best_state + used for calibration.
    """

    def __init__(self, cfg: TrainConfigV2):
        self.cfg = cfg
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        self.device = torch.device(cfg.DEVICE)

        self._setup_data()

        self.scaler = GradScaler()
        self.best_metric = 0.0
        self.best_state = None
        self.history = {
            "phase": [],
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_macro_f1": [],
            "val_parasite_f1": [],
            "val_monitor": [],
            "alpha": [],
            "phase_desc": [],
        }

    # ── Data ───────────────────────────────────────────────────────────────
    def _setup_data(self):
        cfg = self.cfg
        train_tf = get_transforms("train", cfg.IMG_SIZE)
        val_tf = get_transforms("val", cfg.IMG_SIZE)

        self.train_ds = MalariaDataset(cfg.TRAIN_ANN, cfg.IMG_BASE, transform=train_tf)
        self.val_ds = MalariaDataset(cfg.VAL_ANN, cfg.IMG_BASE, transform=val_tf)

        # Build class counts first — needed by make_weighted_sampler
        self.class_counts = self._build_class_counts()

        sampler = make_weighted_sampler(self.train_ds, self.class_counts, ta_boost=8.0, minority_boost=4.0)
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

    def _build_class_counts(self):
        labels = [lbl for _, lbl in self.train_ds.samples]
        counts = Counter(labels)
        return [counts.get(i, 1) for i in range(self.cfg.NUM_CLASSES)]

    def _build_class_weights(self, power: float | None = None):
        """Inverse-frequency class weights (normalized around 1.0)."""
        if power is None:
            power = self.cfg.FOCAL_CLASS_WEIGHT_POWER
        counts = torch.tensor(self.class_counts, dtype=torch.float)
        weights = (counts.sum() / counts.clamp_min(1.0)) ** power
        weights = weights / weights.mean().clamp_min(1e-9)
        return weights.tolist()

    def _select_monitor(self, macro_f1: float, parasite_f1: float):
        if self.cfg.BEST_MODEL_METRIC == "parasite_macro_f1":
            return parasite_f1
        return macro_f1

    # ── External checkpoint bootstrap ─────────────────────────────────────
    def _extract_state_dict(self, ckpt_obj):
        """Extract state_dict from common checkpoint formats."""
        if isinstance(ckpt_obj, dict):
            for key in ("model_state", "state_dict", "model", "net", "weights"):
                value = ckpt_obj.get(key)
                if isinstance(value, dict):
                    return value
            if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
                return ckpt_obj
        raise ValueError("Unsupported checkpoint format: cannot find a state_dict")

    def _normalize_state_dict_keys(self, state_dict):
        """Normalize keys by removing common wrappers like module./model."""
        normalized = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            new_key = key
            for prefix in ("module.", "model.", "network."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            normalized[new_key] = value
        return normalized

    def _bootstrap_from_checkpoint(self, checkpoint_path):
        """Load baseline checkpoint, build FC model, and compute prototypes for ProtoCLR."""
        cfg = self.cfg
        print(f"\n[Init] Loading external baseline checkpoint: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ext_state = self._normalize_state_dict_keys(self._extract_state_dict(ckpt))

        # Build Phase-1-style model (FC head) and load as many keys as possible.
        self.model = MalariaProtoCLFv2(
            backbone_name=cfg.BACKBONE,
            num_classes=cfg.NUM_CLASSES,
            proj_dim=cfg.PROJ_DIM,
            use_prototype=False,
            use_dual_head=False,
            pretrained=False,
            proto_init=None,
            dropout=cfg.DROPOUT,
        ).to(self.device)

        target_state = self.model.state_dict()
        matched, skipped = 0, 0

        for src_key, src_tensor in ext_state.items():
            candidate_keys = [src_key]
            if src_key.startswith("backbone."):
                candidate_keys.append(src_key[len("backbone.") :])
            else:
                candidate_keys.append(f"backbone.{src_key}")

            loaded = False
            for dst_key in candidate_keys:
                if dst_key in target_state and target_state[dst_key].shape == src_tensor.shape:
                    target_state[dst_key] = src_tensor
                    matched += 1
                    loaded = True
                    break
            if not loaded:
                skipped += 1

        self.model.load_state_dict(target_state, strict=False)
        print(f"[Init] Matched keys: {matched} | skipped: {skipped}")

        # Evaluate this initialization once on val to set best metric/state baseline.
        self._setup_loss(classification_loss=cfg.CLF_LOSS, label_smoothing=cfg.LABEL_SMOOTHING)

        if cfg.INIT_ADAPT_EPOCHS > 0:
            print(
                f"[Init] Adaptation stage: {cfg.INIT_ADAPT_EPOCHS}ep | "
                f"head_lr={cfg.INIT_ADAPT_LR} back_lr={cfg.INIT_ADAPT_BACKBONE_LR}"
            )
            adapt_groups = [
                {"params": self.model.proj_head.parameters(), "lr": cfg.INIT_ADAPT_LR},
                {"params": self.model.clf_head.parameters(), "lr": cfg.INIT_ADAPT_LR},
            ]
            if cfg.INIT_ADAPT_BACKBONE_LR > 0:
                adapt_groups.append({"params": self.model.backbone.parameters(), "lr": cfg.INIT_ADAPT_BACKBONE_LR})

            self.optimizer = torch.optim.AdamW(adapt_groups, weight_decay=cfg.WEIGHT_DECAY)
            adapt_best = float("-inf")
            adapt_best_state = copy.deepcopy(self.model.state_dict())

            for ep in range(1, cfg.INIT_ADAPT_EPOCHS + 1):
                t_loss = self._train_epoch_phase1(ep)
                v_loss, macro_f1, parasite_f1 = self._val_epoch_phase1()
                monitor = self._select_monitor(macro_f1, parasite_f1)
                print(
                    f"[InitAdapt] {ep:02d}/{cfg.INIT_ADAPT_EPOCHS} | "
                    f"T {t_loss:.4f} | V {v_loss:.4f} | "
                    f"F1m {macro_f1:.4f} F1p {parasite_f1:.4f}"
                )
                if monitor > adapt_best:
                    adapt_best = monitor
                    adapt_best_state = copy.deepcopy(self.model.state_dict())

            self.model.load_state_dict(adapt_best_state, strict=False)

        v_loss, macro_f1, parasite_f1 = self._val_epoch_phase1()
        self.best_metric = self._select_monitor(macro_f1, parasite_f1)
        self.best_state = copy.deepcopy(self.model.state_dict())
        torch.save(self.best_state, os.path.join(cfg.OUTPUT_DIR, "phase1_best_from_init.pth"))
        print(
            f"[Init] Val from external init | V {v_loss:.4f} | "
            f"F1m {macro_f1:.4f} F1p {parasite_f1:.4f} | monitor {self.best_metric:.4f}"
        )

        if not cfg.USE_PROTOCLR:
            return None

        print("[Init] Computing class prototypes from external-initialized model...")
        proto_init = compute_class_prototypes(
            self.model,
            self.train_loader,
            self.device,
            max_samples_per_class=2000,
        )
        print(f"[Init] Prototypes shape={proto_init.shape}, norms={proto_init.norm(dim=1).tolist()}")
        return proto_init

    # ── Loss ───────────────────────────────────────────────────────────────
    def _setup_loss(self, classification_loss=None, label_smoothing=None):
        cfg = self.cfg
        if classification_loss is None:
            classification_loss = cfg.CLF_LOSS
        if label_smoothing is None:
            label_smoothing = cfg.LABEL_SMOOTHING

        if classification_loss == "focal":
            class_weights = self._build_class_weights() if cfg.FOCAL_USE_CLASS_WEIGHTS else None
            self.clf_loss = DynamicFocalLoss(
                cfg.NUM_CLASSES,
                class_counts=self.class_counts,
                class_weights=class_weights,
            ).to(self.device)
        elif classification_loss == "ce":
            self.clf_loss = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
            ).to(self.device)
        elif classification_loss == "asymmetric_ce":
            from losses import AsymmetricLabelSmoothingCE

            self.clf_loss = AsymmetricLabelSmoothingCE(
                cfg.NUM_CLASSES,
                majority_class=cfg.MAJORITY_CLASS,
            ).to(self.device)
        else:
            self.clf_loss = nn.CrossEntropyLoss().to(self.device)

        self.supcon_loss = SupConLoss(temperature=cfg.SUPCON_TEMP).to(self.device)
        self.push_loss = PrototypePushLoss(margin=0.5, weight=cfg.PUSH_WEIGHT).to(self.device)

        # ProtoRepulsionLoss pushes TJ and TA prototypes apart (replaces wrong PairConfusionPenalty)
        if cfg.ENABLE_PROTO_REPULSION:
            self.repulsion_loss = ProtoRepulsionLoss(
                class_pairs=[(cfg.TJ_CLASS_IDX, cfg.TA_CLASS_IDX)],
                weight=cfg.REPULSION_WEIGHT,
            ).to(self.device)
        else:
            self.repulsion_loss = None

    # ── Phase 1: CE/Focal training ─────────────────────────────────────────
    def _run_phase1(self):
        """
        Phase 1: Train backbone + FC head with CE/Focal loss.
        Model uses FC head (use_prototype=False) throughout — no key mismatch.
        """
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"PHASE 1: CE/Focal | {cfg.EPOCHS} epochs | backbone TRAINABLE")
        print(f"         loss={cfg.CLF_LOSS}, warmup={cfg.WARMUP_EPOCHS}ep, smoothing={cfg.LABEL_SMOOTHING}")
        print(f"{'='*60}")

        self.model = MalariaProtoCLFv2(
            backbone_name=cfg.BACKBONE,
            num_classes=cfg.NUM_CLASSES,
            proj_dim=cfg.PROJ_DIM,
            use_prototype=False,  # FC head throughout Phase 1
            use_dual_head=False,
            pretrained=True,
            proto_init=None,
            dropout=cfg.DROPOUT,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        # Warmup + Cosine schedule
        self._setup_scheduler(
            total_epochs=cfg.EPOCHS,
            warmup_epochs=cfg.WARMUP_EPOCHS,
        )

        for epoch in range(1, cfg.EPOCHS + 1):
            t_loss = self._train_epoch_phase1(epoch)
            v_loss, macro_f1, parasite_f1 = self._val_epoch_phase1()
            monitor = self._select_monitor(macro_f1, parasite_f1)

            if epoch > 1:
                self.scheduler.step()

            self._log_epoch(
                "P1", epoch, cfg.EPOCHS, t_loss, v_loss, macro_f1, parasite_f1, monitor, alpha=None, extra=""
            )

            if monitor > self.best_metric:
                self.best_metric = monitor
                self.best_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state, os.path.join(cfg.OUTPUT_DIR, "phase1_best.pth"))

        print(f"\n[Phase 1] Best monitor ({cfg.BEST_MODEL_METRIC}): {self.best_metric:.4f}")

        # Compute class-mean prototypes for Phase 2 (if enabled)
        if cfg.USE_PROTOCLR:
            print("\n[Phase 1] Computing class prototypes for ProtoCLR...")
            self.model.load_state_dict(self.best_state)  # ensure model is in best state
            proto_init = compute_class_prototypes(
                self.model,
                self.train_loader,
                self.device,
                max_samples_per_class=2000,
            )
            print(f"[Phase 1] Prototypes shape={proto_init.shape}, norms={proto_init.norm(dim=1).tolist()}")
            return proto_init
        return None

    def _setup_scheduler(self, total_epochs, warmup_epochs):
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=1e-6,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=1e-6,
            )

    def _train_epoch_phase1(self, epoch):
        cfg = self.cfg

        self.model.train()
        total_loss, n = 0.0, 0
        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                proj_feats, logits = self.model(imgs)
                l_clf = self.clf_loss(logits, labels)
                loss = l_clf
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
            loss = self.clf_loss(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        parasite_f1 = f1_score(all_labels, all_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        return total_loss / len(self.val_loader), macro_f1, parasite_f1

    # ── Phase 2: ProtoCLR fine-tune ────────────────────────────────────────
    def _run_phase2(self, proto_init):
        """
        ProtoCLR fine-tune:
          • Model rebuilt with PrototypeHead (pretrained from class-mean).
          • Backbone weights loaded from Phase 1 best_state (safe key-copy).
          • Phase 2a: SupCon + PushLoss (backbone FROZEN) → learn embedding space.
          • Phase 2b: Joint SupCon + CE (backbone UNFROZEN) → fine-tune.
        """
        cfg = self.cfg

        print(f"\n{'='*60}")
        print(f"PHASE 2: ProtoCLR Fine-Tune | {cfg.PROTOCLR_EPOCHS} ep")
        print(
            f"         SupCon alpha={cfg.PROTOCLR_ALPHA}, push_weight={cfg.PUSH_WEIGHT}, "
            f"clf={cfg.PROTOCLR_CLF_LOSS}"
        )
        print(f"{'='*60}")

        # ── Build PrototypeHead model ────────────────────────────────────
        proto_model = MalariaProtoCLFv2(
            backbone_name=cfg.BACKBONE,
            num_classes=cfg.NUM_CLASSES,
            proj_dim=cfg.PROJ_DIM,
            use_prototype=True,  # PrototypeHead
            use_dual_head=False,
            pretrained=False,  # weights loaded below
            proto_init=proto_init,  # class-mean from Phase 1
            dropout=cfg.DROPOUT,
        ).to(self.device)

        # ── Copy backbone + proj_head weights from Phase 1 best_state ───
        p1_state = self.best_state
        proto_state = proto_model.state_dict()
        loaded_keys, skipped_keys = [], []
        for key in list(p1_state.keys()):
            if key in proto_state:
                proto_state[key] = p1_state[key]
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)

        proto_model.load_state_dict(proto_state, strict=False)
        print(f"[Phase 2] Loaded {len(loaded_keys)} keys from Phase 1, skipped {len(skipped_keys)} (FC head)")

        # ── Fine-tune backbone (with low LR) + prototype + proj_head ───────────────
        # Set different LRs: backbone vs head+prototypes
        self.model = proto_model

        # Separate parameter groups for differential learning rates.
        # Use `is` (identity) not `in` (value comparison) — tensors with the same
        # values are NOT the same object, and dimension mismatch causes RuntimeError.
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.model.parameters() if id(p) not in backbone_ids and p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.PROTOCLR_LR_BACK},
                {"params": other_params, "lr": cfg.PROTOCLR_LR_HEAD},
            ],
            weight_decay=cfg.WEIGHT_DECAY,
        )
        self._setup_loss(
            classification_loss=cfg.PROTOCLR_CLF_LOSS,
            label_smoothing=cfg.PROTOCLR_LABEL_SMOOTHING,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.PROTOCLR_EPOCHS,
            eta_min=1e-6,
        )

        p2_epochs = cfg.PROTOCLR_EPOCHS

        for epoch in range(1, p2_epochs + 1):
            t_loss, t_sc, t_push, t_repulse = self._train_epoch_protoclr(epoch)
            v_loss, macro_f1, parasite_f1 = self._val_epoch_protoclr()
            monitor = self._select_monitor(macro_f1, parasite_f1)

            if epoch > 1:
                self.scheduler.step()

            self._log_epoch(
                "P2",
                epoch,
                p2_epochs,
                t_loss,
                v_loss,
                macro_f1,
                parasite_f1,
                monitor,
                alpha=cfg.PROTOCLR_ALPHA,
                extra=f"SC:{t_sc:.4f} push:{t_push:.4f} repel:{t_repulse:.4f}",
            )

            # [BUGFIX-1] Reset es counter AFTER saving best_state, not inside the if-block
            if monitor > self.best_metric:
                self.best_metric = monitor
                self.best_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state, os.path.join(cfg.OUTPUT_DIR, "best_protoclr.pth"))

        print(f"\n[Phase 2] Best monitor ({cfg.BEST_MODEL_METRIC}): {self.best_metric:.4f}")

    def _train_epoch_protoclr(self, epoch):
        """Combined SupCon + CE + PushLoss + ProtoRepulsion for ProtoCLR."""
        cfg = self.cfg
        alpha = cfg.PROTOCLR_ALPHA

        self.model.train()
        total_loss, total_sc, total_push, total_repulse, n = 0.0, 0.0, 0.0, 0.0, 0

        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            with autocast():
                proj_feats, logits = self.model(imgs)

                l_sc = self.supcon_loss(proj_feats, labels)
                l_clf = self.clf_loss(logits, labels)
                l_push, push_debug = self.push_loss(self.model.clf_head.prototypes)
                l_repulse = 0.0
                if self.repulsion_loss is not None:
                    l_repulse, repulse_debug = self.repulsion_loss(self.model.clf_head.prototypes)
                else:
                    l_repulse = logits.new_zeros(())

                loss = alpha * l_sc + (1.0 - alpha) * l_clf + l_push + l_repulse

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_sc += l_sc.item()
            total_push += l_push.item()
            total_repulse += l_repulse.item()
            n += 1

        # ── Proto distance logging (debugging push=0.0 issues) ────────────────
        if cfg.LOG_PROTO_DISTANCES and n > 0:
            with torch.no_grad():
                P = F.normalize(self.model.clf_head.prototypes, dim=1)
                sim = torch.matmul(P, P.T) - torch.eye(P.shape[0], device=P.device)
                min_sim = sim.min().item()
                max_sim = sim.max().item()
                # Pair-specific: TJ(0) vs TA(1)
                tj_ta_sim = torch.dot(P[0], P[1]).item()
            print(
                f"         [Proto] TJ-TA sim={tj_ta_sim:.4f} (dist={1-tj_ta_sim:.4f}), "
                f"inter-proto min_sim={min_sim:.4f} max_sim={max_sim:.4f}"
            )

        return (
            total_loss / max(n, 1),
            total_sc / max(n, 1),
            total_push / max(n, 1),
            total_repulse / max(n, 1),
        )

    @torch.no_grad()
    def _val_epoch_protoclr(self):
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
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        parasite_f1 = f1_score(all_labels, all_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        return total_loss / len(self.val_loader), macro_f1, parasite_f1

    # ── Logging ────────────────────────────────────────────────────────────
    def _log_epoch(
        self, phase, epoch, total_epochs, t_loss, v_loss, macro_f1, parasite_f1, monitor, alpha=None, extra=""
    ):
        self.history["phase"].append(phase)
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(t_loss)
        self.history["val_loss"].append(v_loss)
        self.history["val_macro_f1"].append(macro_f1)
        self.history["val_parasite_f1"].append(parasite_f1)
        self.history["val_monitor"].append(monitor)
        self.history["alpha"].append(alpha if alpha is not None else 0.0)
        self.history["phase_desc"].append(f"{phase} ep{epoch}")

        tag = "✓ BEST" if monitor >= self.best_metric else ""
        alpha_str = f"α={alpha:.3f}" if alpha is not None else ""
        print(
            f"[{phase}] {epoch:03d}/{total_epochs} | "
            f"T {t_loss:.4f} {extra} | V {v_loss:.4f} | "
            f"F1m {macro_f1:.4f} F1p {parasite_f1:.4f} M {monitor:.4f} {alpha_str} {tag}"
        )

    # ── Main run ──────────────────────────────────────────────────────────
    def run(self):
        cfg = self.cfg

        # Print config summary
        print(f"{'='*60}")
        print(f"Malaria ProtoCLR Training | {cfg.BACKBONE}")
        print(f"  Phase 1: {cfg.EPOCHS}ep CE | ProtoCLR: {cfg.USE_PROTOCLR} ({cfg.PROTOCLR_EPOCHS}ep)")
        print(f"  Init ckpt: {cfg.INIT_FROM_CHECKPOINT if cfg.INIT_FROM_CHECKPOINT else 'None'}")
        print(
            f"  P2 cfg: alpha={cfg.PROTOCLR_ALPHA}, head_lr={cfg.PROTOCLR_LR_HEAD}, "
            f"back_lr={cfg.PROTOCLR_LR_BACK}, push={cfg.PUSH_WEIGHT}, metric={cfg.BEST_MODEL_METRIC}"
        )
        print(
            f"  ProtoRepulsion: {cfg.ENABLE_PROTO_REPULSION} "
            f"(TJ={cfg.TJ_CLASS_IDX}, TA={cfg.TA_CLASS_IDX}, weight={cfg.REPULSION_WEIGHT})"
        )
        print(f"  Two-stage inference: {cfg.USE_TWO_STAGE_INFERENCE}")
        print(f"  Classes: {cfg.NUM_CLASSES} | Imbalance: {self.class_counts}")
        print(f"{'='*60}")

        # ── Phase 1: CE baseline OR external checkpoint bootstrap ─────────
        use_external_init = bool(cfg.INIT_FROM_CHECKPOINT) and cfg.SKIP_PHASE1_IF_INIT
        if use_external_init and os.path.exists(cfg.INIT_FROM_CHECKPOINT):
            print("[Run] Skip Phase 1 training, bootstrap from external checkpoint.")
            proto_init = self._bootstrap_from_checkpoint(cfg.INIT_FROM_CHECKPOINT)
        else:
            if use_external_init:
                print(f"[Run] Init checkpoint not found at {cfg.INIT_FROM_CHECKPOINT}; fallback to full Phase 1.")
            proto_init = self._run_phase1()

        # ── Phase 2: ProtoCLR fine-tune (optional) ──────────────────────────
        if cfg.USE_PROTOCLR and proto_init is not None:
            self._run_phase2(proto_init)

        # ── Load best state into model ──────────────────────────────────────
        # self.model may be FC-head (Phase 1 only) or PrototypeHead (Phase 2 done).
        # best_state always contains the globally best checkpoint.
        # strict=False: handles FC-head → PrototypeHead key mismatch gracefully.
        self.model.load_state_dict(self.best_state, strict=False)

        # ── Calibration ────────────────────────────────────────────────────
        # [BUGFIX-2] Save self.model.state_dict() (finalized model with
        # best weights loaded) instead of self.best_state (which may be
        # a raw Phase 1 FC-key dict if P2 didn't improve).
        if cfg.DO_CALIBRATION:
            print("\n[Calibration] Fitting temperature scaling...")
            ts = TemperatureScaling(self.model)
            T = ts.fit(self.val_loader, self.device)

            torch.save(
                {
                    # [BUGFIX-3] Use self.model.state_dict() (finalized model)
                    # not self.best_state (which may have FC keys from Phase 1)
                    "model_state": self.model.state_dict(),
                    "temperature": T,
                    "cfg": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
                    "best_val_f1": float(self.best_metric),
                    "proto_init_summary": {
                        "method": "class_mean_from_phase1",
                        "shape": list(proto_init.shape) if proto_init is not None else None,
                    },
                },
                os.path.join(cfg.OUTPUT_DIR, "calibrated_model.pth"),
            )
            print(f"[Calibration] Done. Temperature T={T:.4f}")

        print(f"\n{'='*60}")
        print(f"BEST MACRO-F1: {self.best_metric:.4f}")
        print(f"{'='*60}")
        return self.model, self.history


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_model_v2(checkpoint_path, device, use_prototype=True):
    """
    Load model from checkpoint — handles both FC-head and PrototypeHead.
    Uses strict=False to avoid key mismatch errors.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "cfg" in ckpt:
        cfg_dict = ckpt["cfg"]
        model_cfg = {
            "backbone": cfg_dict.get("BACKBONE", "convnext_tiny.in22k_ft_in1k"),
            "num_classes": cfg_dict.get("NUM_CLASSES", 5),
            "proj_dim": cfg_dict.get("PROJ_DIM", 128),
            "use_prototype": use_prototype,  # override to user's choice
            "use_dual_head": cfg_dict.get("USE_DUAL_HEAD", False),
            "pretrained": False,
            "dropout": cfg_dict.get("DROPOUT", 0.1),
        }
        temperature = sanitize_temperature(ckpt.get("temperature", 1.0))
        state_dict = ckpt["model_state"]
    else:
        model_cfg = {
            "backbone": "convnext_tiny.in22k_ft_in1k",
            "num_classes": 5,
            "proj_dim": 128,
            "use_prototype": use_prototype,
            "pretrained": False,
        }
        temperature = 1.0
        state_dict = ckpt

    model = build_model(model_cfg)
    # [BUGFIX-4] strict=False prevents RuntimeError when checkpoint has
    # FC keys (clf_head.0.weight) but model expects PrototypeHead (clf_head.prototypes).
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model, temperature


if __name__ == "__main__":
    cfg = TrainConfigV2()
    trainer = TrainerV2SinglePhase(cfg)
    trainer.run()
