"""
train_v2_improved.py
────────────────────
Cải thiện dựa trên phân tích kết quả:
  1. Giảm alpha (SupCon weight) để CLF loss được học nhiều hơn
  2. EfficientNet-B1 làm backbone (đã proven tốt hơn trên dataset này)
  3. Tăng EPOCHS + LR scheduler ReduceLROnPlateau thay Cosine
  4. Augmentation mạnh hơn cho minority classes (RandAugment)
  5. Class-weighted sampler + mixup augmentation cho parasite classes
  6. Thêm Dropout sau projection head để giảm overfit
  7. TJ có support nhỏ nhất (41) → dùng class weight cao hơn trong focal loss
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
from collections import Counter

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

from calibration import TemperatureScaling
from dataset import CLASS_NAMES, NUM_CLASSES, MalariaDataset, get_transforms
from losses import DynamicFocalLoss, SupConLoss


# ─────────────────────────────────────────────
# Improved ProjectionHead with Dropout
# ─────────────────────────────────────────────
class ProjectionHeadV2(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class PrototypeHead(nn.Module):
    def __init__(self, feat_dim, num_classes, temperature=10.0):
        super().__init__()
        self.temperature = temperature
        self.prototypes  = nn.Parameter(
            F.normalize(torch.randn(num_classes, feat_dim), dim=1)
        )
    def forward(self, z):
        return torch.matmul(z, F.normalize(self.prototypes, dim=1).T) * self.temperature


class MalariaProtoCLFv2(nn.Module):
    def __init__(self, backbone_name, num_classes=5, proj_dim=128,
                 use_prototype=True, pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.proj_head = ProjectionHeadV2(feat_dim, 512, proj_dim, dropout)
        self.clf_head  = (PrototypeHead(proj_dim, num_classes)
                          if use_prototype else nn.Linear(feat_dim, num_classes))
        self.use_prototype = use_prototype

    def forward(self, x):
        feats      = self.backbone(x)
        proj_feats = self.proj_head(feats)
        logits     = self.clf_head(proj_feats if self.use_prototype else feats)
        return proj_feats, logits


# ─────────────────────────────────────────────
# Mixup helper (for minority classes)
# ─────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4, parasite_only=True, parasite_idx=None):
    """Apply Mixup only between parasite class samples."""
    if parasite_only and parasite_idx is not None:
        mask = torch.isin(y, torch.tensor(parasite_idx, device=y.device))
        if mask.sum() < 2:
            return x, y, y, 1.0
        x_para, y_para = x[mask], y[mask]
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x_para.size(0), device=x.device)
        x_mixed         = lam * x_para + (1 - lam) * x_para[idx]
        x[mask]         = x_mixed
        return x, y, y[mask][idx], lam
    else:
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1-lam)*x[idx], y, y[idx], lam


# ─────────────────────────────────────────────
# Config V2
# ─────────────────────────────────────────────
class TrainConfigV2:
    BASE_DIR  = "/kaggle/input/datasets/khanhtq2101/malaria-parasite/final_malaria_full_class_classification_cropped/5 classes - May 2025"
    IMG_BASE  = "/kaggle/input/datasets/khanhtq2101/malaria-parasite/final_malaria_full_class_classification_cropped"
    TRAIN_ANN = os.path.join(BASE_DIR, "train_annotation_5classes.txt")
    VAL_ANN   = os.path.join(BASE_DIR, "val_annotation_5classes.txt")
    TEST_ANN  = os.path.join(BASE_DIR, "test_annotation_5classes.txt")
    OUTPUT_DIR = "/kaggle/working/malaria_proto_v2"

    # ── Key changes vs V1 ──
    # Dùng lại EfficientNet-B1 đã proven tốt hơn
    BACKBONE = "efficientnet_b1.ra4_e3600_r240_in1k"
    # Alternatives nếu muốn thử:
    # "tf_efficientnetv2_s.in21k_ft_in1k"  (EfficientNetV2-S, pretrain IN21k)
    # "convnext_tiny.in22k_ft_in1k"

    NUM_CLASSES     = 5
    PROJ_DIM        = 128
    USE_PROTOTYPE   = True
    IMG_SIZE        = 240          # EfficientNet-B1 native size

    EPOCHS          = 80
    BATCH_SIZE      = 32
    LR              = 2e-4
    WEIGHT_DECAY    = 1e-4
    WARMUP_EPOCHS   = 5
    LR_BACKBONE     = 2e-5
    DROPOUT         = 0.25

    # ── Critical change: giảm alpha SupCon ──
    # V1 alpha=0.4 → CLF loss chỉ 0.6×focal → model learn prototype nhưng không học classifier tốt
    # V2 alpha=0.25 → CLF loss 0.75× → classifier được ưu tiên hơn
    SUPCON_TEMP     = 0.07
    ALPHA           = 0.25         # ← GIẢM từ 0.4 xuống 0.25
    CLF_LOSS        = "focal"
    MAJORITY_CLASS  = 4

    USE_MIXUP       = True
    MIXUP_ALPHA     = 0.4

    DO_CALIBRATION  = True
    SEED            = 42
    NUM_WORKERS     = 4
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_BEST_METRIC = "macro_f1"


# ─────────────────────────────────────────────
# Trainer V2
# ─────────────────────────────────────────────
class TrainerV2:
    def __init__(self, cfg: TrainConfigV2):
        self.cfg = cfg
        torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        self.device = torch.device(cfg.DEVICE)
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self.scaler  = GradScaler("cuda")
        self.best_metric = 0.0
        self.best_state  = None
        self.history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    def _setup_data(self):
        cfg = self.cfg
        train_tf = get_transforms("train", cfg.IMG_SIZE)
        val_tf   = get_transforms("val",   cfg.IMG_SIZE)
        self.train_ds = MalariaDataset(cfg.TRAIN_ANN, cfg.IMG_BASE, transform=train_tf)
        self.val_ds   = MalariaDataset(cfg.VAL_ANN,   cfg.IMG_BASE, transform=val_tf)

        labels = [lbl for _, lbl in self.train_ds.samples]
        counts = Counter(labels)
        self.class_counts = [counts.get(i, 1) for i in range(cfg.NUM_CLASSES)]

        # Weighted sampler: oversample minority
        weights = [len(labels) / counts[lbl] for lbl in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

        self.train_loader = DataLoader(self.train_ds, batch_size=cfg.BATCH_SIZE,
                                       sampler=sampler, num_workers=cfg.NUM_WORKERS,
                                       pin_memory=True, drop_last=True)
        self.val_loader   = DataLoader(self.val_ds,   batch_size=cfg.BATCH_SIZE * 2,
                                       shuffle=False,  num_workers=cfg.NUM_WORKERS,
                                       pin_memory=True)

    def _setup_model(self):
        cfg = self.cfg
        self.model = MalariaProtoCLFv2(
            backbone_name  = cfg.BACKBONE,
            num_classes    = cfg.NUM_CLASSES,
            proj_dim       = cfg.PROJ_DIM,
            use_prototype  = cfg.USE_PROTOTYPE,
            pretrained     = True,
            dropout        = cfg.DROPOUT,
        ).to(self.device)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[Model] {cfg.BACKBONE} | {total/1e6:.1f}M params")

    def _freeze_backbone(self):
        for p in self.model.backbone.parameters(): p.requires_grad = False
        print("[Trainer] Backbone FROZEN")

    def _unfreeze_backbone(self):
        for p in self.model.backbone.parameters(): p.requires_grad = True
        cfg = self.cfg
        self.optimizer = torch.optim.AdamW([
            {"params": self.model.backbone.parameters(),  "lr": cfg.LR_BACKBONE},
            {"params": self.model.proj_head.parameters(), "lr": cfg.LR},
            {"params": self.model.clf_head.parameters(),  "lr": cfg.LR},
        ], weight_decay=cfg.WEIGHT_DECAY)
        # ReduceLROnPlateau: giảm LR khi macro_f1 không tăng
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=8, verbose=True)
        print(f"[Trainer] Backbone UNFROZEN | backbone_lr={cfg.LR_BACKBONE}")

    def _setup_loss(self):
        cfg = self.cfg
        self.supcon_loss = SupConLoss(temperature=cfg.SUPCON_TEMP).to(self.device)
        self.focal_loss  = DynamicFocalLoss(cfg.NUM_CLASSES,
                                            self.class_counts).to(self.device)

    def _setup_optimizer(self):
        cfg = self.cfg
        self._freeze_backbone()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
        )
        self.scheduler = None   # warm-up phase: no scheduler

    def _train_epoch(self, epoch):
        cfg = self.cfg
        self.model.train()
        total_loss = total_sc = total_clf = 0.0
        n = 0

        for imgs, labels in self.train_loader:
            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)

            # Mixup for parasite classes only
            if cfg.USE_MIXUP:
                imgs, labels_a, labels_b, lam = mixup_data(
                    imgs, labels, alpha=cfg.MIXUP_ALPHA,
                    parasite_only=True,
                    parasite_idx=[0, 1, 2, 3]
                )

            self.optimizer.zero_grad()
            with autocast("cuda"):
                proj_feats, logits = self.model(imgs)
                # SupCon always uses original labels
                lsc  = self.supcon_loss(proj_feats, labels)
                # Focal: mixup → mix losses
                if cfg.USE_MIXUP and lam < 1.0:
                    lclf = lam * self.focal_loss(logits, labels_a) + \
                           (1-lam) * self.focal_loss(logits, labels_b)
                else:
                    lclf = self.focal_loss(logits, labels)
                loss = cfg.ALPHA * lsc + (1.0 - cfg.ALPHA) * lclf

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item(); total_sc += lsc.item()
            total_clf  += lclf.item() if isinstance(lclf, torch.Tensor) else lclf
            n += 1

        return total_loss/n, total_sc/n, total_clf/n

    @torch.no_grad()
    def _val_epoch(self):
        from sklearn.metrics import f1_score
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for imgs, labels in self.val_loader:
            imgs = imgs.to(self.device); labels = labels.to(self.device)
            proj_feats, logits = self.model(imgs)
            lsc  = self.supcon_loss(proj_feats, labels)
            lclf = self.focal_loss(logits, labels)
            loss = self.cfg.ALPHA * lsc + (1.0 - self.cfg.ALPHA) * lclf
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        val_loss = total_loss / len(self.val_loader)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return val_loss, macro_f1

    def run(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"Training V2: {cfg.BACKBONE} | alpha={cfg.ALPHA} | Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, cfg.EPOCHS + 1):
            if epoch == cfg.WARMUP_EPOCHS + 1:
                self._unfreeze_backbone()

            t_loss, t_sc, t_clf = self._train_epoch(epoch)
            v_loss, macro_f1    = self._val_epoch()

            if self.scheduler and epoch > cfg.WARMUP_EPOCHS:
                self.scheduler.step(macro_f1)

            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(v_loss)
            self.history["val_macro_f1"].append(macro_f1)

            if macro_f1 > self.best_metric:
                self.best_metric = macro_f1
                self.best_state  = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state,
                           os.path.join(cfg.OUTPUT_DIR, "best_model_v2.pth"))
                tag = "✓ BEST"
            else:
                tag = ""

            print(f"Epoch {epoch:03d}/{cfg.EPOCHS} | "
                  f"T {t_loss:.4f}(SC:{t_sc:.4f} CLF:{t_clf:.4f}) | "
                  f"V {v_loss:.4f} | F1 {macro_f1:.4f} {tag}")

        self.model.load_state_dict(self.best_state)

        if cfg.DO_CALIBRATION:
            print("\n[Calibration] Fitting temperature scaling...")
            ts = TemperatureScaling(self.model)
            T  = ts.fit(self.val_loader, self.device)
            torch.save({
                "model_state": self.best_state,
                "temperature": T,
                "cfg": cfg.__dict__,
            }, os.path.join(cfg.OUTPUT_DIR, "calibrated_model_v2.pth"))
            print(f"[Done] Best macro-F1={self.best_metric:.4f}  T={T:.4f}")

        return self.model


if __name__ == "__main__":
    cfg     = TrainConfigV2()
    trainer = TrainerV2(cfg)
    trainer.run()
