"""
model.py - Prototype Contrastive Classification Model v2
═════════════════════════════════════════════════════════
Architecture:
  Backbone (timm) → Global Pool → Projection Head (MLP, L2-norm) → Prototype Head
                                                              ↘ Classification Head (FC)

Changes from v1:
  • MalariaProtoCLFv2: get_embeddings() đúng cách, thêm LayerNorm + Dropout
  • compute_class_prototypes(): trích embeddings từ pretrained backbone, tính class-mean
  • init_prototypes_from_pretrained(): khởi tạo prototypes = class-mean embeddings
  • DualHeadCLF: hybrid prototype + FC head, compare hoặc ensemble được
  • PrototypeHead.get_distances(): trả cosine distance cho prototype-based confidence

Supports backbones:
  - convnext_tiny.in22k_ft_in1k       (RECOMMENDED)
  - convnextv2_tiny.fcmae_ft_in22k_in1k
  - efficientnet_b1.ra4_e3600_r240_in1k
  - efficientnet_b2.ra_in1k
  - efficientnetv2_s.in21k_ft_in1k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────
# Projection Head
# ─────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """MLP projection head — LayerNorm + GELU + Dropout, L2-normalized output."""

    def __init__(self, in_dim, hidden_dim=512, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)


# ─────────────────────────────────────────────
# Prototype Head
# ─────────────────────────────────────────────
class PrototypeHead(nn.Module):
    """
    Cosine-similarity based classification via learnable prototypes.
    Mỗi class có 1 prototype vector; logits = temperature * cosine_sim.
    """

    def __init__(self, feat_dim, num_classes, temperature=10.0, init_prototypes=None):
        super().__init__()
        self.temperature = temperature
        if init_prototypes is not None:
            # pretrained prototypes: đã L2-normalized
            init = init_prototypes if init_prototypes.dim() == 1 else F.normalize(init_prototypes, dim=1)
            if init.shape[0] != num_classes:
                raise ValueError(f"init_prototypes shape {init.shape} mismatch with num_classes={num_classes}")
            self.prototypes = nn.Parameter(F.normalize(init, dim=1))
        else:
            self.prototypes = nn.Parameter(F.normalize(torch.randn(num_classes, feat_dim), dim=1))

    def forward(self, z):
        """z: (B, feat_dim) — L2-normalized projection vectors."""
        proto_norm = F.normalize(self.prototypes, dim=1)  # (num_classes, feat_dim)
        sim = torch.matmul(z, proto_norm.T)              # (B, num_classes)
        return sim * self.temperature

    def get_distances(self, z):
        """
        Trả về cosine distance (1 - sim) cho tất cả classes.
        Dùng cho prototype-based confidence scoring.
        z: (B, feat_dim)
        Returns: (B, num_classes) cosine distance
        """
        proto_norm = F.normalize(self.prototypes, dim=1)
        sim = torch.matmul(z, proto_norm.T)
        return 1 - sim  # cosine distance

    def get_distances_to_target(self, z, target_idx):
        """Cosine distance tới prototype của class target_idx."""
        proto_norm = F.normalize(self.prototypes, dim=1)
        sim = torch.matmul(z, proto_norm[:, target_idx:target_idx+1])  # (B, 1)
        return 1 - sim.squeeze(-1)


# ─────────────────────────────────────────────
# Dual Head (Prototype + FC)
# ─────────────────────────────────────────────
class DualHeadCLF(nn.Module):
    """
    Hybrid head: kết hợp Prototype Head và FC Head.
    • proto_logits  = cosine_sim(embedding, prototype) * temperature
    • fc_logits     = Linear(embedding, num_classes)
    • combined      = alpha * proto + (1-alpha) * fc
    """

    def __init__(self, feat_dim, num_classes, temperature=10.0,
                 proto_init=None, fc_dropout=0.2, blend_alpha=0.5):
        super().__init__()
        self.blend_alpha = blend_alpha
        self.proto_head = PrototypeHead(
            feat_dim, num_classes, temperature, init_prototypes=proto_init
        )
        self.fc_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(fc_dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, z, return_both=False):
        proto_logits = self.proto_head(z)
        fc_logits    = self.fc_head(z)
        combined     = self.blend_alpha * proto_logits + (1 - self.blend_alpha) * fc_logits
        if return_both:
            return combined, proto_logits, fc_logits
        return combined


# ─────────────────────────────────────────────
# Main Model v2
# ─────────────────────────────────────────────
class MalariaProtoCLFv2(nn.Module):
    """
    MalariaProtoCLF version 2.

    Args:
        backbone_name:      timm model name
        num_classes:        5
        proj_dim:           projection output dim (default 128)
        use_prototype:      True → PrototypeHead, False → FC Head
        use_dual_head:      True → DualHeadCLF (prototype + FC blend)
        pretrained:         load pretrained backbone weights
        proto_init:         (num_classes, proj_dim) tensor hoặc None
        blend_alpha:        weight for prototype logits in dual head
        dropout:            dropout rate in projection head
    """

    def __init__(self,
                 backbone_name="convnext_tiny.in22k_ft_in1k",
                 num_classes=5,
                 proj_dim=128,
                 use_prototype=True,
                 use_dual_head=False,
                 pretrained=True,
                 proto_init=None,
                 blend_alpha=0.5,
                 dropout=0.1):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        self.proj_head = ProjectionHead(feat_dim, hidden_dim=512, out_dim=proj_dim, dropout=dropout)

        if use_dual_head:
            self.clf_head = DualHeadCLF(
                feat_dim=proj_dim,
                num_classes=num_classes,
                temperature=10.0,
                proto_init=proto_init,
                blend_alpha=blend_alpha,
            )
        elif use_prototype:
            self.clf_head = PrototypeHead(
                feat_dim=proj_dim,
                num_classes=num_classes,
                temperature=10.0,
                init_prototypes=proto_init,
            )
        else:
            self.clf_head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes),
            )

        self.use_prototype = use_prototype
        self.use_dual_head = use_dual_head
        self.feat_dim      = feat_dim
        self.proj_dim      = proj_dim

    def forward(self, x, return_both_logits=False):
        """
        x: (B, 3, H, W)
        return_both_logits: True → DualHeadCLF trả về cả proto + fc logits
        Returns: (proj_feats, logits) hoặc (proj_feats, logits, proto_logits, fc_logits)
        """
        feats      = self.backbone(x)
        proj_feats = self.proj_head(feats)

        if return_both_logits and self.use_dual_head:
            logits, proto_logits, fc_logits = self.clf_head(proj_feats, return_both=True)
            return proj_feats, logits, proto_logits, fc_logits
        else:
            logits = self.clf_head(proj_feats)
            return proj_feats, logits

    @torch.no_grad()
    def get_embeddings(self, x):
        """
        Trả về projection embeddings và logits.
        Dùng cho prototype initialization và analysis.
        """
        self.eval()
        return self.forward(x)

    @torch.no_grad()
    def get_proto_distances(self, x):
        """
        Trả về cosine distance tới tất cả prototypes.
        Dùng cho prototype-based confidence scoring.
        Returns: (B, num_classes) cosine distance
        """
        self.eval()
        proj_feats, logits = self.forward(x)
        if hasattr(self.clf_head, 'get_distances'):
            return self.clf_head.get_distances(proj_feats)
        return None


# ─────────────────────────────────────────────
# Backward-compatible alias (giữ nguyên tên cũ)
# ─────────────────────────────────────────────
class MalariaProtoCLF(MalariaProtoCLFv2):
    """Backward-compatible alias."""
    pass


# ─────────────────────────────────────────────
# Prototype Initialization Utilities
# ─────────────────────────────────────────────
def compute_class_prototypes(
    model: MalariaProtoCLFv2,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples_per_class: int = 2000,
) -> torch.Tensor:
    """
    Trích embeddings của training set, tính class-mean embeddings,
    L2-normalize → dùng làm initial prototypes cho PrototypeHead.

    Args:
        model:               MalariaProtoCLFv2 đã có backbone + proj_head trained
        dataloader:          DataLoader của training set (không cần label transform)
        device:              torch device
        max_samples_per_class: giới hạn samples per class để tránh OOM
    Returns:
        prototypes:         (num_classes, proj_dim) tensor, L2-normalized
    """
    model.eval()
    from collections import defaultdict
    all_embeddings: dict[int, list[torch.Tensor]] = defaultdict(list)

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs   = imgs.to(device)
            labels_np = labels.numpy()

            proj_feats, _ = model.get_embeddings(imgs)  # (B, proj_dim)
            proj_feats = F.normalize(proj_feats, dim=1)

            for emb, lbl in zip(proj_feats.cpu(), labels_np):
                if len(all_embeddings[int(lbl)]) < max_samples_per_class:
                    all_embeddings[int(lbl)].append(emb)

    # Compute mean embedding per class
    prototypes_list = []
    for cls_idx in sorted(all_embeddings.keys()):
        embs = torch.stack(all_embeddings[cls_idx])   # (N, proj_dim)
        mean_emb = embs.mean(dim=0)
        mean_emb = F.normalize(mean_emb, dim=0)
        prototypes_list.append(mean_emb)

    prototypes = torch.stack(prototypes_list)  # (num_classes, proj_dim)
    return prototypes


def compute_class_prototypes_from_backbone(
    backbone: nn.Module,
    proj_head: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples_per_class: int = 2000,
) -> torch.Tensor:
    """
    Trích embeddings TỪ BACKBONE THUẦN (chưa train), tính class-mean.
    Dùng để init prototypes ngay từ đầu mà không cần Phase 1.

    Cách dùng:
        proto_init = compute_class_prototypes_from_backbone(backbone, proj_head,
                                                            train_loader, device)
        # proto_init.shape = (5, 128)
        model = MalariaProtoCLFv2(..., proto_init=proto_init)
    """
    backbone.eval()
    proj_head.eval()
    from collections import defaultdict
    all_embeddings: dict[int, list[torch.Tensor]] = defaultdict(list)

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels_np = labels.numpy()
            feats = backbone(imgs)
            proj  = proj_head(feats)
            proj  = F.normalize(proj, dim=1)

            for emb, lbl in zip(proj.cpu(), labels_np):
                if len(all_embeddings[int(lbl)]) < max_samples_per_class:
                    all_embeddings[int(lbl)].append(emb)

    prototypes_list = []
    for cls_idx in sorted(all_embeddings.keys()):
        embs = torch.stack(all_embeddings[cls_idx])
        mean_emb = embs.mean(dim=0)
        mean_emb = F.normalize(mean_emb, dim=0)
        prototypes_list.append(mean_emb)

    return torch.stack(prototypes_list)


# ─────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────
def build_model(cfg: dict) -> MalariaProtoCLFv2:
    """
    Build model từ config dict.
    Hỗ trợ cả config cũ (v1) và config mới (v2).
    """
    use_dual_head = cfg.get("use_dual_head", False)
    proto_init    = cfg.get("proto_init", None)

    model = MalariaProtoCLFv2(
        backbone_name  = cfg.get("backbone", "convnext_tiny.in22k_ft_in1k"),
        num_classes    = cfg.get("num_classes", 5),
        proj_dim       = cfg.get("proj_dim", 128),
        use_prototype  = cfg.get("use_prototype", True),
        use_dual_head  = use_dual_head,
        pretrained     = cfg.get("pretrained", True),
        proto_init     = proto_init,
        blend_alpha    = cfg.get("blend_alpha", 0.5),
        dropout        = cfg.get("dropout", 0.1),
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {cfg.get('backbone')} | params: {total/1e6:.1f}M | "
          f"trainable: {trainable/1e6:.1f}M | dual_head={use_dual_head}")
    return model
