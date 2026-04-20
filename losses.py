"""
losses.py - Custom losses for Confidence-Calibrated Prototype Contrastive Learning
  1. SupConLoss  - Supervised Contrastive Loss (Khosla et al., 2020)
  2. FocalLoss   - Focal Loss with per-class dynamic gamma
  3. AsymmetricLabelSmoothingCE - Label smoothing heavier on majority class
  4. CombinedLoss - Joint SupCon + classification loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (https://arxiv.org/abs/2004.11362).
    Expects normalized feature vectors of shape (B, D) or (B, views, D).

    Changes for v2:
      • Optional online loss normalization (running-mean based) to balance
        gradient magnitude with CE/focal loss in joint training.
      • Clips temperature-aware similarity to avoid overflow.
    """

    def __init__(self, temperature=0.07, contrast_mode="all", normalize_loss=False, target_loss=0.5):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.normalize_loss = normalize_loss
        self.target_loss = target_loss
        self.register_buffer("_run_mean", torch.tensor(0.0))
        self._momentum = 0.9

    def forward(self, features, labels):
        """
        features: (B, D) – already L2-normalized projection outputs
        labels:   (B,)   – integer class labels
        """
        device = features.device
        B = features.shape[0]

        # similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # mask diagonal (self-contrast)
        logits_mask = ~torch.eye(B, dtype=torch.bool, device=device)

        # positive mask: same label, exclude self
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T) & logits_mask

        # for numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        # mean over positives per anchor
        pos_count = pos_mask.float().sum(dim=1)
        mean_log_prob = (pos_mask.float() * log_prob).sum(dim=1) / (pos_count + 1e-9)

        loss = -mean_log_prob
        loss = loss[pos_count > 0].mean()  # only anchors that have positives

        # ── Online loss normalization ──
        # If enabled, scale loss so that its running-mean stays near target_loss.
        # This prevents SupCon loss (~1.75 in v1) from being dwarfed by CE (~0.001).
        if self.normalize_loss and self.training:
            self._run_mean = self._momentum * self._run_mean + (1 - self._momentum) * loss.detach()
            scale = self.target_loss / (self._run_mean + 1e-9)
            loss = loss * scale.clamp(0.1, 10.0)

        return loss


class DynamicFocalLoss(nn.Module):
    """
    Focal Loss with per-class gamma.
    class_counts: list/tensor of sample counts per class (used to set gamma).
    gamma_min / gamma_max: clamp range for computed gamma values.
    """

    def __init__(
        self,
        num_classes,
        class_counts=None,
        class_weights=None,
        gamma_min=0.5,
        gamma_max=5.0,
        reduction="mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

        if class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float)
            freq = counts / counts.sum()
            # rarer class → higher gamma
            raw_gamma = 1.0 / (freq + 1e-9)
            raw_gamma = raw_gamma / raw_gamma.max() * gamma_max
            gammas = raw_gamma.clamp(gamma_min, gamma_max)
        else:
            gammas = torch.ones(num_classes) * 2.0

        self.register_buffer("gammas", gammas)

        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float)
            # Keep average weight near 1.0 to avoid exploding loss scale.
            weights = weights / weights.mean().clamp_min(1e-9)
        else:
            weights = torch.ones(num_classes, dtype=torch.float)
        self.register_buffer("class_weights", weights)

    def forward(self, logits, targets):
        """logits: (B, C), targets: (B,)"""
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # gather p_t and gamma for each sample's true class
        p_t = probs.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
        gamma_t = self.gammas[targets]  # (B,)
        alpha_t = self.class_weights[targets]  # (B,)

        focal_weight = (1.0 - p_t) ** gamma_t
        ce_loss = -log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLabelSmoothingCE(nn.Module):
    """
    Label Smoothing CE that applies HIGHER smoothing to the majority class
    and LOWER (or zero) smoothing to minority classes.
    majority_class: index of the dominant class (e.g. Unparasitized = 4)
    smooth_majority: smoothing ε for majority class (default 0.2)
    smooth_minority: smoothing ε for minority classes (default 0.05)
    """

    def __init__(self, num_classes, majority_class=4, smooth_majority=0.2, smooth_minority=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.majority_class = majority_class
        # build per-class smoothing vector
        eps = torch.full((num_classes,), smooth_minority)
        eps[majority_class] = smooth_majority
        self.register_buffer("eps", eps)

    def forward(self, logits, targets):
        """logits: (B, C), targets: (B,)"""
        log_probs = F.log_softmax(logits, dim=1)

        # one-hot with asymmetric label smoothing
        B, C = logits.shape
        smooth_targets = torch.zeros_like(log_probs)
        eps_per_sample = self.eps[targets]  # (B,)

        for i in range(B):
            e = eps_per_sample[i].item()
            smooth_targets[i] = e / C
            smooth_targets[i, targets[i]] = 1.0 - e + e / C

        loss = -(smooth_targets * log_probs).sum(dim=1)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Joint loss: alpha * SupConLoss + (1-alpha) * ClassificationLoss
    classification_loss: "focal" | "asymmetric_ce" | "ce"
    """

    def __init__(
        self, num_classes, class_counts=None, supcon_temp=0.07, alpha=0.5, classification_loss="focal", majority_class=4
    ):
        super().__init__()
        self.alpha = alpha
        self.supcon = SupConLoss(temperature=supcon_temp)

        if classification_loss == "focal":
            self.clf_loss = DynamicFocalLoss(num_classes, class_counts)
        elif classification_loss == "asymmetric_ce":
            self.clf_loss = AsymmetricLabelSmoothingCE(num_classes, majority_class=majority_class)
        else:
            self.clf_loss = nn.CrossEntropyLoss()

    def forward(self, proj_feats, logits, labels):
        """
        proj_feats: L2-normalized projection vectors (B, D)
        logits:     raw classification logits (B, C)
        labels:     (B,)
        """
        loss_supcon = self.supcon(proj_feats, labels)
        loss_clf = self.clf_loss(logits, labels)
        return self.alpha * loss_supcon + (1.0 - self.alpha) * loss_clf, loss_supcon.item(), loss_clf.item()


# ─────────────────────────────────────────────
# Prototype Push-Away Loss
# ─────────────────────────────────────────────
class ProtoRepulsionLoss(nn.Module):
    """
    Maximizes cosine distance between specific class prototypes.

    Unlike PairConfusionPenalty (penalizes logit overlap per-sample),
    this loss directly pushes prototypes apart in embedding space.

    Use when two classes are easily confused → make their centers far apart.
    For malaria: TJ vs TA (class 0 vs 1) are morphologically similar.

    Loss = -cosine_similarity(proto_i, proto_j)  (push similarity → -1)
    """

    def __init__(self, class_pairs: list[tuple[int, int]], weight: float = 0.1):
        """
        Args:
            class_pairs: list of (class_i, class_j) pairs to push apart
            weight: scalar weight for the combined loss
        """
        super().__init__()
        self.class_pairs = [(int(a), int(b)) for a, b in class_pairs]
        self.weight = weight

    def forward(self, prototypes: nn.Parameter) -> tuple[torch.Tensor, dict]:
        """
        prototypes: (C, D) L2-normalized prototype parameters
        Returns: (weighted_loss, debug_dict)
        """
        P = F.normalize(prototypes, dim=1)  # (C, D)
        total = torch.zeros((), device=P.device)
        activations = 0
        pair_details = {}

        for i, j in self.class_pairs:
            sim = torch.dot(P[i], P[j])  # cosine similarity (normalized)
            # Repulsion: penalize when sim > -margin (i.e. too close or same direction)
            # We want sim ≤ -margin → proto far apart (≥ 90° for margin=0)
            margin = 0.1
            violation = F.relu(sim + margin)  # zero when sim ≤ -margin (good)
            total = total + violation
            activations += (violation.item() > 1e-6)
            pair_details[f"pair_{i}_{j}"] = {
                "sim": sim.item(),
                "dist": (1 - sim).item(),  # cosine distance
                "violation": violation.item(),
            }

        n_pairs = len(self.class_pairs)
        loss = (total / n_pairs) * self.weight
        return loss, {"total_violations": activations, "n_pairs": n_pairs, **pair_details}


# ─────────────────────────────────────────────
# Prototype Push-Away Loss (refactored)
# ─────────────────────────────────────────────
class PrototypePushLoss(nn.Module):
    """
    Penalises excessive cosine similarity between all pairs of class prototypes.

    Usage:
        push_loss = PrototypePushLoss(margin=0.5, weight=0.05)
        loss = push_loss(prototype_params)

    V2 changes from v1:
      • Default margin raised from 0.3 → 0.5 (0.3 was too strict → never fired)
      • Reports activation rate for debugging
      • Uses mean instead of sum (scale-independent)
    """

    def __init__(self, margin: float = 0.5, weight: float = 0.05):
        """
        Args:
            margin:  desired minimum cosine distance between any two prototypes.
                     cos_dist = 1 - sim ≥ margin → sim ≤ 1 - margin
                     Default 0.5 means prototypes must be ≥ 60° apart.
            weight:  scalar weight multiplied onto the loss in the combined objective.
        """
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(self, prototypes: nn.Parameter) -> tuple[torch.Tensor, dict]:
        """
        prototypes: (C, D) L2-normalized class prototype vectors.
        Returns: (weighted_loss, debug_dict)
        """
        P = F.normalize(prototypes, dim=1)  # (C, D)
        sim = torch.matmul(P, P.T)  # (C, C)
        # Zero out diagonal (self-similarity = 1.0 by definition)
        sim = sim - torch.eye(P.shape[0], device=P.device)
        # Penalise any pair with similarity above (1 - margin)
        threshold = 1.0 - self.margin  # sim > threshold → violation
        violation = torch.clamp_min(sim - threshold, min=0.0)
        # Average over all off-diagonal pairs
        n_pairs = P.shape[0] * (P.shape[0] - 1)
        loss = violation.sum() / n_pairs

        # Debug info
        off_diag = sim.abs()  # ignore sign for count
        n_violations = (violation > 1e-6).sum().item()
        return self.weight * loss, {
            "raw_loss": loss.item(),
            "n_violations": n_violations,
            "max_sim": sim.max().item(),
            "min_sim": sim.min().item(),
            "mean_sim": sim.abs().mean().item(),
        }


class PairConfusionPenalty(nn.Module):
    """
    Directional margin penalty for a confusion pair.

    For samples whose true label is `target_class_idx`, penalize when
    `confusing_class_idx` logit is too close to (or higher than) target logit.

    Loss per sample:
        relu(logit_confusing - logit_target + margin)
    """

    def __init__(self, target_class_idx: int, confusing_class_idx: int, margin: float = 0.1):
        super().__init__()
        self.target_class_idx = int(target_class_idx)
        self.confusing_class_idx = int(confusing_class_idx)
        self.margin = float(margin)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C)
        targets: (B,)
        """
        mask = targets == self.target_class_idx
        if not mask.any():
            return logits.new_zeros(())

        target_logits = logits[mask, self.target_class_idx]
        confusing_logits = logits[mask, self.confusing_class_idx]
        loss = F.relu(confusing_logits - target_logits + self.margin)
        return loss.mean()


# ─────────────────────────────────────────────
# Smoothed Early Stopping
# ─────────────────────────────────────────────
class SmoothedEarlyStopping:
    """
    Tracks a rolling smoothed value of a monitored metric and stops when
    no improvement exceeds `min_delta` for `patience` consecutive updates.

    Compared to naive best-check:
      • Smoothing reduces sensitivity to noisy validation fluctuations.
      • Patience gives the model a real chance to recover vs 1-epoch stall.

    Usage:
        es = SmoothedEarlyStopping(patience=8, min_delta=0.002, smoothing=5)
        for epoch in range(1, 61):
            val_f1 = evaluate()
            if es(val_f1):
                print("Early stopping!")
                break
            trainer.train()
    """

    def __init__(self, patience: int = 8, min_delta: float = 0.002, smoothing: int = 5):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing = smoothing
        self.history: list[float] = []

    def __call__(self, value: float) -> bool:
        """
        Returns True if training should stop.
        """
        self.history.append(value)
        if len(self.history) < self.smoothing:
            return False

        smoothed = sum(self.history[-self.smoothing :]) / self.smoothing

        if len(self.history) == self.smoothing:
            self._best_smoothed = smoothed
            self._wait = 0
            return False

        delta = smoothed - self._best_smoothed
        if delta > self.min_delta:
            self._best_smoothed = smoothed
            self._wait = 0
        else:
            self._wait += 1

        return self._wait >= self.patience

    @property
    def best_smoothed(self) -> float:
        return getattr(self, "_best_smoothed", float("-inf"))
