"""
calibration.py - Post-hoc Temperature Scaling + Prototype-based Confidence Scoring
════════════════════════════════════════════════════════════════════════════════
V2 additions:
  • PrototypeConfidenceScorer: thay softmax-max bằng prototype-based metrics
    - margin_confidence: pred_prob - second_prob
    - proto_distance_ratio: dist_to_pred / (dist_to_pred + dist_to_second)
    - normalized_entropy: 1 - H(p) / log(n_classes)
    - proto_margin: dist_to_pred - dist_to_true (âm = confident đúng, dương = nhầm)
  • CalibrationWithProtoConfidence: TemperatureScaling + prototype distance metrics
  • compute_ece: đã có từ v1
  • reliability_diagram_data: đã có từ v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np


# ─────────────────────────────────────────────
# Temperature Scaling (giữ nguyên từ v1)
# ─────────────────────────────────────────────
class TemperatureScaling(nn.Module):
    """
    Wraps a trained MalariaProtoCLF and learns a single scalar T
    that minimizes NLL on the validation set.
    """

    def __init__(self, model):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        _, logits = self.model(x)
        return self.scale(logits)

    def scale(self, logits):
        return logits / self.temperature.clamp(min=1e-2)

    def fit(self, val_loader, device, lr=0.01, max_iter=100):
        """Optimize temperature on val_loader."""
        self.model.eval()
        self.model.to(device)
        self.to(device)

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                _, logits = self.model(imgs)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        all_logits = all_logits.to(device)
        all_labels = all_labels.to(device)

        def closure():
            optimizer.zero_grad()
            scaled = self.scale(all_logits)
            loss = nll_criterion(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"[TemperatureScaling] Optimal T = {self.temperature.item():.4f}")
        return self.temperature.item()


# ─────────────────────────────────────────────
# Prototype-based Confidence Scoring (V2)
# ─────────────────────────────────────────────
class PrototypeConfidenceScorer:
    """
    Thay vì dùng softmax-max confidence thuần, dùng prototype distances
    để đo lường uncertainty một cách có ý nghĩa hơn.

    Cơ sở lý thuyết:
      • Cosine distance giữa embedding và prototype cho biết embedding đó
        "gần" hay "xa" class center → phản ánh genuine similarity
      • Softmax score bị ảnh hưởng bởi temperature và magnitude của logits
      • Prototype distance trực tiếp hơn: khoảng cách = 1 - cosine_similarity

    Metrics:
      1. margin_confidence     = P_pred - P_second       (softmax-based margin)
      2. proto_margin          = dist_pred - dist_true   (prototype-based margin)
                                 < 0 → embedding gần pred hơn true
                                 > 0 → embedding gần true hơn pred → uncertain
      3. proto_distance_ratio  = dist_pred / (dist_pred + dist_second)
                                 → 0.5 = equidistant (highly uncertain)
                                 → 0.0 = very close to predicted prototype
      4. normalized_entropy   = 1 - H(p) / log(n_classes)
                                 0 = certain, 1 = maximum uncertainty
      5. calibration_score     = confidence - accuracy (binned, cho ECE)
    """

    def __init__(self, model: nn.Module, temperature: float = 1.0):
        """
        model: MalariaProtoCLFv2 đã trained
        temperature: temperature đã fit (mặc định 1.0 = không scale)
        """
        self.model = model
        self.temperature = temperature
        self.model.eval()

    @torch.no_grad()
    def score(self, images: torch.Tensor, labels: torch.Tensor | None = None):
        """
        Compute tất cả confidence metrics cho batch images.

        Args:
            images: (B, 3, H, W)
            labels: (B,) optional — true labels để tính proto_margin
        Returns:
            dict với keys:
              "softmax_probs":     (B, C) softmax probabilities
              "predictions":       (B,) predicted class indices
              "max_confidence":   (B,) max softmax probability
              "margin_confidence": (B,) pred_prob - second_prob
              "proto_distances":  (B, C) cosine distances to prototypes
              "proto_margin":     (B,) dist_pred - dist_true (labels required)
              "proto_ratio":      (B,) dist_pred / (dist_pred + dist_second)
              "entropy":          (B,) normalized entropy [0..1]
              "correct":          (B,) bool — prediction matches true label
        """
        device = next(self.model.parameters()).device
        images = images.to(device)

        # Forward pass
        proj_feats, logits = self.model(images)  # (B, proj_dim), (B, C)

        # Scale temperature
        logits_scaled = logits / self.temperature
        probs = F.softmax(logits_scaled, dim=-1)  # (B, C)

        predictions = probs.argmax(dim=1)  # (B,)

        # Softmax-based metrics
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        max_conf = sorted_probs[:, 0]                                    # (B,)
        margin_conf = sorted_probs[:, 0] - sorted_probs[:, 1]           # (B,)

        # Normalized entropy
        eps = 1e-9
        entropy = -(probs * torch.log(probs + eps)).sum(dim=1)         # (B,)
        n_classes = probs.shape[1]
        entropy = entropy / np.log(n_classes)                          # normalize to [0,1]
        norm_entropy = 1 - entropy                                     # high = certain

        # Prototype distances
        if hasattr(self.model.clf_head, 'get_distances'):
            proto_distances = self.model.clf_head.get_distances(proj_feats)  # (B, C)
        else:
            # FC head fallback: dùng softmax scores reversed
            proto_distances = 1 - probs

        proto_dist_pred = proto_distances.gather(1, predictions.unsqueeze(1)).squeeze(1)  # (B,)
        proto_dist_second_list = []
        for i in range(len(predictions)):
            mask = torch.ones(n_classes, device=device)
            mask[predictions[i]] = 0
            second_idx = (proto_distances[i] * mask).argmin()
            proto_dist_second_list.append(proto_distances[i, second_idx])
        proto_dist_second = torch.stack(proto_dist_second_list)  # (B,)

        # Proto distance ratio: 0.5 = equidistant (uncertain)
        proto_ratio = proto_dist_pred / (proto_dist_pred + proto_dist_second + eps)

        results = {
            "softmax_probs":     probs.cpu().numpy(),
            "predictions":       predictions.cpu().numpy(),
            "max_confidence":    max_conf.cpu().numpy(),
            "margin_confidence": margin_conf.cpu().numpy(),
            "proto_distances":   proto_distances.cpu().numpy(),
            "proto_ratio":       proto_ratio.cpu().numpy(),
            "entropy":           norm_entropy.cpu().numpy(),
        }

        # Prototype margin (requires true labels)
        if labels is not None:
            labels = labels.to(device)
            proto_dist_true = proto_distances.gather(1, labels.unsqueeze(1)).squeeze(1)  # (B,)
            proto_margin = proto_dist_pred - proto_dist_true                            # (B,)
            results["proto_margin"] = proto_margin.cpu().numpy()
            results["correct"] = (predictions == labels).cpu().numpy()

        return results

    @torch.no_grad()
    def per_class_analysis(self, images: torch.Tensor, labels: torch.Tensor,
                           output_dir: str):
        """
        Phân tích confidence metrics cho từng class.
        Xuất CSV và histogram plots.
        """
        import os
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        scores = self.score(images, labels)
        df = pd.DataFrame({
            "true_label": labels.numpy(),
            "pred_label": scores["predictions"],
            "max_conf":   scores["max_confidence"],
            "margin":     scores["margin_confidence"],
            "proto_margin": scores.get("proto_margin", np.zeros(len(labels))),
            "proto_ratio":  scores["proto_ratio"],
            "entropy":      scores["entropy"],
            "correct":      scores.get("correct", np.zeros(len(labels))),
        })

        from dataset import CLASS_NAMES
        df["class_name"] = df["true_label"].map(CLASS_NAMES)

        # Save CSV
        df.to_csv(os.path.join(output_dir, "confidence_scores.csv"), index=False)
        print(f"[ConfidenceScorer] Saved confidence_scores.csv ({len(df)} samples)")

        # Histogram per class
        class_names = [CLASS_NAMES[i] for i in sorted(labels.unique().tolist())]
        n = len(class_names)
        fig, axes = plt.subplots(2, n, figsize=(4*n, 8))

        for idx, cls_name in enumerate(class_names):
            mask = df["true_label"] == list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(cls_name)]
            sub = df[mask]

            # Proto ratio histogram
            ax = axes[0, idx]
            ax.hist(sub["proto_ratio"], bins=20, alpha=0.7, color="steelblue")
            ax.set_title(f"{cls_name}\nproto_ratio", fontsize=10)
            ax.set_xlabel("Proto Ratio"); ax.set_ylabel("Count")
            if len(sub) > 0:
                ax.axvline(sub["proto_ratio"].mean(), color="red", linestyle="--",
                           label=f"mean={sub['proto_ratio'].mean():.3f}")
                ax.legend(fontsize=8)

            # Proto margin histogram
            ax = axes[1, idx]
            ax.hist(sub["proto_margin"], bins=20, alpha=0.7, color="orange")
            ax.set_title(f"{cls_name}\nproto_margin", fontsize=10)
            ax.set_xlabel("Proto Margin (dist_pred - dist_true)"); ax.set_ylabel("Count")
            ax.axvline(0, color="green", linestyle="--", label="margin=0")
            if len(sub) > 0:
                ax.axvline(sub["proto_margin"].mean(), color="red", linestyle="--",
                           label=f"mean={sub['proto_margin'].mean():.3f}")
                ax.legend(fontsize=8)

        plt.suptitle("Prototype-based Confidence Analysis per Class", fontsize=13)
        plt.tight_layout()
        fname = os.path.join(output_dir, "proto_confidence_analysis.png")
        plt.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[ConfidenceScorer] Saved proto_confidence_analysis.png")

        # Confusion: proto_ratio vs accuracy
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Proto ratio distribution: correct vs incorrect
        for group_name, color, mask in [
            ("Correct", "green", df["correct"]),
            ("Incorrect", "red", ~df["correct"]),
        ]:
            data = df.loc[mask, "proto_ratio"]
            axes[0].hist(data, bins=20, alpha=0.6, color=color, label=f"{group_name} (n={mask.sum()})", density=True)
        axes[0].set_xlabel("Proto Ratio"); axes[0].set_ylabel("Density")
        axes[0].set_title("Proto Ratio: Correct vs Incorrect"); axes[0].legend()

        # Proto margin vs correct
        for group_name, color, mask in [
            ("Correct", "green", df["correct"]),
            ("Incorrect", "red", ~df["correct"]),
        ]:
            data = df.loc[mask, "proto_margin"]
            axes[1].hist(data, bins=20, alpha=0.6, color=color, label=f"{group_name} (n={mask.sum()})", density=True)
        axes[1].set_xlabel("Proto Margin (dist_pred - dist_true)"); axes[1].set_ylabel("Density")
        axes[1].set_title("Proto Margin: Correct vs Incorrect"); axes[1].legend()
        axes[1].axvline(0, color="black", linestyle="--")

        fname2 = os.path.join(output_dir, "proto_confidence_calibration.png")
        plt.savefig(fname2, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[ConfidenceScorer] Saved proto_confidence_calibration.png")

        # Summary stats
        print("\n[ConfidenceScorer] Summary:")
        print(f"  Proto Ratio    — correct mean={df[df['correct']]['proto_ratio'].mean():.4f}, "
              f"incorrect mean={df[~df['correct']]['proto_ratio'].mean():.4f}")
        print(f"  Proto Margin    — correct mean={df[df['correct']]['proto_margin'].mean():.4f}, "
              f"incorrect mean={df[~df['correct']]['proto_margin'].mean():.4f}")
        print(f"  Proto Ratio < 0.6 (uncertain): {(df['proto_ratio'] < 0.6).sum()} samples "
              f"({(df['proto_ratio'] < 0.6).mean()*100:.1f}%)")
        print(f"  Proto Margin > 0 (dist_pred > dist_true): {(df['proto_margin'] > 0).sum()} samples")

        return df


# ─────────────────────────────────────────────
# ECE utilities (giữ nguyên từ v1)
# ─────────────────────────────────────────────
def compute_ece(probs, labels, n_bins=15):
    """
    Expected Calibration Error.
    probs:  numpy (N, C) softmax probabilities
    labels: numpy (N,) int true labels
    Returns ECE scalar.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() / len(labels) * abs(acc - conf)

    return ece


def reliability_diagram_data(probs, labels, n_bins=15):
    """Returns (bin_midpoints, accuracies, confidences, counts) for plotting."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)

    bins      = np.linspace(0, 1, n_bins + 1)
    midpoints = (bins[:-1] + bins[1:]) / 2
    accs, confs, counts = [], [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (confidences >= lo) & (confidences < hi)
        counts.append(mask.sum())
        if mask.sum() == 0:
            accs.append(0.0)
            confs.append(midpoints[i])
        else:
            accs.append(correct[mask].mean())
            confs.append(confidences[mask].mean())

    return midpoints, np.array(accs), np.array(confs), np.array(counts)
