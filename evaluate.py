"""
evaluate.py - Full evaluation pipeline with:
  - Per-class Precision, Recall, F1
  - Weighted P/R/F1 for parasite classes only
  - Macro/Weighted averages
  - Confusion matrix
  - ECE + Reliability Diagram
  - Misclassification analysis (confidence bias investigation)

Usage:
    python evaluate.py --checkpoint /kaggle/working/malaria_proto_clf/calibrated_model.pth \
                       --test_ann   /path/to/test_annotation_5classes.txt \
                       --img_base   /path/to/base_dir \
                       --output_dir /kaggle/working/eval_results
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from calibration import TemperatureScaling, compute_ece, reliability_diagram_data
from dataset import CLASS_NAMES, NUM_CLASSES, MalariaDataset, get_transforms
from model import build_model

PARASITE_INDICES = [0, 1, 2, 3]  # TA, TJ, S, G
MAJORITY_INDEX = 4  # Unparasitized


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "cfg" in ckpt:
        cfg_dict = ckpt["cfg"]
        model_cfg = {
            "backbone": cfg_dict.get("BACKBONE", "convnext_tiny.in22k_ft_in1k"),
            "num_classes": cfg_dict.get("NUM_CLASSES", 5),
            "proj_dim": cfg_dict.get("PROJ_DIM", 128),
            "use_prototype": cfg_dict.get("USE_PROTOTYPE", True),
            "pretrained": False,
        }
        temperature = ckpt.get("temperature", 1.0)
        state_dict = ckpt["model_state"]
    else:
        # bare state dict
        model_cfg = {
            "backbone": "convnext_tiny.in22k_ft_in1k",
            "num_classes": 5,
            "proj_dim": 128,
            "use_prototype": True,
            "pretrained": False,
        }
        temperature = 1.0
        state_dict = ckpt

    model = build_model(model_cfg)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    ts = TemperatureScaling(model)
    ts.temperature.data = torch.tensor([temperature])
    ts.to(device)
    return ts, temperature


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(ts_model, loader, device):
    all_probs, all_preds, all_labels, all_paths = [], [], [], []

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        logits = ts_model(imgs)
        probs = F.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return all_probs, all_preds, all_labels


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_metrics(probs, preds, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    class_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    # ── Per-class report ──
    print("" + "=" * 65)
    print("CLASSIFICATION REPORT (all classes)")
    print("=" * 65)
    report = classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0)
    print(report)

    # Save report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # ── Parasite-only weighted metrics ──
    para_mask = np.isin(labels, PARASITE_INDICES)
    if para_mask.sum() > 0:
        para_labels = labels[para_mask]
        para_preds = preds[para_mask]
        para_names = [CLASS_NAMES[i] for i in PARASITE_INDICES]

        print("" + "=" * 65)
        print("PARASITE-ONLY WEIGHTED METRICS (TA, TJ, S, G)")
        print("=" * 65)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            para_labels, para_preds, labels=PARASITE_INDICES, average="weighted", zero_division=0
        )
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            para_labels, para_preds, labels=PARASITE_INDICES, average="macro", zero_division=0
        )
        print(f"  Weighted  P: {p_w:.4f} | R: {r_w:.4f} | F1: {f1_w:.4f}")
        print(f"  Macro     P: {p_m:.4f} | R: {r_m:.4f} | F1: {f1_m:.4f}")

        # Per parasite class
        print("  Per-class breakdown (parasite classes only):")
        p_cls, r_cls, f1_cls, sup = precision_recall_fscore_support(
            para_labels, para_preds, labels=PARASITE_INDICES, zero_division=0
        )
        print(f"  {'Class':<18} {'P':>8} {'R':>8} {'F1':>8} {'Support':>8}")
        print(f"  {'-'*54}")
        for i, idx in enumerate(PARASITE_INDICES):
            print(f"  {CLASS_NAMES[idx]:<18} {p_cls[i]:>8.4f} " f"{r_cls[i]:>8.4f} {f1_cls[i]:>8.4f} {int(sup[i]):>8}")

    # ── ECE ──
    ece = compute_ece(probs, labels)
    print(f"[Calibration] Expected Calibration Error (ECE): {ece:.4f}")

    # Summary JSON
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    summary = {
        "overall_accuracy": float((preds == labels).mean()),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "ece": float(ece),
    }
    if para_mask.sum() > 0:
        summary["parasite_weighted_f1"] = float(f1_w)
        summary["parasite_macro_f1"] = float(f1_m)

    with open(os.path.join(output_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary] {summary}")

    return summary


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────
def plot_confusion_matrix(preds, labels, output_dir):
    class_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)", fontsize=13)
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Oranges", xticklabels=class_names, yticklabels=class_names, ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix (normalized)", fontsize=13)
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    out = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved confusion matrix → {out}")


def plot_reliability_diagram(probs, labels, output_dir, title_suffix=""):
    midpoints, accs, confs, counts = reliability_diagram_data(probs, labels)
    ece = compute_ece(probs, labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(midpoints, accs, width=1 / 15, alpha=0.6, color="steelblue", label="Accuracy per bin", align="center")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Reliability Diagram {title_suffix}(ECE = {ece:.4f})", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    out = os.path.join(output_dir, f"reliability_diagram{title_suffix}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved reliability diagram → {out}")


def plot_confidence_distribution(probs, preds, labels, output_dir):
    """
    Per-class confidence distribution — highlights confidence bias.
    Shows how confident the model is for correct vs incorrect predictions.
    """
    confidences = probs.max(axis=1)
    correct = preds == labels
    class_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(4 * NUM_CLASSES, 4), sharey=True)
    for c_idx in range(NUM_CLASSES):
        mask = labels == c_idx
        if mask.sum() == 0:
            continue
        ax = axes[c_idx]
        conf_correct = confidences[mask & correct]
        conf_incorrect = confidences[mask & ~correct]
        ax.hist(conf_correct, bins=20, alpha=0.7, color="green", label=f"Correct (n={len(conf_correct)})", density=True)
        ax.hist(
            conf_incorrect, bins=20, alpha=0.7, color="red", label=f"Incorrect (n={len(conf_incorrect)})", density=True
        )
        ax.set_title(f"Class: {class_names[c_idx]}", fontsize=11)
        ax.set_xlabel("Max Confidence")
        ax.legend(fontsize=8)
        if c_idx == 0:
            ax.set_ylabel("Density")

    plt.suptitle("Confidence Distribution per Class(Correct vs Incorrect Predictions)", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "confidence_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved confidence distribution → {out}")


def plot_per_class_f1(preds, labels, output_dir):
    class_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    _, _, f1s, supports = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    colors = ["tomato" if i in PARASITE_INDICES else "steelblue" for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_names, f1s, color=colors, edgecolor="black", linewidth=0.7)
    for bar, f1, sup in zip(bars, f1s, supports):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"F1={f1:.3f}(n={int(sup)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Score(Red = Parasite, Blue = Unparasitized)", fontsize=13)
    ax.axhline(np.mean(f1s), color="gray", linestyle="--", label=f"Macro avg F1 = {np.mean(f1s):.3f}")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(output_dir, "per_class_f1.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved per-class F1 → {out}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def evaluate(checkpoint_path, test_ann, img_base, output_dir, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    # Load model
    ts_model, temperature = load_model(checkpoint_path, device)
    print(f"[Eval] Temperature: {temperature:.4f}")

    # Dataset
    test_tf = get_transforms("val")
    test_ds = MalariaDataset(test_ann, img_base, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Inference
    probs, preds, labels = run_inference(ts_model, test_loader, device)

    # Metrics
    summary = compute_metrics(probs, preds, labels, output_dir)

    # Plots
    plot_confusion_matrix(preds, labels, output_dir)
    plot_reliability_diagram(probs, labels, output_dir)
    plot_confidence_distribution(probs, preds, labels, output_dir)
    plot_per_class_f1(preds, labels, output_dir)

    print(f"[Eval] All results saved to: {output_dir}")
    return summary, probs, preds, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_ann", required=True)
    parser.add_argument("--img_base", required=True)
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        test_ann=args.test_ann,
        img_base=args.img_base,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
