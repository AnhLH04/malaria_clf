"""
misclassification_analysis.py
──────────────────────────────
Phân tích chi tiết các cells bị phân loại sai:
- Hiển thị score cho TẤT CẢ lớp (không chỉ predicted label)
- Highlight confusion giữa các cặp lớp (TA↔TJ, S↔TJ...)
- Export CSV đầy đủ để phân tích
- Visualize top misclassified samples với probability bar

Usage (Kaggle cell):
    from misclassification_analysis import analyze_misclassifications
    analyze_misclassifications(
        checkpoint_path = "/kaggle/working/malaria_proto_clf/calibrated_model.pth",
        test_ann        = "...test_annotation_5classes.txt",
        img_base        = "...base_dir",
        output_dir      = "/kaggle/working/eval_results",
    )
"""

import json
import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from PIL import Image

from calibration import TemperatureScaling, sanitize_temperature
from dataset import CLASS_NAMES, NUM_CLASSES, MalariaDataset, get_transforms
from model import build_model

CLASS_LIST = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
PARASITE_IDX = [0, 1, 2, 3]
COLORS_BAR = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#95a5a6"]


# ─────────────────────────────────────────────
# Load model (reuse from evaluate.py)
# ─────────────────────────────────────────────
def _load_ts_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "cfg" in ckpt:
        cfg_dict = ckpt["cfg"]
        model_cfg = {
            "backbone": cfg_dict.get("BACKBONE", "convnextv2_nano.fcmae_ft_in22k_in1k"),
            "num_classes": cfg_dict.get("NUM_CLASSES", 5),
            "proj_dim": cfg_dict.get("PROJ_DIM", 128),
            "use_prototype": cfg_dict.get("USE_PROTOTYPE", True),
            "pretrained": False,
        }
        temperature = sanitize_temperature(ckpt.get("temperature", 1.0))
        state_dict = ckpt["model_state"]
    else:
        model_cfg = {
            "backbone": "convnextv2_nano.fcmae_ft_in22k_in1k",
            "num_classes": 5,
            "proj_dim": 128,
            "use_prototype": True,
            "pretrained": False,
        }
        temperature = 1.0
        state_dict = ckpt

    model = build_model(model_cfg)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    ts = TemperatureScaling(model)
    ts.temperature.data = torch.tensor([temperature])
    ts.eval().to(device)
    return ts


# ─────────────────────────────────────────────
# Collect full-score inference
# ─────────────────────────────────────────────
@torch.no_grad()
def _run_full_inference(ts_model, loader, device, dataset):
    """Returns DataFrame with columns: path, true_label, pred_label, score_TA, score_TJ, ..."""
    all_rows = []

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        logits = ts_model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()

        start = batch_idx * loader.batch_size
        end = start + len(labels)
        paths = [dataset.samples[i][0] for i in range(start, min(end, len(dataset)))]

        for i, (path, true_lbl) in enumerate(zip(paths, labels.numpy())):
            pred_lbl = probs[i].argmax()
            row = {
                "path": path,
                "true_idx": int(true_lbl),
                "true_label": CLASS_NAMES[int(true_lbl)],
                "pred_idx": int(pred_lbl),
                "pred_label": CLASS_NAMES[int(pred_lbl)],
                "correct": int(true_lbl) == int(pred_lbl),
                "max_conf": float(probs[i].max()),
            }
            for c_idx in range(NUM_CLASSES):
                row[f"score_{CLASS_NAMES[c_idx]}"] = float(probs[i, c_idx])
            all_rows.append(row)

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────
# Confusion pair analysis
# ─────────────────────────────────────────────
def _confusion_pair_stats(df_wrong, output_dir):
    """Print + save stats for each (true→pred) confusion pair."""
    score_cols = [f"score_{CLASS_NAMES[i]}" for i in range(NUM_CLASSES)]

    print("MISCLASSIFICATION ANALYSIS — Per Confusion Pair")
    print("=" * 70)

    pairs = df_wrong.groupby(["true_label", "pred_label"])
    pair_rows = []

    for (true_lbl, pred_lbl), group in pairs:
        n = len(group)
        avg_scores = {col: group[col].mean() for col in score_cols}
        row = {"true→pred": f"{true_lbl}→{pred_lbl}", "count": n}
        row.update({col.replace("score_", "avg_"): f"{v:.4f}" for col, v in avg_scores.items()})
        pair_rows.append(row)

        print(f"[{true_lbl} → {pred_lbl}] n={n}")
        print(f"  {'Class':<18} {'Avg Score':>10}")
        for col, v in avg_scores.items():
            marker = " ◄TRUE" if col == f"score_{true_lbl}" else (" ◄PRED" if col == f"score_{pred_lbl}" else "")
            print(f"    {col.replace('score_',''):<16} {v:>10.4f}{marker}")

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(os.path.join(output_dir, "confusion_pairs.csv"), index=False)
    print(f"[Saved] confusion_pairs.csv")
    return pair_df


# ─────────────────────────────────────────────
# Visualize misclassified samples
# ─────────────────────────────────────────────
def _visualize_misclassified(df_wrong, output_dir, max_per_pair=6, img_size=96):
    """Grid visualization: each row = one confusion pair, cols = sample images."""
    score_cols = [f"score_{CLASS_NAMES[i]}" for i in range(NUM_CLASSES)]
    pairs = list(df_wrong.groupby(["true_label", "pred_label"]))

    # Only show parasite confusion pairs (exclude UN→X and X→UN unless interesting)
    parasite_pairs = [(key, grp) for key, grp in pairs if key[0] != "Unparasitized" or key[1] != "Unparasitized"]

    for (true_lbl, pred_lbl), group in parasite_pairs:
        samples = group.head(max_per_pair)
        n = len(samples)
        if n == 0:
            continue

        fig = plt.figure(figsize=(4 * n, 5.5))
        fig.suptitle(
            f"Misclassified: True={true_lbl} → Pred={pred_lbl}  (n={len(group)} total)",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )

        for col_idx, (_, row) in enumerate(samples.iterrows()):
            ax_img = plt.subplot2grid((2, n), (0, col_idx))
            ax_bar = plt.subplot2grid((2, n), (1, col_idx))

            # Image
            try:
                img = Image.open(row["path"]).convert("RGB").resize((img_size, img_size))
                ax_img.imshow(img)
            except Exception:
                ax_img.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax_img.axis("off")
            ax_img.set_title(f"conf={row['max_conf']:.3f} |" f"T:{row['true_label']} P:{row['pred_label']}", fontsize=8)

            # Probability bar — all classes
            scores = [row[f"score_{CLASS_NAMES[i]}"] for i in range(NUM_CLASSES)]
            bars = ax_bar.bar(CLASS_LIST, scores, color=COLORS_BAR, width=0.6)
            ax_bar.set_ylim(0, 1.05)
            ax_bar.set_ylabel("Score", fontsize=7)
            ax_bar.tick_params(axis="x", labelsize=7)
            ax_bar.tick_params(axis="y", labelsize=6)

            # Annotate values on bars
            for bar, score in zip(bars, scores):
                if score > 0.01:
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{score:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                    )

            # Highlight true/pred class bars
            true_idx = CLASS_LIST.index(true_lbl)
            pred_idx = CLASS_LIST.index(pred_lbl)
            bars[true_idx].set_edgecolor("green")
            bars[true_idx].set_linewidth(2.5)
            bars[pred_idx].set_edgecolor("red")
            bars[pred_idx].set_linewidth(2.5)

        plt.tight_layout()
        fname = f"misclassified_{true_lbl}_to_{pred_lbl}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved {fname}")


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
def analyze_misclassifications(checkpoint_path, test_ann, img_base, output_dir, batch_size=64):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts_model = _load_ts_model(checkpoint_path, device)

    test_tf = get_transforms("val")
    test_ds = MalariaDataset(test_ann, img_base, transform=test_tf)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("[Analysis] Running full-score inference...")
    df = _run_full_inference(ts_model, loader, device, test_ds)

    # Save full predictions CSV
    df.to_csv(os.path.join(output_dir, "all_predictions.csv"), index=False)
    print(f"[Saved] all_predictions.csv ({len(df)} rows)")

    # Misclassified subset
    df_wrong = df[~df["correct"]].copy()
    df_wrong.to_csv(os.path.join(output_dir, "misclassified.csv"), index=False)
    print(f"[Saved] misclassified.csv ({len(df_wrong)} wrong out of {len(df)})")

    # Confusion pair analysis (with avg scores per class)
    _confusion_pair_stats(df_wrong, output_dir)

    # Visual grid for each confusion pair
    print("[Analysis] Generating misclassification visualizations...")
    _visualize_misclassified(df_wrong, output_dir)

    print(f"[Done] All analysis saved to: {output_dir}")
    return df, df_wrong


# ─────────────────────────────────────────────
# Bonus: score histogram per confusion pair
# ─────────────────────────────────────────────
def plot_score_histograms(df, output_dir, focus_class_idx=1):
    """
    Phân tích distribution của score cho 1 lớp cụ thể (mặc định TJ = 1)
    để thấy rõ overlap giữa correct và incorrect samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    focus_name = CLASS_NAMES[focus_class_idx]
    score_col = f"score_{focus_name}"

    df_focus = df[df["true_idx"] == focus_class_idx].copy()
    if len(df_focus) == 0:
        print(f"No samples for class {focus_name}")
        return

    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(4 * NUM_CLASSES, 4), sharey=True)
    for c_idx in range(NUM_CLASSES):
        ax = axes[c_idx]
        col = f"score_{CLASS_NAMES[c_idx]}"

        correct_scores = df_focus[df_focus["correct"]][col].values
        incorrect_scores = df_focus[~df_focus["correct"]][col].values

        if len(correct_scores) > 0:
            ax.hist(
                correct_scores,
                bins=20,
                alpha=0.7,
                color="green",
                label=f"Correct (n={len(correct_scores)})",
                density=True,
            )
        if len(incorrect_scores) > 0:
            ax.hist(
                incorrect_scores,
                bins=20,
                alpha=0.7,
                color="red",
                label=f"Wrong (n={len(incorrect_scores)})",
                density=True,
            )

        ax.set_title(f"Score: {CLASS_NAMES[c_idx]}", fontsize=10)
        ax.set_xlabel("Probability")
        ax.legend(fontsize=7)
        if c_idx == 0:
            ax.set_ylabel("Density")

    fig.suptitle(
        f"Score Distributions for True Class = {focus_name} " f"(Correct vs Misclassified)", fontsize=12, y=1.01
    )
    plt.tight_layout()
    fname = f"score_hist_{focus_name}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {fname}")
