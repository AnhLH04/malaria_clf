# Malaria Prototype Contrastive Classifier

**Confidence-Calibrated 2-Stage Detection via Prototype Contrastive Learning**

## File Structure

```
malaria_proto_clf_src/
├── dataset.py           # MalariaDataset, transforms, TwoViewTransform
├── model.py             # MalariaProtoCLF: Backbone → Projection → Prototype Head
├── losses.py            # SupConLoss, DynamicFocalLoss, AsymmetricLabelSmoothing, CombinedLoss
├── calibration.py       # TemperatureScaling, compute_ece, reliability_diagram_data
├── train.py             # Trainer class + TrainConfig
├── evaluate.py          # Full evaluation: metrics + all plots
├── inference_pipeline.py# End-to-end YOLO11 → Classifier pipeline
└── kaggle_notebook.py   # Cell-by-cell runner for Kaggle
```

## Architecture Overview

```
Input Image
    │
    ▼
[YOLO11 Detector] → crops bounding boxes of all cells
    │
    ▼
[MalariaProtoCLF]
    │
    ├─ Backbone (ConvNeXt-Tiny / EfficientNet-B1/B2)
    │       ↓ feature vector (768-dim)
    ├─ Projection Head (MLP, 768→512→128, L2-norm)
    │       ↓ z ∈ unit sphere
    ├─ Prototype Head (learnable prototype per class, cosine sim × T)
    │       ↓ logits
    └─ [Temperature Scaling] → calibrated probabilities
```

## Loss Function

```
L = α × SupConLoss(z, labels) + (1-α) × DynamicFocalLoss(logits, labels)
```
- **α = 0.4** by default (tune via `cfg.ALPHA`)
- `DynamicFocalLoss`: per-class γ inversely proportional to class frequency
- `SupConLoss`: pulls same-class embeddings together, pushes different classes apart

## Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Backbone | ConvNeXt-Tiny (in22k pretrained) | Best for medical imaging per benchmark |
| Classifier head | Prototype (cosine-distance) | Interpretable; avoids overconfident softmax |
| Loss | SupCon + Focal | SupCon fixes embedding collapse; Focal handles imbalance |
| Calibration | Temperature Scaling | Simple, effective post-hoc ECE reduction |
| Sampler | WeightedRandomSampler | Ensures minority classes seen equally in training |

## Class Labels

| Index | Name | Type |
|-------|------|------|
| 0 | TA | Parasite (Trophozoite/Ring Asexual) |
| 1 | TJ | Parasite (Trophozoite Juvenile) |
| 2 | S  | Parasite (Schizont) |
| 3 | G  | Parasite (Gametocyte) |
| 4 | Unparasitized | Healthy |

## Annotation Format

```
relative/path/to/cell.jpg  label_idx
test/051Overlay002/rbc_parasitized_F_S1/cell.jpg 2
test/051Overlay002/rbc_unparasitized/cell.jpg 4
```

## Quick Start

```python
from train    import Trainer, TrainConfig
from evaluate import evaluate

cfg = TrainConfig()
cfg.BACKBONE = "convnext_tiny.in22k_ft_in1k"  # or your EfficientNet
cfg.TRAIN_ANN = "/path/to/train_annotation_5classes.txt"
cfg.VAL_ANN   = "/path/to/val_annotation_5classes.txt"
cfg.TEST_ANN  = "/path/to/test_annotation_5classes.txt"
cfg.IMG_BASE  = "/path/to/base_dir"

trainer = Trainer(cfg)
trainer.run()

evaluate(
    checkpoint_path="best_checkpoint/calibrated_model.pth",
    test_ann=cfg.TEST_ANN,
    img_base=cfg.IMG_BASE,
    output_dir="eval_results",
)
```

## Outputs

After `evaluate()`:
- `classification_report.txt` — per-class P/R/F1
- `summary_metrics.json` — overall + parasite-only weighted metrics + ECE
- `confusion_matrix.png` — counts + normalized
- `reliability_diagram.png` — calibration visualization
- `confidence_distribution.png` — bias investigation per class
- `per_class_f1.png` — bar chart F1 per class
