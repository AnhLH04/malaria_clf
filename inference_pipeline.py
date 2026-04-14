"""
inference_pipeline.py - End-to-end inference: YOLO11 detection → Prototype CLF

This module takes raw microscopy images (not cropped cells), runs YOLO11
to detect all cells, crops each detected cell, then classifies via the
calibrated Prototype Contrastive model.

Usage:
    from inference_pipeline import MalariaFullPipeline

    pipeline = MalariaFullPipeline(
        yolo_weights   = "/path/to/yolo11_malaria.pt",
        clf_checkpoint = "/kaggle/working/malaria_proto_clf/calibrated_model.pth",
    )

    results = pipeline.infer_image("slide001.jpg")
    pipeline.visualize("slide001.jpg", results, save_path="slide001_output.jpg")
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO

from dataset     import CLASS_NAMES, NUM_CLASSES, get_transforms
from model       import build_model
from calibration import TemperatureScaling


COLORS = {
    0: (0,   165, 255),   # TA        – orange
    1: (255,   0, 128),   # TJ        – magenta
    2: (0,   200,   0),   # S         – green
    3: (0,   0,   255),   # G         – red
    4: (220, 220, 220),   # Unparasitized – gray
}


class MalariaFullPipeline:
    """
    2-Stage malaria detection + classification pipeline.

    yolo_weights:   path to YOLO11 .pt weights (cell detector)
    clf_checkpoint: path to calibrated classification model (.pth)
    yolo_conf:      detection confidence threshold
    yolo_iou:       NMS IoU threshold
    clf_batch_size: batch size for classification GPU calls
    """

    def __init__(self,
                 yolo_weights,
                 clf_checkpoint,
                 yolo_conf=0.25,
                 yolo_iou=0.45,
                 clf_batch_size=64,
                 device=None):

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── YOLO detector ─────────────────────
        self.detector   = YOLO(yolo_weights)
        self.yolo_conf  = yolo_conf
        self.yolo_iou   = yolo_iou

        # ── Classifier ────────────────────────
        self.ts_model, self.temperature = self._load_clf(clf_checkpoint)
        self.clf_batch_size = clf_batch_size
        self.clf_transform  = get_transforms("val", img_size=224)

    def _load_clf(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if "cfg" in ckpt:
            cfg_dict = ckpt["cfg"]
            model_cfg = {
                "backbone":      cfg_dict.get("BACKBONE", "convnext_tiny.in22k_ft_in1k"),
                "num_classes":   cfg_dict.get("NUM_CLASSES", 5),
                "proj_dim":      cfg_dict.get("PROJ_DIM", 128),
                "use_prototype": cfg_dict.get("USE_PROTOTYPE", True),
                "pretrained":    False,
            }
            temperature = ckpt.get("temperature", 1.0)
            state_dict  = ckpt["model_state"]
        else:
            model_cfg  = {"backbone": "convnext_tiny.in22k_ft_in1k",
                          "num_classes": 5, "proj_dim": 128,
                          "use_prototype": True, "pretrained": False}
            temperature = 1.0
            state_dict  = ckpt

        model = build_model(model_cfg)
        model.load_state_dict(state_dict)
        model.eval().to(self.device)

        ts = TemperatureScaling(model)
        ts.temperature.data = torch.tensor([temperature])
        ts.eval().to(self.device)
        return ts, temperature

    # ── Detection ─────────────────────────────
    def detect_cells(self, image_path):
        """
        Returns list of dicts: {bbox: [x1,y1,x2,y2], det_conf: float}
        """
        results = self.detector.predict(
            image_path,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            detections.append({"bbox": [x1, y1, x2, y2], "det_conf": conf})
        return detections

    # ── Crop & preprocess ─────────────────────
    def _crop_cells(self, img_bgr, detections, pad=4):
        """Crop detected bboxes from BGR image, return list of PIL images."""
        H, W = img_bgr.shape[:2]
        crops = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
            crop = img_bgr[y1:y2, x1:x2]
            pil  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crops.append(pil)
        return crops

    # ── Classify in batches ────────────────────
    @torch.no_grad()
    def _classify_crops(self, crops):
        """Returns (probs, pred_labels) arrays for list of PIL crops."""
        if not crops:
            return np.array([]), np.array([])

        all_probs = []
        for i in range(0, len(crops), self.clf_batch_size):
            batch_pil = crops[i: i + self.clf_batch_size]
            tensors   = torch.stack(
                [self.clf_transform(img) for img in batch_pil]
            ).to(self.device)

            logits = self.ts_model(tensors)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)
        pred_labels = all_probs.argmax(axis=1)
        return all_probs, pred_labels

    # ── Full inference ─────────────────────────
    def infer_image(self, image_path):
        """
        Full pipeline on a single raw slide image.
        Returns list of result dicts:
          {bbox, det_conf, class_idx, class_name, confidence, all_probs}
        """
        img_bgr    = cv2.imread(image_path)
        detections = self.detect_cells(image_path)

        if not detections:
            return []

        crops             = self._crop_cells(img_bgr, detections)
        all_probs, labels = self._classify_crops(crops)

        results = []
        for i, det in enumerate(detections):
            c_idx  = int(labels[i])
            results.append({
                "bbox":       det["bbox"],
                "det_conf":   det["det_conf"],
                "class_idx":  c_idx,
                "class_name": CLASS_NAMES[c_idx],
                "confidence": float(all_probs[i, c_idx]),
                "all_probs":  all_probs[i].tolist(),
            })

        return results

    # ── Visualization ─────────────────────────
    def visualize(self, image_path, results, save_path=None, show=False):
        """Draw bboxes with class labels and confidence scores."""
        img = cv2.imread(image_path)
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            c_idx           = r["class_idx"]
            color           = COLORS.get(c_idx, (200, 200, 200))
            label           = f"{r['class_name']} {r['confidence']:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(img, label, (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        if save_path:
            cv2.imwrite(save_path, img)
            print(f"[Pipeline] Saved visualization → {save_path}")
        if show:
            cv2.imshow("Malaria Detection", img)
            cv2.waitKey(0)
        return img

    # ── Batch inference on folder / annotation ─
    def infer_annotation(self, annotation_file, img_base, output_dir, batch_size=64):
        """
        Classify pre-cropped cells from an annotation file (no YOLO needed).
        Used for evaluating classification only.
        Returns (probs, preds, labels).
        """
        from dataset     import MalariaDataset
        from torch.utils.data import DataLoader

        test_tf = get_transforms("val")
        ds      = MalariaDataset(annotation_file, img_base, transform=test_tf)
        loader  = DataLoader(ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs   = imgs.to(self.device)
                logits = self.ts_model(imgs)
                probs  = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_preds.extend(probs.argmax(axis=1))
                all_labels.extend(labels.numpy())

        return (np.concatenate(all_probs),
                np.array(all_preds),
                np.array(all_labels))
