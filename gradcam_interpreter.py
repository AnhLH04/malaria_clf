"""
gradcam_interpreter.py - GradCAM + Prototype Similarity Heatmap
───────────────────────────────────────────────────────────────
Hai loại interpretability:
  1. GradCAM trên backbone features → heatmap tại prototype similarity layer
  2. Prototype Similarity Heatmap → cosine distance giữa embedding và prototype

Cách dùng:
    from gradcam_interpreter import GradCAMInterpreter, PrototypeHeatmapGenerator

    # GradCAM
    interp = GradCAMInterpreter(model, target_layer_name="backbone.stages.3")
    heatmap = interp.generate(image_tensor, class_idx=0)  # class_idx = 0..4

    # Prototype similarity breakdown (cho toàn bộ classes)
    sim_scores = interp.get_prototype_similarities(image_tensor)
    # sim_scores = [0.12, 0.85, 0.03, 0.01, 0.00] cho 5 class
    # → Model CONFIDENT vì class 1 cao vượt trội

    # Prototype heatmap: spatial similarity map với mỗi prototype
    proto_hm_gen = PrototypeHeatmapGenerator(model)
    spatial_maps = proto_hm_gen.generate_spatial_maps(image_tensor)
    # spatial_maps[0].shape = (H, W) — spatial similarity với prototype TA
"""

import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image


# ─────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────
class GradCAM:
    """
    Generic GradCAM cho bất kỳ model có forward(x) → logits.
    Dùng gradient của logits[class_idx] w.r.t. feature maps.

    target_layer: nn.Module — layer mà ta muốn hook gradient/activation.
                  ConvNeXtV2: "backbone.stages.3" hoặc "backbone.stages.2"
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = copy.deepcopy(model)  # copy để không ảnh hưởng train state
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        self._register_hooks()

    def _get_target_layer(self) -> torch.nn.Module:
        """Navigate model.children() để tìm layer theo tên."""
        parts = self.target_layer_name.split(".")
        module = self.model
        for part in parts:
            module = getattr(module, part)
        return module

    def _register_hooks(self):
        target = self._get_target_layer()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    @torch.enable_grad()
    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        """
        input_tensor: (1, C, H, W) — single image
        class_idx:     None → dùng argmax của logits
                      int  → specific class
        Returns: (H, W) numpy heatmap [0..1]
        """
        self.model.zero_grad()
        _, logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        # weights = global average pooling over gradients
        grads = self.gradients  # (1, C, H, W)
        acts = self.activations  # (1, C, H', W')

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Upsample to input size
        h, w = input_tensor.shape[2:]
        cam = cv2.resize(cam, (w, h))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam


class GradCAMInterpreter:
    """
    Wrapper cao hơn: chạy GradCAM + overlay lên ảnh gốc.
    Đồng thời trả về prototype similarity scores.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "backbone.stages.3"):
        self.model = model
        self.target_layer_name = target_layer_name
        self._cam = GradCAM(model, target_layer_name)

    @torch.no_grad()
    def get_prototype_similarities(self, input_tensor: torch.Tensor):
        """
        Trả về cosine similarity giữa projected embedding và TẤT CẢ prototypes.
        input_tensor: (1, C, H, W)
        Returns: list[float] — similarity scores cho 5 class
        """
        feats, logits = self.model(input_tensor)
        # feats đã L2-normalized từ projection head
        # logits = sim * temperature → softmax để được probability
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        return probs.tolist()

    @torch.enable_grad()
    def generate_overlay(self, image_tensor: torch.Tensor, class_idx: int | None = None, alpha: float = 0.4):
        """
        Tạo ảnh overlay: heatmap GradCAM + original image.
        Trả về PIL Image.
        """
        cam = self._cam.generate(image_tensor, class_idx)

        # Chuyển tensor → numpy image
        img_np = image_tensor.squeeze().cpu().numpy()
        img_np = img_np.transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        # Heatmap colormap
        cmap = cm.get_cmap("jet")
        heatmap_np = cmap(cam)[:, :, :3]  # RGBA → RGB

        # Blend
        blended = (1 - alpha) * img_np + alpha * heatmap_np
        blended = (blended * 255).astype(np.uint8)
        return Image.fromarray(blended)

    def generate_batch(self, images: torch.Tensor, class_indices: list[int]):
        """
        images: (B, C, H, W)
        class_indices: list[int] cùng length với batch
        Trả về list các PIL overlay images.
        """
        overlays = []
        for i in range(len(images)):
            inp = images[i : i + 1].to(next(self.model.parameters()).device)
            overlay = self.generate_overlay(inp, class_indices[i])
            overlays.append(overlay)
        return overlays


# ─────────────────────────────────────────────
# Prototype Spatial Similarity Heatmap
# ─────────────────────────────────────────────
class PrototypeHeatmapGenerator:
    """
    Tạo spatial similarity map: với mỗi prototype (class),
    tính cosine similarity tại từng spatial position.

    Hoạt động bằng cách:
    1. Hook vào output của backbone để lấy feature maps (C, H', W')
    2. Với mỗi prototype vector, tính cosine sim với từng channel
    3. Tổng hợp lại → (H', W') similarity map

    Điều này cho thấy model "nhìn" vào vùng nào trên ảnh để quyết định
    class → thể hiện rõ hơn ambiguous cells.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = copy.deepcopy(model)
        self.model.eval()

    def _extract_spatial_feature_map(self, inp: torch.Tensor) -> torch.Tensor:
        """Return backbone features as spatial map (B, C, H, W)."""
        backbone = self.model.backbone

        if hasattr(backbone, "forward_features"):
            feats = backbone.forward_features(inp)
        else:
            feats = backbone(inp)

        # Some backbones may already return pooled vectors (B, C).
        # Keep code robust by treating them as a 1x1 spatial map.
        if feats.dim() == 2:
            feats = feats.unsqueeze(-1).unsqueeze(-1)

        if feats.dim() != 4:
            raise RuntimeError(f"Expected spatial features with shape (B, C, H, W), got {tuple(feats.shape)}")

        return feats

    def _get_fc_weight(self):
        """Return final linear weight for FC-head fallback."""
        fc = self.model.clf_head
        if hasattr(fc, "weight"):
            return fc.weight

        # MalariaProtoCLFv2 FC head is nn.Sequential(..., Linear)
        linear_layers = [m for m in fc.modules() if isinstance(m, torch.nn.Linear)]
        if not linear_layers:
            raise RuntimeError("Cannot find Linear layer in clf_head for FC fallback")
        return linear_layers[-1].weight

    @torch.no_grad()
    def generate_spatial_maps(self, input_tensor: torch.Tensor):
        """
        input_tensor: (1, C, H, W)
        Returns: dict {class_name: (H', W') numpy array}
                 — spatial cosine similarity với mỗi class prototype
        """
        inp = input_tensor.to(next(self.model.parameters()).device)

        # Forward để lấy feature maps
        feats = self._extract_spatial_feature_map(inp)  # (1, feat_dim, H', W')

        # Project từng spatial position
        spatial_tokens = feats.flatten(2).transpose(1, 2)  # (1, H'*W', feat_dim)
        proj = self.model.proj_head  # MLP: feat_dim → proj_dim
        spatial_feats = proj(spatial_tokens)  # (1, H'*W', proj_dim)
        spatial_feats = F.normalize(spatial_feats, dim=-1)  # L2 normalize

        # Prototype vectors (đã normalize)
        if hasattr(self.model, "clf_head") and hasattr(self.model.clf_head, "prototypes"):
            prototypes = F.normalize(self.model.clf_head.prototypes, dim=1)  # (num_classes, proj_dim)
        else:
            # FC head fallback: dùng last linear weights như class proxies.
            prototypes = F.normalize(self._get_fc_weight(), dim=1)  # (num_classes, D)

        # Ensure feature/prototype dimensions match.
        if spatial_feats.shape[-1] != prototypes.shape[-1]:
            spatial_feats = F.normalize(spatial_tokens, dim=-1)
            if spatial_feats.shape[-1] != prototypes.shape[-1]:
                raise RuntimeError(
                    "Dimension mismatch between spatial features and class vectors: "
                    f"{spatial_feats.shape[-1]} vs {prototypes.shape[-1]}"
                )

        # Cosine similarity: (1, H'*W', proj_dim) @ (num_classes, proj_dim).T
        #                     = (1, H'*W', num_classes)
        sim = torch.matmul(spatial_feats, prototypes.T).squeeze(0)  # (H'*W', num_classes)
        sim = ((sim + 1.0) / 2.0).clamp(0.0, 1.0)
        sim = sim.cpu().numpy()  # mỗi cột = similarity map flatten

        H, W = feats.shape[2:]
        from dataset import CLASS_NAMES

        result = {}
        for cls_idx in range(prototypes.shape[0]):
            cls_name = CLASS_NAMES.get(cls_idx, f"Class_{cls_idx}")
            result[cls_name] = sim[:, cls_idx].reshape(H, W)
            # Upsample to input size
            result[cls_name] = cv2.resize(result[cls_name], (input_tensor.shape[3], input_tensor.shape[2]))

        return result

    @torch.no_grad()
    def overlay_spatial_maps(self, input_tensor: torch.Tensor, alpha: float = 0.5):
        """
        Tạo grid overlay: mỗi cell = spatial map cho 1 class.
        Returns: PIL Image (grid 1×5 hoặc 2×3).
        """
        spatial_maps = self.generate_spatial_maps(input_tensor)

        # Denormalize input
        img_np = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = np.clip(img_np * std + mean, 0, 1)

        import matplotlib.pyplot as plt

        n_classes = len(spatial_maps)
        cols = min(5, n_classes)
        rows = (n_classes + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[a] for a in axes]

        for idx, (cls_name, sim_map) in enumerate(spatial_maps.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col]

            ax.imshow(img_np)
            im = ax.imshow(sim_map, cmap="jet", alpha=alpha, vmin=0, vmax=1)
            ax.set_title(f"{cls_name}", fontsize=11, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Hide unused
        for idx in range(n_classes, rows * cols):
            row = idx // cols
            col = idx % cols
            if row < len(axes) and col < len(axes[row]):
                axes[row][col].axis("off")

        plt.tight_layout()
        from io import BytesIO

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        plt.close()
        return Image.open(buf)


# ─────────────────────────────────────────────
# Unified Interpreter cho inference pipeline
# ─────────────────────────────────────────────
class MalariaInterpreter:
    """
    Unified interface: kết hợp GradCAM + Prototype Similarity + Spatial Maps.
    Dùng trong evaluate / misclassification analysis để giải thích kết quả.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "backbone.stages.3"):
        self.gradcam = GradCAMInterpreter(model, target_layer_name)
        self.spatial = PrototypeHeatmapGenerator(model)

    @torch.no_grad()
    def explain_sample(self, image_tensor: torch.Tensor, true_label: int | None = None):
        """
        Trả về dict chứa đầy đủ thông tin giải thích cho 1 sample.

        Returns:
            {
                "proto_similarities": [0.12, 0.85, 0.03, 0.01, 0.00],  # 5 class probs
                "predicted_class": 1,
                "confidence": 0.85,
                "entropy": 0.42,         # lower = more confident
                "margin": 0.82,           # pred_prob - second_prob
                "gradcam_overlay": PIL.Image,
                "spatial_maps": dict,     # class_name → (H, W) array
            }
        """
        from dataset import CLASS_NAMES

        device = next(self.model.parameters()).device

        # 1. Prototype similarities (softmax probabilities)
        sim_scores = self.gradcam.get_prototype_similarities(image_tensor)
        probs = np.array(sim_scores)
        pred = int(probs.argmax())
        conf = float(probs.max())

        # 2. Confidence metrics
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])

        # 3. GradCAM overlay
        gradcam_overlay = self.gradcam.generate_overlay(image_tensor, pred)

        # 4. Spatial maps
        spatial_maps = self.spatial.generate_spatial_maps(image_tensor)

        return {
            "proto_similarities": sim_scores,
            "predicted_class": pred,
            "predicted_name": CLASS_NAMES.get(pred, "?"),
            "true_label": true_label,
            "true_name": CLASS_NAMES.get(true_label, "?") if true_label is not None else None,
            "confidence": conf,
            "entropy": entropy,
            "margin": margin,
            "gradcam_overlay": gradcam_overlay,
            "spatial_maps": spatial_maps,
        }

    def explain_batch(self, images: torch.Tensor, labels: list[int] | None = None):
        """Explain nhiều samples, trả về list dict."""
        results = []
        for i in range(len(images)):
            inp = images[i : i + 1].to(next(self.gradcam.model.parameters()).device)
            true_lbl = labels[i] if labels is not None else None
            results.append(self.explain_sample(inp, true_lbl))
        return results
