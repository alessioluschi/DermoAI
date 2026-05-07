"""
GradCAM++ explainer for EfficientNet-B0 melanoma classifier.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

logger = logging.getLogger(__name__)

# 3x3 grid region names (row, col) -> Italian description
_REGION_NAMES: dict[tuple[int, int], str] = {
    (0, 0): "zona superiore-sinistra",
    (0, 1): "zona superiore-centrale",
    (0, 2): "zona superiore-destra",
    (1, 0): "zona centrale-sinistra",
    (1, 1): "zona centrale della lesione",
    (1, 2): "zona centrale-destra",
    (2, 0): "zona inferiore-sinistra",
    (2, 1): "zona inferiore-centrale",
    (2, 2): "zona inferiore-destra",
}


class GradCAMExplainer:
    """Generates GradCAM++ activation maps for melanoma classification.

    Supports GradCAM++, GradCAM, and ScoreCAM methods via pytorch-grad-cam.
    Automatically handles fp16/fp32 conversion required for gradient computation.
    Frees CUDA memory after each heatmap generation.

    Args:
        model: The EfficientNet-B0 classifier model (on target device).
        method: CAM method — "gradcam++", "gradcam", or "scorecam". Default: "gradcam++".
        device: Compute device. Auto-detected if None.
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "gradcam++",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.method = method.lower()
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._cam_class = self._select_cam_class()
        logger.info(f"GradCAMExplainer initialized | method={self.method}")

    def _select_cam_class(self):
        """Select pytorch-grad-cam implementation class.

        Returns:
            CAM class (GradCAMPlusPlus, GradCAM, or ScoreCAM).
        """
        mapping = {
            "gradcam++": GradCAMPlusPlus,
            "gradcam": GradCAM,
            "scorecam": ScoreCAM,
        }
        if self.method not in mapping:
            logger.warning(
                f"Unknown CAM method '{self.method}', defaulting to GradCAM++"
            )
            return GradCAMPlusPlus
        return mapping[self.method]

    def _preprocess_image(
        self, image_path: Path
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Preprocess image for CAM computation.

        Args:
            image_path: Path to input image.

        Returns:
            Tuple of (rgb_float_array [H,W,3] in [0,1], preprocessed_tensor).
        """
        pil_image = Image.open(image_path).convert("RGB").resize((224, 224))
        rgb_array = np.array(pil_image, dtype=np.float32) / 255.0

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        # GradCAM requires float32 for gradient computation
        tensor = preprocess(pil_image).unsqueeze(0).float().to(self.device)
        return rgb_array, tensor

    def generate(
        self,
        image_path: Union[str, Path],
        model: nn.Module,
        target_class: Optional[int] = None,
    ) -> dict:
        """Generate GradCAM++ heatmap and overlay for an input image.

        Temporarily converts fp16 model to fp32 for gradient computation,
        then restores fp16. Calls torch.cuda.empty_cache() after generation.

        Args:
            image_path: Path to the input image.
            model: The EfficientNet-B0 classifier (modified in-place for fp32).
            target_class: Target class index (None = use predicted class).

        Returns:
            dict with keys:
                - heatmap (np.ndarray): Grayscale activation map [0,1], shape [H,W].
                - overlay (np.ndarray): RGB overlay on original image, shape [H,W,3].
                - focus_regions (str): Italian description of activated regions.
                - activation_percentage (float): Percentage of image with activation > 0.5.

        Raises:
            FileNotFoundError: If image does not exist.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        rgb_array, tensor = self._preprocess_image(image_path)
        target_layers = [model.features[-1]]
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

        # GradCAM requires float32 — temporarily convert from fp16
        was_half = next(model.parameters()).dtype == torch.float16
        if was_half:
            model.float()

        try:
            with self._cam_class(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=tensor, targets=targets)
        finally:
            # Always restore original precision
            if was_half:
                model.half()

        heatmap = grayscale_cam[0]  # [H, W] float32 in [0, 1]
        overlay = show_cam_on_image(rgb_array, heatmap, use_rgb=True)

        focus_regions = self._describe_focus_regions(heatmap)
        activation_percentage = float((heatmap > 0.5).mean() * 100)

        # Release gradient buffers
        torch.cuda.empty_cache()
        gc.collect()

        logger.info(
            f"GradCAM++ | focus='{focus_regions}' | "
            f"activation={activation_percentage:.1f}%"
        )

        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "focus_regions": focus_regions,
            "activation_percentage": activation_percentage,
        }

    def _describe_focus_regions(
        self, heatmap: np.ndarray, threshold: float = 0.5
    ) -> str:
        """Describe activated regions as human-readable text using 3x3 grid.

        Args:
            heatmap: Grayscale activation map in [0, 1].
            threshold: Mean activation threshold per cell.

        Returns:
            Italian description of activated grid cells.
        """
        h, w = heatmap.shape
        cell_h, cell_w = h // 3, w // 3

        active_regions = []
        for row in range(3):
            for col in range(3):
                r0, r1 = row * cell_h, (row + 1) * cell_h
                c0, c1 = col * cell_w, (col + 1) * cell_w
                cell = heatmap[r0:r1, c0:c1]
                if cell.mean() > threshold:
                    active_regions.append(_REGION_NAMES[(row, col)])

        if not active_regions:
            # Fall back to the single highest-activation cell
            max_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
            row = min(int(max_idx[0] * 3 / h), 2)
            col = min(int(max_idx[1] * 3 / w), 2)
            active_regions = [_REGION_NAMES[(row, col)]]

        if len(active_regions) == 1:
            return active_regions[0]
        elif len(active_regions) == 2:
            return f"{active_regions[0]} e {active_regions[1]}"
        else:
            return ", ".join(active_regions[:-1]) + f" e {active_regions[-1]}"

    def save_visualization(
        self,
        image_path: Union[str, Path],
        gradcam_result: dict,
        output_path: Union[str, Path],
    ) -> None:
        """Save side-by-side 3-panel visualization: original | heatmap | overlay.

        Args:
            image_path: Path to original image.
            gradcam_result: Output dictionary from generate().
            output_path: Path for the output PNG file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        original = np.array(
            Image.open(image_path).convert("RGB").resize((224, 224))
        )
        heatmap = gradcam_result["heatmap"]
        overlay = gradcam_result["overlay"]
        heatmap_rgb = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        axes[0].imshow(original)
        axes[0].set_title("Original Image", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(heatmap_rgb)
        axes[1].set_title(
            f"GradCAM++ Heatmap\n(activation: {gradcam_result['activation_percentage']:.1f}%)",
            fontsize=11,
        )
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Overlay\nFocus: {gradcam_result['focus_regions']}", fontsize=11
        )
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"GradCAM visualization saved → {output_path}")
