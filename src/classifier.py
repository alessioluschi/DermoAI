"""
Melanoma Classifier using EfficientNet-B0 with fp16 inference on RTX 3080.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

logger = logging.getLogger(__name__)


class MelanomaClassifier:
    """EfficientNet-B0 classifier for melanoma vs nevus classification.

    Loads a pre-trained EfficientNet-B0 model and performs inference using
    fp16 precision on CUDA for optimal performance on RTX 3080 (Ampere).
    Handles both state_dict formats: pure and wrapped with 'model_state_dict'.

    Args:
        weights_path: Path to the .pth checkpoint file.
        num_classes: Number of output classes. Default: 2.
        class_names: List of class names in training order (index 0 = melanoma).
            Default: ["melanoma", "nevus"].
        device: Device string. Default: "cuda:0" with CPU fallback.
        square_crop: If True, apply dynamic CenterCrop to shorter side before
            resize. Default: True.
        colour_normalisation: If True, normalise colour toward ISIC statistics
            before inference (for domain-shift mitigation). Default: False.
        melanoma_threshold: Asymmetric decision threshold for predict_tta()
            (applied to class index 0 = melanoma). Values below 0.5 increase
            sensitivity. Default: 0.40.
        tta_n_augmentations: Number of TTA augmentations used in predict_tta().
            Maximum 5. Default: 5.
    """

    # ISIC HAM10000 colour statistics (RGB, float 0–1)
    _ISIC_MEAN: list[float] = [0.7635, 0.5461, 0.5705]
    _ISIC_STD:  list[float] = [0.1409, 0.1526, 0.1700]

    def __init__(
        self,
        weights_path: Union[str, Path],
        num_classes: int = 2,
        class_names: list[str] | None = None,
        device: str = "cuda:0",
        square_crop: bool = True,
        colour_normalisation: bool = False,
        melanoma_threshold: float = 0.40,
        tta_n_augmentations: int = 5,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.num_classes = num_classes
        self.class_names = class_names or ["melanoma", "nevus"]
        self.square_crop = square_crop
        self.colour_normalisation = colour_normalisation
        self.melanoma_threshold = melanoma_threshold
        self.tta_n_augmentations = min(tta_n_augmentations, 5)
        self.device = self._setup_device(device)
        self.model = self._build_model()
        self._load_weights()
        self.transform = self._build_transform()
        if self.colour_normalisation:
            logger.info("Colour normalisation: ENABLED (ISIC target mean/std)")
        else:
            logger.debug("Colour normalisation: DISABLED")
        logger.info(
            f"MelanomaClassifier ready on {self.device} | "
            f"classes={self.class_names} | square_crop={self.square_crop}"
        )

    def _setup_device(self, requested: str) -> torch.device:
        """Setup compute device with CPU fallback if CUDA unavailable."""
        if torch.cuda.is_available():
            return torch.device(requested)
        logger.warning("CUDA unavailable — falling back to CPU")
        return torch.device("cpu")

    def _build_model(self) -> nn.Module:
        """Build EfficientNet-B0 with custom 2-class head.

        Returns:
            Model on target device in fp16 (CUDA) or fp32 (CPU).
        """
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, self.num_classes),
        )
        if self.device.type == "cuda":
            model = model.half()  # fp16 on Ampere — faster + less VRAM
        return model.to(self.device)

    def _load_weights(self) -> None:
        """Load checkpoint weights supporting both state_dict formats.

        Supported formats:
            - Pure state_dict: torch.save(model.state_dict(), path)
            - Wrapped dict: torch.save({"model_state_dict": ...}, path)

        Raises:
            FileNotFoundError: If weights file does not exist.
            RuntimeError: If state_dict cannot be loaded into model.
        """
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}\n"
                "Place the .pth file in the models/ directory."
            )

        try:
            checkpoint = torch.load(
                self.weights_path,
                map_location=self.device,
                weights_only=True,
            )
        except Exception:
            # Fallback for older checkpoints with non-tensor objects
            checkpoint = torch.load(
                self.weights_path,
                map_location=self.device,
            )

        # Detect format: wrapped dict vs pure state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            logger.info("Checkpoint format: wrapped dict with 'model_state_dict'")
        else:
            state_dict = checkpoint
            logger.info("Checkpoint format: pure state_dict")

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        logger.info(f"Loaded weights from {self.weights_path}")
        mapping = " | ".join(
            f"index {i} → '{name}'" for i, name in enumerate(self.class_names)
        )
        logger.warning(f"Class mapping: {mapping} — verify this matches the training checkpoint order")

    def _build_transform(self) -> transforms.Compose:
        """Build ImageNet-standard preprocessing pipeline (post-crop).

        The optional square CenterCrop is applied dynamically in predict()
        before this transform, so the input here is always square.

        Returns:
            Composed transform: Resize(224) → ToTensor → Normalize.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def normalize_to_isic(image: Image.Image) -> Image.Image:
        """Normalise image colour statistics toward ISIC dataset distribution.

        Applies channel-wise linear normalisation to match ISIC/HAM10000
        training data statistics, reducing scanner-induced domain shift.

        Algorithm (per RGB channel c):
            arr_norm = (arr - test_mean) / (test_std + 1e-8)
            arr_out  = clip(arr_norm * ISIC_STD[c] + ISIC_MEAN[c], 0, 1)

        Args:
            image: Input PIL Image (RGB mode).

        Returns:
            Colour-normalised PIL Image with the same size and mode.
        """
        arr = np.array(image, dtype=np.float32) / 255.0
        out = np.zeros_like(arr)
        for c in range(3):
            ch = arr[:, :, c]
            test_mean = float(ch.mean())
            test_std = float(ch.std())
            arr_norm = (ch - test_mean) / (test_std + 1e-8)
            out[:, :, c] = np.clip(
                arr_norm * MelanomaClassifier._ISIC_STD[c] + MelanomaClassifier._ISIC_MEAN[c],
                0.0,
                1.0,
            )
        return Image.fromarray((out * 255).astype(np.uint8))

    def predict(self, image_path: Union[str, Path]) -> dict:
        """Run fp16 inference on a single image.

        Args:
            image_path: Path to the input image file.

        Returns:
            dict with keys:
                - class (str): Predicted class name.
                - confidence (float): Confidence score for predicted class.
                - probabilities (dict[str, float]): Per-class probabilities.
                - preprocessing_info (dict): Preprocessing metadata.

        Raises:
            FileNotFoundError: If image does not exist.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (W, H)
        W, H = original_size

        # Dynamic square crop: crop to shorter side before resize
        crop_size = min(H, W)
        square_crop_applied = self.square_crop and (H != W)
        if square_crop_applied:
            image = transforms.CenterCrop(crop_size)(image)
            logger.info(
                f"Preprocessing: CenterCrop({crop_size}) → Resize(224, 224) "
                f"[{W}×{H} → {crop_size}×{crop_size} → 224×224]"
            )
        else:
            logger.debug(
                f"Preprocessing: Resize(224, 224) (image already square: {W}×{H})"
            )

        tensor = self.transform(image).unsqueeze(0)
        if self.device.type == "cuda":
            tensor = tensor.half().to(self.device)
        else:
            tensor = tensor.to(self.device)

        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type, enabled=(self.device.type == "cuda")
        ):
            logits = self.model(tensor)
            probs = torch.softmax(logits.float(), dim=1)

        probs_np = probs.cpu().numpy()[0]
        predicted_idx = int(probs_np.argmax())

        result = {
            "class": self.class_names[predicted_idx],
            "confidence": float(probs_np[predicted_idx]),
            "probabilities": {
                name: float(prob)
                for name, prob in zip(self.class_names, probs_np)
            },
            "preprocessing_info": {
                "original_size": original_size,
                "square_crop_applied": square_crop_applied,
                "crop_size": crop_size if square_crop_applied else None,
                "model_input_size": (224, 224),
                "normalization": "ImageNet",
                "precision": "fp16" if self.device.type == "cuda" else "fp32",
            },
        }

        logger.info(
            f"Prediction: {result['class']} "
            f"(confidence={result['confidence']:.3f}) "
            f"| image={image_path.name}"
        )
        return result

    def predict_tta(self, image_path: Union[str, Path]) -> dict:
        """Run Test-Time Augmentation inference with asymmetric threshold.

        Applies up to 5 deterministic augmentations, averages the softmax
        probability vectors, and applies an asymmetric melanoma threshold on
        class index 0 (melanoma). Colour normalisation is applied if enabled
        at construction time.

        TTA augmentations (applied after square crop + colour normalisation):
            1. Standard:  Resize(224, 224)
            2. HFlip:     HorizontalFlip + Resize(224, 224)
            3. VFlip:     VerticalFlip + Resize(224, 224)
            4. Rot90:     Rotation(90°) + Resize(224, 224)
            5. Rot180:    Rotation(180°) + Resize(224, 224)

        Asymmetric threshold: if P(class_0) > melanoma_threshold → melanoma,
        else → nevus. A threshold < 0.5 biases toward sensitivity (fewer FN).

        Args:
            image_path: Path to the input image file.

        Returns:
            dict with keys:
                - class (str): Predicted class name.
                - confidence (float): Mean TTA probability for predicted class.
                - probabilities (dict[str, float]): Per-class mean probabilities.
                - tta_mean_probs (list[float]): Full mean softmax vector.
                - melanoma_threshold (float): Threshold applied.
                - n_augmentations (int): Number of augmentations used.
                - preprocessing_info (dict): Preprocessing metadata.

        Raises:
            FileNotFoundError: If image does not exist.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        crop_size = min(H, W)
        square_crop_applied = self.square_crop and H != W

        # Step 1: square crop
        if square_crop_applied:
            image = transforms.CenterCrop(crop_size)(image)

        # Step 2: colour normalisation
        if self.colour_normalisation:
            image = MelanomaClassifier.normalize_to_isic(image)

        # Step 3: 5 deterministic augmentation pipelines (pre-ToTensor)
        aug_pipelines = [
            transforms.Compose([transforms.Resize((224, 224))]),
            transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.Resize((224, 224))]),
            transforms.Compose([transforms.RandomVerticalFlip(p=1.0), transforms.Resize((224, 224))]),
            transforms.Compose([transforms.RandomRotation((90, 90)), transforms.Resize((224, 224))]),
            transforms.Compose([transforms.RandomRotation((180, 180)), transforms.Resize((224, 224))]),
        ]
        to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        n_augs = min(self.tta_n_augmentations, len(aug_pipelines))
        all_probs: list[np.ndarray] = []

        for aug in aug_pipelines[:n_augs]:
            aug_img = aug(image)
            tensor = to_tensor_norm(aug_img).unsqueeze(0)
            if self.device.type == "cuda":
                tensor = tensor.half().to(self.device)
            else:
                tensor = tensor.to(self.device)

            with torch.no_grad(), torch.amp.autocast(
                device_type=self.device.type, enabled=(self.device.type == "cuda")
            ):
                logits = self.model(tensor)
                probs = torch.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy()[0])

        # Step 4: average and apply asymmetric threshold
        mean_probs = np.mean(all_probs, axis=0)
        p_melanoma = float(mean_probs[0])  # class index 0 = melanoma

        if p_melanoma > self.melanoma_threshold:
            predicted_idx = 0
        else:
            predicted_idx = 1
        predicted_class = self.class_names[predicted_idx]

        logger.info(
            f"TTA mean P({self.class_names[0]})={p_melanoma:.4f} "
            f"P({self.class_names[1]})={float(mean_probs[1]):.4f} "
            f"threshold={self.melanoma_threshold} → {predicted_class}"
        )

        return {
            "class": predicted_class,
            "confidence": float(mean_probs[predicted_idx]),
            "probabilities": {
                name: float(prob)
                for name, prob in zip(self.class_names, mean_probs)
            },
            "tta_mean_probs": mean_probs.tolist(),
            "melanoma_threshold": self.melanoma_threshold,
            "n_augmentations": n_augs,
            "preprocessing_info": {
                "original_size": (W, H),
                "square_crop_applied": square_crop_applied,
                "crop_size": crop_size,
                "colour_normalisation": self.colour_normalisation,
                "model_input_size": (224, 224),
                "normalization": "ImageNet",
                "precision": "fp16" if self.device.type == "cuda" else "fp32",
            },
        }

    def diagnose_prediction(self, image_path: Union[str, Path]) -> dict:
        """Run predict() and return extended preprocessing diagnostics.

        Useful for verifying that CenterCrop and Resize are applied correctly
        to a specific image before using it in the full pipeline.

        Args:
            image_path: Path to the input image file.

        Returns:
            All keys from predict() plus:
                - preprocessing_steps (list[str]): Human-readable description
                  of each transform applied in order.
        """
        result = self.predict(image_path)
        info = result["preprocessing_info"]
        W, H = info["original_size"]

        steps: list[str] = []
        if info["square_crop_applied"]:
            c = info["crop_size"]
            steps.append(f"CenterCrop({c})  [{W}×{H} → {c}×{c}]")
        steps.append(f"Resize(224, 224)  [→ 224×224]")
        steps.append("ToTensor")
        steps.append(
            "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
        )

        result["preprocessing_steps"] = steps
        return result

    def get_target_layer(self) -> nn.Module:
        """Return the GradCAM target layer (last features block).

        Returns:
            model.features[-1] — EfficientNet-B0 MBConv block at index 8.
        """
        return self.model.features[-1]

    def offload(self) -> None:
        """Move model to CPU and free CUDA memory.

        Call this after classification when GradCAM is not needed immediately.
        """
        self.model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug("MelanomaClassifier offloaded from GPU")
