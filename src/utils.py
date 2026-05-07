"""
Utility functions for DermoAI Pipeline.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with VRAM-aware formatting.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured root logger.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Silence noisy third-party loggers that inherit from root at INFO
    for noisy in ("httpx", "httpcore", "huggingface_hub", "huggingface_hub.utils._http"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger = logging.getLogger("dermoai")
    logger.info("DermoAI Pipeline logging initialized")
    return logger


def setup_device() -> torch.device:
    """Force cuda:0 if available, fallback to CPU.

    Returns:
        torch.device for computation.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.getLogger("dermoai").info(
            f"GPU: {name} | VRAM: {vram:.1f}GB | CUDA compute: "
            f"{torch.cuda.get_device_capability(0)}"
        )
        return device
    logging.getLogger("dermoai").warning("CUDA not available — running on CPU (slow)")
    return torch.device("cpu")


def get_vram_status() -> dict[str, float]:
    """Get current VRAM usage statistics.

    Returns:
        dict with keys: total_gb, used_gb, free_gb.
        Returns zeros if CUDA not available.
    """
    if not torch.cuda.is_available():
        return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0}
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    used_bytes = total_bytes - free_bytes
    return {
        "total_gb": total_bytes / 1e9,
        "used_gb": used_bytes / 1e9,
        "free_gb": free_bytes / 1e9,
    }


def load_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_image(path: Union[str, Path]) -> tuple[Image.Image, torch.Tensor]:
    """Load image and return both PIL and tensor forms.

    Args:
        path: Path to image file.

    Returns:
        Tuple of (PIL Image RGB, normalized tensor [1, 3, 224, 224]).

    Raises:
        FileNotFoundError: If image does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    pil_image = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    tensor = transform(pil_image).unsqueeze(0)
    return pil_image, tensor


def save_json(data: Any, path: Union[str, Path]) -> None:
    """Save data as JSON file, creating parent directories as needed.

    Args:
        data: JSON-serializable data.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def create_side_by_side(
    images: list[np.ndarray],
    titles: list[str],
    figsize: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Create a side-by-side montage of images.

    Args:
        images: List of RGB numpy arrays [H, W, 3].
        titles: List of titles for each image.
        figsize: Optional figure size tuple.

    Returns:
        RGB numpy array of the montage.
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    result = np.array(Image.open(buf).convert("RGB"))
    return result


def format_report_markdown(report_dict: dict) -> str:
    """Format a report dictionary as a Markdown string.

    Args:
        report_dict: Report dictionary with sections and metadata.

    Returns:
        Formatted Markdown string.
    """
    lines = ["# REFERTO DERMATOLOGICO - ANALISI ABC", ""]

    sections = report_dict.get("sections", {})
    section_labels = {
        "asymmetry": "A - Asimmetria",
        "borders": "B - Bordi",
        "color": "C - Colore",
        "conclusion": "CONCLUSIONE",
    }

    for key, label in section_labels.items():
        text = sections.get(key, "")
        if text:
            lines.append(f"**{label}:**")
            lines.append(text)
            lines.append("")

    risk = report_dict.get("risk_level", "")
    if risk:
        lines.append(f"**Livello di Rischio:** {risk.upper()}")
        lines.append("")

    sources = report_dict.get("rag_sources_used", [])
    if sources:
        lines.append("**Fonti RAG Utilizzate:**")
        for src in sources:
            lines.append(f"- {src}")
        lines.append("")

    meta = report_dict.get("metadata", {})
    if meta:
        lines.append("---")
        lines.append(f"*Modello: {meta.get('model_used', 'N/A')} | "
                     f"Quantizzazione: {meta.get('quantization_used', 'N/A')} | "
                     f"VRAM: {meta.get('vram_used_gb', 0):.1f}GB | "
                     f"Tempo: {meta.get('generation_time_s', 0):.1f}s*")
        lines.append("")

    lines.append(
        "> **NOTA:** Questo referto è generato da un sistema AI di supporto "
        "e NON sostituisce il giudizio clinico di un dermatologo."
    )
    return "\n".join(lines)
