"""
Quantitative dermoscopy feature extractor for ABCDE prompt enrichment.

CPU-only module (numpy, scipy, scikit-image, scikit-learn).
No torch / CUDA dependency — runs after GradCAM offload with zero VRAM cost.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ─── Defaults ────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "heatmap_threshold": 0.30,
    "kmeans_k": 5,
    "kmeans_random_state": 42,
    "asymmetry_thresholds": [0.2, 0.4, 0.6],
    "border_thresholds": [0.4, 0.6, 0.8],
    "heterogeneity_thresholds": [0.15, 0.30, 0.50],
    "gradcam_peak_threshold": 0.60,
    "top_n_regions": 3,
}

# ─── Grid labels (3×3, Italian) ──────────────────────────────────────────────
_GRID_NAMES: dict[tuple[int, int], str] = {
    (0, 0): "superiore-sinistro",
    (0, 1): "superiore-centrale",
    (0, 2): "superiore-destro",
    (1, 0): "centrale-sinistro",
    (1, 1): "centrale",
    (1, 2): "centrale-destro",
    (2, 0): "inferiore-sinistro",
    (2, 1): "inferiore-centrale",
    (2, 2): "inferiore-destro",
}


class DermoscopyFeatureExtractor:
    """Quantitative dermoscopy feature extractor for ABCDE prompt enrichment.

    Stateless — no constructor parameters needed.  All configuration is
    supplied via the ``config`` dict passed to :meth:`extract_all`, with
    safe defaults for every key.

    All methods operate on CPU using numpy / scipy / scikit-image / scikit-learn.
    OpenCV is used opportunistically for contour extraction if importable;
    ``scipy.ndimage`` is the fallback.

    Example:
        >>> extractor = DermoscopyFeatureExtractor()
        >>> features = extractor.extract_all(pil_image, heatmap_array)
        >>> print(features["prompt_block"])
    """

    # ── Method 1: Asymmetry ───────────────────────────────────────────────────

    def compute_asymmetry(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        threshold: float = 0.30,
        asymmetry_thresholds: Optional[list[float]] = None,
    ) -> dict:
        """Compute lesion asymmetry along horizontal and vertical axes.

        Uses the GradCAM++ heatmap thresholded at *threshold* as a binary
        lesion mask.  Falls back to Otsu thresholding when the heatmap mask
        is smaller than 100 pixels.

        Args:
            image: RGB image as float32 np.ndarray (H, W, 3), values in [0, 1].
            heatmap: GradCAM++ activation map (H, W), values in [0, 1].
            threshold: Binary mask threshold applied to *heatmap*.
            asymmetry_thresholds: Three breakpoints [t1, t2, t3] that separate
                the four label categories.  Defaults to [0.2, 0.4, 0.6].

        Returns:
            dict with keys:

            - ``asymmetry_x`` (float): Horizontal-axis fold asymmetry, [0, 1].
            - ``asymmetry_y`` (float): Vertical-axis fold asymmetry, [0, 1].
            - ``asymmetry_mean`` (float): Mean of x and y asymmetry.
            - ``asymmetry_label`` (str): Italian category label.
        """
        if asymmetry_thresholds is None:
            asymmetry_thresholds = [0.2, 0.4, 0.6]

        mask = heatmap >= threshold

        # Otsu fallback when heatmap gives too few foreground pixels
        if int(mask.sum()) < 100:
            logger.warning(
                "compute_asymmetry: heatmap mask < 100 px — falling back to Otsu"
            )
            gray = (
                0.299 * image[:, :, 0]
                + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]
            )
            try:
                from skimage.filters import threshold_otsu
                mask = gray > threshold_otsu(gray)
            except Exception:
                # Minimal fallback: assume lesion is darker than mean
                mask = gray < float(gray.mean())

        if int(mask.sum()) < 100:
            logger.warning(
                "compute_asymmetry: mask still < 100 px after fallback — returning defaults"
            )
            return {
                "asymmetry_x": 0.0,
                "asymmetry_y": 0.0,
                "asymmetry_mean": 0.0,
                "asymmetry_label": "simmetrica",
            }

        m = mask.astype(np.float32)

        def _fold_asymmetry(arr: np.ndarray, axis: int) -> float:
            """IoU-based asymmetry after folding along *axis*.

            axis=0 → fold along horizontal axis (compare top / bottom halves).
            axis=1 → fold along vertical axis (compare left / right halves).
            """
            if axis == 0:
                mid = arr.shape[0] // 2
                half_a = arr[:mid, :]
                half_b = np.flipud(arr[mid:, :])
            else:
                mid = arr.shape[1] // 2
                half_a = arr[:, :mid]
                half_b = np.fliplr(arr[:, mid:])

            ha, wa = half_a.shape
            hb, wb = half_b.shape
            th = max(ha, hb)
            tw = max(wa, wb)

            pad_a = np.zeros((th, tw), dtype=np.float32)
            pad_b = np.zeros((th, tw), dtype=np.float32)
            pad_a[:ha, :wa] = half_a
            pad_b[:hb, :wb] = half_b

            intersection = float((pad_a * pad_b).sum())
            union = float(np.maximum(pad_a, pad_b).sum())
            return 0.0 if union < 1e-6 else float(1.0 - intersection / union)

        asym_x = _fold_asymmetry(m, axis=0)   # horizontal-axis fold → top/bottom
        asym_y = _fold_asymmetry(m, axis=1)   # vertical-axis fold   → left/right
        asym_mean = (asym_x + asym_y) / 2.0

        t1, t2, t3 = asymmetry_thresholds
        if asym_mean < t1:
            label = "simmetrica"
        elif asym_mean < t2:
            label = "lievemente asimmetrica"
        elif asym_mean < t3:
            label = "asimmetrica"
        else:
            label = "marcatamente asimmetrica"

        return {
            "asymmetry_x": round(asym_x, 4),
            "asymmetry_y": round(asym_y, 4),
            "asymmetry_mean": round(asym_mean, 4),
            "asymmetry_label": label,
        }

    # ── Method 2: Border irregularity ────────────────────────────────────────

    def compute_border_irregularity(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.30,
        border_thresholds: Optional[list[float]] = None,
    ) -> dict:
        """Compute lesion border irregularity using circularity.

        Uses GradCAM++ heatmap (thresholded) as a binary lesion mask and
        computes the contour-based circularity score.  OpenCV is preferred;
        ``scipy.ndimage.binary_erosion`` is the fallback.

        Args:
            heatmap: GradCAM++ activation map (H, W), values in [0, 1].
            threshold: Binary mask threshold.
            border_thresholds: Three circularity breakpoints [t1, t2, t3]
                separating the four border label categories.
                Default: [0.4, 0.6, 0.8].

        Returns:
            dict with keys:

            - ``irregularity_score`` (float): 1 − circularity, clipped to [0, 1].
            - ``circularity`` (float): (4π·area) / perimeter², clipped to [0, 1].
            - ``border_label`` (str): Italian category label.
        """
        if border_thresholds is None:
            border_thresholds = [0.4, 0.6, 0.8]

        mask = (heatmap >= threshold).astype(np.uint8)
        area = int(mask.sum())

        if area < 100:
            logger.warning(
                "compute_border_irregularity: mask < 100 px — returning defaults"
            )
            return {
                "irregularity_score": 0.0,
                "circularity": 1.0,
                "border_label": "regolari",
            }

        # Contour perimeter — OpenCV preferred, scipy fallback
        perimeter: float = 0.0
        try:
            import cv2  # optional dependency
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                raise ValueError("No contours found by cv2")
            contour = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(contour, closed=True))
        except ImportError:
            # scipy fallback: boundary = mask XOR eroded mask
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(mask.astype(bool))
            boundary = mask.astype(bool) & ~eroded
            perimeter = float(boundary.sum())

        if perimeter < 1e-6:
            return {
                "irregularity_score": 0.0,
                "circularity": 1.0,
                "border_label": "regolari",
            }

        circularity = min(float((4.0 * math.pi * area) / (perimeter ** 2)), 1.0)
        irregularity_score = round(float(np.clip(1.0 - circularity, 0.0, 1.0)), 4)
        circularity = round(circularity, 4)

        t1, t2, t3 = border_thresholds   # [0.4, 0.6, 0.8]
        if circularity > t3:
            border_label = "regolari"
        elif circularity > t2:
            border_label = "lievemente irregolari"
        elif circularity > t1:
            border_label = "irregolari"
        else:
            border_label = "molto irregolari"

        return {
            "irregularity_score": irregularity_score,
            "circularity": circularity,
            "border_label": border_label,
        }

    # ── Method 3: Colour features ─────────────────────────────────────────────

    def compute_colour_features(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        threshold: float = 0.30,
        kmeans_k: int = 5,
        kmeans_random_state: int = 42,
        heterogeneity_thresholds: Optional[list[float]] = None,
    ) -> dict:
        """Compute colour heterogeneity features within the lesion mask.

        Converts masked pixels to CIE Lab colour space and clusters them
        with K-Means.  Each centroid is mapped to a clinical colour name.

        Args:
            image: RGB image as float32 np.ndarray (H, W, 3), values in [0, 1].
            heatmap: GradCAM++ activation map (H, W), values in [0, 1].
            threshold: Binary mask threshold.
            kmeans_k: Number of K-Means clusters.
            kmeans_random_state: Random state seed for reproducibility.
            heterogeneity_thresholds: Three breakpoints [t1, t2, t3] for the
                heterogeneity label.  Default: [0.15, 0.30, 0.50].

        Returns:
            dict with keys:

            - ``n_dominant_colours`` (int): Clusters with relative area ≥ 5 %.
            - ``colour_percentages`` (dict): Colour name → percentage area.
            - ``dominant_colour`` (str): Colour with the highest percentage.
            - ``heterogeneity_index`` (float): Normalised mean pairwise centroid
                distance in Lab space, in [0, 1].
            - ``heterogeneity_label`` (str): Italian category label.
            - ``colour_summary`` (str): Human-readable Italian summary.
        """
        if heterogeneity_thresholds is None:
            heterogeneity_thresholds = [0.15, 0.30, 0.50]

        mask = heatmap >= threshold
        n_pixels = int(mask.sum())

        if n_pixels < 100:
            logger.warning(
                "compute_colour_features: mask < 100 px — returning defaults"
            )
            return {
                "n_dominant_colours": 1,
                "colour_percentages": {"marrone": 100.0},
                "dominant_colour": "marrone",
                "heterogeneity_index": 0.0,
                "heterogeneity_label": "omogenea",
                "colour_summary": "marrone (100%)",
            }

        from skimage.color import rgb2lab  # noqa: PLC0415
        from sklearn.cluster import KMeans  # noqa: PLC0415

        lab_image = rgb2lab(image)               # (H, W, 3), L in [0,100]
        masked_pixels = lab_image[mask]          # (N, 3)

        k = min(kmeans_k, n_pixels)
        km = KMeans(n_clusters=k, n_init=10, random_state=kmeans_random_state)
        labels = km.fit_predict(masked_pixels)
        centroids = km.cluster_centers_          # (k, 3) in Lab

        counts = np.bincount(labels, minlength=k)
        pct_arr = counts / counts.sum() * 100.0  # float64 array

        def _lab_to_name(lab: np.ndarray) -> str:
            L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
            if L > 75 and abs(a) < 10 and abs(b) < 10:
                return "bianco/grigio"
            if L < 30:
                return "nero"
            if a > 15 and 30 <= L <= 60:
                return "rosso/rosa"
            if b > 15 and a > 5 and 30 <= L <= 60:
                return "marrone scuro"
            if b > 10 and L > 50:
                return "marrone chiaro"
            if b < -5:
                return "blu/grigio-bluastro"
            return "marrone"

        # Accumulate percentages by colour name (merge identical names)
        colour_pct: dict[str, float] = {}
        for idx in range(k):
            if pct_arr[idx] >= 5.0:
                name = _lab_to_name(centroids[idx])
                colour_pct[name] = colour_pct.get(name, 0.0) + round(float(pct_arr[idx]), 1)

        # Heterogeneity: mean pairwise Lab distance between centroids / 100
        if k > 1:
            dists = [
                float(np.linalg.norm(centroids[i] - centroids[j]))
                for i in range(k)
                for j in range(i + 1, k)
            ]
            het_index = round(float(np.clip(float(np.mean(dists)) / 100.0, 0.0, 1.0)), 4)
        else:
            het_index = 0.0

        t1, t2, t3 = heterogeneity_thresholds
        if het_index < t1:
            het_label = "omogenea"
        elif het_index < t2:
            het_label = "lievemente eterogenea"
        elif het_index < t3:
            het_label = "eterogenea"
        else:
            het_label = "marcatamente eterogenea"

        sorted_colours = sorted(colour_pct.items(), key=lambda kv: -kv[1])
        dominant = sorted_colours[0][0] if sorted_colours else "marrone"
        colour_summary = ", ".join(
            f"{cname} ({cpct:.0f}%)" for cname, cpct in sorted_colours
        )

        return {
            "n_dominant_colours": len(sorted_colours),
            "colour_percentages": {cname: cpct for cname, cpct in sorted_colours},
            "dominant_colour": dominant,
            "heterogeneity_index": het_index,
            "heterogeneity_label": het_label,
            "colour_summary": colour_summary,
        }

    # ── Method 4: GradCAM distribution ───────────────────────────────────────

    def compute_gradcam_distribution(
        self,
        heatmap: np.ndarray,
        peak_threshold: float = 0.60,
        top_n: int = 3,
        activation_threshold: float = 0.30,
    ) -> dict:
        """Analyse GradCAM++ activation distribution on a 3×3 spatial grid.

        Args:
            heatmap: GradCAM++ activation map (H, W), values in [0, 1].
            peak_threshold: Minimum mean activation for a region to appear in
                ``peak_regions``.
            top_n: Number of top regions included in ``focus_description``.
            activation_threshold: Threshold for ``activation_percentage``.

        Returns:
            dict with keys:

            - ``grid_activations`` (dict): Italian region name → mean activation.
            - ``peak_regions`` (list[str]): Regions with mean activation above
                *peak_threshold*.
            - ``activation_percentage`` (float): % of heatmap pixels above
                *activation_threshold*.
            - ``focus_description`` (str): Italian narrative (top-N regions).
        """
        H, W = heatmap.shape
        cell_h, cell_w = H // 3, W // 3

        region_values: list[tuple[str, float]] = []
        for row in range(3):
            for col in range(3):
                r0 = row * cell_h
                r1 = (row + 1) * cell_h if row < 2 else H
                c0 = col * cell_w
                c1 = (col + 1) * cell_w if col < 2 else W
                mean_val = float(heatmap[r0:r1, c0:c1].mean())
                region_values.append((_GRID_NAMES[(row, col)], mean_val))

        grid_activations = {name: round(val, 4) for name, val in region_values}
        peak_regions = [name for name, val in region_values if val >= peak_threshold]
        activation_pct = round(float((heatmap >= activation_threshold).mean() * 100.0), 2)

        # Top-N regions for focus_description (exclude near-zero regions)
        top_regions = [
            (name, val)
            for name, val in sorted(region_values, key=lambda kv: -kv[1])[:top_n]
            if val > 0.01
        ]
        if top_regions:
            parts = [f"zona {name} ({val:.2f})" for name, val in top_regions]
            if len(parts) == 1:
                focus_description = parts[0]
            elif len(parts) == 2:
                focus_description = f"{parts[0]} e {parts[1]}"
            else:
                focus_description = ", ".join(parts[:-1]) + f" e {parts[-1]}"
        else:
            focus_description = "nessuna regione attivata"

        return {
            "grid_activations": grid_activations,
            "peak_regions": peak_regions,
            "activation_percentage": activation_pct,
            "focus_description": focus_description,
        }

    # ── Method 5: Aggregate ───────────────────────────────────────────────────

    def extract_all(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        config: Optional[dict] = None,
    ) -> dict:
        """Run all four feature extractors and build the BioMistral prompt block.

        Each sub-extractor is wrapped in a try/except so that a failure in one
        does not abort the pipeline.  Failed sub-dicts are returned as ``None``.

        Args:
            image: Input PIL Image (RGB mode).
            heatmap: GradCAM++ activation map (H, W), values in [0, 1].
            config: Optional feature_extractor config dict.  Missing keys fall
                back to ``_DEFAULTS``.

        Returns:
            dict with keys ``asymmetry``, ``border``, ``colour``,
            ``gradcam_grid`` (each a sub-dict or ``None``), and
            ``prompt_block`` (str, always present).
        """
        cfg = {**_DEFAULTS, **(config or {})}
        threshold: float = float(cfg["heatmap_threshold"])

        # Convert PIL → float32 numpy (H, W, 3) in [0, 1]
        img_np = np.array(image.convert("RGB"), dtype=np.float32) / 255.0

        # Align heatmap spatial resolution to image
        h_img, w_img = img_np.shape[:2]
        if heatmap.shape != (h_img, w_img):
            hm_pil = Image.fromarray(
                (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
            ).resize((w_img, h_img), Image.BILINEAR)
            heatmap = np.array(hm_pil, dtype=np.float32) / 255.0

        asymmetry_res: Optional[dict] = None
        border_res: Optional[dict] = None
        colour_res: Optional[dict] = None
        gradcam_res: Optional[dict] = None

        try:
            asymmetry_res = self.compute_asymmetry(
                img_np, heatmap,
                threshold=threshold,
                asymmetry_thresholds=list(cfg["asymmetry_thresholds"]),
            )
        except Exception as exc:
            logger.warning(f"Feature extractor — asymmetry failed: {exc}")

        try:
            border_res = self.compute_border_irregularity(
                heatmap,
                threshold=threshold,
                border_thresholds=list(cfg["border_thresholds"]),
            )
        except Exception as exc:
            logger.warning(f"Feature extractor — border failed: {exc}")

        try:
            colour_res = self.compute_colour_features(
                img_np, heatmap,
                threshold=threshold,
                kmeans_k=int(cfg["kmeans_k"]),
                kmeans_random_state=int(cfg["kmeans_random_state"]),
                heterogeneity_thresholds=list(cfg["heterogeneity_thresholds"]),
            )
        except Exception as exc:
            logger.warning(f"Feature extractor — colour failed: {exc}")

        try:
            gradcam_res = self.compute_gradcam_distribution(
                heatmap,
                peak_threshold=float(cfg["gradcam_peak_threshold"]),
                top_n=int(cfg["top_n_regions"]),
                activation_threshold=threshold,
            )
        except Exception as exc:
            logger.warning(f"Feature extractor — gradcam_distribution failed: {exc}")

        return {
            "asymmetry": asymmetry_res,
            "border": border_res,
            "colour": colour_res,
            "gradcam_grid": gradcam_res,
            "prompt_block": _build_prompt_block(
                asymmetry_res, border_res, colour_res, gradcam_res
            ),
        }


# ─── Prompt builder (module-level helper) ────────────────────────────────────

def _build_prompt_block(
    asymmetry: Optional[dict],
    border: Optional[dict],
    colour: Optional[dict],
    gradcam: Optional[dict],
) -> str:
    """Format all extracted features as an Italian text block for BioMistral.

    Args:
        asymmetry: Output of :meth:`compute_asymmetry` or ``None``.
        border: Output of :meth:`compute_border_irregularity` or ``None``.
        colour: Output of :meth:`compute_colour_features` or ``None``.
        gradcam: Output of :meth:`compute_gradcam_distribution` or ``None``.

    Returns:
        Formatted Italian text block ready to inject into the ABCDE prompt.
    """
    lines = ["CARATTERISTICHE QUANTITATIVE DELL'IMMAGINE:"]

    if asymmetry:
        lines.append(
            f"- Asimmetria: asse orizzontale {asymmetry['asymmetry_x']:.2f}, "
            f"asse verticale {asymmetry['asymmetry_y']:.2f}\n"
            f"  \u2192 {asymmetry['asymmetry_label']}"
        )

    if border:
        lines.append(
            f"- Irregolarit\u00e0 del bordo: {border['irregularity_score']:.2f}"
            f" \u2192 {border['border_label']}"
        )

    if colour:
        lines.append(
            f"- Colori dominanti: {colour['colour_summary']}\n"
            f"  Eterogeneit\u00e0 cromatica: {colour['heterogeneity_index']:.2f}"
            f" \u2192 {colour['heterogeneity_label']}"
        )

    if gradcam:
        peak_str = (
            ", ".join(gradcam["peak_regions"]) if gradcam["peak_regions"] else "nessuna"
        )
        lines.append(
            f"- Attivazione GradCAM++: {gradcam['focus_description']}\n"
            f"  Regioni principali: {peak_str}\n"
            f"  Area attivata: {gradcam['activation_percentage']:.1f}%"
        )

    return "\n".join(lines)


# ─── Standalone test entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Standalone test for DermoscopyFeatureExtractor."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    pil_img = Image.open(args.image).convert("RGB")
    W, H = pil_img.size

    # Synthetic centre-weighted Gaussian heatmap for standalone testing
    ys = np.linspace(-1, 1, H)
    xs = np.linspace(-1, 1, W)
    xx, yy = np.meshgrid(xs, ys)
    dummy_heatmap = np.exp(-3.0 * (xx ** 2 + yy ** 2)).astype(np.float32)

    extractor = DermoscopyFeatureExtractor()
    features = extractor.extract_all(pil_img, dummy_heatmap)

    # Serialise: convert any remaining numpy scalars
    def _to_python(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    features_clean = json.loads(
        json.dumps(features, default=_to_python)
    )
    print(json.dumps(features_clean, indent=2, ensure_ascii=False))
