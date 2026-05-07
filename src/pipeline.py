"""
DermoAI Pipeline — End-to-end orchestration with VRAM management.
Supports single image, batch, and multi-model comparison modes.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch

from .utils import (
    format_report_markdown,
    get_vram_status,
    load_config,
    save_json,
    setup_device,
    setup_logging,
)

logger = logging.getLogger(__name__)


def _save_confusion_matrix_png(
    tp: int, fn: int, fp: int, tn: int, save_path: Path
) -> None:
    """Save a 2×2 confusion matrix as a PNG file.

    Args:
        tp: True positives (predicted MM, actual MM).
        fn: False negatives (predicted NV, actual MM).
        fp: False positives (predicted MM, actual NV).
        tn: True negatives (predicted NV, actual NV).
        save_path: Destination path for the PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = [[tp, fn], [fp, tn]]
    max_val = max(tp, fn, fp, tn, 1)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=max_val)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred MM", "Pred NV"], fontsize=11)
    ax.set_yticklabels(["Actual MM", "Actual NV"], fontsize=11)
    ax.set_title("Confusion Matrix — Fine-tuned Model", fontsize=12)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            color = "white" if val > max_val * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix PNG saved → {save_path}")


_BIAS_NOTE = (
    "⚠️ BIAS NOTE: EfficientNet-B0 trained on ISIC augmented with StyleGAN. "
    "ISIC has known demographic biases (Fitzpatrick I-III over-representation). "
    "Performance may be lower on darker skin types (IV-VI)."
)


class DermAIPipeline:
    """End-to-end dermatological AI pipeline with VRAM-aware orchestration.

    Manages sequential loading/offloading of GPU components:
        EfficientNet-B0 (fp16) → GradCAM++ → RAG (CPU) →
        BioMistral-7B (8bit/4bit) → QualitativeEval (CPU) →
        BERTScore (optional, after LLM offload)

    Args:
        config_path: Path to config/config.yaml.
    """

    def __init__(
        self,
        config_path: Union[str, Path] = "config/config.yaml",
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.device = setup_device()
        logger.info(_BIAS_NOTE)
        self._log_vram("pipeline init")

    def _log_vram(self, step: str) -> None:
        """Log current VRAM status if monitoring is enabled."""
        if not self.config.get("hardware", {}).get("monitor_vram", True):
            return
        vram = get_vram_status()
        logger.info(
            f"[VRAM @ {step}] "
            f"total={vram['total_gb']:.1f}GB | "
            f"used={vram['used_gb']:.1f}GB | "
            f"free={vram['free_gb']:.1f}GB"
        )

    def _empty_cache(self) -> None:
        """Free CUDA memory between pipeline stages."""
        if self.config.get("hardware", {}).get("empty_cache_between_modules", True):
            torch.cuda.empty_cache()
            gc.collect()

    def _load_classifier(self):
        """Instantiate and return MelanomaClassifier."""
        from .classifier import MelanomaClassifier

        model_cfg = self.config["model"]
        prep_cfg = self.config.get("preprocessing", {})
        return MelanomaClassifier(
            weights_path=model_cfg["weights_path"],
            num_classes=model_cfg["num_classes"],
            class_names=model_cfg["class_names"],
            square_crop=prep_cfg.get("square_crop", True),
            colour_normalisation=prep_cfg.get("colour_normalisation", False),
            melanoma_threshold=prep_cfg.get("melanoma_threshold", 0.40),
            tta_n_augmentations=prep_cfg.get("tta_n_augmentations", 5),
        )

    def _load_rag(self):
        """Instantiate RAGRetriever (CPU-only, no VRAM)."""
        from .rag_retriever import RAGRetriever

        rag_cfg = self.config.get("rag", {})
        if not rag_cfg.get("enabled", True):
            return None
        return RAGRetriever(rag_cfg)

    def _load_report_generator(self):
        """Instantiate ABCDEReportGenerator."""
        from .report_generator import ABCDEReportGenerator

        llm_cfg = {**self.config["llm"], **self.config.get("report_generator", {})}
        template_path = (
            self.config_path.parent.parent / llm_cfg["template"]
            # if "template" in llm_cfg
            # else self.config_path.parent / "abcde_prompt_template.txt"
        )
        return ABCDEReportGenerator(
            config=llm_cfg,
            template_path=template_path,
        )

    def _load_qualitative_evaluator(self):
        """Instantiate QualitativeEvaluator (CPU)."""
        from .qualitative_evaluator import QualitativeEvaluator

        return QualitativeEvaluator(self.config.get("qualitative", {}))

    def _load_evaluator(self):
        """Instantiate ReportEvaluator (NLG metrics)."""
        from .evaluator import ReportEvaluator

        return ReportEvaluator(self.config.get("evaluation", {}))

    def _export_classification_report(
        self,
        clf_rows: list[dict],
        output_dir: Union[str, Path],
    ) -> None:
        """Export classification results CSV, confusion matrix PNG, and metrics MD.

        Args:
            clf_rows: List of dicts with keys: image_id (str), predicted (str),
                confidence (float), ground_truth (str | None).
            output_dir: Directory to write all outputs.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(clf_rows)
        df.to_csv(output_dir / "classification_results.csv", index=False)
        logger.info(
            f"Classification results saved → {output_dir / 'classification_results.csv'}"
        )

        gt_df = df[df["ground_truth"].notna() & (df["ground_truth"] != "")].copy()
        if gt_df.empty:
            logger.info(
                "No ground truth labels — skipping confusion matrix and metrics"
            )
            return

        gt_df["ground_truth"] = gt_df["ground_truth"].str.lower().str.strip()
        gt_df["predicted"] = gt_df["predicted"].str.lower().str.strip()

        y_true = (gt_df["ground_truth"] == "melanoma").astype(int).tolist()
        y_pred = (gt_df["predicted"] == "melanoma").astype(int).tolist()
        y_score = [
            row["confidence"] if row["predicted"] == "melanoma" else 1.0 - row["confidence"]
            for _, row in gt_df.iterrows()
        ]

        from sklearn.metrics import (
            accuracy_score, confusion_matrix, f1_score, roc_auc_score,
            ConfusionMatrixDisplay,
        )

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float("nan")  # single class in GT

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["nevus", "melanoma"]
        ).plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix (n={len(gt_df)})")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        # Markdown metrics report
        lines = [
            "# Classification Metrics",
            "",
            f"**Samples evaluated:** {len(gt_df)} / {len(df)}",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Accuracy | {accuracy:.3f} |",
            f"| Sensitivity (recall melanoma) | {sensitivity:.3f} |",
            f"| Specificity (recall nevus) | {specificity:.3f} |",
            f"| F1-score (melanoma) | {f1:.3f} |",
            f"| AUC-ROC | {auc:.3f} |",
            "",
            "## Confusion Matrix",
            "",
            "| | Pred. nevus | Pred. melanoma |",
            "|---|---|---|",
            f"| **GT nevus** | {tn} | {fp} |",
            f"| **GT melanoma** | {fn} | {tp} |",
        ]
        (output_dir / "classification_metrics.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        logger.info(f"Classification metrics exported → {output_dir}")

    def run(
        self,
        image_path: Union[str, Path],
        ground_truth: Optional[str] = None,
        output_dir: Union[str, Path] = "outputs",
    ) -> dict:
        """Run the full pipeline on a single image.

        Pipeline steps:
            1. Classify (EfficientNet-B0 fp16) → empty_cache
            2. GradCAM++ → empty_cache
            3. RAG retrieve (CPU)
            4. Generate ABCDE report (LLM 8bit→4bit) → offload → empty_cache
            5. Qualitative evaluation (CPU, always)
            6. NLG metrics (only if ground_truth provided)
            7. Save outputs

        Args:
            image_path: Path to input image.
            ground_truth: Optional ground truth report text for NLG metrics.
            output_dir: Base output directory.

        Returns:
            dict with classification, gradcam, report, qualitative, and optionally
            nlg_metrics.
        """
        from .gradcam import GradCAMExplainer

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        stem = image_path.stem

        logger.info(f"{'='*60}")
        logger.info(f"Pipeline start | image={image_path.name}")
        logger.info(f"{'='*60}")

        # Step 1: Classification
        self._log_vram("before classification")
        classifier = self._load_classifier()
        prep_cfg = self.config.get("preprocessing", {})
        predict_fn = classifier.predict_tta if prep_cfg.get("tta_enabled", True) else classifier.predict
        classification = predict_fn(image_path)
        logger.info(
            f"[1/6] Classification: {classification['class']} "
            f"({classification['confidence']:.1%})"
        )
        self._empty_cache()
        self._log_vram("after classification")

        # Step 2: GradCAM++
        self._log_vram("before gradcam")
        explainer = GradCAMExplainer(
            model=classifier.model,
            method=self.config["gradcam"]["method"],
            device=self.device,
        )
        gradcam = explainer.generate(image_path, classifier.model)
        logger.info(
            f"[2/6] GradCAM++: focus='{gradcam['focus_regions']}' | "
            f"activation={gradcam['activation_percentage']:.1f}%"
        )

        # Save GradCAM visualization
        gradcam_out = output_dir / "gradcam" / f"{stem}_gradcam.png"
        explainer.save_visualization(image_path, gradcam, gradcam_out)
        self._empty_cache()
        self._log_vram("after gradcam")

        # Step 3: RAG (CPU)
        rag_context, rag_references, rag_sources = "", "", []
        rag = self._load_rag()
        if rag and rag.is_available():
            query = rag.build_query(classification, gradcam)
            chunks = rag.retrieve(query)
            rag_context, rag_references = rag.format_rag_context(chunks)
            rag_sources = [f"{c['source']} | {c['reference']}" for c in chunks]
            logger.info(f"[3/6] RAG: retrieved {len(chunks)} chunks")
        else:
            logger.info("[3/6] RAG: unavailable or disabled")

        # Step 3b: Feature extraction (CPU, no VRAM impact)
        image_features = None
        feat_cfg = self.config.get("feature_extractor", {})
        if feat_cfg.get("enabled", True):
            from PIL import Image as _PilImage  # type: ignore[import]  # noqa: PLC0415
            from .feature_extractor import DermoscopyFeatureExtractor  # noqa: PLC0415
            pil_image = _PilImage.open(image_path).convert("RGB")
            image_features = DermoscopyFeatureExtractor().extract_all(
                image=pil_image,
                heatmap=gradcam["heatmap"],
                config=feat_cfg,
            )
            asym_val = (image_features.get("asymmetry") or {}).get("asymmetry_mean", "N/A")
            bord_val = (image_features.get("border") or {}).get("irregularity_score", "N/A")
            logger.info(
                f"[3b/6] Feature extraction | asymmetry_mean={asym_val} | irregularity={bord_val}"
            )

        # Step 4: Report generation (LLM)
        self._log_vram("before LLM")
        generator = self._load_report_generator()
        report = generator.generate_report(
            classification, gradcam,
            image_features=image_features,
            rag_context=rag_context,
            rag_references=rag_references,
            rag_sources=rag_sources,
        )
        logger.info(
            f"[4/6] Report generated | "
            f"risk={report['risk_level']} | "
            f"quant={report['metadata']['quantization_used']} | "
            f"time={report['metadata']['generation_time_s']:.1f}s"
        )
        self._empty_cache()
        self._log_vram("after LLM offload")

        # Step 5: Qualitative evaluation (CPU, always)
        qual_eval = self._load_qualitative_evaluator()
        qualitative = qual_eval.auto_evaluate(report, classification, gradcam)
        logger.info(
            f"[5/6] Qualitative: {qualitative['total']}/{qualitative['max_possible']} "
            f"({qualitative['percentage']:.0f}%)"
        )

        # Step 6: NLG metrics (only with ground truth)
        nlg_metrics = None
        if ground_truth:
            self._log_vram("before BERTScore")
            evaluator = self._load_evaluator()
            nlg_metrics = evaluator.evaluate(report["report_text"], ground_truth)
            logger.info(
                f"[6/6] NLG metrics | "
                f"ROUGE-L={nlg_metrics.get('rougeL', 0):.3f} | "
                f"BERTScore={nlg_metrics.get('bertscore_f1', 0):.3f}"
            )
            self._empty_cache()

        # Save outputs
        self._save_run_outputs(
            stem, output_dir, image_path, classification,
            gradcam, report, qualitative, nlg_metrics
        )

        result = {
            "image": str(image_path),
            "classification": classification,
            "gradcam": {k: v for k, v in gradcam.items() if k != "heatmap" and k != "overlay"},
            "report": report,
            "qualitative": qualitative,
            "image_features": image_features,
        }
        if nlg_metrics:
            result["nlg_metrics"] = nlg_metrics

        logger.info(f"Pipeline complete | image={image_path.name}")
        return result

    def _save_run_outputs(
        self, stem, output_dir, image_path,
        classification, gradcam, report, qualitative, nlg_metrics
    ) -> None:
        """Save all pipeline outputs to disk."""
        # JSON result
        result_data = {
            "image": str(image_path),
            "classification": classification,
            "gradcam": {
                k: v for k, v in gradcam.items()
                if k not in ("heatmap", "overlay")
            },
            "report_metadata": report.get("metadata", {}),
            "risk_level": report.get("risk_level", ""),
            "qualitative": qualitative,
        }
        if nlg_metrics:
            result_data["nlg_metrics"] = nlg_metrics

        save_json(result_data, output_dir / "reports" / f"{stem}_result.json")

        # Markdown report
        md_text = format_report_markdown(report)
        md_path = output_dir / "reports" / f"{stem}_report.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_text, encoding="utf-8")

        logger.info(f"Outputs saved → {output_dir}")

    def run_batch(
        self,
        image_dir: Union[str, Path],
        ground_truth_dir: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "outputs",
    ) -> dict:
        """Run pipeline on all images in a directory.

        LLM loaded ONCE for the entire batch (more efficient).
        BERTScore computed after the complete batch.

        Args:
            image_dir: Directory containing input images.
            ground_truth_dir: Optional directory with JSON ground truth files.
            output_dir: Base output directory.

        Returns:
            dict with batch_results (list) and aggregate statistics.
        """
        from .classifier import MelanomaClassifier
        from .gradcam import GradCAMExplainer
        from .qualitative_evaluator import QualitativeEvaluator
        from .report_generator import ABCDEReportGenerator

        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        run_dir = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in image_extensions
        )

        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")

        logger.info(f"Batch mode | images={len(image_paths)}")

        # Load LLM once for the batch
        generator = self._load_report_generator()
        generator.load_model()

        classifier = self._load_classifier()
        rag = self._load_rag()
        qual_eval = QualitativeEvaluator(self.config.get("qualitative", {}))
        explainer = GradCAMExplainer(
            model=classifier.model,
            method=self.config["gradcam"]["method"],
            device=self.device,
        )
        prep_cfg = self.config.get("preprocessing", {})
        predict_fn = classifier.predict_tta if prep_cfg.get("tta_enabled", True) else classifier.predict

        from .feature_extractor import DermoscopyFeatureExtractor  # noqa: PLC0415
        feat_extractor = DermoscopyFeatureExtractor()
        feat_cfg = self.config.get("feature_extractor", {})
        feat_enabled = feat_cfg.get("enabled", True)

        batch_results = []
        generated_reports = []
        reference_reports = []
        image_ids = []
        qualitative_evals = []
        clf_rows = []

        for i, img_path in enumerate(image_paths, 1):
            logger.info(f"[{i}/{len(image_paths)}] Processing {img_path.name}")
            stem = img_path.stem

            try:
                classification = predict_fn(img_path)
                self._empty_cache()

                gradcam = explainer.generate(img_path, classifier.model)
                gradcam_out = run_dir / "gradcam" / f"{stem}_gradcam.png"
                explainer.save_visualization(img_path, gradcam, gradcam_out)
                self._empty_cache()

                image_features = None
                if feat_enabled:
                    from PIL import Image as _PilImage  # noqa: PLC0415
                    _pil = _PilImage.open(img_path).convert("RGB")
                    image_features = feat_extractor.extract_all(
                        image=_pil,
                        heatmap=gradcam["heatmap"],
                        config=feat_cfg,
                    )

                rag_context, rag_references, rag_sources = "", "", []
                if rag and rag.is_available():
                    query = rag.build_query(classification, gradcam)
                    chunks = rag.retrieve(query)
                    rag_context, rag_references = rag.format_rag_context(chunks)
                    rag_sources = [f"{c['source']} | {c['reference']}" for c in chunks]

                report = generator.generate_report(
                    classification, gradcam,
                    image_features=image_features,
                    rag_context=rag_context,
                    rag_references=rag_references,
                    rag_sources=rag_sources,
                )

                qualitative = qual_eval.auto_evaluate(report, classification, gradcam)
                qualitative_evals.append(qualitative)
                save_json(
                    {"classification": classification, "report": report,
                     "qualitative": qualitative},
                    run_dir / "reports" / f"{stem}_result.json"
                )
                md_path = run_dir / "reports" / f"{stem}_report.md"
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(format_report_markdown(report), encoding="utf-8")

                batch_results.append({
                    "image_id": stem,
                    "classification": classification["class"],
                    "confidence": classification["confidence"],
                    "risk_level": report["risk_level"],
                    "qualitative_total": qualitative["total"],
                    "qualitative_pct": qualitative["percentage"],
                })
                generated_reports.append(report["report_text"])
                image_ids.append(stem)

                # Collect ground truth if available
                gt_class: Optional[str] = None
                if ground_truth_dir:
                    gt_path = Path(ground_truth_dir) / f"{stem}.json"
                    if gt_path.exists():
                        with open(gt_path, encoding="utf-8") as f:
                            gt_data = json.load(f)
                        if "report" in gt_data:
                            reference_reports.append(gt_data["report"])
                            gt_class = gt_data.get("classification")
                        else:
                            first_val = next(iter(gt_data.values()), {})
                            ref = first_val.get("report", "") if isinstance(first_val, dict) else ""
                            reference_reports.append(ref or None)
                            gt_class = first_val.get("classification") if isinstance(first_val, dict) else None
                    else:
                        reference_reports.append(None)

                clf_rows.append({
                    "image_id": stem,
                    "predicted": classification["class"],
                    "confidence": classification["confidence"],
                    "ground_truth": gt_class,
                })

            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
                batch_results.append({"image_id": stem, "error": str(e)})

        # Offload LLM after batch
        generator.offload()
        self._empty_cache()
        self._log_vram("after batch LLM offload")

        # Ensure evaluation directory exists before any writes
        eval_dir = run_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # BERTScore after batch if ground truth available
        nlg_batch_df = None
        if reference_reports and any(r is not None for r in reference_reports):
            valid = [
                (gen, ref, iid)
                for gen, ref, iid in zip(generated_reports, reference_reports, image_ids)
                if ref is not None
            ]
            if valid:
                gen_list, ref_list, id_list = zip(*valid)
                evaluator = self._load_evaluator()
                nlg_batch_df = evaluator.evaluate_batch(
                    list(gen_list), list(ref_list), list(id_list)
                )
                nlg_batch_df.to_csv(
                    eval_dir / "nlg_metrics.csv",
                    index=False,
                )
                logger.info(f"Batch NLG metrics saved")

        # Qualitative export
        qual_eval.manual_evaluate_template(image_ids, eval_dir)
        if qualitative_evals:
            qual_eval.export_qualitative_report(
                qualitative_evals, image_ids, eval_dir
            )

        # Classification results + metrics
        if clf_rows:
            self._export_classification_report(clf_rows, eval_dir)

        import pandas as pd
        summary_df = pd.DataFrame(batch_results)
        summary_df.to_csv(eval_dir / "batch_summary.csv", index=False)

        logger.info(
            f"Batch complete | processed={len(batch_results)} | "
            f"errors={sum(1 for r in batch_results if 'error' in r)} | "
            f"output={run_dir}"
        )

        return {
            "batch_results": batch_results,
            "nlg_metrics": nlg_batch_df.to_dict() if nlg_batch_df is not None else None,
            "total_processed": len(batch_results),
            "run_dir": str(run_dir),
        }

    def run_comparison(
        self,
        image_path: Union[str, Path],
        models_list: list[str],
        ground_truth: Optional[str] = None,
        output_dir: Union[str, Path] = "outputs",
    ) -> dict:
        """Compare multiple LLMs on the same image.

        Critical VRAM constraint: LLMs loaded SEQUENTIALLY (one at a time).
        Sequence: classify (once) → gradcam (once) → RAG (once) →
            for each LLM: load → generate → offload → empty_cache

        Args:
            image_path: Path to input image.
            models_list: List of model names to compare.
            ground_truth: Optional ground truth report text.
            output_dir: Base output directory.

        Returns:
            dict with per-model reports and cross-model evaluation results.
        """
        from .classifier import MelanomaClassifier
        from .cross_model_evaluator import CrossModelEvaluator
        from .gradcam import GradCAMExplainer
        from .qualitative_evaluator import QualitativeEvaluator
        from .report_generator import ABCDEReportGenerator

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        stem = image_path.stem

        logger.info(f"Comparison mode | models={models_list}")

        # Classify + GradCAM once
        classifier = self._load_classifier()
        prep_cfg = self.config.get("preprocessing", {})
        predict_fn = classifier.predict_tta if prep_cfg.get("tta_enabled", True) else classifier.predict
        classification = predict_fn(image_path)
        self._empty_cache()
        self._log_vram("after classification (comparison)")

        explainer = GradCAMExplainer(
            model=classifier.model,
            method=self.config["gradcam"]["method"],
            device=self.device,
        )
        gradcam = explainer.generate(image_path, classifier.model)
        explainer.save_visualization(
            image_path, gradcam,
            output_dir / "gradcam" / f"{stem}_gradcam.png"
        )
        self._empty_cache()

        # RAG once (same context for all models)
        rag_context, rag_references, rag_sources = "", "", []
        rag = self._load_rag()
        if rag and rag.is_available():
            query = rag.build_query(classification, gradcam)
            chunks = rag.retrieve(query)
            rag_context, rag_references = rag.format_rag_context(chunks)
            rag_sources = [f"{c['source']} | {c['reference']}" for c in chunks]

        # Sequential LLM generation
        qual_eval = QualitativeEvaluator(self.config.get("qualitative", {}))
        reports_dict: dict[str, str] = {}
        qualitative_evals: dict[str, dict] = {}
        model_reports: dict[str, dict] = {}

        for model_name in models_list:
            logger.info(f"{'─'*40}")
            logger.info(f"Loading model: {model_name}")
            self._log_vram(f"before {model_name}")

            cfg = {**self.config["llm"], **self.config.get("report_generator", {})}
            cfg["model_name"] = model_name
            _tpl = (
                self.config_path.parent.parent / cfg["template"]
                # if "template" in cfg
                # else self.config_path.parent / "abcde_prompt_template.txt"
            )
            generator = ABCDEReportGenerator(
                config=cfg,
                template_path=_tpl,
            )

            report = generator.generate_report(
                classification, gradcam,
                rag_context=rag_context,
                rag_references=rag_references,
                rag_sources=rag_sources,
            )

            generator.offload()
            self._empty_cache()
            self._log_vram(f"after {model_name} offload")

            qualitative = qual_eval.auto_evaluate(report, classification, gradcam)
            reports_dict[model_name] = report["report_text"]
            qualitative_evals[model_name] = qualitative
            model_reports[model_name] = report

            logger.info(
                f"Model {model_name}: risk={report['risk_level']} | "
                f"qualitative={qualitative['total']}/{qualitative['max_possible']}"
            )

        # Cross-model evaluation
        qual_scores = {
            m: qualitative_evals[m]["total"] for m in models_list
        }
        cross_eval = CrossModelEvaluator(
            qualitative_scores=qual_scores,
            config=self.config.get("cross_model", {}),
        )
        cross_result = cross_eval.cross_model_evaluate(reports_dict)

        # Ground truth NLG metrics if available
        gt_metrics = None
        if ground_truth:
            evaluator = self._load_evaluator()
            gt_metrics = {}
            for model_name, report_text in reports_dict.items():
                gt_metrics[model_name] = evaluator.evaluate(report_text, ground_truth)
                logger.info(
                    f"NLG {model_name}: "
                    f"ROUGE-L={gt_metrics[model_name].get('rougeL', 0):.3f}"
                )
            self._empty_cache()

        # Build final table
        final_table = cross_eval.build_final_table(
            reports_dict, qualitative_evals, cross_result, gt_metrics
        )

        # Export
        cross_eval.export_cross_model_results(
            cross_result, output_dir / "evaluation"
        )
        final_table.to_csv(
            output_dir / "evaluation" / "final_comparison_table.csv"
        )

        logger.info("Comparison complete")
        return {
            "classification": classification,
            "gradcam": {k: v for k, v in gradcam.items() if k not in ("heatmap", "overlay")},
            "model_reports": {
                m: {"risk_level": r["risk_level"], "metadata": r["metadata"]}
                for m, r in model_reports.items()
            },
            "qualitative_evals": qualitative_evals,
            "cross_model": cross_result,
            "final_table": final_table.to_dict(),
        }

    def run_eval_only(
        self,
        run_dir: Union[str, Path],
        gt_dir: Optional[Union[str, Path]] = None,
    ) -> dict:
        """Run NLG evaluation on an existing run checkpoint.

        Reads ``*_result.json`` files from ``run_dir/reports/``, extracts
        ``report_text``, matches ground-truth by stem from ``gt_dir``, and
        runs ROUGE + BLEU + BERTScore evaluation.

        Args:
            run_dir: Existing pipeline run directory (must contain
                ``reports/*_result.json`` files).
            gt_dir: Directory with ground-truth JSON files named
                ``{stem}.json`` each containing ``{"report": "..."}``.

        Returns:
            dict with ``nlg_metrics`` (dict) and ``output_dir`` (str).
        """
        run_dir = Path(run_dir)
        reports_dir = run_dir / "reports"
        if not reports_dir.exists():
            raise FileNotFoundError(f"reports/ not found in {run_dir}")

        result_files = sorted(reports_dir.glob("*_result.json"))
        if not result_files:
            raise FileNotFoundError(f"No *_result.json files in {reports_dir}")

        logger.info(
            f"eval-only | run_dir={run_dir} | found={len(result_files)} result files"
        )

        gen_list: list[str] = []
        ref_list: list[str] = []
        id_list: list[str] = []
        qual_evals: list[dict] = []
        qual_stems: list[str] = []
        clf_rows: list[dict] = []

        for rf in result_files:
            stem = rf.stem.replace("_result", "")
            with open(rf, encoding="utf-8") as fh:
                data = json.load(fh)

            # Collect qualitative scores regardless of GT availability
            qual = data.get("qualitative")
            if qual and "scores" in qual:
                qual_evals.append(qual)
                qual_stems.append(stem)

            # Collect predicted classification (always present)
            clf_entry = data.get("classification", {})
            gt_class: Optional[str] = None

            report_text = data.get("report", {}).get("report_text", "")
            if not report_text:
                logger.warning(f"Empty report_text in {rf.name} — skipping")
                continue

            ref_text: Optional[str] = None
            if gt_dir:
                gt_path = Path(gt_dir) / f"{stem}.json"
                logger.debug(f"Looking for GT: {gt_path.resolve()}")
                if gt_path.exists():
                    with open(gt_path, encoding="utf-8") as fh:
                        gt_data = json.load(fh)
                    # Support flat {"report": "..."} and nested {"stem.jpg": {"report": "..."}}
                    if "report" in gt_data:
                        ref_text = gt_data["report"]
                        gt_class = gt_data.get("classification")
                    else:
                        first_val = next(iter(gt_data.values()), {})
                        if isinstance(first_val, dict):
                            ref_text = first_val.get("report", "")
                            gt_class = first_val.get("classification")
                    if not ref_text:
                        logger.warning(
                            f"GT file found for {stem} but 'report' key missing or empty: {gt_path}"
                        )
                else:
                    logger.warning(f"Ground truth file not found: {gt_path}")

            clf_rows.append({
                "image_id": stem,
                "predicted": clf_entry.get("class", ""),
                "confidence": clf_entry.get("confidence", 0.0),
                "ground_truth": gt_class,
            })

            if not ref_text:
                logger.warning(f"Skipping {stem} — no ground truth available")
                continue

            gen_list.append(report_text)
            ref_list.append(ref_text)
            id_list.append(stem)

        if not gen_list:
            raise RuntimeError(
                "No valid (generated, reference) pairs found. "
                "Ensure --ground-truth-dir contains JSON files matching result stems."
            )

        logger.info(f"eval-only | evaluating {len(gen_list)} pairs")
        evaluator = self._load_evaluator()
        nlg_df = evaluator.evaluate_batch(gen_list, ref_list, id_list)

        eval_dir = run_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        nlg_df.to_csv(eval_dir / "nlg_metrics.csv", index=False)
        evaluator.export_results(nlg_df, eval_dir)

        # Qualitative export from stored result files
        from .qualitative_evaluator import QualitativeEvaluator
        qual_evaluator = QualitativeEvaluator(self.config.get("qualitative", {}))
        qual_evaluator.manual_evaluate_template(qual_stems, eval_dir)
        if qual_evals:
            qual_evaluator.export_qualitative_report(qual_evals, qual_stems, eval_dir)

        # Classification results + metrics
        if clf_rows:
            self._export_classification_report(clf_rows, eval_dir)

        logger.info(f"eval-only complete | n={len(nlg_df)} | output={eval_dir}")
        return {
            "nlg_metrics": nlg_df.to_dict(),
            "output_dir": str(eval_dir),
        }

    def vram_check(self) -> None:
        """Print VRAM status and hardware configuration summary."""
        vram = get_vram_status()
        print("\n" + "="*60)
        print("VRAM / HARDWARE CHECK")
        print("="*60)
        if torch.cuda.is_available():
            print(f"GPU:        {torch.cuda.get_device_name(0)}")
            print(f"VRAM Total: {vram['total_gb']:.2f} GB")
            print(f"VRAM Used:  {vram['used_gb']:.2f} GB")
            print(f"VRAM Free:  {vram['free_gb']:.2f} GB")
            cap = torch.cuda.get_device_capability(0)
            print(f"Compute:    {cap[0]}.{cap[1]}")
        else:
            print("CUDA: NOT AVAILABLE — Pipeline will run in CPU fallback mode")

        cfg_hw = self.config.get("hardware", {})
        print(f"\nConfig device:     {cfg_hw.get('device', 'N/A')}")
        print(f"Config VRAM total: {cfg_hw.get('vram_total_gb', 'N/A')} GB")
        print(f"Config margin:     {cfg_hw.get('vram_safety_margin_gb', 'N/A')} GB")
        print("\nExpected VRAM usage:")
        print("  EfficientNet-B0 (fp16):  ~0.3 GB")
        print("  GradCAM++:               ~0.5 GB")
        print("  S-PubMedBert RAG (CPU):   0.0 GB (CPU)")
        print("  BioMistral-7B (8-bit):   ~7.0 GB")
        print("  BioMistral-7B (4-bit):   ~4.0 GB [fallback]")
        print("  BERTScore deberta-xl:    ~1.5 GB [after LLM offload]")
        print("  BERTScore distilbert:    ~0.25 GB [VRAM < 2GB fallback]")
        print("="*60 + "\n")

    def build_rag_kb(self) -> None:
        """Build the RAG knowledge base (one-time operation)."""
        import subprocess
        logger.info("Building RAG knowledge base...")
        kb_script = Path("rag_kb/build_kb.py")
        if not kb_script.exists():
            raise FileNotFoundError(
                f"build_kb.py not found at {kb_script}. "
                "Ensure the rag_kb/ directory is present."
            )
        result = subprocess.run(
            [sys.executable, str(kb_script)],
            capture_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError("build_kb.py failed. Check logs above.")
        logger.info("RAG knowledge base built successfully.")

    def finetune_classifier(
        self,
        finetune_dir: Union[str, Path],
        epochs: int = 10,
        lr: float = 1e-3,
        output_dir: Union[str, Path] = "finetune/outputs",
    ) -> Path:
        """Fine-tune only the classifier head (classifier.1) of EfficientNet-B0.

        Freezes the entire backbone and trains only classifier.1
        (nn.Linear(1280, 2) = 2562 parameters). Converts the model to fp32
        for stable gradient updates. Uses WeightedRandomSampler for class
        balance in the 80/20 train/val split.

        Expected finetune_dir layout (ImageFolder-compatible):
            finetune_dir/
                melanoma/    img1.jpg, img2.jpg, ...
                nevus/       img1.jpg, img2.jpg, ...

        Args:
            finetune_dir: Root directory passed to torchvision.datasets.ImageFolder.
            epochs: Training epochs. Default: 10.
            lr: Adam learning rate. Default: 1e-3.
            output_dir: Directory to save efficientnet_b0_finetuned.pth.

        Returns:
            Path to the saved fine-tuned weights file.

        Raises:
            ValueError: If fewer than 6 images are found in finetune_dir.
        """
        import torch.nn as nn
        from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it, **kw): return it  # noqa: E731

        from .classifier import MelanomaClassifier

        finetune_dir = Path(finetune_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prep_cfg = self.config.get("preprocessing", {})
        colour_norm = prep_cfg.get("colour_normalisation", False)
        finetune_cfg = self.config.get("finetune", {})
        unfreeze_last_block = finetune_cfg.get("unfreeze_last_block", False)
        use_focal_loss = finetune_cfg.get("focal_loss", False)
        focal_alpha = finetune_cfg.get("focal_alpha", 0.75)
        focal_gamma = finetune_cfg.get("focal_gamma", 2.0)
        lr_backbone = finetune_cfg.get("lr_backbone", lr * 0.1)

        class _FocalLoss(nn.Module):
            def __init__(self, alpha: float, gamma: float) -> None:
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                import torch.nn.functional as F
                ce = F.cross_entropy(logits, targets, reduction="none")
                pt = torch.exp(-ce)
                alpha_t = torch.where(
                    targets == 0,
                    torch.full_like(ce, self.alpha),
                    torch.full_like(ce, 1.0 - self.alpha),
                )
                return (alpha_t * (1.0 - pt) ** self.gamma * ce).mean()

        def _make_transform(augment: bool) -> transforms.Compose:
            steps = [
                transforms.Lambda(lambda img: transforms.CenterCrop(min(img.size))(img)),
            ]
            if colour_norm:
                steps.append(transforms.Lambda(MelanomaClassifier.normalize_to_isic))
            if augment:
                steps += [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
                ]
            steps += [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
            return transforms.Compose(steps)

        # Index dataset (no transform) — used only for split indices and class metadata
        index_ds = ImageFolder(str(finetune_dir))
        n_total = len(index_ds)
        if n_total < 6:
            raise ValueError(
                f"Fine-tune dataset too small: {n_total} images in {finetune_dir}. "
                "Minimum 6 images required (at least 3 per class)."
            )

        # 80/20 train/val split
        n_val = max(1, int(n_total * 0.2))
        n_train = n_total - n_val
        _split = random_split(index_ds, [n_train, n_val])
        train_idx = list(_split[0].indices)
        val_idx = list(_split[1].indices)

        # Two separate datasets with correct transforms for each split
        train_ds = Subset(
            ImageFolder(str(finetune_dir), transform=_make_transform(augment=True)), train_idx
        )
        val_ds = Subset(
            ImageFolder(str(finetune_dir), transform=_make_transform(augment=False)), val_idx
        )

        # WeightedRandomSampler for class balance in training
        all_targets = index_ds.targets
        train_targets = [all_targets[i] for i in train_idx]
        class_counts = [max(train_targets.count(c), 1) for c in range(len(index_ds.classes))]
        sample_weights = [1.0 / class_counts[t] for t in train_targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

        logger.info(
            f"Fine-tune | classes={index_ds.classes} | "
            f"total={n_total} (train={n_train}, val={n_val}) | "
            f"colour_norm={colour_norm} | epochs={epochs} | lr={lr} | "
            f"unfreeze_last_block={unfreeze_last_block} | focal_loss={use_focal_loss}"
        )

        # Load original weights, convert to fp32 for stable gradient updates
        model_cfg = self.config["model"]
        _base_clf = MelanomaClassifier(
            weights_path=model_cfg["weights_path"],
            num_classes=model_cfg["num_classes"],
            class_names=model_cfg["class_names"],
            square_crop=False,  # crop handled by transform above
        )
        model = _base_clf.model.float().to(self.device)

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Always unfreeze classifier head
        for param in model.classifier[1].parameters():
            param.requires_grad = True
        # Optionally unfreeze last backbone block (features.8)
        if unfreeze_last_block:
            for param in model.features[8].parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        unfreeze_desc = "features.8 + classifier.1" if unfreeze_last_block else "classifier.1 only"
        logger.info(f"Trainable parameters: {n_trainable} ({unfreeze_desc})")

        if unfreeze_last_block:
            optimizer = torch.optim.Adam([
                {"params": model.features[8].parameters(), "lr": lr_backbone},
                {"params": model.classifier[1].parameters(), "lr": lr},
            ])
        else:
            optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=lr)
        criterion = _FocalLoss(focal_alpha, focal_gamma) if use_focal_loss else nn.CrossEntropyLoss()
        save_path = output_dir / "efficientnet_b0_finetuned.pth"
        best_val_f1 = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = train_correct = 0
            for bx, by in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bx.size(0)
                train_correct += (logits.argmax(1) == by).sum().item()

            model.eval()
            val_loss = val_correct = val_tp = val_fp = val_fn = 0
            with torch.no_grad():
                for bx, by in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                    bx, by = bx.to(self.device), by.to(self.device)
                    logits = model(bx)
                    preds = logits.argmax(1)
                    val_loss += criterion(logits, by).item() * bx.size(0)
                    val_correct += (preds == by).sum().item()
                    val_tp += ((preds == 0) & (by == 0)).sum().item()
                    val_fp += ((preds == 0) & (by == 1)).sum().item()
                    val_fn += ((preds == 1) & (by == 0)).sum().item()

            train_acc = train_correct / n_train
            val_acc = val_correct / n_val
            val_prec = val_tp / (val_tp + val_fp) if (val_tp + val_fp) else 0.0
            val_sens = val_tp / (val_tp + val_fn) if (val_tp + val_fn) else 0.0
            val_f1 = (
                2 * val_prec * val_sens / (val_prec + val_sens)
                if (val_prec + val_sens) else 0.0
            )
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss/n_train:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss/n_val:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
            )
            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), save_path)
                logger.info(f"  → Best checkpoint saved (val_f1={val_f1:.3f})")

        logger.info(
            f"Fine-tuning complete | best_val_f1={best_val_f1:.3f} | weights={save_path}"
        )

        # Update in-memory config so run_finetune_pipeline loads fine-tuned weights
        self.config["model"]["weights_path"] = str(save_path)

        _base_clf.offload()
        self._empty_cache()
        return save_path

    def run_finetune_pipeline(
        self,
        finetune_dir: Union[str, Path] = "finetune/images/",
        test_dir: Union[str, Path] = "finetune/test_images/",
        output_dir: Union[str, Path] = "finetune/outputs/",
    ) -> dict:
        """Orchestrate the full domain-shift mitigation workflow.

        Steps:
            1. Fine-tune classifier.1 on finetune_dir (labelled images).
            2. Load fine-tuned model with TTA + colour_norm from config.
            3. Evaluate on test_dir (subfolders: melanoma/, nevus/).
            4. Compute TP/FN/FP/TN and derived metrics.
            5. Save finetune_metrics.json + confusion_matrix.png.

        Expected test_dir layout:
            test_dir/
                melanoma/    ground truth positive
                nevus/       ground truth negative

        If test_dir is absent or empty, evaluation is skipped and only
        the fine-tuned weights path is returned.

        Args:
            finetune_dir: Labelled training images (ImageFolder layout).
            test_dir: Labelled test images (melanoma/ and nevus/ subdirs).
            output_dir: Directory for fine-tuned weights and metrics.

        Returns:
            dict with:
                - finetune_weights (str): Path to saved fine-tuned weights.
                - metrics (dict): Classification metrics (if test_dir found).
        """
        from .classifier import MelanomaClassifier

        finetune_dir = Path(finetune_dir)
        test_dir = Path(test_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"{'='*60}")
        logger.info("Domain-shift mitigation | --finetune mode")
        logger.info(f"{'='*60}")

        prep_cfg = self.config.get("preprocessing", {})
        finetune_cfg = self.config.get("finetune", {})

        # Step 1: Fine-tune
        finetune_weights = self.finetune_classifier(
            finetune_dir=finetune_dir,
            epochs=finetune_cfg.get("epochs", 10),
            lr=finetune_cfg.get("lr", 1e-3),
            output_dir=output_dir,
        )

        result: dict = {"finetune_weights": str(finetune_weights)}

        # Step 2: Load fine-tuned classifier with config-driven TTA settings
        model_cfg = self.config["model"]
        ft_clf = MelanomaClassifier(
            weights_path=finetune_weights,
            num_classes=model_cfg["num_classes"],
            class_names=model_cfg["class_names"],
            square_crop=prep_cfg.get("square_crop", True),
            colour_normalisation=prep_cfg.get("colour_normalisation", False),
            melanoma_threshold=prep_cfg.get("melanoma_threshold", 0.40),
            tta_n_augmentations=prep_cfg.get("tta_n_augmentations", 5),
        )

        # Step 3: Validate test directory
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        mm_dir = test_dir / "melanoma"
        nv_dir = test_dir / "nevus"

        if not (test_dir.exists() and mm_dir.exists() and nv_dir.exists()):
            logger.warning(
                f"Test directory missing or incomplete: {test_dir}. "
                "Expected melanoma/ and nevus/ subdirs. Skipping evaluation."
            )
            ft_clf.offload()
            self._empty_cache()
            return result

        mm_images = sorted(p for p in mm_dir.iterdir() if p.suffix.lower() in image_extensions)
        nv_images = sorted(p for p in nv_dir.iterdir() if p.suffix.lower() in image_extensions)

        if not (mm_images or nv_images):
            logger.warning(f"No images found in {test_dir}. Skipping evaluation.")
            ft_clf.offload()
            self._empty_cache()
            return result

        # Step 4: Predict on test images
        tta_enabled = prep_cfg.get("tta_enabled", True)
        predict_fn = ft_clf.predict_tta if tta_enabled else ft_clf.predict
        logger.info(
            f"Evaluating | melanoma={len(mm_images)} | nevus={len(nv_images)} | "
            f"mode={'TTA' if tta_enabled else 'standard'}"
        )

        tp = fn = fp = tn = 0
        per_image: list[dict] = []

        for img_path in mm_images:
            try:
                pred = predict_fn(img_path)
                if pred["class"] == "melanoma":
                    tp += 1
                else:
                    fn += 1
                per_image.append({
                    "image": img_path.name, "actual": "melanoma",
                    "predicted": pred["class"], "confidence": pred["confidence"],
                })
            except Exception as e:
                logger.error(f"Prediction failed {img_path.name}: {e}")

        for img_path in nv_images:
            try:
                pred = predict_fn(img_path)
                if pred["class"] == "nevus":
                    tn += 1
                else:
                    fp += 1
                per_image.append({
                    "image": img_path.name, "actual": "nevus",
                    "predicted": pred["class"], "confidence": pred["confidence"],
                })
            except Exception as e:
                logger.error(f"Prediction failed {img_path.name}: {e}")

        total = tp + fn + fp + tn
        accuracy = (tp + tn) / total if total else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) else 0.0
        )

        metrics = {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn, "total": total,
            "accuracy": round(accuracy, 4),
            "sensitivity_recall": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "melanoma_threshold": prep_cfg.get("melanoma_threshold", 0.40),
            "tta_enabled": tta_enabled,
            "tta_n_augmentations": prep_cfg.get("tta_n_augmentations", 5),
            "per_image": per_image,
        }

        # Step 5: Console confusion matrix
        print(f"\n{'='*50}")
        print("FINE-TUNED MODEL — TEST RESULTS")
        print(f"{'='*50}")
        print(f"  {'':22s} Pred MM  Pred NV")
        print(f"  {'Actual MM (melanoma)':22s} {tp:6d}   {fn:6d}")
        print(f"  {'Actual NV (nevus)':22s} {fp:6d}   {tn:6d}")
        print(f"\n  Accuracy:    {accuracy:.3f}")
        print(f"  Sensitivity: {sensitivity:.3f}  (melanoma recall)")
        print(f"  Specificity: {specificity:.3f}")
        print(f"  Precision:   {precision:.3f}")
        print(f"  F1 Score:    {f1:.3f}")
        print(f"{'='*50}\n")

        # Save outputs
        metrics_path = output_dir / "finetune_metrics.json"
        save_json(metrics, metrics_path)
        logger.info(f"Metrics saved → {metrics_path}")

        _save_confusion_matrix_png(tp, fn, fp, tn, output_dir / "confusion_matrix.png")

        ft_clf.offload()
        self._empty_cache()

        result["metrics"] = {k: v for k, v in metrics.items() if k != "per_image"}
        logger.info(
            f"Fine-tune pipeline complete | "
            f"acc={accuracy:.3f} | sens={sensitivity:.3f} | "
            f"spec={specificity:.3f} | f1={f1:.3f}"
        )
        return result


def main() -> None:
    """CLI entry point for the DermoAI Pipeline."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="DermoAI Pipeline — Dermatological lesion classification and reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline --image data/test_images/img.jpg
  python -m src.pipeline --batch data/test_images/
  python -m src.pipeline --image img.jpg --compare BioMistral/BioMistral-7B,mistralai/Mistral-7B-Instruct-v0.1
  python -m src.pipeline --image img.jpg --ground-truth referti/img.json
  python -m src.pipeline --vram-check
  python -m src.pipeline --build-rag-kb
  python -m src.pipeline --finetune finetune/images/
  python -m src.pipeline --finetune finetune/images/ --test finetune/test_images/ --output finetune/outputs/
  python -m src.pipeline --eval-only outputs/run_20240101_120000 --ground-truth-dir data/ground_truth_reports/
""",
    )

    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--batch", type=str, help="Directory of images for batch processing")
    parser.add_argument("--compare", type=str, help="Comma-separated list of LLM model names to compare")
    parser.add_argument("--ground-truth", type=str, help="Path to ground truth JSON or text file (single image mode)")
    parser.add_argument("--ground-truth-dir", type=str, help="Directory with ground truth JSON files for batch mode (matched by filename stem)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--vram-check", action="store_true", help="Print VRAM status and exit")
    parser.add_argument("--build-rag-kb", action="store_true", help="Build RAG knowledge base and exit")
    parser.add_argument(
        "--finetune", type=str, nargs="?", const="finetune/images/", metavar="DIR",
        help="Fine-tune classifier head on labelled images (default dir: finetune/images/)",
    )
    parser.add_argument(
        "--test", type=str, default="finetune/test_images/", metavar="DIR",
        help="Test images dir for --finetune evaluation (default: finetune/test_images/)",
    )
    parser.add_argument(
        "--output", type=str, default="finetune/outputs/", metavar="DIR",
        help="Output dir for --finetune weights and metrics (default: finetune/outputs/)",
    )
    parser.add_argument(
        "--eval-only", type=str, metavar="RUN_DIR",
        help="Run NLG evaluation on an existing run directory (use with --ground-truth-dir)",
    )

    args = parser.parse_args()

    pipeline = DermAIPipeline(config_path=args.config)

    if args.vram_check:
        pipeline.vram_check()
        return

    if args.build_rag_kb:
        pipeline.build_rag_kb()
        return

    ground_truth_text = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if gt_path.exists():
            if gt_path.suffix == ".json":
                with open(gt_path) as f:
                    gt_data = json.load(f)
                # Support dict {image_id: {report: ...}} or {report: ...}
                if "report" in gt_data:
                    ground_truth_text = gt_data["report"]
                else:
                    # Take first value's report
                    ground_truth_text = next(iter(gt_data.values())).get("report", "")
            else:
                ground_truth_text = gt_path.read_text(encoding="utf-8")

    if args.image:
        if args.compare:
            models = [m.strip() for m in args.compare.split(",")]
            result = pipeline.run_comparison(
                args.image, models, ground_truth_text, args.output_dir
            )
            print(f"\n✓ Comparison complete. Results in {args.output_dir}/evaluation/")
            print(f"  Ranking: {[r['model'] for r in result['cross_model']['composite_ranking']]}")
        else:
            result = pipeline.run(args.image, ground_truth_text, args.output_dir)
            print(f"\n✓ Pipeline complete.")
            print(f"  Classification: {result['classification']['class']} "
                  f"({result['classification']['confidence']:.1%})")
            print(f"  Risk level: {result['report']['risk_level']}")
            print(f"  Qualitative: {result['qualitative']['total']}/30 "
                  f"({result['qualitative']['percentage']:.0f}%)")

    elif args.batch:
        gt_dir = Path(args.ground_truth_dir) if args.ground_truth_dir else None
        result = pipeline.run_batch(args.batch, gt_dir, args.output_dir)
        print(f"\n✓ Batch complete: {result['total_processed']} images processed.")
        print(f"  Output: {result['run_dir']}")

    elif args.finetune:
        result = pipeline.run_finetune_pipeline(args.finetune, args.test, args.output)
        print(f"\n✓ Fine-tune complete. Weights: {result['finetune_weights']}")
        if "metrics" in result:
            m = result["metrics"]
            print(
                f"  Accuracy: {m['accuracy']:.3f} | "
                f"Sensitivity: {m['sensitivity_recall']:.3f} | "
                f"F1: {m['f1_score']:.3f}"
            )
            print(f"  Results → {args.output}")

    elif args.eval_only:
        gt_dir = Path(args.ground_truth_dir) if args.ground_truth_dir else None
        result = pipeline.run_eval_only(args.eval_only, gt_dir)
        print(f"\n✓ Evaluation complete.")
        print(f"  Output: {result['output_dir']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
