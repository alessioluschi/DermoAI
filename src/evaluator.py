"""
NLG Metrics Evaluator: ROUGE, BLEU, BERTScore.
Only used when ground truth reports are available.
Loads BERTScore AFTER LLM offload to avoid VRAM conflicts.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Union

import pandas as pd
import torch

logger = logging.getLogger(__name__)


class ReportEvaluator:
    """Evaluates generated reports against ground truth using NLG metrics.

    Computes ROUGE-1/2/L, BLEU, and BERTScore. BERTScore uses
    deberta-xlarge-mnli (primary) or distilbert-base-uncased (fallback if
    VRAM < 2GB). Must be called AFTER LLM has been offloaded.

    Args:
        config: Configuration dict from config.yaml (evaluation section).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._bertscore_model: str | None = None
        self._bert_fallback: str = config.get(
            "bertscore_model_fallback", "distilbert-base-uncased"
        )
        self._offline: bool = bool(config.get("bertscore_offline", False))
        logger.info(
            "ReportEvaluator initialized (CPU-ready, loads BERTScore on demand)"
            + (" [offline mode]" if self._offline else "")
        )

    @contextmanager
    def _offline_ctx(self):
        """Temporarily set HF offline env vars when bertscore_offline is True.

        Prevents BERTScorer from attempting any network access during model
        loading, falling back to the local HuggingFace cache instead.
        """
        if not self._offline:
            yield
            return
        _keys = ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE")
        _saved = {k: os.environ.get(k) for k in _keys}
        for k in _keys:
            os.environ[k] = "1"
        try:
            yield
        finally:
            for k, v in _saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _get_vram_free_gb(self) -> float:
        """Get free VRAM in GB.

        Returns:
            Free VRAM or float('inf') if CUDA unavailable.
        """
        if not torch.cuda.is_available():
            return float("inf")
        free_bytes, _ = torch.cuda.mem_get_info(0)
        return free_bytes / 1e9

    def _select_bertscore_model(self) -> str:
        """Select BERTScore model based on available VRAM.

        Returns:
            Model identifier string for bert_score library.
        """
        free_vram = self._get_vram_free_gb()
        primary = self.config.get("bertscore_model", "microsoft/deberta-xlarge-mnli")
        fallback = self.config.get(
            "bertscore_model_fallback", "distilbert-base-uncased"
        )

        if free_vram >= 2.0:
            logger.info(
                f"VRAM free: {free_vram:.1f}GB ≥ 2GB — using primary BERTScore: {primary}"
            )
            return primary
        else:
            logger.warning(
                f"VRAM free: {free_vram:.1f}GB < 2GB — using fallback BERTScore: {fallback}"
            )
            return fallback

    def _compute_rouge(
        self, generated: str, reference: str
    ) -> dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L scores.

        Args:
            generated: Generated report text.
            reference: Ground truth report text.

        Returns:
            dict with rouge1, rouge2, rougeL F1 scores.
        """
        from rouge_score import rouge_scorer

        rouge_types = self.config.get(
            "rouge_types", ["rouge1", "rouge2", "rougeL"]
        )
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=False)
        scores = scorer.score(reference, generated)
        return {k: scores[k].fmeasure for k in rouge_types}

    def _compute_bleu(self, generated: str, reference: str) -> float:
        """Compute sentence-level BLEU score.

        Args:
            generated: Generated report text.
            reference: Ground truth report text.

        Returns:
            BLEU score as float.
        """
        import sacrebleu

        result = sacrebleu.sentence_bleu(generated, [reference])
        return result.score / 100.0  # Normalize to [0, 1]

    def _compute_bertscore(
        self, generated: list[str], reference: list[str], model: str
    ) -> dict[str, float]:
        """Compute BERTScore P/R/F1 for a batch.

        Args:
            generated: List of generated report texts.
            reference: List of reference report texts.
            model: BERTScore model identifier.

        Returns:
            dict with precision, recall, f1 averages.
        """
        from bert_score import BERTScorer

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Replace empty strings — older bert_score versions call
        # build_inputs_with_special_tokens on slow BertTokenizer for "" → AttributeError
        _PLACEHOLDER = "."
        _MAX_CHARS = 2000
        gen_safe = [(t[:_MAX_CHARS] if t.strip() else _PLACEHOLDER) for t in generated]
        ref_safe = [(t[:_MAX_CHARS] if t.strip() else _PLACEHOLDER) for t in reference]

        def _score_with(m: str):
            with self._offline_ctx():
                scorer = BERTScorer(model_type=m, lang="it", device=device)
            # Patch model_max_length to 512 — avoids Rust tokenizer OverflowError
            # when tokenizer.model_max_length is unset (very large int or inf)
            scorer._tokenizer.model_max_length = 512
            return scorer.score(gen_safe, ref_safe)

        try:
            P, R, F1 = _score_with(model)
        except Exception as primary_exc:
            logger.warning(
                f"BERTScore with {model!r} failed ({primary_exc}) — "
                f"retrying with fallback {self._bert_fallback!r}"
            )
            P, R, F1 = _score_with(self._bert_fallback)
        return {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }

    def evaluate(self, generated: str, reference: str) -> dict[str, float]:
        """Evaluate a single generated report against ground truth.

        Args:
            generated: Generated report text.
            reference: Ground truth report text.

        Returns:
            dict with rouge1, rouge2, rougeL, bleu, bertscore_precision,
            bertscore_recall, bertscore_f1.
        """
        rouge_scores = self._compute_rouge(generated, reference)
        bleu_score = self._compute_bleu(generated, reference)

        model = self._select_bertscore_model()
        bert_scores = self._compute_bertscore([generated], [reference], model)
        self._bertscore_model = model

        return {**rouge_scores, "bleu": bleu_score, **bert_scores}

    def evaluate_batch(
        self,
        generated_list: list[str],
        reference_list: list[str],
        image_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Evaluate a batch of generated reports against ground truth.

        Uses chunked BERTScore computation (chunk size 10) for batches > 20.

        Args:
            generated_list: List of generated report texts.
            reference_list: List of ground truth report texts.
            image_ids: Optional list of image identifiers.

        Returns:
            DataFrame with one row per image and metric columns.
        """
        if len(generated_list) != len(reference_list):
            raise ValueError(
                f"Mismatch: {len(generated_list)} generated vs "
                f"{len(reference_list)} references"
            )

        ids = image_ids or [f"image_{i:03d}" for i in range(len(generated_list))]

        # ROUGE and BLEU per-sample
        rows = []
        for gen, ref, img_id in zip(generated_list, reference_list, ids):
            rouge = self._compute_rouge(gen, ref)
            bleu = self._compute_bleu(gen, ref)
            rows.append({"image_id": img_id, **rouge, "bleu": bleu})

        df = pd.DataFrame(rows)

        # BERTScore in chunks of 10 — use BERTScorer with model_max_length patch
        # to avoid OverflowError in the Rust tokenizer backend
        bert_model = self._select_bertscore_model()
        chunk_size = 10
        all_bert = {"bertscore_precision": [], "bertscore_recall": [], "bertscore_f1": []}

        from bert_score import BERTScorer
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _PLACEHOLDER = "."
        _MAX_CHARS = 2000

        def _make_scorer(m: str) -> BERTScorer:
            with self._offline_ctx():
                s = BERTScorer(model_type=m, lang="it", device=device)
            s._tokenizer.model_max_length = 512
            return s

        try:
            bert_scorer = _make_scorer(bert_model)
        except Exception as exc:
            logger.warning(
                f"BERTScorer init failed for {bert_model!r} ({exc}) — "
                f"falling back to {self._bert_fallback!r}"
            )
            bert_scorer = _make_scorer(self._bert_fallback)

        for i in range(0, len(generated_list), chunk_size):
            chunk_gen = [
                (t[:_MAX_CHARS] if t.strip() else _PLACEHOLDER)
                for t in generated_list[i:i + chunk_size]
            ]
            chunk_ref = [
                (t[:_MAX_CHARS] if t.strip() else _PLACEHOLDER)
                for t in reference_list[i:i + chunk_size]
            ]
            P, R, F1 = bert_scorer.score(chunk_gen, chunk_ref)
            all_bert["bertscore_precision"].extend(P.tolist())
            all_bert["bertscore_recall"].extend(R.tolist())
            all_bert["bertscore_f1"].extend(F1.tolist())
            logger.info(f"BERTScore chunk {i // chunk_size + 1} done")

        for key, values in all_bert.items():
            df[key] = values

        self._bertscore_model = bert_model
        logger.info(f"Batch evaluation complete | n={len(df)}")
        return df

    def generate_comparison_table(
        self, results_dict: dict[str, dict]
    ) -> pd.DataFrame:
        """Build a comparison table from results of multiple models.

        Args:
            results_dict: dict mapping model_name -> metrics dict.

        Returns:
            DataFrame with model names as index and metrics as columns.
        """
        rows = []
        for model_name, metrics in results_dict.items():
            row = {"model": model_name}
            row.update(metrics)
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")

    def export_results(
        self,
        results_df: pd.DataFrame,
        output_dir: Union[str, Path],
    ) -> None:
        """Export evaluation results: CSV, bar charts, heatmap, Markdown summary.

        Args:
            results_df: DataFrame from evaluate_batch().
            output_dir: Directory to write outputs.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        csv_path = output_dir / "nlg_metrics.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved metrics CSV → {csv_path}")

        # Bar chart of mean scores
        metric_cols = [c for c in results_df.columns if c != "image_id"]
        means = results_df[metric_cols].mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        means.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title("Mean NLG Metrics")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "nlg_bar_chart.png", dpi=150)
        plt.close()

        # Heatmap of per-image metrics
        if len(results_df) > 1:
            heatmap_data = results_df[metric_cols].set_index(
                results_df.get("image_id", results_df.index)
            )
            fig, ax = plt.subplots(figsize=(12, max(4, len(heatmap_data) * 0.4)))
            sns.heatmap(
                heatmap_data,
                annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=ax,
            )
            ax.set_title("NLG Metrics Heatmap")
            plt.tight_layout()
            plt.savefig(output_dir / "nlg_heatmap.png", dpi=150)
            plt.close()

        # Markdown summary
        summary_lines = [
            "# NLG Metrics Evaluation",
            "",
            f"**Sample count:** {len(results_df)}",
            f"**BERTScore model:** {self._bertscore_model or 'N/A'}",
            "",
            "## Statistics",
            "",
            results_df[metric_cols].describe().to_markdown(),
            "",
        ]
        (output_dir / "nlg_summary.md").write_text(
            "\n".join(summary_lines), encoding="utf-8"
        )
        logger.info(f"NLG results exported → {output_dir}")
