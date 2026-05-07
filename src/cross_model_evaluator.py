"""
Cross-Model Evaluator — compares LLM reports without ground truth.
Measures consensus and agreement, NOT clinical accuracy.
"""
from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Union

import pandas as pd
import torch

logger = logging.getLogger(__name__)

_METHODOLOGICAL_NOTE = (
    "⚠️ METHODOLOGICAL NOTE: Cross-model metrics measure CONSISTENCY between "
    "reports from different models, NOT clinical accuracy. Evaluating clinical "
    "accuracy requires ground truth reports written by dermatologists."
)


class CrossModelEvaluator:
    """Evaluates and ranks multiple LLM-generated reports without ground truth.

    Computes pairwise similarity, leave-one-out consensus scores, and an
    NxN agreement matrix. Produces composite ranking.

    Critical VRAM constraint: Never loads two LLMs simultaneously.
    The report generation must be done sequentially (load→generate→offload)
    before calling this evaluator.

    Args:
        qualitative_scores: Optional dict mapping model_name -> qualitative score (0-30).
        config: Configuration dict from config.yaml (cross_model section).
    """

    def __init__(
        self,
        qualitative_scores: dict[str, float] | None = None,
        config: dict | None = None,
    ) -> None:
        self.qualitative_scores = qualitative_scores or {}
        self.config = config or {}
        self._ranking_weights = self.config.get(
            "ranking_weights",
            {"consensus_bertscore": 0.40, "qualitative_total": 0.40, "mean_pairwise_agreement": 0.20},
        )
        logger.info("CrossModelEvaluator initialized")

    def _get_vram_free_gb(self) -> float:
        """Get free VRAM in GB."""
        if not torch.cuda.is_available():
            return float("inf")
        free_bytes, _ = torch.cuda.mem_get_info(0)
        return free_bytes / 1e9

    def _compute_rouge_bleu(self, text_a: str, text_b: str) -> dict[str, float]:
        """Compute bidirectional ROUGE/BLEU between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            dict with rouge1, rouge2, rougeL, bleu (mean of both directions).
        """
        from rouge_score import rouge_scorer
        import sacrebleu

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=False
        )
        ab = scorer.score(text_b, text_a)
        ba = scorer.score(text_a, text_b)
        bleu_ab = sacrebleu.sentence_bleu(text_a, [text_b]).score / 100.0
        bleu_ba = sacrebleu.sentence_bleu(text_b, [text_a]).score / 100.0

        return {
            "rouge1": (ab["rouge1"].fmeasure + ba["rouge1"].fmeasure) / 2,
            "rouge2": (ab["rouge2"].fmeasure + ba["rouge2"].fmeasure) / 2,
            "rougeL": (ab["rougeL"].fmeasure + ba["rougeL"].fmeasure) / 2,
            "bleu": (bleu_ab + bleu_ba) / 2,
        }

    def _compute_bertscore_pair(self, text_a: str, text_b: str) -> float:
        """Compute bidirectional BERTScore F1 between two texts.

        Uses distilbert to avoid VRAM issues during cross-model evaluation.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Mean bidirectional BERTScore F1.
        """
        import bert_score

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_type = "distilbert-base-uncased"  # Lightweight for cross-model

        P_ab, R_ab, F1_ab = bert_score.score(
            [text_a], [text_b], model_type=model_type, lang="it",
            device=device, verbose=False,
        )
        P_ba, R_ba, F1_ba = bert_score.score(
            [text_b], [text_a], model_type=model_type, lang="it",
            device=device, verbose=False,
        )
        return float((F1_ab.mean() + F1_ba.mean()) / 2)

    def cross_model_evaluate(
        self, reports_dict: dict[str, str]
    ) -> dict:
        """Run full cross-model evaluation pipeline.

        Steps:
            1. Pairwise metrics (ROUGE/BLEU/BERTScore) for all model pairs.
            2. Consensus scoring via leave-one-out pseudo-reference.
            3. NxN agreement matrix (BERTScore F1).
            4. Composite ranking with configurable weights.

        Args:
            reports_dict: dict mapping model_name -> report_text.

        Returns:
            dict with keys:
                - pairwise_metrics (list[dict]): Per-pair metric rows.
                - consensus_scores (dict[str, dict]): Per-model consensus metrics.
                - agreement_matrix (pd.DataFrame): NxN BERTScore F1 matrix.
                - composite_ranking (list[dict]): Models ranked by composite score.
                - methodological_note (str): Mandatory disclaimer.
        """
        model_names = list(reports_dict.keys())
        n = len(model_names)

        if n < 2:
            logger.warning("Need ≥ 2 models for cross-model evaluation")
            return {
                "pairwise_metrics": [],
                "consensus_scores": {},
                "agreement_matrix": pd.DataFrame(),
                "composite_ranking": [],
                "methodological_note": _METHODOLOGICAL_NOTE,
            }

        # 1. Pairwise metrics
        pairwise_rows = []
        logger.info(f"Computing pairwise metrics for {n} models ({n*(n-1)//2} pairs)...")
        for model_a, model_b in itertools.combinations(model_names, 2):
            rouge_bleu = self._compute_rouge_bleu(
                reports_dict[model_a], reports_dict[model_b]
            )
            bert_f1 = self._compute_bertscore_pair(
                reports_dict[model_a], reports_dict[model_b]
            )
            pairwise_rows.append({
                "model_a": model_a,
                "model_b": model_b,
                **rouge_bleu,
                "bertscore_f1": bert_f1,
            })
            logger.debug(f"Pair ({model_a}, {model_b}): BERTScore={bert_f1:.3f}")

        # 2. NxN agreement matrix
        logger.info("Building agreement matrix...")
        matrix_data = {}
        for model_a in model_names:
            matrix_data[model_a] = {}
            for model_b in model_names:
                if model_a == model_b:
                    matrix_data[model_a][model_b] = 1.0
                else:
                    # Find from pairwise_rows
                    for row in pairwise_rows:
                        if (row["model_a"] == model_a and row["model_b"] == model_b) or \
                           (row["model_a"] == model_b and row["model_b"] == model_a):
                            matrix_data[model_a][model_b] = row["bertscore_f1"]
                            break

        agreement_matrix = pd.DataFrame(matrix_data).reindex(
            index=model_names, columns=model_names
        )

        # 3. Leave-one-out consensus scoring
        logger.info("Computing consensus scores (leave-one-out)...")
        consensus_scores = {}
        for target_model in model_names:
            other_models = [m for m in model_names if m != target_model]
            if not other_models:
                continue
            # Pseudo-reference: concatenation of other models' reports
            pseudo_ref = " ".join(reports_dict[m] for m in other_models)
            bert_vs_consensus = self._compute_bertscore_pair(
                reports_dict[target_model], pseudo_ref
            )
            consensus_scores[target_model] = {
                "consensus_bertscore_f1": bert_vs_consensus,
                "mean_pairwise_bertscore": float(
                    agreement_matrix.loc[target_model, other_models].mean()
                ),
            }

        # 4. Composite ranking
        logger.info("Computing composite ranking...")
        ranking_rows = []
        for model in model_names:
            cs = consensus_scores.get(model, {})
            qual_raw = self.qualitative_scores.get(model, 0)
            qual_norm = qual_raw / 30.0  # Normalize to [0, 1]

            composite = (
                cs.get("consensus_bertscore_f1", 0) * self._ranking_weights["consensus_bertscore"]
                + qual_norm * self._ranking_weights["qualitative_total"]
                + cs.get("mean_pairwise_bertscore", 0) * self._ranking_weights["mean_pairwise_agreement"]
            )

            ranking_rows.append({
                "model": model,
                "consensus_bertscore": cs.get("consensus_bertscore_f1", 0),
                "qualitative_norm": qual_norm,
                "mean_pairwise_agreement": cs.get("mean_pairwise_bertscore", 0),
                "composite_score": composite,
            })

        ranking_rows.sort(key=lambda x: x["composite_score"], reverse=True)
        for i, row in enumerate(ranking_rows, 1):
            row["rank"] = i

        return {
            "pairwise_metrics": pairwise_rows,
            "consensus_scores": consensus_scores,
            "agreement_matrix": agreement_matrix,
            "composite_ranking": ranking_rows,
            "methodological_note": _METHODOLOGICAL_NOTE,
        }

    def export_cross_model_results(
        self,
        cross_model_result: dict,
        output_dir: Union[str, Path],
    ) -> None:
        """Export cross-model results: CSVs, heatmap, ranking, Markdown.

        Args:
            cross_model_result: Output from cross_model_evaluate().
            output_dir: Directory to write all outputs.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Pairwise metrics CSV
        if cross_model_result["pairwise_metrics"]:
            pd.DataFrame(cross_model_result["pairwise_metrics"]).to_csv(
                output_dir / "pairwise_metrics.csv", index=False
            )

        # Consensus ranking CSV
        pd.DataFrame(cross_model_result["composite_ranking"]).to_csv(
            output_dir / "consensus_ranking.csv", index=False
        )

        # Agreement heatmap (green=high, red=low)
        matrix: pd.DataFrame = cross_model_result["agreement_matrix"]
        if not matrix.empty:
            fig, ax = plt.subplots(
                figsize=(max(6, len(matrix) * 1.5), max(5, len(matrix) * 1.2))
            )
            sns.heatmap(
                matrix,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                vmin=0.5,
                vmax=1.0,
                ax=ax,
                linewidths=0.5,
            )
            ax.set_title("Cross-Model Agreement Matrix (BERTScore F1)")
            plt.tight_layout()
            plt.savefig(output_dir / "agreement_heatmap.png", dpi=150)
            plt.close()
            logger.info(f"Agreement heatmap saved → {output_dir / 'agreement_heatmap.png'}")

        # Markdown summary with mandatory methodological note
        ranking = cross_model_result["composite_ranking"]
        lines = [
            "# Cross-Model Evaluation Summary",
            "",
            cross_model_result["methodological_note"],
            "",
            "## Composite Ranking",
            "",
            "| Rank | Modello | Consensus BERTScore | Qualitative Norm | Mean Agreement | Composite Score |",
            "|------|---------|--------------------|--------------------|----------------|-----------------|",
        ]
        for row in ranking:
            lines.append(
                f"| {row['rank']} | {row['model']} | "
                f"{row['consensus_bertscore']:.3f} | "
                f"{row['qualitative_norm']:.3f} | "
                f"{row['mean_pairwise_agreement']:.3f} | "
                f"{row['composite_score']:.3f} |"
            )

        lines += [
            "",
            "## Weight Notes",
            "",
            f"- Consensus BERTScore: {self._ranking_weights['consensus_bertscore']:.0%}",
            f"- Qualitative Total: {self._ranking_weights['qualitative_total']:.0%}",
            f"- Mean Pairwise Agreement: {self._ranking_weights['mean_pairwise_agreement']:.0%}",
            "",
        ]

        (output_dir / "cross_model_summary.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        logger.info(f"Cross-model results exported → {output_dir}")

    def build_final_table(
        self,
        reports_dict: dict[str, str],
        qualitative_evaluations: dict[str, dict],
        cross_model_result: dict,
        ground_truth_metrics: dict[str, dict] | None = None,
    ) -> pd.DataFrame:
        """Build the final comparison table (with or without ground truth).

        Args:
            reports_dict: Model name → report text.
            qualitative_evaluations: Model name → qualitative eval dict.
            cross_model_result: Output from cross_model_evaluate().
            ground_truth_metrics: Optional model name → NLG metrics dict.

        Returns:
            DataFrame with models as rows and all metrics as columns.
        """
        from .qualitative_evaluator import QUALITATIVE_RUBRIC

        criteria = list(QUALITATIVE_RUBRIC.keys())
        rows = []

        for row in cross_model_result["composite_ranking"]:
            model = row["model"]
            qual = qualitative_evaluations.get(model, {})
            qual_scores = qual.get("scores", {})

            data = {
                "Model": model,
                "Rank": row["rank"],
            }

            # Qualitative scores
            for c in criteria:
                data[c.replace("_", " ").title()] = qual_scores.get(c, 0)
            data["Qualitative Total"] = qual.get("total", 0)

            # Cross-model metrics
            data["Pairwise ROUGE-L (xm)"] = None
            data["Pairwise BERTScore (xm)"] = row["mean_pairwise_agreement"]
            data["Cross-Model Agreement"] = row["mean_pairwise_agreement"]
            data["Consensus BERTScore"] = row["consensus_bertscore"]
            data["Composite Score"] = row["composite_score"]

            # Optional ground truth NLG metrics
            if ground_truth_metrics and model in ground_truth_metrics:
                gt = ground_truth_metrics[model]
                data["ROUGE-1"] = gt.get("rouge1", None)
                data["ROUGE-2"] = gt.get("rouge2", None)
                data["ROUGE-L"] = gt.get("rougeL", None)
                data["BLEU"] = gt.get("bleu", None)
                data["BERTScore F1"] = gt.get("bertscore_f1", None)

            rows.append(data)

        return pd.DataFrame(rows).set_index("Model")
