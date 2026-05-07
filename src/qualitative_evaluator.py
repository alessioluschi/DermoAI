"""
Qualitative Evaluator — rubric-based scoring of ABCDE reports.
Runs entirely on CPU, no VRAM impact. Always available.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

QUALITATIVE_RUBRIC: dict[str, dict[int, str]] = {
    "completeness": {
        5: "Tutte le 3 sezioni ABC + conclusione presenti e sviluppate",
        4: "Tutte presenti, 1-2 poco sviluppate",
        3: "1 sezione mancante o 3+ poco sviluppate",
        2: "2 sezioni mancanti",
        1: "3+ sezioni mancanti o referto destrutturato",
    },
    "clinical_consistency": {
        5: "Perfetta coerenza melanoma→alto rischio / nevo→basso rischio",
        4: "Coerente con 1 lieve incongruenza",
        3: "Parzialmente coerente",
        2: "Scarsa coerenza con classificazione CNN",
        1: "Contraddittorio con classificazione CNN",
    },
    "specificity": {
        5: "8+ termini clinici da lista config",
        4: "5-7 termini clinici",
        3: "3-4 termini clinici",
        2: "1-2 termini clinici",
        1: "0 termini clinici",
    },
    "gradcam_integration": {
        5: "Riferimenti precisi alle regioni attivate dal GradCAM",
        4: "Riferimenti generali ma coerenti",
        3: "Menzione vaga",
        2: "Nessun riferimento spaziale",
        1: "Informazioni spaziali contraddittorie",
    },
    "actionability": {
        5: "Livello rischio chiaro + azione specifica",
        4: "Livello rischio + raccomandazione generica",
        3: "Solo livello rischio",
        2: "Conclusione vaga",
        1: "Nessuna conclusione",
    },
    "disclaimer_present": {
        5: "Disclaimer AI completo e ben posizionato",
        1: "Disclaimer assente",
    },
}

MAX_SCORE: int = 30  # 6 criteria × max 5 points


class QualitativeEvaluator:
    """CPU-based qualitative scoring of ABCDE reports using a 6-criterion rubric.

    Evaluates completeness, clinical consistency, clinical specificity,
    GradCAM integration, actionability, and disclaimer presence.
    Always runs regardless of VRAM availability.

    Args:
        config: Configuration dict from config.yaml (qualitative section).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.clinical_terms: list[str] = config.get("clinical_terms", [])
        self.action_terms: list[str] = config.get("action_terms", [])
        logger.info(
            f"QualitativeEvaluator initialized | "
            f"clinical_terms={len(self.clinical_terms)} | "
            f"action_terms={len(self.action_terms)}"
        )

    # ------------------------------------------------------------------ #
    # Individual criterion scorers
    # ------------------------------------------------------------------ #

    def _score_completeness(self, sections: dict[str, str]) -> int:
        """Score based on number and depth of ABC sections + conclusion present.

        ABC sections require at least 1 sentence. Conclusion requires >2 sentences.

        Args:
            sections: dict mapping section keys to text.

        Returns:
            Score 1-5.
        """
        import re as _re

        abc_keys = ["asymmetry", "borders", "color"]
        all_keys = abc_keys + ["conclusion"]

        def _count_sentences(text: str) -> int:
            return len([s for s in _re.split(r"[.!?]+", text) if s.strip()])

        filled = [k for k in all_keys if sections.get(k, "").strip()]
        n_filled = len(filled)

        # ABC: at least 1 sentence each; conclusion: >2 sentences
        abc_ok = all(
            _count_sentences(sections.get(k, "")) >= 1 for k in abc_keys if k in filled
        )
        conclusion_developed = (
            "conclusion" in filled
            and _count_sentences(sections["conclusion"]) >= 2
        )

        if n_filled == 4 and abc_ok and conclusion_developed:
            return 5
        elif n_filled == 4 and abc_ok:
            return 4
        elif n_filled == 3:
            return 3
        elif n_filled == 2:
            return 2
        return 1

    def _score_clinical_consistency(
        self, report_text: str, classification: str
    ) -> int:
        """Score coherence between CNN classification and risk level in report.

        Args:
            report_text: Full report text.
            classification: CNN prediction ("melanoma" or "nevus").

        Returns:
            Score 1-5.
        """
        text_lower = report_text.lower()
        is_melanoma = classification == "melanoma"

        high_risk_words = ["alto rischio", "rischio alto", "biopsia", "escissione", "urgente", "maligno"]
        low_risk_words = ["basso rischio", "rischio basso", "benigno", "monitoraggio", "follow-up"]
        contradictions = ["basso rischio" if is_melanoma else "alto rischio",
                          "benigno" if is_melanoma else "maligno"]

        has_contradiction = any(w in text_lower for w in contradictions)
        matches_expected = any(
            w in text_lower for w in (high_risk_words if is_melanoma else low_risk_words)
        )

        if has_contradiction:
            return 1
        if not matches_expected:
            return 2

        # Count supporting evidence
        n_matches = sum(
            1 for w in (high_risk_words if is_melanoma else low_risk_words)
            if w in text_lower
        )
        if n_matches >= 3:
            return 5
        elif n_matches >= 2:
            return 4
        return 3

    def _score_specificity(self, report_text: str) -> int:
        """Score clinical terminology usage.

        Args:
            report_text: Full report text.

        Returns:
            Score 1-5 based on count of clinical terms found.
        """
        text_lower = report_text.lower()
        found = sum(1 for term in self.clinical_terms if term.lower() in text_lower)

        if found >= 8:
            return 5
        elif found >= 5:
            return 4
        elif found >= 3:
            return 3
        elif found >= 1:
            return 2
        return 1

    def _score_gradcam_integration(
        self, report_text: str, focus_regions: str
    ) -> int:
        """Score spatial reference to GradCAM activation regions.

        Args:
            report_text: Full report text.
            focus_regions: Focus region description from GradCAM.

        Returns:
            Score 1-5.
        """
        text_lower = report_text.lower()
        focus_lower = focus_regions.lower()

        # Extract keywords from focus description
        spatial_words = [
            "centrale", "superiore", "inferiore", "sinistro", "destro",
            "periferica", "periferia", "bordo", "margine", "centro",
        ] + focus_lower.split()

        spatial_matches = sum(1 for w in spatial_words if w in text_lower)
        has_gradcam_mention = any(
            w in text_lower for w in ["gradcam", "attivazione", "attenzione", "focus", "mappa"]
        )

        if spatial_matches >= 3 and has_gradcam_mention:
            return 5
        elif spatial_matches >= 2:
            return 4
        elif spatial_matches >= 1:
            return 3
        elif any(w in text_lower for w in ["regione", "zona", "area"]):
            return 2
        return 1

    _RISK_LEVEL_RE = re.compile(
        r"(?:rischio\s*[:\s]*\s*(?:alto|basso|moderato)"
        r"|(?:alto|basso|moderato)\s+rischio)",
        re.IGNORECASE,
    )

    def _score_actionability(self, report_text: str) -> int:
        """Score clarity of risk level and clinical recommendation.

        Args:
            report_text: Full report text.

        Returns:
            Score 1-5.
        """
        text_lower = report_text.lower()
        has_risk_level = bool(self._RISK_LEVEL_RE.search(text_lower))
        has_specific_action = any(
            term.lower() in text_lower for term in self.action_terms
            if term.lower() in ["biopsia", "escissione", "follow-up", "rivalutazione"]
        )
        has_generic_action = any(
            term.lower() in text_lower for term in self.action_terms
        )

        if has_risk_level and has_specific_action:
            return 5
        elif has_risk_level and has_generic_action:
            return 4
        elif has_risk_level:
            return 3
        elif has_generic_action:
            return 2
        return 1

    def _score_disclaimer(self, report_text: str) -> int:
        """Score presence of AI disclaimer.

        Args:
            report_text: Full report text.

        Returns:
            5 if disclaimer found, 1 otherwise.
        """
        disclaimer_phrases = [
            "non sostituisce",
            "sistema ai",
            "non è un sostituto",
            "supporto",
            "giudizio clinico",
        ]
        text_lower = report_text.lower()
        found = sum(1 for phrase in disclaimer_phrases if phrase in text_lower)
        return 5 if found >= 2 else (3 if found == 1 else 1)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def auto_evaluate(
        self,
        report_dict: dict,
        classification_result: dict,
        gradcam_result: dict,
    ) -> dict:
        """Automatically score a report using the 6-criterion rubric.

        Args:
            report_dict: Output from ABCDEReportGenerator.generate_report().
            classification_result: Output from MelanomaClassifier.predict().
            gradcam_result: Output from GradCAMExplainer.generate().

        Returns:
            dict with keys:
                - scores (dict[str, int]): Per-criterion scores.
                - total (int): Sum of all scores.
                - max_possible (int): Always 30.
                - percentage (float): total / max_possible * 100.
                - details (dict[str, str]): Rubric description for each score.
        """
        report_text = report_dict.get("report_text", "")
        sections = report_dict.get("sections", {})
        classification = classification_result.get("class", "nevus")
        focus_regions = gradcam_result.get("focus_regions", "zona centrale")

        scores = {
            "completeness": self._score_completeness(sections),
            "clinical_consistency": self._score_clinical_consistency(
                report_text, classification
            ),
            "specificity": self._score_specificity(report_text),
            "gradcam_integration": self._score_gradcam_integration(
                report_text, focus_regions
            ),
            "actionability": self._score_actionability(report_text),
            "disclaimer_present": self._score_disclaimer(report_text),
        }

        total = sum(scores.values())
        details = {
            criterion: QUALITATIVE_RUBRIC[criterion].get(score, "")
            for criterion, score in scores.items()
        }

        logger.info(
            f"Qualitative eval: {total}/{MAX_SCORE} "
            f"({total / MAX_SCORE * 100:.1f}%) | scores={scores}"
        )

        return {
            "scores": scores,
            "total": total,
            "max_possible": MAX_SCORE,
            "percentage": total / MAX_SCORE * 100,
            "details": details,
        }

    def manual_evaluate_template(
        self,
        image_ids: list[str],
        output_dir: Union[str, Path] = "outputs/evaluation",
    ) -> None:
        """Generate a CSV template for manual scoring.

        Args:
            image_ids: List of image identifiers.
            output_dir: Directory to write the template CSV.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        columns = [
            "image_id",
            "completeness",
            "clinical_consistency",
            "specificity",
            "gradcam_integration",
            "actionability",
            "disclaimer_present",
            "evaluator_notes",
        ]
        rows = [
            {
                "image_id": img_id,
                "completeness": "",
                "clinical_consistency": "",
                "specificity": "",
                "gradcam_integration": "",
                "actionability": "",
                "disclaimer_present": "",
                "evaluator_notes": "",
            }
            for img_id in image_ids
        ]
        df = pd.DataFrame(rows, columns=columns)
        path = output_dir / "manual_scoring_template.csv"
        df.to_csv(path, index=False)
        logger.info(f"Manual scoring template saved → {path} ({len(image_ids)} rows)")

    def load_manual_scores(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        """Load manually filled scoring CSV.

        Args:
            csv_path: Path to the filled manual scoring CSV.

        Returns:
            DataFrame with image_id and numeric score columns.

        Raises:
            FileNotFoundError: If CSV does not exist.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Manual scores CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        score_cols = [
            "completeness", "clinical_consistency", "specificity",
            "gradcam_integration", "actionability", "disclaimer_present",
        ]
        for col in score_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def aggregate_scores(self, evaluations: list[dict]) -> pd.DataFrame:
        """Aggregate a list of auto_evaluate results into summary statistics.

        Args:
            evaluations: List of dicts from auto_evaluate().

        Returns:
            DataFrame with mean/std/min/max per criterion.
        """
        rows = []
        for ev in evaluations:
            rows.append(ev["scores"])

        df = pd.DataFrame(rows)
        return df.agg(["mean", "std", "min", "max"]).round(3)

    def export_qualitative_report(
        self,
        evaluations: list[dict],
        image_ids: list[str] | None = None,
        output_dir: Union[str, Path] = "outputs/evaluation",
    ) -> None:
        """Export qualitative evaluation: radar charts, box plot, CSV, Markdown.

        Args:
            evaluations: List of dicts from auto_evaluate().
            image_ids: Optional list of image identifiers.
            output_dir: Directory to write all outputs.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ids = image_ids or [f"image_{i:03d}" for i in range(len(evaluations))]

        criteria = list(QUALITATIVE_RUBRIC.keys())
        n_criteria = len(criteria)
        angles = np.linspace(0, 2 * np.pi, n_criteria, endpoint=False).tolist()
        angles += angles[:1]  # close polygon

        # --- Per-report radar charts ---
        for img_id, ev in zip(ids, evaluations):
            values = [ev["scores"].get(c, 0) for c in criteria]
            values += values[:1]

            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
            ax.plot(angles, values, "b-", linewidth=2)
            ax.fill(angles, values, alpha=0.25, color="blue")
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(criteria, size=8)
            ax.set_ylim(0, 5)
            ax.set_title(
                f"{img_id}\nTotale: {ev['total']}/{ev['max_possible']} "
                f"({ev['percentage']:.0f}%)",
                size=9,
            )
            plt.tight_layout()
            plt.savefig(output_dir / f"radar_{img_id}.png", dpi=120)
            plt.close()

        # --- Aggregate radar chart ---
        all_scores = [[ev["scores"].get(c, 0) for c in criteria] for ev in evaluations]
        import numpy as np_arr
        mean_scores = np.array(all_scores).mean(axis=0).tolist()
        mean_scores += mean_scores[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
        ax.plot(angles, mean_scores, "r-", linewidth=2)
        ax.fill(angles, mean_scores, alpha=0.25, color="red")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria, size=9)
        ax.set_ylim(0, 5)
        ax.set_title(f"Media Aggregata (n={len(evaluations)})", size=10)
        plt.tight_layout()
        plt.savefig(output_dir / "radar_aggregato.png", dpi=150)
        plt.close()

        # --- Box plot ---
        rows = [ev["scores"] for ev in evaluations]
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(10, 5))
        df.boxplot(ax=ax)
        ax.set_title("Distribuzione Punteggi Qualitativi")
        ax.set_ylabel("Score (1-5)")
        ax.set_ylim(0, 6)
        plt.tight_layout()
        plt.savefig(output_dir / "boxplot_qualitative.png", dpi=150)
        plt.close()

        # --- CSV ---
        csv_rows = []
        for img_id, ev in zip(ids, evaluations):
            row = {"image_id": img_id, **ev["scores"],
                   "total": ev["total"], "percentage": round(ev["percentage"], 1)}
            csv_rows.append(row)
        pd.DataFrame(csv_rows).to_csv(
            output_dir / "qualitative_scores.csv", index=False
        )

        # --- Markdown summary ---
        agg = self.aggregate_scores(evaluations)
        summary = [
            "# Qualitative Report Evaluation",
            "",
            f"**Evaluated samples:** {len(evaluations)}",
            "",
            "## Statistics by Criterion",
            "",
            agg.to_markdown(),
            "",
            "## Rubric",
            "",
        ]
        for criterion, rubric in QUALITATIVE_RUBRIC.items():
            summary.append(f"### {criterion}")
            for score, desc in sorted(rubric.items(), reverse=True):
                summary.append(f"- **{score}**: {desc}")
            summary.append("")

        (output_dir / "qualitative_summary.md").write_text(
            "\n".join(summary), encoding="utf-8"
        )
        logger.info(f"Qualitative report exported → {output_dir}")
