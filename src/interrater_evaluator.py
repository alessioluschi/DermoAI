"""
Inter-Rater Agreement Evaluator.

Computes categorical agreement (Fleiss' kappa, Krippendorff's alpha, ICC(3,k))
and NLG-level inter-rater BERTScore for multi-rater ground truth reports.
Compares AI-generated reports against inter-clinician variability.
"""
from __future__ import annotations

import json
import logging
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class InterRaterEvaluator:
    """Evaluates inter-rater agreement and AI-vs-rater similarity.

    Args:
        config: Configuration dict from config.yaml (interrater section).
    """

    # Ordinal encoding maps
    ASYMMETRY_MAP: dict[str, int] = {
        "simmetrica": 0,
        "lievemente asimmetrica": 1,
        "asimmetrica": 2,
        "marcatamente asimmetrica": 3,
    }
    BORDER_MAP: dict[str, int] = {
        "regolari": 0,
        "lievemente irregolari": 1,
        "irregolari": 2,
        "molto irregolari": 3,
    }
    HETEROGENEITY_MAP: dict[str, int] = {
        "omogenea": 0,
        "lievemente eterogenea": 1,
        "eterogenea": 2,
        "marcatamente eterogenea": 3,
    }

    KAPPA_THRESHOLDS: list[tuple[float, str]] = [
        (0.00, "poor"),
        (0.20, "slight"),
        (0.40, "fair"),
        (0.60, "moderate"),
        (0.80, "substantial"),
        (1.01, "almost perfect"),
    ]

    def __init__(self, config: dict) -> None:
        self.config = config
        self.min_raters = config.get("min_raters_required", 2)
        self.min_images_for_categorical = config.get("min_images_for_categorical", 5)
        self.bootstrap_iterations = config.get("bootstrap_iterations", 1000)
        self.bootstrap_seed = config.get("bootstrap_seed", 42)

    def load_reports(self, gt_dir: str) -> dict:
        """Load ground truth reports from multi-rater directory structure.

        Args:
            gt_dir: Path to ground_truth_reports/ directory.

        Returns:
            Nested dict {image_id: {rater_id: {"report_text": str}}}.

        Raises:
            ValueError: If fewer than 2 raters or empty image intersection.
        """
        gt_path = Path(gt_dir)
        rater_dirs = sorted([d for d in gt_path.iterdir() if d.is_dir()])

        if len(rater_dirs) < 2:
            raise ValueError(
                f"Need at least 2 rater subdirectories, found {len(rater_dirs)} in {gt_dir}"
            )

        # Collect per-rater image sets
        rater_images: dict[str, dict[str, str]] = {}
        for rater_dir in rater_dirs:
            rater_id = rater_dir.name
            rater_images[rater_id] = {}
            for json_file in sorted(rater_dir.glob("*.json")):
                image_id = json_file.stem
                try:
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)
                    if "report" in data:
                        report_text = data["report"]
                    else:
                        first_val = next(iter(data.values()), {})
                        report_text = first_val.get("report", "") if isinstance(first_val, dict) else ""
                    if not report_text:
                        logger.warning(f"Skipping {json_file}: empty or missing 'report'")
                        continue
                    rater_images[rater_id][image_id] = report_text
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning(f"Skipping {json_file}: {exc}")

        all_rater_ids = list(rater_images.keys())
        n_raters = len(all_rater_ids)

        # Intersect image sets
        image_sets = [set(rater_images[r].keys()) for r in all_rater_ids]
        common_images = set.intersection(*image_sets)
        all_images = set.union(*image_sets)

        for img_id in sorted(all_images - common_images):
            present_in = sum(1 for s in image_sets if img_id in s)
            logger.warning(
                f"{img_id} found in {present_in}/{n_raters} raters — "
                "excluded from inter-rater analysis."
            )

        if not common_images:
            raise ValueError(
                "Intersection of image sets across all raters is empty."
            )

        # Build result dict
        reports: dict[str, dict[str, dict[str, str]]] = {}
        for image_id in sorted(common_images):
            reports[image_id] = {}
            for rater_id in all_rater_ids:
                reports[image_id][rater_id] = {
                    "report_text": rater_images[rater_id][image_id]
                }

        logger.info(
            f"Loaded {len(common_images)} images x {n_raters} raters from {gt_dir}"
        )
        return reports

    def extract_categorical(self, report_text: str) -> dict:
        """Extract ordinal categorical labels from ABC report sections.

        Args:
            report_text: Full Italian ABC report text.

        Returns:
            Dict with asymmetry_level, border_level, heterogeneity_level.
        """
        text_lower = report_text.lower()

        def _extract_section(header: str) -> str | None:
            pattern = re.escape(header.lower())
            match = re.search(pattern, text_lower)
            if not match:
                return None
            start = match.end()
            next_marker = re.search(r"\*\*", text_lower[start:])
            if next_marker:
                return text_lower[start:start + next_marker.start()]
            return text_lower[start:]

        # Asymmetry
        asymmetry_level = None
        section_a = _extract_section("**A - Asimmetria:**")
        if section_a is not None:
            # Skip non-assessable cases (N/D, non valutabile, lesione troppo grande)
            is_na = ("n/d" in section_a or "non valutabil" in section_a
                     or "non possibil" in section_a
                     or "troppo grande" in section_a)
            if is_na:
                # If text also contains "asimmetri" treat as assessable
                if "asimmetri" in section_a:
                    asymmetry_level = "asimmetrica"
                else:
                    asymmetry_level = "non valutabile"
            elif "marcata" in section_a or "marcatamente" in section_a or "molto" in section_a:
                asymmetry_level = "marcatamente asimmetrica"
            elif "asimmetri" in section_a:
                asymmetry_level = "asimmetrica"
            elif "lieve" in section_a:
                asymmetry_level = "lievemente asimmetrica"
            elif "simmetri" in section_a:
                asymmetry_level = "simmetrica"
            elif re.search(r"\b(2|due)\s*(ass[ei])", section_a):
                # "su 2 assi", "sì, su 2 assi" → asymmetric
                asymmetry_level = "asimmetrica"
            elif re.search(r"\bsu\s+(un\s+)?ass[ei]", section_a) or "sì" in section_a:
                # "su un asse", "sì, su un asse" → asymmetric
                asymmetry_level = "asimmetrica"

        # Border — handles free-text clinical descriptions
        border_level = None
        section_b = _extract_section("**B - Bordi:**")
        if section_b is not None:
            # Non-assessable
            is_na_b = ("non valutabil" in section_b or "n/d" in section_b
                       or "troppo grande" in section_b)
            if is_na_b:
                border_level = "non valutabile"
            else:
                has_irreg = "irregol" in section_b
                has_mal_def = "mal defini" in section_b
                has_estrema = "estremamente" in section_b or "molto irregol" in section_b
                has_cut_off = "cut-off" in section_b or "cutoff" in section_b
                has_strie = "strie" in section_b
                has_atipic = "atipic" in section_b
                has_lieve = "lievemente" in section_b or "relativamente" in section_b
                has_ben_def = "ben defini" in section_b or "nett" in section_b
                has_regol = "regol" in section_b or "omogen" in section_b
                has_gradual = "sfumano gradualmente" in section_b
                has_per_lo_piu = "per lo pi" in section_b
                has_sfumati = "sfumat" in section_b
                has_espansivi = "espansiv" in section_b

                severe_markers = sum([has_estrema, has_cut_off, has_strie, has_atipic])

                has_variegato = "variegat" in section_b
                has_no_malig = "non presenta" in section_b or "assenza" in section_b

                if has_estrema or (has_irreg and severe_markers >= 2):
                    border_level = "molto irregolari"
                elif has_mal_def or (has_irreg and not has_lieve and not has_gradual):
                    border_level = "irregolari"
                elif (has_lieve or has_per_lo_piu or has_gradual
                      or (has_irreg and has_gradual)
                      or (has_variegato and has_no_malig)
                      or has_sfumati or has_espansivi):
                    border_level = "lievemente irregolari"
                elif has_ben_def or has_regol:
                    border_level = "regolari"

        # Colour — maps colour count to heterogeneity ordinal
        heterogeneity_level = None
        section_c = _extract_section("**C - Colore:**")
        if section_c is not None:
            # Try explicit heterogeneity terms first (AI-generated reports)
            if "marcata" in section_c and "eterog" in section_c:
                heterogeneity_level = "marcatamente eterogenea"
            elif "eterog" in section_c:
                heterogeneity_level = "eterogenea"
            elif "lieve" in section_c and ("eterog" in section_c or "variabil" in section_c):
                heterogeneity_level = "lievemente eterogenea"
            elif "omogen" in section_c or "uniforme" in section_c:
                heterogeneity_level = "omogenea"
            else:
                # Colour-count mapping (clinical rater style)
                count_map = {
                    "cinque": 5, "sei": 6, "sette": 7,
                    "quattro": 4, "tre": 3, "due": 2, "un ": 1, "uno": 1,
                }
                n_colours = 0
                for word, val in count_map.items():
                    if word in section_c:
                        n_colours = max(n_colours, val)
                        break

                # Try digit-based counts: "2 colori", "3 colors", standalone digits
                if n_colours == 0:
                    digit_match = re.search(r"\b(\d+)\b", section_c)
                    if digit_match:
                        n_colours = int(digit_match.group(1))

                # "multipli"/"molti"/"diversi"/"moltitudine"/"molteplici" → at least 3
                if n_colours == 0:
                    multi_words = ("multipli", "multipl", "molti", "diversi",
                                   "moltitudine", "multicolor", "molteplic")
                    if any(w in section_c for w in multi_words):
                        n_colours = 3
                    elif "pochi" in section_c:
                        n_colours = 2

                # Fallback: count listed colour names by commas/separators
                if n_colours == 0:
                    colour_names = (
                        "marrone", "nero", "blu", "bianco", "rosso", "rosa",
                        "grigio", "nocciola", "brunastro", "biancastro",
                        "grigiastro", "rosato", "bluastro", "rossastro",
                    )
                    found = sum(1 for c in colour_names if c in section_c)
                    if found > 0:
                        n_colours = found

                has_regression = ("regressione" in section_c
                                  or "grigio" in section_c
                                  or "grigiastro" in section_c)

                if n_colours >= 4 or (n_colours >= 3 and has_regression):
                    heterogeneity_level = "marcatamente eterogenea"
                elif n_colours == 3 or (n_colours == 2 and has_regression):
                    heterogeneity_level = "eterogenea"
                elif n_colours == 2:
                    heterogeneity_level = "lievemente eterogenea"
                elif n_colours == 1:
                    heterogeneity_level = "omogenea"

        return {
            "asymmetry_level": asymmetry_level,
            "border_level": border_level,
            "heterogeneity_level": heterogeneity_level,
        }

    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret kappa using Landis & Koch (1977) thresholds.

        Args:
            kappa: Fleiss' kappa value.

        Returns:
            Interpretation string.
        """
        if kappa < 0.00:
            return "poor"
        for threshold, label in self.KAPPA_THRESHOLDS:
            if kappa < threshold:
                return label
        return "almost perfect"

    def _interpret_icc(self, icc: float) -> str:
        """Interpret ICC using Koo & Li (2016) thresholds.

        Args:
            icc: Intraclass correlation coefficient value.

        Returns:
            Interpretation string.
        """
        if icc < 0.50:
            return "poor"
        if icc < 0.75:
            return "moderate"
        if icc < 0.90:
            return "good"
        return "excellent"

    def compute_categorical_agreement(self, reports: dict) -> dict:
        """Compute Fleiss' kappa, Krippendorff's alpha, and ICC(3,k) for categorical fields.

        Computes three agreement metrics on ordinal integer matrices
        (images x raters) for asymmetry_level, border_level, and
        heterogeneity_level. ICC(3,k) is the two-way mixed effects,
        average measurement, absolute agreement form (Koo & Li, 2016).

        Args:
            reports: Output from load_reports().

        Returns:
            Dict with fleiss_kappa, krippendorff_alpha, icc_3k,
            interpretation (nested per-metric), and n_images_per_field.
        """
        import krippendorff
        import pandas as pd
        from pingouin import intraclass_corr
        from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

        image_ids = sorted(reports.keys())
        rater_ids = sorted(next(iter(reports.values())).keys())
        n_raters = len(rater_ids)

        field_configs = [
            ("asymmetry_level", self.ASYMMETRY_MAP),
            ("border_level", self.BORDER_MAP),
            ("heterogeneity_level", self.HETEROGENEITY_MAP),
        ]

        result: dict = {
            "fleiss_kappa": {},
            "krippendorff_alpha": {},
            "icc_3k": {},
            "interpretation": {},
            "n_images_per_field": {},
        }

        for field_name, ordinal_map in field_configs:
            # Build matrix: n_images x n_raters
            rows: list[list[int]] = []
            valid_image_ids: list[str] = []
            for image_id in image_ids:
                row: list[int] = []
                skip = False
                for rater_id in rater_ids:
                    report_text = reports[image_id][rater_id]["report_text"]
                    cats = self.extract_categorical(report_text)
                    val = cats[field_name]
                    if val is None:
                        logger.warning(
                            f"{field_name}: None for {image_id}/{rater_id} — "
                            "excluding image from this field."
                        )
                        skip = True
                        break
                    if val == "non valutabile":
                        logger.debug(
                            f"{field_name}: non valutabile for "
                            f"{image_id}/{rater_id} — excluding image."
                        )
                        skip = True
                        break
                    if val not in ordinal_map:
                        logger.warning(
                            f"{field_name}: unknown value '{val}' for "
                            f"{image_id}/{rater_id} — excluding image."
                        )
                        skip = True
                        break
                    row.append(ordinal_map[val])
                if not skip:
                    rows.append(row)
                    valid_image_ids.append(image_id)

            n_valid = len(rows)
            if n_valid < self.min_images_for_categorical:
                logger.warning(
                    f"{field_name}: only {n_valid} images with complete ratings "
                    f"(need {self.min_images_for_categorical}) — skipping field."
                )
                continue

            matrix = np.array(rows)  # (n_images, n_raters)

            # Fleiss' kappa
            table, _ = aggregate_raters(matrix)
            fk = fleiss_kappa(table, method="fleiss")

            # Krippendorff's alpha (ordinal)
            # krippendorff expects (n_raters, n_images)
            ka = krippendorff.alpha(
                reliability_data=matrix.T,
                level_of_measurement="ordinal",
            )

            # ICC(3,k) — two-way mixed, average measurement, absolute agreement
            icc_value: float | None = None
            icc_ci_low: float | None = None
            icc_ci_high: float | None = None
            icc_pval: float | None = None
            try:
                # ICC requires variance across targets; skip if all ratings identical
                if matrix.var(ddof=1).sum() == 0:
                    raise ValueError("zero variance in ratings — ICC undefined")

                # Melt matrix to long format for pingouin
                df_long = pd.DataFrame(
                    [
                        {"image_id": valid_image_ids[i], "rater_id": rater_ids[j], "rating": int(matrix[i, j])}
                        for i in range(matrix.shape[0])
                        for j in range(matrix.shape[1])
                    ]
                )
                icc_result = intraclass_corr(
                    data=df_long,
                    targets="image_id",
                    raters="rater_id",
                    ratings="rating",
                )
                # ICC(C,k) = ICC3k: two-way mixed, average, consistency
                # Pingouin versions use different Type labels
                icc3k_row = icc_result[
                    icc_result["Type"].isin(["ICC3k", "ICC(C,k)"])
                ]
                if icc3k_row.empty and len(icc_result) >= 6:
                    icc3k_row = icc_result.iloc[[5]]
                if icc3k_row.empty:
                    raise ValueError(
                        f"ICC3k row not found in pingouin output "
                        f"(types: {icc_result['Type'].tolist()})"
                    )
                icc_value = float(icc3k_row["ICC"].iloc[0])
                # CI column name varies: "CI95%" or "CI95"
                ci_col = "CI95%" if "CI95%" in icc_result.columns else "CI95"
                try:
                    ci95 = icc3k_row[ci_col].iloc[0]
                    icc_ci_low = float(ci95[0])
                    icc_ci_high = float(ci95[1])
                except (TypeError, IndexError, KeyError):
                    icc_ci_low = None
                    icc_ci_high = None
                try:
                    icc_pval = float(icc3k_row["pval"].iloc[0])
                except (TypeError, IndexError, KeyError):
                    icc_pval = None
            except Exception as exc:
                logger.debug(
                    f"ICC(3,k) computation failed for {field_name}: {exc}"
                )

            interp_fk = self._interpret_kappa(fk)
            interp_ka = self._interpret_kappa(ka)
            interp_icc = self._interpret_icc(icc_value) if icc_value is not None else None

            result["fleiss_kappa"][field_name] = float(fk)
            result["krippendorff_alpha"][field_name] = float(ka)
            result["icc_3k"][field_name] = {
                "value": icc_value,
                "ci95_low": icc_ci_low,
                "ci95_high": icc_ci_high,
                "p_value": icc_pval,
            }
            result["interpretation"][field_name] = {
                "fleiss": interp_fk,
                "krippendorff": interp_ka,
                "icc_3k": interp_icc,
            }
            result["n_images_per_field"][field_name] = n_valid

        # Macro means
        if result["fleiss_kappa"]:
            fk_vals = list(result["fleiss_kappa"].values())
            ka_vals = list(result["krippendorff_alpha"].values())
            result["fleiss_kappa"]["macro_mean"] = float(np.mean(fk_vals))
            result["krippendorff_alpha"]["macro_mean"] = float(np.mean(ka_vals))
            icc_vals = [
                v["value"] for v in result["icc_3k"].values()
                if v["value"] is not None
            ]
            result["icc_3k"]["macro_mean"] = float(np.mean(icc_vals)) if icc_vals else None

        return result

    # Canonical metric order used across all NLG methods
    NLG_METRICS = [
        "bertscore_f1", "bertscore_precision", "bertscore_recall",
        "rouge1", "rouge2", "rougeL", "bleu",
    ]
    NLG_METRIC_LABELS = {
        "bertscore_f1": "BERTScore F1",
        "bertscore_precision": "BERTScore Precision",
        "bertscore_recall": "BERTScore Recall",
        "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2",
        "rougeL": "ROUGE-L",
        "bleu": "BLEU",
    }

    def _compute_rouge_bleu(
        self, cands: list[str], refs: list[str]
    ) -> dict[str, list[float]]:
        """Compute ROUGE-1/2/L and BLEU for candidate-reference pairs.

        Args:
            cands: Candidate texts.
            refs: Reference texts.

        Returns:
            Dict mapping metric name to list of scores.
        """
        from rouge_score import rouge_scorer
        import sacrebleu

        rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=False
        )
        r1: list[float] = []
        r2: list[float] = []
        rL: list[float] = []
        bl: list[float] = []
        for cand, ref in zip(cands, refs):
            rs = rouge.score(ref, cand)
            r1.append(rs["rouge1"].fmeasure)
            r2.append(rs["rouge2"].fmeasure)
            rL.append(rs["rougeL"].fmeasure)
            bl.append(sacrebleu.sentence_bleu(cand, [ref]).score / 100.0)
        return {"rouge1": r1, "rouge2": r2, "rougeL": rL, "bleu": bl}

    def _bootstrap_ci95(
        self, values: np.ndarray, rng: np.random.Generator
    ) -> dict[str, float]:
        """Compute mean, std, and bootstrap 95% CI for an array of values."""
        boot_means = []
        for _ in range(self.bootstrap_iterations):
            sample = rng.choice(values, size=len(values), replace=True)
            boot_means.append(float(np.mean(sample)))
        boot_arr = np.array(boot_means)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci95_low": float(np.percentile(boot_arr, 2.5)),
            "ci95_high": float(np.percentile(boot_arr, 97.5)),
        }

    def compute_nlg_interrater(
        self, reports: dict, model_name: str
    ) -> dict:
        """Compute pairwise NLG metrics between all rater pairs.

        Metrics: BERTScore (precision/recall/F1), ROUGE-1/2/L, BLEU.

        Args:
            reports: Output from load_reports().
            model_name: HuggingFace model for BERTScore.

        Returns:
            Dict with per_image scores and aggregate statistics per metric.
        """
        from bert_score import BERTScorer

        image_ids = sorted(reports.keys())
        rater_ids = sorted(next(iter(reports.values())).keys())

        # Collect all pairs for batch scoring
        all_cands: list[str] = []
        all_refs: list[str] = []

        for image_id in image_ids:
            texts = [reports[image_id][r]["report_text"] for r in rater_ids]
            for i, j in combinations(range(len(texts)), 2):
                all_cands.append(texts[i])
                all_refs.append(texts[j])

        # BERTScore batch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        scorer = BERTScorer(model_type=model_name, lang="it", device=device)
        scorer._tokenizer.model_max_length = 512
        P, R, F1 = scorer.score(all_cands, all_refs)

        del scorer
        torch.cuda.empty_cache()

        # ROUGE and BLEU per-pair
        rouge_bleu = self._compute_rouge_bleu(all_cands, all_refs)

        all_metric_scores: dict[str, list[float]] = {
            "bertscore_f1": F1.tolist(),
            "bertscore_precision": P.tolist(),
            "bertscore_recall": R.tolist(),
            **rouge_bleu,
        }

        # Organize per-image
        per_image: dict = {}
        idx = 0
        image_means: dict[str, list[float]] = {m: [] for m in self.NLG_METRICS}
        for image_id in image_ids:
            n_rater_texts = len(rater_ids)
            n_pairs = n_rater_texts * (n_rater_texts - 1) // 2
            img_data: dict = {"n_pairs": n_pairs}
            for metric in self.NLG_METRICS:
                pairwise = all_metric_scores[metric][idx:idx + n_pairs]
                img_data[f"pairwise_{metric}"] = pairwise
                img_data[f"mean_{metric}"] = float(np.mean(pairwise))
                image_means[metric].append(float(np.mean(pairwise)))
            idx += n_pairs
            per_image[image_id] = img_data

        # Aggregate with bootstrap CI95
        rng = np.random.default_rng(seed=self.bootstrap_seed)
        aggregate: dict = {}
        for metric in self.NLG_METRICS:
            aggregate[metric] = self._bootstrap_ci95(
                np.array(image_means[metric]), rng
            )

        return {
            "per_image": per_image,
            "aggregate": aggregate,
        }

    def compare_ai_vs_interrater(
        self,
        reports: dict,
        ai_reports_dir: str,
        model_name: str,
        interrater_result: dict,
    ) -> dict:
        """Compare AI reports against inter-rater variability.

        Metrics: BERTScore (precision/recall/F1), ROUGE-1/2/L, BLEU.
        Mann-Whitney U test on BERTScore F1.

        Args:
            reports: Output from load_reports().
            ai_reports_dir: Directory with AI-generated *_result.json files.
            model_name: HuggingFace model for BERTScore.
            interrater_result: Output from compute_nlg_interrater().

        Returns:
            Dict with per_image AI scores, aggregate, and Mann-Whitney test.
        """
        from bert_score import BERTScorer
        from scipy.stats import mannwhitneyu

        image_ids = sorted(reports.keys())
        rater_ids = sorted(next(iter(reports.values())).keys())
        ai_dir = Path(ai_reports_dir)

        # Load AI reports
        ai_texts: dict[str, str] = {}
        for image_id in image_ids:
            ai_file = ai_dir / f"{image_id}_result.json"
            if not ai_file.exists():
                logger.warning(f"AI report missing: {ai_file} — skipping {image_id}")
                continue
            try:
                with open(ai_file, encoding="utf-8") as f:
                    data = json.load(f)
                ai_texts[image_id] = data["report"]["report_text"]
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(f"Error reading {ai_file}: {exc} — skipping {image_id}")

        valid_ids = [img for img in image_ids if img in ai_texts]

        # Collect all AI-vs-rater pairs
        all_ai_cands: list[str] = []
        all_rater_refs: list[str] = []
        for image_id in valid_ids:
            ai_text = ai_texts[image_id]
            for rater_id in rater_ids:
                all_ai_cands.append(ai_text)
                all_rater_refs.append(reports[image_id][rater_id]["report_text"])

        # BERTScore batch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        scorer = BERTScorer(model_type=model_name, lang="it", device=device)
        scorer._tokenizer.model_max_length = 512
        P, R, F1 = scorer.score(all_ai_cands, all_rater_refs)

        del scorer
        torch.cuda.empty_cache()

        # ROUGE and BLEU per-pair
        rouge_bleu = self._compute_rouge_bleu(all_ai_cands, all_rater_refs)

        all_metric_scores: dict[str, list[float]] = {
            "bertscore_f1": F1.tolist(),
            "bertscore_precision": P.tolist(),
            "bertscore_recall": R.tolist(),
            **rouge_bleu,
        }

        # Organize per-image
        per_image: dict = {}
        idx = 0
        image_means: dict[str, list[float]] = {m: [] for m in self.NLG_METRICS}
        flat_ai_f1: list[float] = []
        for image_id in valid_ids:
            n = len(rater_ids)
            img_data: dict = {}
            for metric in self.NLG_METRICS:
                scores_for_img = all_metric_scores[metric][idx:idx + n]
                img_data[f"ai_vs_raters_{metric}"] = scores_for_img
                img_data[f"ai_mean_{metric}"] = float(np.mean(scores_for_img))
                image_means[metric].append(float(np.mean(scores_for_img)))
            flat_ai_f1.extend(all_metric_scores["bertscore_f1"][idx:idx + n])
            idx += n
            per_image[image_id] = img_data

        # Aggregate with bootstrap CI95
        rng = np.random.default_rng(seed=self.bootstrap_seed)
        aggregate: dict = {}
        for metric in self.NLG_METRICS:
            aggregate[metric] = self._bootstrap_ci95(
                np.array(image_means[metric]), rng
            )

        # Mann-Whitney U test on BERTScore F1
        flat_interrater_f1: list[float] = []
        for image_id in valid_ids:
            if image_id in interrater_result["per_image"]:
                flat_interrater_f1.extend(
                    interrater_result["per_image"][image_id]["pairwise_bertscore_f1"]
                )

        U, p = mannwhitneyu(
            flat_ai_f1, flat_interrater_f1, alternative="two-sided"
        )
        effect_size_r = 1 - (2 * U) / (len(flat_ai_f1) * len(flat_interrater_f1))

        if p > 0.05:
            interpretation = (
                f"AI reports are not significantly different from "
                f"inter-clinician variability (p={p:.3f})"
            )
        else:
            interpretation = (
                f"AI reports differ significantly from "
                f"inter-clinician variability (p={p:.3f})"
            )

        return {
            "per_image": per_image,
            "aggregate": aggregate,
            "mann_whitney": {
                "U_statistic": float(U),
                "p_value": float(p),
                "effect_size_r": float(effect_size_r),
                "interpretation": interpretation,
            },
        }

    def run(
        self, gt_dir: str, ai_reports_dir: str, output_dir: str
    ) -> dict:
        """Orchestrate full inter-rater evaluation.

        Args:
            gt_dir: Path to ground_truth_reports/ directory.
            ai_reports_dir: Path to AI-generated reports directory.
            output_dir: Output directory for results.

        Returns:
            Full aggregated results dict.
        """
        model_name = self.config.get(
            "bertscore_model", "microsoft/deberta-xlarge-mnli"
        )
        fallback_model = self.config.get(
            "bertscore_fallback", "distilbert-base-uncased"
        )

        # 1. Load reports
        reports = self.load_reports(gt_dir)
        n_images = len(reports)
        n_raters = len(next(iter(reports.values())))

        # 2. Categorical agreement
        categorical = self.compute_categorical_agreement(reports)

        # 3. NLG inter-rater BERTScore
        try:
            interrater_nlg = self.compute_nlg_interrater(reports, model_name)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            logger.warning(
                f"BERTScore OOM with {model_name} — retrying with {fallback_model}"
            )
            interrater_nlg = self.compute_nlg_interrater(reports, fallback_model)
            model_name = fallback_model

        # 4. AI vs inter-rater comparison
        try:
            ai_comparison = self.compare_ai_vs_interrater(
                reports, ai_reports_dir, model_name, interrater_nlg
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            logger.warning(
                f"BERTScore OOM with {model_name} — retrying with {fallback_model}"
            )
            ai_comparison = self.compare_ai_vs_interrater(
                reports, ai_reports_dir, fallback_model, interrater_nlg
            )

        # Save outputs
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "categorical_agreement.json", "w", encoding="utf-8") as f:
            json.dump(categorical, f, indent=2, ensure_ascii=False)

        with open(out_path / "nlg_interrater.json", "w", encoding="utf-8") as f:
            json.dump(interrater_nlg, f, indent=2, ensure_ascii=False)

        with open(out_path / "ai_vs_interrater.json", "w", encoding="utf-8") as f:
            json.dump(ai_comparison, f, indent=2, ensure_ascii=False)

        # Build summary markdown
        md_lines = [
            f"# Inter-Rater Agreement Report",
            "",
            f"**Images:** {n_images} | **Raters:** {n_raters}",
            "",
            "## Categorical Agreement (Fleiss' kappa / Krippendorff's alpha / ICC(3,k))",
            "",
            "| Field | Fleiss' kappa | Krippendorff's alpha | ICC(3,k) [CI95%] | Interpretation (k / alpha / ICC) | N images |",
            "|---|---|---|---|---|---|",
        ]

        for field in ["asymmetry_level", "border_level", "heterogeneity_level"]:
            if field in categorical.get("fleiss_kappa", {}):
                fk = categorical["fleiss_kappa"][field]
                ka = categorical["krippendorff_alpha"][field]
                icc_data = categorical["icc_3k"][field]
                interp = categorical["interpretation"][field]
                n_img = categorical["n_images_per_field"][field]
                label = field.replace("_level", "").replace("_", " ").title()
                if icc_data["value"] is not None:
                    if icc_data["ci95_low"] is not None and icc_data["ci95_high"] is not None:
                        icc_str = f"{icc_data['value']:.3f} [{icc_data['ci95_low']:.3f}, {icc_data['ci95_high']:.3f}]"
                    else:
                        icc_str = f"{icc_data['value']:.3f}"
                    interp_icc = interp["icc_3k"]
                else:
                    icc_str = "N/A"
                    interp_icc = "N/A"
                md_lines.append(
                    f"| {label} | {fk:.3f} | {ka:.3f} | {icc_str} "
                    f"| {interp['fleiss']} / {interp['krippendorff']} / {interp_icc} | {n_img} |"
                )

        md_lines.extend([
            "",
            "## NLG Metrics: Inter-Rater vs AI Pipeline",
            "",
            "| Metric | Inter-rater Mean +/- Std | Inter-rater CI95 "
            "| AI Mean +/- Std | AI CI95 |",
            "|---|---|---|---|---|",
        ])
        for metric in self.NLG_METRICS:
            ir = interrater_nlg["aggregate"][metric]
            ai = ai_comparison["aggregate"][metric]
            label = self.NLG_METRIC_LABELS[metric]
            md_lines.append(
                f"| {label} "
                f"| {ir['mean']:.3f} +/- {ir['std']:.3f} "
                f"| [{ir['ci95_low']:.3f}, {ir['ci95_high']:.3f}] "
                f"| {ai['mean']:.3f} +/- {ai['std']:.3f} "
                f"| [{ai['ci95_low']:.3f}, {ai['ci95_high']:.3f}] |"
            )

        mw = ai_comparison["mann_whitney"]
        md_lines.extend([
            "",
            "## Statistical Comparison (BERTScore F1)",
            "",
            f"- **Mann-Whitney U:** {mw['U_statistic']:.1f}",
            f"- **p-value:** {mw['p_value']:.3f}",
            f"- **Effect size (r):** {mw['effect_size_r']:.3f}",
            f"- **{mw['interpretation']}**",
            "",
            "---",
            "",
            "95% CIs: non-parametric bootstrap (1,000 iterations, "
            "seed=42, percentile method).",
            "Statistical comparison: Mann-Whitney U, two-sided (on BERTScore F1).",
            "Categorical agreement: Landis & Koch (1977).",
            "ICC(3,k): two-way mixed effects, average measurement, "
            "absolute agreement. Interpreted per Koo & Li (2016).",
        ])

        (out_path / "interrater_summary.md").write_text(
            "\n".join(md_lines), encoding="utf-8"
        )

        # Build summary CSV
        csv_lines = ["metric,field,value"]
        for field in ["asymmetry_level", "border_level", "heterogeneity_level"]:
            if field in categorical.get("fleiss_kappa", {}):
                csv_lines.append(
                    f"fleiss_kappa,{field},{categorical['fleiss_kappa'][field]:.4f}"
                )
                csv_lines.append(
                    f"krippendorff_alpha,{field},{categorical['krippendorff_alpha'][field]:.4f}"
                )
                icc_data = categorical["icc_3k"][field]
                if icc_data["value"] is not None:
                    csv_lines.append(f"icc_3k_value,{field},{icc_data['value']:.4f}")
                    csv_lines.append(f"icc_3k_ci_low,{field},{icc_data['ci95_low']:.4f}")
                    csv_lines.append(f"icc_3k_ci_high,{field},{icc_data['ci95_high']:.4f}")
                    csv_lines.append(
                        f"icc_3k_interpretation,{field},{categorical['interpretation'][field]['icc_3k']}"
                    )
                else:
                    csv_lines.append(f"icc_3k_value,{field},")
                    csv_lines.append(f"icc_3k_ci_low,{field},")
                    csv_lines.append(f"icc_3k_ci_high,{field},")
                    csv_lines.append(f"icc_3k_interpretation,{field},")
        for metric in self.NLG_METRICS:
            ir = interrater_nlg["aggregate"][metric]
            ai = ai_comparison["aggregate"][metric]
            csv_lines.append(f"interrater_{metric},mean,{ir['mean']:.4f}")
            csv_lines.append(f"interrater_{metric},std,{ir['std']:.4f}")
            csv_lines.append(f"ai_{metric},mean,{ai['mean']:.4f}")
            csv_lines.append(f"ai_{metric},std,{ai['std']:.4f}")
        csv_lines.append(
            f"mann_whitney,p_value,{mw['p_value']:.4f}"
        )
        csv_lines.append(
            f"mann_whitney,effect_size_r,{mw['effect_size_r']:.4f}"
        )

        (out_path / "interrater_summary.csv").write_text(
            "\n".join(csv_lines) + "\n", encoding="utf-8"
        )

        # Console output
        print("=" * 72)
        print(f"INTER-RATER AGREEMENT  (n={n_images}, raters={n_raters})")
        print("=" * 72)
        print("Categorical agreement (Fleiss kappa / Krippendorff alpha / ICC(3,k)):")
        for field, label in [
            ("asymmetry_level", "Asymmetry"),
            ("border_level", "Border"),
            ("heterogeneity_level", "Colour het."),
        ]:
            if field in categorical.get("fleiss_kappa", {}):
                fk = categorical["fleiss_kappa"][field]
                ka = categorical["krippendorff_alpha"][field]
                interp = categorical["interpretation"][field]
                icc_data = categorical["icc_3k"][field]
                if icc_data["value"] is not None:
                    if icc_data["ci95_low"] is not None and icc_data["ci95_high"] is not None:
                        icc_str = (
                            f"ICC={icc_data['value']:.3f} "
                            f"[{icc_data['ci95_low']:.3f}, {icc_data['ci95_high']:.3f}] "
                            f"[{interp['icc_3k']}]"
                        )
                    else:
                        icc_str = f"ICC={icc_data['value']:.3f} [{interp['icc_3k']}]"
                else:
                    icc_str = "ICC=N/A"
                print(
                    f"  {label + ':':<16s}"
                    f"k={fk:.3f} [{interp['fleiss']}] / "
                    f"a={ka:.3f} [{interp['krippendorff']}] / "
                    f"{icc_str}"
                )
        print("-" * 72)
        print(f"{'Metric':<22s} {'Inter-rater':>14s} {'AI vs raters':>14s}")
        print(f"{'':<22s} {'Mean +/- Std':>14s} {'Mean +/- Std':>14s}")
        print("-" * 72)
        for metric in self.NLG_METRICS:
            ir = interrater_nlg["aggregate"][metric]
            ai = ai_comparison["aggregate"][metric]
            label = self.NLG_METRIC_LABELS[metric]
            print(
                f"  {label:<20s} {ir['mean']:.3f}+/-{ir['std']:.3f}"
                f"   {ai['mean']:.3f}+/-{ai['std']:.3f}"
            )
        print("-" * 72)
        print(
            f"Mann-Whitney U (BERTScore F1): p={mw['p_value']:.3f}  "
            f"r={mw['effect_size_r']:.3f}"
        )
        print(f"  {mw['interpretation']}")
        print("=" * 72)

        return {
            "categorical": categorical,
            "interrater_nlg": interrater_nlg,
            "ai_comparison": ai_comparison,
        }
