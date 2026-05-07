"""
Deterministic ABC section text generator for dermatological reports.

Converts quantitative features from DermoscopyFeatureExtractor into
Italian clinical text for the A, B, C sections of the report.
No randomness — same input always produces identical output.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_ASYMMETRY_LABEL_MAP: dict[str, str] = {
    "simmetrica": "simmetria strutturale e di pigmentazione, tipica di un pattern benigno",
    "lievemente asimmetrica": "lieve asimmetria nella distribuzione del reticolo pigmentario",
    "asimmetrica": "asimmetria dermoscopica evidente, con possibile network atipico",
    "marcatamente asimmetrica": "marcata asimmetria dermoscopica, con distribuzione caotica della pigmentazione e disorganizzazione delle strutture",
}

_BORDER_TEXTS: dict[str, str] = {
    "regolari": "I margini appaiono regolari e ben definiti, con profilo liscio e assenza di incisure periferiche.",
    "lievemente irregolari": "I margini presentano lieve irregolarità, con rare strie radiali e minime asimmetrie periferiche.",
    "irregolari": "I margini risultano irregolari, con morfologia compatibile con incisure periferiche e possibili pseudopodi.",
    "molto irregolari": "I margini sono marcatamente irregolari, con aspetto frastagliato, reticolo pigmentario interrotto e possibili pseudopodi atipici.",
}

_HETEROGENEITY_TEXTS: dict[str, str] = {
    "omogenea": "La lesione presenta pigmentazione omogenea e regolare",
    "lievemente eterogenea": "La lesione presenta lieve eterogeneità cromatica",
    "eterogenea": "La lesione mostra eterogeneità cromatica con irregolarità di pattern",
    "marcatamente eterogenea": "La lesione presenta marcata eterogeneità cromatica, sospetta per iperpigmentazione irregolare o aree di regressione",
}


class ABCTextGenerator:
    """Converts quantitative dermoscopy features into Italian ABC report sections.

    Stateless — no constructor parameters required.  All logic is
    deterministic: identical ``features`` dicts always produce identical text.

    Example:
        >>> gen = ABCTextGenerator()
        >>> sections = gen.generate_abc_sections(features, classification_result)
        >>> print(sections["asymmetry"])
    """

    def _generate_asymmetry(self, asymmetry: dict) -> str:
        """Generate Section A text from asymmetry feature dict.

        Args:
            asymmetry: Output of DermoscopyFeatureExtractor.compute_asymmetry().

        Returns:
            Italian clinical sentence describing lesion asymmetry.
        """
        ax: float = float(asymmetry.get("asymmetry_x", 0.0))
        ay: float = float(asymmetry.get("asymmetry_y", 0.0))
        label: str = asymmetry.get("asymmetry_label", "simmetrica")
        mapped = _ASYMMETRY_LABEL_MAP.get(label, label)

        if abs(ax - ay) > 0.10:
            return f"La lesione presenta {mapped} lungo l'asse orizzontale e moderata lungo quello verticale."
        return f"La lesione presenta {mapped} lungo entrambi gli assi."

    def _generate_borders(self, border: dict) -> str:
        """Generate Section B text from border feature dict.

        Args:
            border: Output of DermoscopyFeatureExtractor.compute_border_irregularity().

        Returns:
            Italian clinical sentence describing border regularity.
        """
        circ: float = float(border.get("circularity", 1.0))
        label: str = border.get("border_label", "regolari")

        text = _BORDER_TEXTS.get(
            label,
            "I margini presentano irregolarità non meglio classificata.",
        )

        if circ < 0.60:
            text += " Profilo geometrico marcatamente asimmetrico."
        elif circ < 0.75:
            text += " Profilo geometrico ovalare con lieve asimmetria."

        return text

    _NUM_WORDS: dict[int, str] = {
        1: "un", 2: "due", 3: "tre", 4: "quattro",
        5: "cinque", 6: "sei",
    }

    def _generate_colour(self, colour: dict) -> str:
        """Generate Section C text from colour feature dict.

        Args:
            colour: Output of DermoscopyFeatureExtractor.compute_colour_features().

        Returns:
            Italian clinical sentence describing colour heterogeneity.
        """
        het_label: str = colour.get("heterogeneity_label", "omogenea")
        colour_pct: dict[str, float] = colour.get("colour_percentages", {})
        n_dom: int = int(colour.get("n_dominant_colours", 1))

        het_text = _HETEROGENEITY_TEXTS.get(
            het_label,
            "La lesione presenta eterogeneità cromatica",
        )

        # Colour composition — only colours > 5%, sorted descending, no percentages
        sorted_colours = sorted(
            [(name, pct) for name, pct in colour_pct.items() if pct > 5.0],
            key=lambda kv: -kv[1],
        )
        n_colors = len(sorted_colours)
        n_word = self._NUM_WORDS.get(n_colors, str(n_colors))

        if sorted_colours:
            names = [name for name, _ in sorted_colours]
            if len(names) == 1:
                colour_list = names[0]
            elif len(names) == 2:
                colour_list = f"{names[0]} e {names[1]}"
            else:
                colour_list = ", ".join(names[:-1]) + f" e {names[-1]}"
            text = f"{het_text}. Presenti {n_word} colori: {colour_list}."
        else:
            text = f"{het_text}."

        if n_dom >= 3 and het_label in ("eterogenea", "marcatamente eterogenea"):
            text += (
                f" La presenza di {n_word} tonalità distinte aumenta il "
                "profilo di sospetto dermoscopico."
            )

        if colour_pct.get("blu/grigio-bluastro", 0.0) > 5.0:
            text += (
                " Si segnala la presenza di aree blu-grigie, "
                "compatibili con velo blu-biancastro o depositi di melanina "
                "in derma profondo."
            )

        return text

    def generate_abc_sections(
        self,
        features: dict,
        classification_result: dict,  # noqa: ARG002  — reserved for future label-aware logic
    ) -> dict[str, str]:
        """Generate Italian A, B, C report sections from quantitative features.

        Args:
            features: Output of DermoscopyFeatureExtractor.extract_all().
                Expected sub-dicts: ``asymmetry``, ``border``, ``colour``.
                Missing sub-dicts fall back to safe default text.
            classification_result: Output of MelanomaClassifier.predict().
                Currently unused; reserved for future label-aware refinements.

        Returns:
            dict with keys:

            - ``asymmetry`` (str): Italian text for Section A.
            - ``borders`` (str): Italian text for Section B.
            - ``color`` (str): Italian text for Section C.

        Raises:
            ValueError: If ``features`` is None.
        """
        if features is None:
            raise ValueError("features must not be None")

        asymmetry_data: Optional[dict] = features.get("asymmetry")
        border_data: Optional[dict] = features.get("border")
        colour_data: Optional[dict] = features.get("colour")

        if asymmetry_data:
            asymmetry_text = self._generate_asymmetry(asymmetry_data)
        else:
            logger.warning("ABCTextGenerator: asymmetry features missing — using fallback text")
            asymmetry_text = "Dati di asimmetria non disponibili per questa immagine."

        if border_data:
            borders_text = self._generate_borders(border_data)
        else:
            logger.warning("ABCTextGenerator: border features missing — using fallback text")
            borders_text = "Dati sui bordi non disponibili per questa immagine."

        if colour_data:
            colour_text = self._generate_colour(colour_data)
        else:
            logger.warning("ABCTextGenerator: colour features missing — using fallback text")
            colour_text = "Dati cromatici non disponibili per questa immagine."

        return {
            "asymmetry": asymmetry_text,
            "borders": borders_text,
            "color": colour_text,
        }


# ─── Standalone test entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Standalone test for ABCTextGenerator."
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Path to a JSON file containing DermoscopyFeatureExtractor output.",
    )
    args = parser.parse_args()

    try:
        with open(args.features, "r", encoding="utf-8") as fh:
            features_data = json.load(fh)
    except FileNotFoundError:
        print(f"ERROR: features file not found: {args.features}", file=sys.stderr)
        sys.exit(1)

    gen = ABCTextGenerator()
    sections = gen.generate_abc_sections(
        features=features_data,
        classification_result={"class": "nevus", "confidence": 0.8},
    )

    print("=== A - Asimmetria ===")
    print(sections["asymmetry"])
    print()
    print("=== B - Bordi ===")
    print(sections["borders"])
    print()
    print("=== C - Colore ===")
    print(sections["color"])
