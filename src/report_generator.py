"""
ABCDE Dermatological Report Generator using BioMistral-7B on RTX 3080.
"""
from __future__ import annotations

import gc
import logging
import re
import time
from pathlib import Path
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)

_TEMPLATE_FALLBACK = """\
## REFERTO DERMATOLOGICO - ANALISI ABC

**A - Asimmetria:**
{asymmetry}

**B - Bordi:**
{borders}

**C - Colore:**
{color}

**CONCLUSIONE:**
{conclusion}

NOTA: Questo referto è generato da un sistema AI di supporto e NON sostituisce
il giudizio clinico di un dermatologo.
"""


class ABCDEReportGenerator:
    """Generates ABCDE dermatological reports using BioMistral-7B.

    Loads BioMistral-7B with 8-bit quantization (primary) or 4-bit (fallback)
    on RTX 3080. Integrates RAG context from PubMed/DermNet into the prompt.
    Supports automatic offload after inference to free VRAM for BERTScore.

    Args:
        config: Configuration dict from config.yaml (llm section).
        template_path: Path to prompt template in config.yaml (llm section).
    """

    def __init__(
        self,
        config: dict,
        template_path: Union[str, Path], # = "config/abcde_prompt_template.txt"
    ) -> None:
        self.config = config
        self.template_path = Path(template_path)
        self.model = None
        self.tokenizer = None
        self._quantization_used: str = ""
        self._template_text = self._load_template()

    def _load_template(self) -> str:
        """Load the ABCDE prompt template from file.

        Returns:
            Template string with {classification}, {confidence}, etc. placeholders.
        """
        if self.template_path.exists():
            with open(self.template_path, "r", encoding="utf-8") as f:
                return f.read()
        logger.warning(
            f"Template not found at {self.template_path}, using built-in fallback"
        )
        return (
            "Sei un dermatologo esperto. Analizza questa lesione cutanea:\n"
            "CNN: {classification} ({confidence:.1%})\n"
            "Dati clinici: {prompt_block}\n"
            "RAG: {rag_context}\n\n"
            "Genera referto ABC strutturato con sezioni A,B,C e CONCLUSIONE.\n"
            "Riferimenti: {rag_references}\n"
            "NOTA: Sistema AI, non sostituisce il dermatologo."
        )

    def _get_vram_free_gb(self) -> float:
        """Get free VRAM in GB.

        Returns:
            Free VRAM in GB, or float('inf') if CUDA unavailable.
        """
        if not torch.cuda.is_available():
            return float("inf")
        free_bytes, _ = torch.cuda.mem_get_info(0)
        return free_bytes / 1e9

    def load_model(self) -> str:
        """Load BioMistral-7B with 8-bit quantization (4-bit fallback on OOM).

        Returns:
            Quantization mode used: "8bit", "4bit", or "template".

        Notes:
            - 8-bit primary: ~7GB VRAM on RTX 3080
            - 4-bit fallback: ~4GB VRAM
            - Template fallback: <3GB VRAM available
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        vram_free = self._get_vram_free_gb()
        logger.info(f"VRAM free before LLM load: {vram_free:.1f}GB")

        if vram_free < 3.0:
            logger.warning(
                f"VRAM too low ({vram_free:.1f}GB < 3GB) — using template fallback"
            )
            self._quantization_used = "template"
            return "template"

        model_name = self.config.get("model_name", "BioMistral/BioMistral-7B")
        device_map = self.config.get("device_map", "cuda:0")

        import os as _os

        # TRANSFORMERS_OFFLINE suppresses the auto_conversion background thread
        # (spawned even with use_safetensors=False in this transformers version).
        _was_offline = _os.environ.get("TRANSFORMERS_OFFLINE")
        _os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Attempt 8-bit (primary for Ampere)
        try:
            logger.info(f"Loading {model_name} in 8-bit (primary mode)...")
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map=device_map,
                torch_dtype=torch.float16,
                use_safetensors=False,
                local_files_only=True,
            )
            self.model.eval()
            self._quantization_used = "8bit"
            logger.info(f"BioMistral-7B loaded in 8-bit | VRAM used: {self._get_vram_used_gb():.1f}GB")
            return "8bit"

        except ValueError as e:
            logger.error(f"8-bit load failed with ValueError ({e}) — using template fallback")
            self._offload_model()
            self._quantization_used = "template"
            return "template"
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.warning(f"8-bit OOM ({e}), retrying with 4-bit...")
            self._offload_model()
        finally:
            # Restore TRANSFORMERS_OFFLINE to its previous state
            if _was_offline is None:
                _os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                _os.environ["TRANSFORMERS_OFFLINE"] = _was_offline

        # Fallback: 4-bit (NF4)
        _os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            logger.info(f"Loading {model_name} in 4-bit (NF4 fallback)...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map=device_map,
                torch_dtype=torch.float16,
                use_safetensors=False,
                local_files_only=True,
            )
            self.model.eval()
            self._quantization_used = "4bit"
            logger.info(f"BioMistral-7B loaded in 4-bit | VRAM used: {self._get_vram_used_gb():.1f}GB")
            return "4bit"

        except ValueError as e:
            logger.error(f"4-bit load failed with ValueError ({e}) — using template fallback")
            self._offload_model()
            self._quantization_used = "template"
            return "template"
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.error(f"4-bit also OOM ({e}) — using template fallback")
            self._offload_model()
            self._quantization_used = "template"
            return "template"
        finally:
            if _was_offline is None:
                _os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                _os.environ["TRANSFORMERS_OFFLINE"] = _was_offline

    def _get_vram_used_gb(self) -> float:
        """Get used VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return (total_bytes - free_bytes) / 1e9

    def _offload_model(self) -> None:
        """Delete model and tokenizer, free CUDA memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()

    def _build_prompt(
        self,
        classification_result: dict,
        gradcam_result: dict,
        rag_context: Optional[str] = None,
        rag_references: Optional[str] = None,
        **extra_fields: str,
    ) -> str:
        """Build the final prompt from template and clinical data.

        Supports both the hybrid template (which uses ``{section_asymmetry}``,
        ``{section_borders}``, ``{section_color}``, ``{gradcam_focus_description}``)
        and the legacy template (which uses ``{prompt_block}``).  Any key present
        in the template but absent from the call raises ``KeyError``; extra keys
        are silently ignored by Python's ``str.format``.

        Args:
            classification_result: Output from MelanomaClassifier.predict().
            gradcam_result: Output from GradCAMExplainer.generate().
            rag_context: Formatted RAG context text.
            rag_references: Formatted RAG references text.
            **extra_fields: Additional placeholder values forwarded to the
                template (e.g. ``section_asymmetry``, ``section_borders``,
                ``section_color``, ``gradcam_focus_description``).

        Returns:
            Complete prompt string ready for LLM inference.
        """

        base_focus = gradcam_result.get("focus_regions", "zona centrale")
        format_kwargs: dict = {
            "classification": classification_result["class"],
            "confidence": classification_result["confidence"],
            # Support both old ({prompt_block}) and new ({gradcam_focus_description}) templates
            "prompt_block": base_focus,
            "gradcam_focus_description": base_focus,
            "rag_context": rag_context or "No RAG context available.",
            "rag_references": rag_references or "No references available.",
            # Defaults for hybrid-template placeholders — overridden via extra_fields in hybrid mode
            "section_asymmetry": "Analisi quantitativa non disponibile.",
            "section_borders": "Analisi quantitativa non disponibile.",
            "section_color": "Analisi quantitativa non disponibile.",
            # Clinical vocabulary for strong template
            "clinical_vocabulary": (
                "asimmetria, pseudopodi, reticolo pigmentario, globuli, "
                "velo blu-biancastro, regressione, pattern, dermoscopia, "
                "pigmentazione, eterogeneità, incisure, strie radiali, "
                "area struttureless, network atipico, punti, iperpigmentazione"
            ),
        }
        format_kwargs.update(extra_fields)
        return self._template_text.format(**format_kwargs)

    def _generate_with_llm(self, prompt: str, temperature: float) -> str:
        """Run LLM inference on the prompt.

        Args:
            prompt: Complete prompt string.
            temperature: Sampling temperature.

        Returns:
            Generated text string (prompt stripped).
        """
        self.tokenizer.truncation_side = "left"  # preserve end of prompt (instruction + conclusion marker)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 1024),
                min_new_tokens=self.config.get("min_new_tokens", 80),
                temperature=temperature,
                top_p=self.config.get("top_p", 0.9),
                repetition_penalty=self.config.get("repetition_penalty", 1.0),
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens (strip prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _parse_sections(self, report_text: str) -> dict[str, str]:
        """Extract ABC sections from generated report text using regex.

        Args:
            report_text: Raw LLM output text.

        Returns:
            dict with keys: asymmetry, borders, color, conclusion.
        """
        patterns = {
            "asymmetry": r"\*\*A\s*[-–]\s*Asimmetri[ae]:\*\*\s*(.*?)(?=\*\*[B-Z]|\Z)",
            "borders": r"\*\*B\s*[-–]\s*Bordi:\*\*\s*(.*?)(?=\*\*[C-Z]|\Z)",
            "color": r"\*\*C\s*[-–]\s*Colore:\*\*\s*(.*?)(?=\*\*CONCLUSIONE|\Z)",
            "conclusion": r"\*\*CONCLUSIONE:\*\*\s*(.*?)(?=RIFERIMENTI|NOTA|\Z)",
        }

        sections = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, report_text, re.DOTALL | re.IGNORECASE)
            sections[key] = match.group(1).strip() if match else ""

        return sections

    def _extract_risk_level(self, report_text: str, classification: str) -> str:
        """Extract risk level from CONCLUSIONE section, enforcing CNN alignment.

        If the extracted level contradicts the CNN classification (e.g. melanoma
        with "basso"), the classification-aligned level is returned instead.

        Args:
            report_text: Generated report text.
            classification: CNN classification ("melanoma" or "nevus").

        Returns:
            Risk level string: "alto", "moderato", or "basso".
        """
        conclusion_match = re.search(
            r"\*\*CONCLUSIONE:\*\*\s*(.*?)(?=RIFERIMENTI|NOTA|\Z)",
            report_text,
            re.DOTALL | re.IGNORECASE,
        )
        search_text = (
            conclusion_match.group(1).lower() if conclusion_match else report_text.lower()
        )

        if "alto" in search_text:
            extracted = "alto"
        elif "moderato" in search_text:
            extracted = "moderato"
        elif "basso" in search_text:
            extracted = "basso"
        else:
            return "alto" if classification == "melanoma" else "basso"

        # Enforce alignment: melanoma must not be "basso"
        if classification == "melanoma" and extracted == "basso":
            logger.warning(
                f"Risk level '{extracted}' contradicts classification "
                f"'{classification}' — forcing 'alto'"
            )
            return "alto"
        if classification == "nevus" and extracted == "alto":
            logger.warning(
                f"Risk level '{extracted}' contradicts classification "
                f"'{classification}' — forcing 'basso'"
            )
            return "basso"

        return extracted

    _ACTION_TERMS = frozenset({
        "biopsia", "escissione", "monitoraggio", "follow-up",
        "dermatoscopia", "controllo", "rivalutazione", "approfondimento",
    })

    # Patterns that indicate the LLM is parroting prompt instructions
    _PARROT_PATTERNS: tuple[re.Pattern, ...] = (
        re.compile(
            r"azioni\s+come\s+[\"\"«]?biopsia[\"\"»]?\s*,\s*[\"\"«]?escissione[\"\"»]?"
            r"\s*o\s*[\"\"«]?valutazione\s+urgente[\"\"»]?\s+sono\s+necessari[ae]",
            re.IGNORECASE,
        ),
        re.compile(
            r"azioni\s+conservative\s+come\s+[\"\"«]?monitoraggio[\"\"»]?",
            re.IGNORECASE,
        ),
        re.compile(r"il livello di rischio\s+DEVE\s+essere", re.IGNORECASE),
    )

    _SPATIAL_WORDS = frozenset({
        "centrale", "superiore", "inferiore", "sinistro", "destro",
        "periferica", "periferia", "bordo", "margine", "centro",
        "quadrante", "regione", "zona", "area",
    })

    def _postprocess_conclusion(
        self,
        conclusion: str,
        abc_sections: dict[str, str],
        classification: str,
        confidence: float,
        gradcam_focus: str = "",
    ) -> str:
        """Post-process LLM conclusion: deduplicate, sanitize, align risk, inject actions.

        Args:
            conclusion: Raw LLM-generated conclusion text.
            abc_sections: Dict with asymmetry/borders/color section text.
            classification: CNN class ("melanoma" or "nevus").
            confidence: CNN confidence score.
            gradcam_focus: GradCAM focus region description for spatial injection.

        Returns:
            Cleaned conclusion string.
        """
        is_melanoma = classification == "melanoma"

        # 1. Deduplicate: remove sentences copied from ABC sections
        abc_words = set(
            " ".join(abc_sections.values()).lower().split()
        )
        sentences = re.split(r"(?<=[.!?])\s+", conclusion)
        unique = []
        for sent in sentences:
            words = sent.strip().lower().split()
            if len(words) < 5:
                unique.append(sent)
                continue
            overlap = sum(1 for w in words if w in abc_words)
            if overlap / len(words) < 0.75:
                unique.append(sent)
        conclusion = " ".join(unique).strip()

        # 2. Remove sentences with hallucinated percentages
        conf_pct = f"{confidence * 100:.1f}"  # e.g. "87.7"
        conf_pct_int = str(round(confidence * 100))  # e.g. "88"
        allowed_pcts = {conf_pct, conf_pct_int}
        sentences = re.split(r"(?<=[.!?])\s+", conclusion)
        cleaned = []
        for sent in sentences:
            pcts_in_sent = re.findall(r"(\d+(?:\.\d+)?)\s*%", sent)
            if pcts_in_sent and not any(p in allowed_pcts for p in pcts_in_sent):
                logger.warning(f"Removed hallucinated percentage: {sent!r}")
                continue
            cleaned.append(sent)
        conclusion = " ".join(cleaned).strip()

        # 3. Remove sentences that parrot prompt instructions
        sentences = re.split(r"(?<=[.!?])\s+", conclusion)
        filtered = []
        for sent in sentences:
            if any(pat.search(sent) for pat in self._PARROT_PATTERNS):
                logger.warning(f"Removed parroted instruction: {sent!r}")
                continue
            filtered.append(sent)
        conclusion = " ".join(filtered).strip()

        # 4. Remove sentences that contradict the CNN classification
        contradiction_words = (
            ["benigno", "benigna", "nevo benigno", "lesione benigna"]
            if is_melanoma
            else ["maligno", "maligna", "malignità", "melanoma primitivo"]
        )
        sentences = re.split(r"(?<=[.!?])\s+", conclusion)
        consistent = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(w in sent_lower for w in contradiction_words):
                logger.warning(f"Removed contradictory sentence: {sent!r}")
                continue
            consistent.append(sent)
        conclusion = " ".join(consistent).strip()

        # 5. Force risk level alignment with CNN classification
        #    Match both "rischio basso" and inverted "basso rischio" patterns
        if is_melanoma:
            conclusion = re.sub(
                r"\b(rischio\s*[:\s]*\s*(?:basso|moderato)"
                r"|(?:basso|moderato)\s+rischio)\b",
                "rischio alto",
                conclusion,
                flags=re.IGNORECASE,
            )
        else:
            conclusion = re.sub(
                r"\b(rischio\s*[:\s]*\s*alto|alto\s+rischio)\b",
                "rischio basso",
                conclusion,
                flags=re.IGNORECASE,
            )

        # 6. Inject action terms if none present
        conclusion_lower = conclusion.lower()
        has_action = any(t in conclusion_lower for t in self._ACTION_TERMS)
        if not has_action:
            if is_melanoma:
                conclusion += (
                    " Si raccomanda biopsia escissionale e "
                    "valutazione dermatologica urgente."
                )
            else:
                conclusion += (
                    " Si raccomanda monitoraggio clinico con "
                    "dermatoscopia di controllo."
                )

        # 7. Ensure risk level is explicitly stated
        if "rischio" not in conclusion.lower():
            risk = "alto" if is_melanoma else "basso"
            conclusion = f"Livello di rischio: {risk}. {conclusion}"

        # 8. Inject GradCAM spatial reference if none present
        if gradcam_focus:
            has_spatial = any(
                w in conclusion.lower() for w in self._SPATIAL_WORDS
            )
            if not has_spatial:
                conclusion += (
                    f" L'attivazione GradCAM++ risulta concentrata su {gradcam_focus}."
                )

        return conclusion.strip()

    def _generate_template_report(
        self,
        classification_result: dict,
        gradcam_result: dict,
        rag_context: Optional[str],
    ) -> str:
        """Generate a structured template report without LLM (VRAM fallback).

        Args:
            classification_result: Classifier output.
            gradcam_result: GradCAM output.
            rag_context: RAG context text (optional).

        Returns:
            Formatted template report string.
        """
        cls = classification_result["class"]
        conf = classification_result["confidence"]
        focus = gradcam_result.get("focus_regions", "zona centrale")
        is_melanoma = cls == "melanoma"

        return _TEMPLATE_FALLBACK.format(
            asymmetry=(
                "La lesione presenta marcata asimmetria lungo entrambi gli assi."
                if is_melanoma else
                "La lesione appare relativamente simmetrica."
            ),
            borders=(
                "I margini risultano irregolari, con possibili pseudopodi e incisure."
                if is_melanoma else
                "I bordi appaiono ben definiti e regolari."
            ),
            color=(
                "Eterogeneità cromatica con tonalità di marrone, nero e possibile area "
                "biancastra di regressione."
                if is_melanoma else
                "Colorazione uniforme, marrone chiaro, senza eterogeneità significativa."
            ),
            conclusion=(
                f"RISCHIO ALTO. Classificazione CNN: {cls} (confidenza: {conf:.1%}). "
                f"Attivazione GradCAM++ concentrata su {focus}. "
                "Escissione e biopsia raccomandate con urgenza. "
                "Consulenza dermatologica specialistica necessaria."
                if is_melanoma else
                f"RISCHIO BASSO. Classificazione CNN: {cls} (confidenza: {conf:.1%}). "
                f"Attivazione GradCAM++ su {focus}. "
                "Follow-up dermatologico periodico consigliato."
            ),
        )

    def generate_report(
        self,
        classification_result: dict,
        gradcam_result: dict,
        image_features: Optional[dict] = None,
        rag_context: Optional[str] = None,
        rag_references: Optional[str] = None,
        rag_sources: Optional[list[str]] = None,
    ) -> dict:
        """Generate ABCDE report using BioMistral-7B or template fallback.

        Manages quantization fallback (8bit → 4bit → template).
        Retries with temperature +0.2 if output is malformed (max 2 retries).
        Offloads model after inference if configured.

        Args:
            classification_result: Output from MelanomaClassifier.predict().
            gradcam_result: Output from GradCAMExplainer.generate().
            rag_context: Formatted RAG context (optional).
            rag_references: Formatted RAG references (optional).
            rag_sources: List of RAG source strings (optional).

        Returns:
            dict with keys:
                - report_text (str): Full report Markdown text.
                - sections (dict): Parsed ABCDE + conclusion sections.
                - risk_level (str): "alto", "moderato", or "basso".
                - rag_sources_used (list[str]): RAG sources included.
                - metadata (dict): Model info, quantization, VRAM, timing, tokens.
        """
        start_time = time.time()
        vram_before = self._get_vram_used_gb()

        # Load model if not already loaded
        if self.model is None:
            quant_mode = self.load_model()
        else:
            quant_mode = self._quantization_used

        # Template fallback (no LLM)
        if quant_mode == "template":
            report_text = self._generate_template_report(
                classification_result, gradcam_result, rag_context
            )
            sections = self._parse_sections(report_text)
            elapsed = time.time() - start_time
            return {
                "report_text": report_text,
                "sections": sections,
                "risk_level": self._extract_risk_level(
                    report_text, classification_result["class"]
                ),
                "rag_sources_used": rag_sources or [],
                "metadata": {
                    "model_used": "template-fallback",
                    "quantization_used": "none",
                    "vram_used_gb": 0.0,
                    "generation_time_s": elapsed,
                    "tokens_generated": 0,
                },
            }

        # ── Hybrid ABC generation ────────────────────────────────────────────
        from src.abc_text_generator import ABCTextGenerator  # lazy import

        abc_sections: dict[str, str] | None = None
        mode = self.config.get("abc_generation_mode", "hybrid")
        use_hybrid = False
        if mode == "hybrid" and image_features is not None:
            try:
                abc_gen = ABCTextGenerator()
                abc_sections = abc_gen.generate_abc_sections(
                    features=image_features,
                    classification_result=classification_result,
                )
                use_hybrid = True
            except Exception as _e:
                logger.warning(
                    "Feature extractor unavailable — falling back to LLM-only "
                    f"ABC generation. Reports may be class-stereotyped. ({_e})"
                )
        elif mode != "hybrid":
            logger.info("abc_generation_mode=llm_only — using LLM-only ABC generation")
        else:
            logger.warning(
                "Feature extractor unavailable — falling back to LLM-only "
                "ABC generation. Reports may be class-stereotyped."
            )

        # ── Build LLM prompt ─────────────────────────────────────────────────
        temperature = self.config.get("temperature", 0.3)
        if use_hybrid:
            gradcam_focus = (
                (image_features.get("gradcam_grid") or {}).get("focus_description")
                or gradcam_result.get("focus_regions", "zona centrale")
            )
            prompt = self._build_prompt(
                classification_result,
                gradcam_result,
                rag_context,
                rag_references,
                gradcam_focus_description=gradcam_focus,
                section_asymmetry=abc_sections["asymmetry"],
                section_borders=abc_sections["borders"],
                section_color=abc_sections["color"]
            )
        else:
            _gradcam_for_prompt = (
                {**gradcam_result, "focus_regions": image_features["prompt_block"]}
                if image_features and image_features.get("prompt_block")
                else gradcam_result
            )
            prompt = self._build_prompt(
                classification_result, _gradcam_for_prompt, rag_context, rag_references,
            )

        # ── LLM generation with retry ────────────────────────────────────────
        report_text = ""
        parsed_conclusion = ""
        for attempt in range(3):
            try:
                t = temperature + 0.2 * attempt
                logger.info(
                    f"LLM generation attempt {attempt + 1}/3 | temperature={t:.1f}"
                )
                llm_output = self._generate_with_llm(prompt, t)

                if use_hybrid:
                    _HEADER_RE = (
                        r"(?:\*\*CONCLUSIONE:\*\*|CONCLUSIONE IN ITALIANO:)"
                    )
                    m = re.search(
                        _HEADER_RE + r"\s*(.*?)(?=RIFERIMENTI:|NOTA:|---|\Z)",
                        llm_output,
                        re.DOTALL | re.IGNORECASE,
                    )
                    if m and m.group(1).strip():
                        parsed_conclusion = m.group(1).strip()
                    else:
                        # Strip echoed header and any trailing boilerplate
                        raw = re.sub(
                            r"^\s*" + _HEADER_RE + r"\s*",
                            "",
                            llm_output,
                            flags=re.IGNORECASE,
                        )
                        raw = re.sub(
                            r"\s*(NOTA:|RIFERIMENTI:|---).*$",
                            "",
                            raw,
                            flags=re.DOTALL | re.IGNORECASE,
                        )
                        parsed_conclusion = raw.strip()
                    if parsed_conclusion:
                        break
                    logger.warning(
                        f"Attempt {attempt + 1}: CONCLUSIONE not parsed, retrying..."
                    )
                else:
                    report_text = llm_output
                    sections = self._parse_sections(report_text)
                    filled = sum(1 for v in sections.values() if v)
                    if filled >= 3:
                        break
                    logger.warning(
                        f"Attempt {attempt + 1}: only {filled}/4 sections parsed, retrying..."
                    )
            except Exception as e:
                logger.error(f"LLM generation error (attempt {attempt + 1}): {e}")
                if attempt == 2 and not use_hybrid:
                    report_text = self._generate_template_report(
                        classification_result, gradcam_result, rag_context
                    )

        # ── Deterministic conclusion fallback (all LLM attempts failed) ────────
        if use_hybrid and not parsed_conclusion:
            cls = classification_result["class"]
            conf = classification_result["confidence"]
            logger.warning(
                "All LLM attempts returned empty conclusion — using deterministic fallback"
            )
            if cls == "melanoma":
                parsed_conclusion = (
                    f"La lesione è classificata come melanoma (confidenza {conf:.1%}), "
                    "con caratteristiche dermoscopiche di sospetto. "
                    "Livello di rischio: alto. "
                    "Si raccomanda valutazione dermatologica urgente e biopsia escissionale."
                )
            else:
                parsed_conclusion = (
                    f"La lesione è classificata come nevo (confidenza {conf:.1%}), "
                    "senza criteri di elevato sospetto all'analisi dermoscopica. "
                    "Livello di rischio: basso. "
                    "Si raccomanda follow-up clinico periodico con dermoscopia."
                )

        # ── Post-process conclusion (dedup, risk alignment, actions) ────────
        if use_hybrid and parsed_conclusion:
            parsed_conclusion = self._postprocess_conclusion(
                conclusion=parsed_conclusion,
                abc_sections=abc_sections,
                classification=classification_result["class"],
                confidence=classification_result["confidence"],
                gradcam_focus=gradcam_focus,
            )

        # ── Assemble final report ────────────────────────────────────────────
        _DISCLAIMER = (
            "NOTA: Questo referto è generato da un sistema AI di supporto "
            "e NON sostituisce\nil giudizio clinico di un dermatologo."
        )
        if use_hybrid:
            report_text = (
                "## REFERTO DERMATOLOGICO - ANALISI ABC\n\n"
                f"**A - Asimmetria:**\n{abc_sections['asymmetry']}\n\n"
                f"**B - Bordi:**\n{abc_sections['borders']}\n\n"
                f"**C - Colore:**\n{abc_sections['color']}\n\n"
                f"**CONCLUSIONE:**\n{parsed_conclusion}\n\n"
                + _DISCLAIMER
            )
            sections = {
                "asymmetry": abc_sections["asymmetry"],
                "borders": abc_sections["borders"],
                "color": abc_sections["color"],
                "conclusion": parsed_conclusion,
            }
        else:
            sections = self._parse_sections(report_text)
        elapsed = time.time() - start_time
        vram_after = self._get_vram_used_gb()

        # Count approximate tokens generated
        tokens_generated = len(report_text.split()) * 1.3 if report_text else 0

        result = {
            "report_text": report_text,
            "sections": sections,
            "risk_level": self._extract_risk_level(
                report_text, classification_result["class"]
            ),
            "rag_sources_used": rag_sources or [],
            "metadata": {
                "model_used": self.config.get("model_name", "BioMistral/BioMistral-7B"),
                "quantization_used": quant_mode,
                "vram_used_gb": vram_after,
                "generation_time_s": elapsed,
                "tokens_generated": int(tokens_generated),
            },
        }

        # Offload after inference if configured
        if self.config.get("offload_after_inference", True):
            logger.info("Offloading BioMistral from GPU...")
            self._offload_model()
            logger.info(
                f"LLM offloaded | VRAM freed: "
                f"{vram_after - self._get_vram_used_gb():.1f}GB"
            )

        return result

    def offload(self) -> None:
        """Explicitly offload model and free CUDA memory.

        Call this manually if offload_after_inference is False.
        """
        self._offload_model()
        logger.info("ABCDEReportGenerator model offloaded")
