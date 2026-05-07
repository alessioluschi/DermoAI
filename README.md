# DermoAI Pipeline

End-to-end AI pipeline for skin lesion classification (nevus vs melanoma) with automatic ABC report generation and multi-dimensional evaluation.

**Target hardware:** NVIDIA RTX 3080 (10GB VRAM, Ampere, CUDA 8.6)

---

## Components

| Component | Description |
|---|---|
| EfficientNet-B0 | Nevus/melanoma classification (fp16, CUDA) |
| GradCAM++ | Activation maps for visual interpretability |
| BioMistral-7B | ABC report generation (8-bit/4-bit quantized) |
| RAG (PubMed + DermNet) | Retrieval from biomedical literature (CPU) |
| QualitativeEvaluator | 6-criterion rubric scoring (CPU, always available) |
| ReportEvaluator | ROUGE/BLEU/BERTScore (requires ground truth) |
| CrossModelEvaluator | Multi-LLM comparison without ground truth |

---

## VRAM Budget per Component

| Component | VRAM | Notes |
|---|---|---|
| EfficientNet-B0 (fp16) | ~300MB | Kept in memory during run |
| GradCAM++ | ~500MB | Peak during backward pass |
| S-PubMedBert (RAG) | 0MB VRAM | Runs on CPU |
| BioMistral-7B (8-bit) | ~7.0GB | Primary mode for RTX 3080 |
| BioMistral-7B (4-bit) | ~4.0GB | Automatic OOM fallback |
| BERTScore deberta-xl | ~1.5GB | Loaded after LLM offload |
| BERTScore distilbert | ~250MB | Fallback if VRAM < 2GB free |

---

## Supported GPUs

| GPU | VRAM | Recommended configuration |
|---|---|---|
| RTX 3080 | 10GB | 8-bit (primary mode) |
| RTX 3060 | 12GB | 8-bit, works better than 3080 |
| RTX 3070 | 8GB | 4-bit mandatory |
| RTX 4090 | 24GB | No quantization needed |
| Tesla T4 | 16GB | 8-bit, great for Colab/cloud |
| CPU only | — | Template-based fallback, no LLM |

---

## Installation

### Windows 

```bat
setup_env.bat
```

### Linux / macOS

```bash
bash setup_env.sh
```

### Manual

```bash
# 1. Create virtualenv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Upgrade PIP
python -m pip install --upgrade pip

# 3. PyTorch with CUDA 12.1 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. bitsandbytes with Ampere support
pip install "bitsandbytes>=0.41.3"

# 5. Remaining dependencies
pip install -r requirements.txt\

# 6. Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Model Configuration

1. Place the EfficientNet-B0 `.pth` file in the `models/` directory:
   ```
   models/efficientnet_b0_melanoma.pth
   ```

2. Update the path in `config/config.yaml` if needed:
   ```yaml
   model:
     weights_path: "models/efficientnet_b0_melanoma.pth"
   ```

The checkpoint must contain an EfficientNet-B0 with classifier `nn.Linear(1280, 2)`.
Both formats are supported:
- `torch.save(model.state_dict(), path)` — pure state_dict
- `torch.save({"model_state_dict": ...}, path)` — dict with key

---

## Report Generation Configuration

### ABC Generation Mode

`config/config.yaml` → `report_generator.abc_generation_mode`

| Value | Behaviour |
|---|---|
| `hybrid` (default) | Sections A, B, C generated deterministically from quantitative features (asymmetry, border, colour). LLM called only for `**CONCLUSIONE:**`. Requires `feature_extractor.enabled: true`. |
| `llm_only` | LLM generates the full report (A + B + C + CONCLUSIONE) from the prompt. Used automatically as fallback when `image_features` is unavailable. |

```yaml
report_generator:
  abc_generation_mode: "hybrid"
```

In `hybrid` mode the conclusion is never empty: if the LLM call fails on all three attempts the section is left blank but A/B/C remain intact. In `llm_only` mode the entire report falls back to a static template when the LLM cannot load.

### Prompt Template

`config/config.yaml` → `llm.template`

Path (relative to project root) of the prompt template sent to BioMistral-7B. Two templates ship with the project:

| File | Purpose |
|---|---|
| `config/abc_prompt_template.txt` | **Default.** Ends with `**CONCLUSIONE:**` — model generates only the conclusion. Used with `hybrid` mode. |
| `config/abcde_prompt_template.txt` | Legacy full-report template. Compatible with `llm_only` mode. |

```yaml
llm:
  template: "config/abc_prompt_template.txt"
```

To use a custom template, create a `.txt` file with the desired placeholders and point `llm.template` at it. Available placeholders:

| Placeholder | Source |
|---|---|
| `{classification}` | CNN predicted class (`melanoma` / `nevus`) |
| `{confidence:.1%}` | CNN confidence score |
| `{gradcam_focus_description}` | GradCAM++ top-N active regions |
| `{section_asymmetry}` | Deterministic Section A text (`hybrid` only) |
| `{section_borders}` | Deterministic Section B text (`hybrid` only) |
| `{section_color}` | Deterministic Section C text (`hybrid` only) |
| `{rag_context}` | Retrieved literature context |
| `{rag_references}` | Formatted literature references |

Placeholders absent from the template are silently ignored; placeholders present in the template but missing from the call raise `KeyError` at runtime.

---

## Building the RAG Knowledge Base

**One-time operation** (~10-20 minutes). Requires internet connection.

```bash
python -m src.pipeline --build-rag-kb
```

or directly:

```bash
python rag_kb/build_kb.py --email your@email.com
```

Downloads data from **4 sources** and builds a local ChromaDB index:

| Source | Type | Notes |
|---|---|---|
| PubMed (Entrez) | Biomedical abstracts | Hybrid MeSH + Title/Abstract query; automatic fallback |
| Europe PMC | Open-access abstracts | Stable REST API, 200 records |
| Open-i NIH | Abstracts + image captions | NIH domain, consistently reachable |
| Semantic Scholar | Papers with abstracts | Free, no API key required |

Source directories are created automatically at startup:

```
rag_kb/
├── pubmed/              # PubMed cache: one {pmid}.json file per record
├── europepmc/           # Europe PMC cache: one {idx}.json file per record
├── openi/               # Open-i NIH cache: one {idx}.json file per record
├── semanticscholar/     # Semantic Scholar cache: one {idx}.json file per record
└── chroma_db/           # ChromaDB index — the only file needed at runtime
```

> **Note:** ISIC Archive is **intentionally excluded** from RAG sources (see Bias section).

---

## Building the RAG KB on a Separate Machine

The RAG KB can be built on **any machine, even without a GPU**. The only requirements are Python and an internet connection during the initial download.

### Why build it elsewhere

- The production machine may have no internet, or you may want to keep its environment clean
- The download process (~10-20 min) requires no CUDA steps
- The `venv_kb` environment weighs ~500MB vs ~15GB for `venv` (with PyTorch + CUDA)

### Setup on the build machine

```bash
# Mac/Linux
bash setup_env_kb.sh

# Windows
setup_env_kb.bat
```

Uses `requirements_kb.txt`: only `chromadb`, `sentence-transformers`, `biopython`, `requests`, `beautifulsoup4`, `tqdm`, `pyyaml`. No torch, no CUDA.

### Building the KB

```bash
source venv_kb/bin/activate        # Windows: call venv_kb\Scripts\activate
python rag_kb/build_kb.py --email your@email.com
```

### What to transfer to the production machine

Copy **only** `rag_kb/chroma_db/` — it is the only directory read at runtime by `RAGRetriever`:

```bash
# From build machine to production machine:
scp -r rag_kb/chroma_db/ user@production:/path/to/derm-ai-pipeline/rag_kb/
```

The `pubmed/`, `europepmc/`, `openi/`, `semanticscholar/` directories contain only raw cache files (one `.json` per record). **Not needed at runtime** and should not be copied to the GPU machine.

### Backup and re-indexing

Keep all source directories on the **build machine** as a backup. If ChromaDB becomes corrupted or you want to change the embedding model, just re-index in ~2-5 min without re-downloading:

```bash
python rag_kb/build_kb.py --rebuild-from-cache
```

Sources to re-index are read from `config/config.yaml` (`rag.sources[].enabled`). Individual sources can be enabled/disabled before rebuilding.

| Directory | Runtime Production | Build machine |
|---|---|---|
| `rag_kb/chroma_db/` | **Required** | Optional (copy) |
| `rag_kb/pubmed/` | Not required | Keep as backup |
| `rag_kb/europepmc/` | Not required | Keep as backup |
| `rag_kb/openi/` | Not required | Keep as backup |
| `rag_kb/semanticscholar/` | Not required | Keep as backup |

---

## Pre-downloading the BERTScore Model on a Separate Machine

The BERTScore model (`microsoft/deberta-xlarge-mnli`, ~3GB) is downloaded automatically from HuggingFace the first time evaluation runs. On the GPU machine (no internet, slow connection, or air-gapped), it can be pre-downloaded elsewhere and transferred.

### 1 — Download on the build machine

```python
# Run once on any machine with internet access (no GPU required)
from transformers import AutoTokenizer, AutoModel

AutoTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")
AutoModel.from_pretrained("microsoft/deberta-xlarge-mnli")
```

Or via CLI:

```bash
pip install huggingface_hub
huggingface-cli download microsoft/deberta-xlarge-mnli
```

### 2 — Locate the cache directory

| OS | Default HuggingFace cache |
|---|---|
| Windows | `C:\Users\{username}\.cache\huggingface\hub\` |
| Linux / macOS | `~/.cache/huggingface/hub/` |

The model directory is named `models--microsoft--deberta-xlarge-mnli`.

### 3 — Copy to the production machine

```bash
# Example: copy from build machine to production over SSH
scp -r ~/.cache/huggingface/hub/models--microsoft--deberta-xlarge-mnli \
    user@production:~/.cache/huggingface/hub/
```

On Windows, copy the folder to `C:\Users\{username}\.cache\huggingface\hub\`.

If you want to store the cache in a non-default location, set the env var before running:

```bat
set HF_HOME=D:\models\huggingface
```

### 4 — Enable offline mode in config

```yaml
evaluation:
  bertscore_offline: true   # prevents any network access during BERTScore loading
```

With `bertscore_offline: true` the pipeline sets `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` only during `BERTScorer` initialisation, then restores the original env state. If the model is not in the cache the run will fail with a clear error rather than hanging on a download.

---

## Configuration Check

```bash
python -m src.pipeline --vram-check
```

---

## Usage

### Single image

```bash
python -m src.pipeline --image data/test_images/lesion.jpg
```

### With ground truth (enables NLG metrics)

```bash
python -m src.pipeline --image data/test_images/lesion.jpg \
    --ground-truth data/ground_truth_reports/lesion.json
```

### Batch

```bash
python -m src.pipeline --batch data/test_images/
```

### Batch with ground truth (enables NLG metrics)

```bash
python -m src.pipeline --batch data/test_images/ \
    --ground-truth-dir data/ground_truth_reports/
```

Files are matched by filename stem: `img1.jpg` → `img1.json`, `img2.jpg` → `img2.json`.
Images without a matching `.json` file are processed normally without NLG metrics.

### Evaluation only (on an existing run)

Re-run NLG evaluation on a previously generated run without repeating GradCAM and report generation:

```bash
python -m src.pipeline --eval-only outputs/run_20240101_120000 \
    --ground-truth-dir data/ground_truth_reports/
```

`--eval-only` accepts the path to an existing run directory (one produced by `--batch`). It reads all `*_result.json` files from `run_dir/reports/`, matches each by filename stem against the JSON files in `--ground-truth-dir`, and computes ROUGE + BLEU + BERTScore. Results are written to `run_dir/evaluation/` (overwriting any previous evaluation files).

Useful when:
- Ground truth was not available at the time of the original run
- You want to re-evaluate with a different BERTScore model after changing `config.yaml`
- BERTScore failed during the original batch and you need to retry without regenerating reports

### Multi-model comparison

```bash
python -m src.pipeline --image data/test_images/lesion.jpg \
    --compare "BioMistral/BioMistral-7B,mistralai/Mistral-7B-Instruct-v0.1"
```

---

## Domain Shift Mitigation (`--finetune` mode)

Use this mode when the model shows degraded performance on your institution's images due to scanner or protocol differences from the training distribution.

### When to use

- Model consistently over-predicts one class on your images
- Images differ significantly from ISIC dermoscopy (different scanner, polarisation, magnification)

### Folder structure

```
finetune/
├── images/              # labelled training images (ImageFolder layout)
│   ├── melanoma/
│   │   └── img1.jpg, img2.jpg, ...
│   └── nevus/
│       └── img1.jpg, img2.jpg, ...
└── test_images/         # labelled test images for evaluation
    ├── melanoma/
    │   └── img1.jpg, ...
    └── nevus/
        └── img1.jpg, ...
```

### Command

```bash
# Minimal — uses default directories
python -m src.pipeline --finetune

# Custom directories
python -m src.pipeline \
    --finetune finetune/images/ \
    --test finetune/test_images/ \
    --output finetune/outputs/
```

### Configuration (`config/config.yaml`)

```yaml
preprocessing:
  colour_normalisation: false    # set true to normalise images toward ISIC colour statistics
  melanoma_threshold: 0.40       # asymmetric TTA threshold — lower = fewer false negatives
  tta_enabled: true              # use 5-augmentation TTA during test evaluation
  tta_n_augmentations: 5         # standard, hflip, vflip, rot90, rot180
```

### What happens internally

1. **Backbone frozen** — all EfficientNet-B0 parameters except `classifier.1` (nn.Linear(1280, 2)) are frozen. Only 2562 parameters are trained.
2. **Transform pipeline** — each training image goes through: `CenterCrop(min(H,W))` → optional ISIC colour normalisation → random augmentation (hflip, vflip, rot15, colorjitter) → `Resize(224,224)` → ImageNet normalisation.
3. **Class balance** — `WeightedRandomSampler` ensures equal per-class sampling even with imbalanced fine-tune sets.
4. **80/20 split** — best checkpoint (highest val accuracy) is saved to `efficientnet_b0_finetuned.pth`.
5. **Test evaluation** — fine-tuned model runs `predict_tta()` with 5 augmentations + asymmetric threshold; produces confusion matrix PNG and `finetune_metrics.json`.

### Outputs

```
finetune/outputs/
├── efficientnet_b0_finetuned.pth   # fine-tuned weights (original weights never overwritten)
├── confusion_matrix.png            # 2×2 confusion matrix (MM/NV)
└── finetune_metrics.json           # accuracy, sensitivity, specificity, precision, F1, per-image
```

### Ethical note

Fine-tuning on a small local dataset reduces domain shift but does **not** correct for demographic biases present in the original ISIC training data. Performance on Fitzpatrick IV-VI skin types should be validated independently. This tool is a clinical decision support aid and does not replace a qualified dermatologist.

---

## Output Structure

Batch runs are saved under a timestamped subdirectory (`run_YYYYMMDD_HHMMSS`) so multiple runs never overwrite each other.

```
outputs/
└── run_{timestamp}/
    ├── gradcam/
    │   └── {stem}_gradcam.png          # 3-panel visualization
    ├── reports/
    │   ├── {stem}_report.md            # ABC report in Markdown
    │   └── {stem}_result.json          # Complete results JSON
    └── evaluation/
        ├── classification_results.csv  # Per-image predicted class + confidence (+ GT if available)
        ├── confusion_matrix.png        # 2×2 confusion matrix (requires GT)
        ├── classification_metrics.md   # Accuracy/sensitivity/specificity/F1/AUC-ROC (requires GT)
        ├── nlg_metrics.csv             # ROUGE/BLEU/BERTScore per image (with GT or --eval-only)
        ├── nlg_bar_chart.png           # Mean NLG metric bar chart
        ├── nlg_heatmap.png             # Per-image NLG metric heatmap
        ├── nlg_summary.md              # NLG statistics Markdown summary
        ├── qualitative_scores.csv      # Rubric scores per image
        ├── manual_scoring_template.csv # Blank template for manual evaluation
        ├── pairwise_metrics.csv        # Cross-model pairwise metrics (--compare mode)
        ├── agreement_heatmap.png       # Cross-model agreement heatmap (--compare mode)
        ├── consensus_ranking.csv       # Composite model ranking (--compare mode)
        └── cross_model_summary.md      # Cross-model report (--compare mode)
```

`--eval-only` writes to the `evaluation/` sub-directory of the target run and overwrites any previous NLG metric files. GradCAM images and reports are never modified.

---

## Ground Truth Format

One JSON file per image, named `{stem}.json` (e.g. `ISIC_0054651.json` for `ISIC_0054651.jpg`). Two formats are supported:

**Flat** (recommended):
```json
{
  "classification": "melanoma",
  "report": "## REFERTO DERMATOLOGICO\n\n**A - Asimmetria:** ...\n\n**CONCLUSIONE:** Rischio ALTO. Biopsia raccomandata."
}
```

**Nested** (also accepted — first value's fields are used):
```json
{
  "ISIC_0054651.jpg": {
    "classification": "melanoma",
    "report": "## REFERTO DERMATOLOGICO\n\n**A - Asimmetria:** ...\n\n**CONCLUSIONE:** Rischio ALTO. Biopsia raccomandata."
  }
}
```

The `classification` field (`"melanoma"` or `"nevus"`) is optional. When present, it enables generation of `confusion_matrix.png` and `classification_metrics.md`. The `report` field is required for NLG metric computation.

---

## Fallback Chain

| Situation | Fallback |
|---|---|
| VRAM < 3GB (LLM) | Template-based without LLM |
| 8-bit OOM | Automatic 4-bit NF4 |
| VRAM < 2GB (BERTScore) | distilbert-base-uncased |
| CUDA unavailable | Full CPU |

---

## ⚠️ Bias and Data Leakage Note

The EfficientNet-B0 model was trained on ISIC images augmented via StyleGAN2-ADA. ISIC has **known demographic biases** (Fitzpatrick I-III over-representation). StyleGAN has learned and potentially amplified these biases. For this reason, the RAG knowledge base uses exclusively sources independent from the training dataset.

System performance may be **lower for patients with darker skin types (Fitzpatrick IV-VI)**. **DO NOT use this system for clinical decisions without qualified medical supervision.**

---

## Medical Disclaimer

This system is a **clinical decision support tool** and **DOES NOT replace** the judgment of a qualified dermatologist. All generated reports must be interpreted by a medical professional.

---

## System Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.1+ (recommended: RTX 3080 10GB or better)
- RAM: 16GB+ (32GB recommended)
- Storage: 20GB+ (models + RAG KB)
- Internet: required for model downloads and RAG KB construction
