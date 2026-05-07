"""
Build RAG Knowledge Base — one-time script.
Sources: PubMed (Entrez) + Europe PMC + Open-i NIH + Semantic Scholar.
DermNet NZ disabled (global outage confirmed).
ISIC explicitly excluded (data leakage risk).
Run: python rag_kb/build_kb.py --email tua@email.com
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_kb")

# ── Configuration ────────────────────────────────────────────────────────────
PUBMED_DIR = Path("rag_kb/pubmed")
DERMNET_DIR = Path("rag_kb/dermnet")
EUROPEPMC_DIR = Path("rag_kb/europepmc")
OPENI_DIR = Path("rag_kb/openi")
SEMANTICSCHOLAR_DIR = Path("rag_kb/semanticscholar")
CHROMA_DIR = Path("rag_kb/chroma_db")
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
COLLECTION_NAME = "dermatology_kb"

# PubMed — hybrid query (MeSH + Title/Abstract) for maximum coverage
PUBMED_QUERY = (
    '(melanoma[MeSH Terms] OR melanoma[Title/Abstract] '
    'OR "melanocytic naevus"[MeSH Terms] OR naevus[Title/Abstract] '
    'OR nevus[Title/Abstract] OR "skin lesion"[Title/Abstract]) '
    'AND (dermoscopy[MeSH Terms] OR dermoscopy[Title/Abstract] '
    'OR dermoscopic[Title/Abstract] OR ABCDE[Title/Abstract] '
    'OR "dermatoscopy"[Title/Abstract]) '
    'AND (diagnosis[Title/Abstract] OR classification[Title/Abstract] '
    'OR detection[Title/Abstract]) '
    'AND ("free full text"[filter] OR "open access"[filter])'
)
# Fallback query — used automatically if the primary query returns 0 results
PUBMED_FALLBACK_QUERY = (
    'melanoma dermoscopy[Title/Abstract] '
    'AND ("free full text"[filter])'
)
PUBMED_MAX_RESULTS = 500
PUBMED_CHUNK_SIZE = 512
PUBMED_CHUNK_OVERLAP = 64

# DermNet NZ — disabled (global outage confirmed)
DERMNET_URLS: list[str] = []
DERMNET_CHUNK_SIZE = 256
DERMNET_CHUNK_OVERLAP = 32
DERMNET_DELAY_S = 1.0
DERMNET_ATTRIBUTION = "Source: DermNet NZ (dermnet.nz)"

# Europe PMC
EUROPEPMC_CHUNK_SIZE = 512
EUROPEPMC_CHUNK_OVERLAP = 64

# Open-i NIH
OPENI_CHUNK_SIZE = 256
OPENI_CHUNK_OVERLAP = 32

# Semantic Scholar
SEMANTICSCHOLAR_CHUNK_SIZE = 512
SEMANTICSCHOLAR_CHUNK_OVERLAP = 64
SEMANTICSCHOLAR_RPS_WITH_KEY = 1.0    # requests/s with API key
SEMANTICSCHOLAR_RPS_NO_KEY = 0.1      # requests/s without key (1 req / 10s)
SEMANTICSCHOLAR_TARGET = 200          # target total papers to retrieve

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    """Create all source cache directories and ChromaDB directory at startup.

    Called once at the beginning of main(). Safe to call multiple times
    (mkdir with exist_ok=True).
    """
    for directory in [
        PUBMED_DIR, DERMNET_DIR, EUROPEPMC_DIR,
        OPENI_DIR, SEMANTICSCHOLAR_DIR, CHROMA_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    logger.debug("All RAG source directories ensured")


def _load_source_config(config_path: Path) -> list[dict]:
    """Read rag.sources from config.yaml to determine enabled sources and paths.

    Args:
        config_path: Path to config/config.yaml.

    Returns:
        List of source config dicts with at minimum 'name', 'enabled', 'path'.
        Falls back to built-in defaults if the file cannot be read.
    """
    _defaults: list[dict] = [
        {"name": "pubmed",          "enabled": True,  "path": str(PUBMED_DIR),
         "chunk_size": PUBMED_CHUNK_SIZE,          "chunk_overlap": PUBMED_CHUNK_OVERLAP},
        {"name": "europepmc",       "enabled": True,  "path": str(EUROPEPMC_DIR),
         "chunk_size": EUROPEPMC_CHUNK_SIZE,       "chunk_overlap": EUROPEPMC_CHUNK_OVERLAP},
        {"name": "openi",           "enabled": True,  "path": str(OPENI_DIR),
         "chunk_size": OPENI_CHUNK_SIZE,           "chunk_overlap": OPENI_CHUNK_OVERLAP},
        {"name": "semanticscholar", "enabled": True,  "path": str(SEMANTICSCHOLAR_DIR),
         "chunk_size": SEMANTICSCHOLAR_CHUNK_SIZE, "chunk_overlap": SEMANTICSCHOLAR_CHUNK_OVERLAP,
         "api_key": "", "requests_per_second": SEMANTICSCHOLAR_RPS_WITH_KEY,
         "requests_per_second_no_key": SEMANTICSCHOLAR_RPS_NO_KEY},
        {"name": "dermnet",         "enabled": False, "path": str(DERMNET_DIR),
         "chunk_size": DERMNET_CHUNK_SIZE,         "chunk_overlap": DERMNET_CHUNK_OVERLAP},
    ]

    if not config_path.exists():
        logger.warning(f"Config not found at {config_path} — using built-in defaults")
        return _defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sources = cfg.get("rag", {}).get("sources", [])
        if not sources:
            raise ValueError("rag.sources is empty in config")
        logger.info(
            f"Source config loaded from {config_path} | "
            f"enabled={[s['name'] for s in sources if s.get('enabled', False)]}"
        )
        return sources
    except Exception as e:
        logger.warning(f"Cannot read source config ({e}) — using built-in defaults")
        return _defaults


def _load_records_from_dir(
    source_dir: Path, combined_name: str = "raw.json"
) -> list[dict]:
    """Load cached records from a source directory.

    Prefers individual per-record .json files (saved as {pmid}.json or
    {idx:04d}.json). Falls back to the combined file (abstracts.json /
    raw.json) if no individual files exist. This makes the function forward-
    and backward-compatible.

    Args:
        source_dir: Path to the source cache directory.
        combined_name: Name of the combined fallback file ('abstracts.json'
            for PubMed, 'raw.json' for all other sources).

    Returns:
        List of record dicts, or [] if the directory or files are missing.
    """
    if not source_dir.exists():
        return []

    # Prefer individual per-record files; skip known combined filenames
    _combined_names = {"abstracts.json", "raw.json"}
    individual = sorted(
        f for f in source_dir.glob("*.json") if f.name not in _combined_names
    )
    if individual:
        records = []
        for fp in individual:
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    records.append(json.load(fh))
            except Exception as e:
                logger.warning(f"Skipping unreadable cache file {fp}: {e}")
        logger.info(f"  Loaded {len(records)} records from {len(individual)} files in {source_dir.name}/")
        return records

    # Fallback: combined file
    combined = source_dir / combined_name
    if combined.exists():
        with open(combined, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        logger.info(f"  Loaded {len(records)} records from {source_dir.name}/{combined_name}")
        return records

    return []


def chunk_text(
    text: str, chunk_size: int, overlap: int, separator: str = "\n\n"
) -> list[str]:
    """Split text into overlapping chunks by approximate word count.

    Args:
        text: Input text to chunk.
        chunk_size: Target chunk size in words.
        overlap: Overlap in words between consecutive chunks.
        separator: Preferred split boundary.

    Returns:
        List of text chunks.
    """
    paragraphs = text.split(separator)
    chunks, current, current_len = [], [], 0

    for para in paragraphs:
        words = para.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep overlap
            overlap_words = current[-overlap:] if overlap > 0 else []
            current = overlap_words + words
            current_len = len(current)
        else:
            current.extend(words)
            current_len += len(words)

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


# ── PubMed Central OA ─────────────────────────────────────────────────────────

def _entrez_search(entrez_module, query: str, retmax: int) -> list[str]:
    """Run Entrez.esearch and return list of PMIDs.

    Args:
        entrez_module: Imported Bio.Entrez module.
        query: PubMed query string.
        retmax: Maximum records to return.

    Returns:
        List of PMID strings (may be empty).
    """
    handle = entrez_module.esearch(
        db="pubmed",
        term=query,
        retmax=retmax,
        usehistory="y",
        sort="relevance",
    )
    results = entrez_module.read(handle)
    handle.close()
    return results["IdList"]


def _parse_abstract_text_blocks(raw_text: str) -> list[dict]:
    """Parse plain-text abstract output from Entrez efetch into record dicts.

    Each record in abstract text mode is separated by blank lines and contains
    a PMID marker. Extracts pmid, title (first substantial line), and the full
    block as abstract text.

    Args:
        raw_text: Raw string from Entrez.efetch(rettype="abstract", retmode="text").

    Returns:
        List of dicts with keys: pmid, title, abstract, year, journal.
    """
    records = []
    # Records are separated by 3+ newlines
    blocks = re.split(r'\n{3,}', raw_text.strip())
    pmid_re = re.compile(r'PMID:\s*(\d+)', re.IGNORECASE)

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        pmid_match = pmid_re.search(block)
        if not pmid_match:
            continue  # Skip malformed blocks without PMID
        pmid = pmid_match.group(1)

        # Best-effort title: second non-empty line (first is usually journal/date)
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        journal = lines[0] if lines else ""
        title = lines[1] if len(lines) > 1 else ""

        # Year: look for 4-digit year in first two lines
        year_match = re.search(r'\b(19|20)\d{2}\b', journal)
        year = year_match.group(0) if year_match else ""

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": block,  # Full block used for RAG text
            "year": year,
            "journal": journal,
        })

    return records


def fetch_pubmed_abstracts(email: str) -> list[dict]:
    """Fetch melanoma/dermoscopy abstracts from PubMed using hybrid query.

    Uses a hybrid MeSH + Title/Abstract query for maximum coverage.
    If the primary query returns 0 results, retries automatically with
    a simpler fallback query and logs a WARNING.

    Fetches with rettype="abstract", retmode="text" in batches of 100
    with 0.4s delay to respect NCBI rate limits.

    Args:
        email: Required by NCBI Entrez API (ToS).

    Returns:
        List of dicts with keys: pmid, title, abstract, year, journal.
        Returns [] on any unrecoverable error (caller handles gracefully).
    """
    try:
        from Bio import Entrez
    except ImportError:
        logger.error("biopython not installed. Run: pip install biopython")
        return []

    Entrez.email = email
    Entrez.tool = "DermoAIPipeline"

    # ── Search phase ──────────────────────────────────────────────────────────
    logger.info(f"PubMed search (hybrid query): {PUBMED_QUERY[:80]}...")
    try:
        pmids = _entrez_search(Entrez, PUBMED_QUERY, PUBMED_MAX_RESULTS)
    except Exception as e:
        logger.error(f"PubMed search failed: {e}")
        return []

    if not pmids:
        logger.warning(
            "Primary query returned 0 results — retrying with fallback query: "
            f"{PUBMED_FALLBACK_QUERY}"
        )
        try:
            pmids = _entrez_search(Entrez, PUBMED_FALLBACK_QUERY, PUBMED_MAX_RESULTS)
        except Exception as e:
            logger.error(f"PubMed fallback search also failed: {e}")
            return []
        if not pmids:
            logger.error("Fallback query also returned 0 results. PubMed unavailable.")
            return []
        logger.warning(f"Using fallback query — found {len(pmids)} records")
    else:
        logger.info(f"Found {len(pmids)} PubMed records")

    # ── Fetch phase (abstract text mode, batches of 100) ─────────────────────
    records: list[dict] = []
    batch_size = 100

    for start in range(0, len(pmids), batch_size):
        batch = pmids[start:start + batch_size]
        try:
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="abstract",
                retmode="text",
            )
            raw_text = fetch_handle.read()
            fetch_handle.close()

            batch_records = _parse_abstract_text_blocks(raw_text)
            records.extend(batch_records)

        except Exception as e:
            logger.warning(
                f"PubMed fetch error (batch start={start}, size={len(batch)}): {e}"
            )

        logger.info(
            f"  Fetched {min(start + batch_size, len(pmids))}/{len(pmids)} records "
            f"({len(records)} parsed so far)..."
        )
        time.sleep(0.4)  # Respect NCBI rate limits

    logger.info(f"PubMed: {len(records)} abstracts fetched successfully")
    return records


def build_pubmed_chunks(records: list[dict]) -> list[dict]:
    """Chunk PubMed abstracts and build document list for ChromaDB.

    Args:
        records: Output from fetch_pubmed_abstracts().

    Returns:
        List of dicts with keys: text, metadata (source, reference, chunk_id).
    """
    documents = []
    for rec in records:
        full_text = f"{rec['title']}. {rec['abstract']}"
        chunks = chunk_text(full_text, PUBMED_CHUNK_SIZE, PUBMED_CHUNK_OVERLAP)
        ref = (
            f"PMID:{rec['pmid']} | {rec['title'][:60]}... | "
            f"{rec['journal']} ({rec['year']})"
        )
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "metadata": {
                    "source": "pubmed",
                    "reference": ref,
                    "chunk_id": f"pubmed_{rec['pmid']}_{i}",
                    "pmid": rec["pmid"],
                    "year": rec["year"],
                },
            })

    # Save one .json file per record (primary cache for --rebuild-from-cache)
    PUBMED_DIR.mkdir(parents=True, exist_ok=True)
    for rec in records:
        rec_path = PUBMED_DIR / f"{rec['pmid']}.json"
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
    # Also save combined file for convenience and backward compat
    with open(PUBMED_DIR / "abstracts.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info(f"PubMed: {len(records)} abstracts → {len(documents)} chunks | cache → {PUBMED_DIR}/")
    return documents


# ── DermNet NZ ───────────────────────────────────────────────────────────────

def scrape_dermnet(urls: list[str]) -> list[dict]:
    """Scrape content from DermNet NZ topic pages.

    Respects robots.txt with 1s delay between requests.
    Adds attribution to each chunk.

    Args:
        urls: List of DermNet NZ topic URLs.

    Returns:
        List of dicts with keys: url, title, content.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("requests/beautifulsoup4 not installed")
        return []

    pages = []
    headers = {
        "User-Agent": "DermoAIPipeline/1.0 (research; contact: researcher@example.com)"
    }

    for url in urls:
        try:
            logger.info(f"Scraping: {url}")
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract title
            title_tag = soup.find("h1") or soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else url

            # Extract main content (avoid nav/footer)
            content_divs = soup.find_all(
                "div",
                class_=lambda c: c and any(
                    kw in c.lower() for kw in ["content", "article", "main", "body"]
                ),
            )
            if not content_divs:
                content_divs = [soup.find("main") or soup.find("body")]

            text_parts = []
            for div in content_divs:
                if div:
                    for tag in div.find_all(["p", "li", "h2", "h3"]):
                        t = tag.get_text(strip=True)
                        if len(t) > 30:
                            text_parts.append(t)

            content = "\n\n".join(text_parts)
            if content:
                pages.append({"url": url, "title": title, "content": content})
                logger.info(f"  → {len(content)} chars extracted")

        except Exception as e:
            logger.warning(f"Scraping failed for {url}: {e}")

        time.sleep(DERMNET_DELAY_S)

    return pages


def build_dermnet_chunks(pages: list[dict]) -> list[dict]:
    """Chunk DermNet pages and build document list for ChromaDB.

    Args:
        pages: Output from scrape_dermnet().

    Returns:
        List of dicts with text and metadata.
    """
    documents = []
    DERMNET_DIR.mkdir(parents=True, exist_ok=True)

    for page in pages:
        url_slug = page["url"].rstrip("/").split("/")[-1]
        chunks = chunk_text(
            page["content"], DERMNET_CHUNK_SIZE, DERMNET_CHUNK_OVERLAP
        )
        ref = f"{DERMNET_ATTRIBUTION} | {page['title']} | {page['url']}"

        # Save raw text
        raw_path = DERMNET_DIR / f"{url_slug}.txt"
        raw_path.write_text(
            f"# {page['title']}\n\n{page['content']}", encoding="utf-8"
        )

        for i, chunk in enumerate(chunks):
            attributed_chunk = f"{chunk}\n\n[{DERMNET_ATTRIBUTION}]"
            documents.append({
                "text": attributed_chunk,
                "metadata": {
                    "source": "dermnet",
                    "reference": ref,
                    "chunk_id": f"dermnet_{url_slug}_{i}",
                    "url": page["url"],
                    "title": page["title"],
                },
            })

    logger.info(f"DermNet: {len(pages)} pages → {len(documents)} chunks")
    return documents


# ── Generic chunk builder for API-based sources ──────────────────────────────

def build_generic_chunks(
    records: list[dict],
    source_name: str,
    chunk_size: int,
    chunk_overlap: int,
    save_dir: Path,
) -> list[dict]:
    """Chunk text records from any API source and build ChromaDB document list.

    Saves raw records as raw.json in save_dir for future --rebuild-from-cache.

    Args:
        records: List of dicts with keys: text, reference. Any extra keys ignored.
        source_name: Source identifier string (e.g. "europepmc").
        chunk_size: Target chunk size in words.
        chunk_overlap: Overlap in words between chunks.
        save_dir: Directory to persist raw.json cache.

    Returns:
        List of ChromaDB-ready document dicts.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save one .json file per record (primary cache for --rebuild-from-cache)
    for idx, rec in enumerate(records):
        rec_path = save_dir / f"{idx:04d}.json"
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
    # Also save combined file for convenience and backward compat
    with open(save_dir / "raw.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    documents = []
    for idx, rec in enumerate(records):
        chunks = chunk_text(rec["text"], chunk_size, chunk_overlap)
        ref = rec.get("reference", f"{source_name}_{idx}")
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "metadata": {
                    "source": source_name,
                    "reference": ref,
                    "chunk_id": f"{source_name}_{idx}_{i}",
                },
            })

    logger.info(f"{source_name}: {len(records)} records → {len(documents)} chunks")
    return documents


# ── Europe PMC ────────────────────────────────────────────────────────────────

def fetch_europepmc() -> list[dict]:
    """Fetch melanoma/dermoscopy records from Europe PMC REST API.

    Stable public API, no key required, no aggressive rate limiting.
    Returns open-access records with title + abstractText.

    Returns:
        List of dicts with keys: text, reference.
        Returns [] on any error (caller handles gracefully).
    """
    try:
        import requests
    except ImportError:
        logger.error("requests not installed. Run: pip install requests")
        return []

    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": "melanoma dermoscopy ABCDE",
        "format": "json",
        "pageSize": 200,
        "resultType": "core",
        "openAccess": "true",
    }

    try:
        logger.info("Fetching Europe PMC...")
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Europe PMC request failed: {e}")
        return []

    results = data.get("resultList", {}).get("result", [])
    records = []
    for r in results:
        title = r.get("title", "").strip()
        abstract = r.get("abstractText", "").strip()
        if not abstract:
            continue
        pmid = r.get("pmid") or r.get("id", "")
        records.append({
            "text": f"{title}. {abstract}" if title else abstract,
            "reference": f"EuropePMC | PMID:{pmid} | {title[:60]}",
        })

    logger.info(f"Europe PMC: {len(records)} records with abstracts (of {len(results)} total)")
    return records


# ── Open-i NIH ────────────────────────────────────────────────────────────────

def fetch_openi() -> list[dict]:
    """Fetch melanoma/dermoscopy records from Open-i NIH API.

    NIH domain — consistently reachable. Extracts abstract + image captions.

    Returns:
        List of dicts with keys: text, reference.
        Returns [] on any error (caller handles gracefully).
    """
    try:
        import requests
    except ImportError:
        logger.error("requests not installed. Run: pip install requests")
        return []

    url = "https://openi.nlm.nih.gov/api/search"
    params = {
        "query": "melanoma dermoscopy",
        "m": 1,
        "n": 100,
        "coll": 4,
    }

    try:
        logger.info("Fetching Open-i NIH...")
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Open-i NIH request failed: {e}")
        return []

    records = []
    for item in data.get("list", []):
        abstract = (item.get("abstract") or "").strip()
        # imgLarge may be a dict with a caption field or a plain string
        img_large = item.get("imgLarge", {})
        caption = ""
        if isinstance(img_large, dict):
            caption = (img_large.get("caption") or "").strip()
        elif isinstance(img_large, str):
            caption = img_large.strip()

        text = " ".join(filter(None, [abstract, caption]))
        if not text:
            continue

        uid = item.get("uid", "")
        title = (item.get("title") or "").strip()
        records.append({
            "text": text,
            "reference": f"Open-i NIH | uid:{uid} | {title[:60]}",
        })

    logger.info(f"Open-i NIH: {len(records)} records with content")
    return records


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def fetch_semanticscholar(
    api_key: str = "",
    rps_with_key: float = SEMANTICSCHOLAR_RPS_WITH_KEY,
    rps_no_key: float = SEMANTICSCHOLAR_RPS_NO_KEY,
    target: int = SEMANTICSCHOLAR_TARGET,
) -> list[dict]:
    """Fetch melanoma/dermoscopy papers from Semantic Scholar Graph API.

    Supports optional API key for higher rate limits. Paginates in batches of
    25 records with a dynamic inter-request delay. Implements exponential
    backoff (60s → 120s → 240s) on HTTP 429 before giving up.

    Args:
        api_key: Optional API key (free at semanticscholar.org/product/api).
        rps_with_key: Requests/second when API key is present.
        rps_no_key: Requests/second without API key (conservative).
        target: Total papers to retrieve (across all pages).

    Returns:
        List of dicts with keys: text, reference.
        Returns [] on any unrecoverable error (caller handles gracefully).
    """
    try:
        import requests
    except ImportError:
        logger.error("requests not installed. Run: pip install requests")
        return []

    if not api_key:
        logger.warning(
            "Semantic Scholar: no API key. Slow retrieval (1 req/10s).\n"
            "  Request a free API key at semanticscholar.org/product/api\n"
            "  to speed up the download."
        )

    headers = {"x-api-key": api_key} if api_key else {}
    delay = 1.0 / (rps_with_key if api_key else rps_no_key)

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    page_size = 25
    max_retries = 3
    records: list[dict] = []
    offset = 0

    while len(records) < target:
        params = {
            "query": "melanoma dermoscopy diagnosis",
            "fields": "title,abstract,year",
            "limit": page_size,
            "offset": offset,
        }

        backoff = 60
        data = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=30)
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Semantic Scholar request error (attempt {attempt + 1}/{max_retries + 1}): "
                        f"{e}. Retry in {backoff}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                logger.warning(f"Semantic Scholar: request failed after {max_retries + 1} attempts: {e}")
                return records

            if resp.status_code == 429:
                if attempt < max_retries:
                    logger.warning(
                        f"Semantic Scholar 429 (offset={offset}, attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {backoff}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                logger.warning(
                    "Semantic Scholar: persistent 429 after 3 attempts — source skipped.\n"
                    "  Consider adding a free API key: "
                    "semanticscholar.org/product/api"
                )
                return records

            try:
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"Semantic Scholar: unexpected response (offset={offset}): {e}")
                return records
            break  # successful response

        if data is None:
            break

        page_data = data.get("data", [])
        if not page_data:
            break  # no more results

        for paper in page_data:
            abstract = (paper.get("abstract") or "").strip()
            if not abstract:
                continue
            title = (paper.get("title") or "").strip()
            year = paper.get("year") or ""
            records.append({
                "text": f"{title}. {abstract}" if title else abstract,
                "reference": f"SemanticScholar | {title[:60]} ({year})",
            })

        offset += page_size
        time.sleep(delay)

    logger.info(f"Semantic Scholar: {len(records)} records with abstracts")
    return records


# ── Load from local cache (no network) ───────────────────────────────────────

def load_pubmed_from_cache() -> list[dict]:
    """Load PubMed abstracts from the local JSON cache.

    Reads the file saved by build_pubmed_chunks() on a previous full run.

    Returns:
        List of abstract dicts (same format as fetch_pubmed_abstracts()).

    Raises:
        SystemExit: If abstracts.json does not exist.
    """
    import json

    cache_file = PUBMED_DIR / "abstracts.json"
    if not cache_file.exists():
        logger.error(
            f"Cache file not found: {cache_file}\n"
            "  Run without --rebuild-from-cache first to download data:\n"
            "    python rag_kb/build_kb.py --email your@email.com"
        )
        sys.exit(1)

    with open(cache_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info(f"PubMed cache loaded: {len(records)} abstracts from {cache_file}")
    return records


def load_dermnet_from_cache() -> list[dict]:
    """Load DermNet pages from local .txt cache files.

    Reads the files saved by build_dermnet_chunks() on a previous full run.
    Reconstructs the URL from the filename slug (e.g. melanoma.txt →
    https://dermnet.nz/topics/melanoma/).

    Returns:
        List of page dicts with keys: url, title, content.

    Raises:
        SystemExit: If no .txt files are found in rag_kb/dermnet/.
    """
    txt_files = sorted(DERMNET_DIR.glob("*.txt"))
    if not txt_files:
        logger.error(
            f"No .txt cache files found in: {DERMNET_DIR}\n"
            "  Run without --rebuild-from-cache first to download data:\n"
            "    python rag_kb/build_kb.py --email your@email.com"
        )
        sys.exit(1)

    pages = []
    for txt_path in txt_files:
        raw = txt_path.read_text(encoding="utf-8")
        lines = raw.split("\n", 2)

        # First line is "# {title}", rest is content
        title = lines[0].lstrip("# ").strip() if lines else txt_path.stem
        content = lines[2].strip() if len(lines) > 2 else ""

        if not content:
            logger.warning(f"Empty content in cache file: {txt_path}, skipping")
            continue

        slug = txt_path.stem
        url = f"https://dermnet.nz/topics/{slug}/"
        pages.append({"url": url, "title": title, "content": content})
        logger.info(f"  Loaded {txt_path.name}: {len(content)} chars")

    logger.info(f"DermNet cache loaded: {len(pages)} pages from {DERMNET_DIR}")
    return pages


def _check_any_cache_exists() -> tuple[bool, bool]:
    """Check which cache sources are available locally.

    Returns:
        Tuple of (pubmed_ok, dermnet_ok).
    """
    pubmed_ok = (PUBMED_DIR / "abstracts.json").exists()
    dermnet_ok = any(DERMNET_DIR.glob("*.txt"))
    return pubmed_ok, dermnet_ok


# ── ChromaDB Indexing ─────────────────────────────────────────────────────────

def index_documents(documents: list[dict]) -> None:
    """Embed documents and index in ChromaDB.

    Args:
        documents: Combined list from PubMed + DermNet.

    Raises:
        ImportError: If chromadb or sentence_transformers not installed.
    """
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)

    if not documents:
        logger.error("No documents to index!")
        sys.exit(1)

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL} (CPU)")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reset collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed and index in batches
    texts = [d["text"] for d in documents]
    metadatas = [d["metadata"] for d in documents]
    ids = [d["metadata"]["chunk_id"] for d in documents]

    batch_size = 64
    total_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(
        f"Indexing {len(documents)} documents in {total_batches} batches..."
    )

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        embeddings = embedder.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids,
        )

        if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= len(texts):
            logger.info(
                f"  Indexed batch {i // batch_size + 1}/{total_batches} "
                f"({min(i + batch_size, len(texts))}/{len(texts)} docs)"
            )

    final_count = collection.count()
    logger.info(f"ChromaDB indexed: {final_count} documents in '{COLLECTION_NAME}'")
    logger.info(f"Knowledge base saved → {CHROMA_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_source(name: str, fetch_fn, *args) -> list[dict]:
    """Run a fetch function with graceful error handling.

    Args:
        name: Human-readable source name for logging.
        fetch_fn: Callable that returns list of records (may return []).
        *args: Arguments forwarded to fetch_fn.

    Returns:
        List of records, or [] on any exception.
    """
    try:
        records = fetch_fn(*args)
        return records if records else []
    except Exception as e:
        logger.warning(f"{name} fetch raised an unexpected error: {e}")
        return []


def main() -> None:
    """Build the full RAG knowledge base."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build DermoAI RAG Knowledge Base\n"
            "Sources: PubMed (Entrez) | Europe PMC | Open-i NIH | Semantic Scholar\n"
            "DermNet NZ: disabled (global outage confirmed)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First run — downloads from internet (~10-20 min):
  python rag_kb/build_kb.py --email your@email.com

  # Re-index only — uses local cache, no internet required (~2-5 min):
  python rag_kb/build_kb.py --rebuild-from-cache

  # Skip specific sources:
  python rag_kb/build_kb.py --email your@email.com --skip-pubmed
""",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="researcher@example.com",
        help="Email for NCBI Entrez API (required by NCBI ToS, ignored with --rebuild-from-cache)",
    )
    parser.add_argument(
        "--skip-pubmed", action="store_true",
        help="Skip PubMed (Entrez) download",
    )
    parser.add_argument(
        "--skip-europepmc", action="store_true",
        help="Skip Europe PMC download",
    )
    parser.add_argument(
        "--skip-openi", action="store_true",
        help="Skip Open-i NIH download",
    )
    parser.add_argument(
        "--skip-semanticscholar", action="store_true",
        help="Skip Semantic Scholar download",
    )
    parser.add_argument(
        "--rebuild-from-cache",
        action="store_true",
        help=(
            "Re-index ChromaDB from local per-record .json cache files — no internet. "
            "Reads enabled sources from --config. Requires a previous full build."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml (used to resolve enabled sources, default: config/config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)

    # Create all source directories up front — safe no-op if they exist
    _ensure_dirs()

    logger.info("=" * 60)
    logger.info("DermoAI RAG Knowledge Base Builder")
    logger.info("=" * 60)
    logger.info("⚠️  ISIC excluded: data leakage (training StyleGAN + EfficientNet-B0)")
    logger.info("⚠️  DermNet NZ: disabled (global outage confirmed)")
    logger.info("Sources: PubMed | Europe PMC | Open-i NIH | Semantic Scholar")
    logger.info("=" * 60)

    all_documents: list[dict] = []
    source_counts: defaultdict[str, int] = defaultdict(int)

    source_configs = _load_source_config(config_path)

    # ── Rebuild-from-cache path (no network) ──────────────────────────────────
    if args.rebuild_from_cache:
        logger.info("\nMode: REBUILD FROM CACHE (no internet access)")

        enabled = [s for s in source_configs if s.get("enabled", False)]

        if not enabled:
            logger.error(
                "No sources are enabled in config. "
                "Set enabled: true for at least one source in config.yaml."
            )
            sys.exit(1)

        total_steps = len(enabled) + 1  # +1 for indexing
        step = 1
        any_cache_found = False

        for src in enabled:
            name = src["name"]
            src_dir = Path(src.get("path", f"rag_kb/{name}"))
            chunk_size = src.get("chunk_size", 512)
            chunk_overlap = src.get("chunk_overlap", 64)
            combined_name = "abstracts.json" if name == "pubmed" else "raw.json"

            logger.info(f"\n[{step}/{total_steps}] Loading {name} from cache ({src_dir})...")
            records = _load_records_from_dir(src_dir, combined_name)

            if not records:
                logger.warning(
                    f"  No cache files found for '{name}' in {src_dir} — skipping.\n"
                    f"  Run a full build first: python rag_kb/build_kb.py --email your@email.com"
                )
                step += 1
                continue

            any_cache_found = True
            if name == "pubmed":
                chunks = build_pubmed_chunks(records)
            else:
                chunks = build_generic_chunks(records, name, chunk_size, chunk_overlap, src_dir)
            all_documents.extend(chunks)
            source_counts[name] = len(chunks)
            step += 1

        if not any_cache_found:
            logger.error(
                "No cache files found for any enabled source.\n"
                "\n"
                "Run a full build first to populate the cache:\n"
                "  python rag_kb/build_kb.py --email your@email.com"
            )
            sys.exit(1)

    # ── Full download path ─────────────────────────────────────────────────────
    else:
        logger.info("\nMode: FULL BUILD (downloading from internet)")

        sources_to_run = []
        if not args.skip_pubmed:
            sources_to_run.append("pubmed")
        if not args.skip_europepmc:
            sources_to_run.append("europepmc")
        if not args.skip_openi:
            sources_to_run.append("openi")
        if not args.skip_semanticscholar:
            sources_to_run.append("semanticscholar")

        total_steps = len(sources_to_run) + 1  # +1 for indexing
        step = 1

        if "pubmed" in sources_to_run:
            logger.info(f"\n[{step}/{total_steps}] Fetching PubMed (Entrez)...")
            pubmed_records = _run_source("PubMed", fetch_pubmed_abstracts, args.email)
            if pubmed_records:
                chunks = build_pubmed_chunks(pubmed_records)
                all_documents.extend(chunks)
                source_counts["pubmed"] = len(chunks)
            else:
                logger.warning("PubMed returned no records — continuing with other sources")
            step += 1

        if "europepmc" in sources_to_run:
            logger.info(f"\n[{step}/{total_steps}] Fetching Europe PMC...")
            records = _run_source("Europe PMC", fetch_europepmc)
            if records:
                chunks = build_generic_chunks(
                    records, "europepmc",
                    EUROPEPMC_CHUNK_SIZE, EUROPEPMC_CHUNK_OVERLAP, EUROPEPMC_DIR,
                )
                all_documents.extend(chunks)
                source_counts["europepmc"] = len(chunks)
            else:
                logger.warning("Europe PMC returned no records — continuing with other sources")
            step += 1

        if "openi" in sources_to_run:
            logger.info(f"\n[{step}/{total_steps}] Fetching Open-i NIH...")
            records = _run_source("Open-i NIH", fetch_openi)
            if records:
                chunks = build_generic_chunks(
                    records, "openi",
                    OPENI_CHUNK_SIZE, OPENI_CHUNK_OVERLAP, OPENI_DIR,
                )
                all_documents.extend(chunks)
                source_counts["openi"] = len(chunks)
            else:
                logger.warning("Open-i NIH returned no records — continuing with other sources")
            step += 1

        if "semanticscholar" in sources_to_run:
            logger.info(f"\n[{step}/{total_steps}] Fetching Semantic Scholar...")
            _ss_cfg = next((s for s in source_configs if s.get("name") == "semanticscholar"), {})
            records = _run_source(
                "Semantic Scholar", fetch_semanticscholar,
                _ss_cfg.get("api_key", ""),
                float(_ss_cfg.get("requests_per_second", SEMANTICSCHOLAR_RPS_WITH_KEY)),
                float(_ss_cfg.get("requests_per_second_no_key", SEMANTICSCHOLAR_RPS_NO_KEY)),
            )
            if records:
                chunks = build_generic_chunks(
                    records, "semanticscholar",
                    SEMANTICSCHOLAR_CHUNK_SIZE, SEMANTICSCHOLAR_CHUNK_OVERLAP,
                    SEMANTICSCHOLAR_DIR,
                )
                all_documents.extend(chunks)
                source_counts["semanticscholar"] = len(chunks)
            else:
                logger.warning("Semantic Scholar returned no records — continuing with other sources")
            step += 1

    # ── Failure guard: at least one source must have produced data ─────────────
    if not all_documents:
        logger.error(
            "No source reachable — no documents to index.\n"
            "Check your internet connection and try again.\n"
            "If the problem persists, try --skip-pubmed or other flags to "
            "isolate the problematic source."
        )
        sys.exit(1)

    # ── Index (shared by both paths) ───────────────────────────────────────────
    logger.info(f"\n[{step}/{total_steps}] Indexing {len(all_documents)} total chunks into ChromaDB...")
    index_documents(all_documents)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Knowledge base build complete!")
    logger.info(
        f"Indexed chunks: "
        f"PubMed={source_counts['pubmed']} | "
        f"EuropePMC={source_counts['europepmc']} | "
        f"OpenI={source_counts['openi']} | "
        f"SemanticScholar={source_counts['semanticscholar']} | "
        f"Total={len(all_documents)}"
    )
    logger.info(f"Location: {CHROMA_DIR}")
    if args.rebuild_from_cache:
        logger.info("Mode:     rebuilt from local cache (no internet)")
    logger.info("=" * 60)
    logger.info("\nNext step: python -m src.pipeline --vram-check")


if __name__ == "__main__":
    main()
