# leasecheck/ingest/make_index.py
"""Ingest pipeline: download Canadian tenancy statutes (HTML **and** PDF),
clean them into plain text, embed with Sentence‑Transformers, and build a
FAISS index that `leasecheck.tools.legality` will query at runtime.

Run from repo root (inside Poetry venv):

    poetry run python -m leasecheck.ingest.make_index \
        --raw   data/statutes_raw \
        --clean data/statutes_clean \
        --dst   leasecheck/tools \
        --provinces ON BC

Re‑run whenever a law updates.  CI can schedule this weekly.  All heavy
embedding work is delegated to `_VectorStore.build_index`, keeping a
single source of truth.
"""

from __future__ import annotations

import argparse
import html
import io
import re
import sys
from pathlib import Path
from typing import Dict, List

import requests
import bs4  # graceful parser fallback
from bs4 import BeautifulSoup
import pdfplumber  # PDF → text

# local import (keeps dependency arrow pointing one way)
from leasecheck.tools.legality import _VectorStore

USER_AGENT = "LeaseCheckBot/0.2 (+https://github.com/yourname/leasecheck)"
TIMEOUT = 45  # seconds for HTTP requests

# ──────────────────────────────────────────────────────────────────────
# 1. Province → list of primary sources
# ──────────────────────────────────────────────────────────────────────
PROVINCE_SOURCES: Dict[str, List[dict]] = {
    "ON": [
        {
            "name": "ont_rta",
            "url": "https://www.ontario.ca/laws/statute/06r17",  # Residential Tenancies Act
        }
    ],
    "BC": [
        {
            "name": "bc_rta",
            "url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/02078_01",
        },
        {
            "name": "bc_policy_37",
            "url": "https://www2.gov.bc.ca/assets/gov/housing-and-tenancy/residential-tenancies/policy-guidelines/gl37.pdf",
            "pdf": True,
        },
    ],
}

# ──────────────────────────────────────────────────────────────────────
# 2. Helper functions
# ──────────────────────────────────────────────────────────────────────


def _http_get(url: str) -> requests.Response:
    """Shared HTTP GET with UA & timeout, raises on non‑200."""
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp


def fetch_html_text(url: str) -> str:
    """Download HTML and return raw string (tags intact)."""
    return _http_get(url).text


def fetch_pdf_text(url: str) -> str:
    """Download PDF and extract its text via **pdfplumber**."""
    resp = _http_get(url)
    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages)


def clean_html(raw: str) -> str:
    """Strip tags/entities & collapse whitespace.

    Tries **lxml** first; falls back to builtin `html.parser` so Windows
    users aren’t forced to compile C extensions.
    """
    try:
        soup = BeautifulSoup(raw, "lxml")
    except bs4.FeatureNotFound:
        soup = BeautifulSoup(raw, "html.parser")
    txt = soup.get_text(separator=" ", strip=True)
    return _collapse_ws(txt)


def _collapse_ws(text: str) -> str:
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# ──────────────────────────────────────────────────────────────────────
# 3. Main routine
# ──────────────────────────────────────────────────────────────────────


def build_index(
    provinces: List[str], raw_dir: Path, clean_dir: Path, dst: Path
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    for prov in provinces:
        sources = PROVINCE_SOURCES.get(prov.upper())
        if not sources:
            print(f"[skip] No sources configured for province {prov}")
            continue

        for src in sources:
            name, url = src["name"], src["url"]
            is_pdf = src.get("pdf", False)
            print(
                f"→ Fetching {prov}/{name} ({'PDF' if is_pdf else 'HTML'}) …",
                flush=True,
            )
            try:
                raw = fetch_pdf_text(url) if is_pdf else fetch_html_text(url)
            except Exception as exc:
                print(f"[error] download failed: {exc}")
                continue

            # save raw bytes or html for audit/debug
            raw_path = raw_dir / prov / f"{name}{'.pdf' if is_pdf else '.html'}"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            if is_pdf:
                raw_path.write_bytes(_http_get(url).content)
            else:
                raw_path.write_text(raw, encoding="utf-8")

            # clean to plain text (HTML) or just collapse whitespace (PDF already text)
            cleaned = _collapse_ws(raw) if is_pdf else clean_html(raw)
            clean_path = clean_dir / prov / f"{name}.txt"
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            clean_path.write_text(cleaned, encoding="utf-8")

    # delegate heavy work to tools.legality (single canonical impl)
    print("✎ Embedding & indexing cleaned text …")
    _VectorStore.build_index(clean_dir, dst)
    print("✓ Done.  Index at", dst)


# ──────────────────────────────────────────────────────────────────────
# 4. CLI entry point
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build FAISS index from statute sources (HTML & PDF)."
    )
    ap.add_argument(
        "--raw",
        type=Path,
        default=Path("data/statutes_raw"),
        help="folder to store raw downloads",
    )
    ap.add_argument(
        "--clean",
        type=Path,
        default=Path("data/statutes_clean"),
        help="folder for cleaned .txt",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=Path("leasecheck/tools"),
        help="output folder for faiss.index & meta.pkl",
    )
    ap.add_argument(
        "--provinces",
        nargs="*",
        default=list(PROVINCE_SOURCES.keys()),
        help="province codes to ingest",
    )

    args = ap.parse_args()
    build_index(
        provinces=[p.upper() for p in args.provinces],
        raw_dir=args.raw,
        clean_dir=args.clean,
        dst=args.dst,
    )
    sys.exit(0)
