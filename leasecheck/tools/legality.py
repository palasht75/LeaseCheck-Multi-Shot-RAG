# leasecheck/tools/legality.py
"""LeaseCheck‑Canada – pure‑Python legality checker

This module provides two public entry points:

* ``build_index(src_dir: str, model_name: str = DEFAULT_MODEL)`` – run once
  to embed all statute text files under *src_dir* and write the FAISS
  binary index + a metadata pickle next to this file.
* ``check_legality(clause: str, province: str)`` – fast runtime helper
  for the LLM tool‑call.  It lazily loads the FAISS index, searches for
  the most relevant statute chunks, applies a minimal heuristic, and
  returns the structure required by the multi‑shot prompt.

To keep the demo lightweight we also hard‑code a few regex shortcuts for
well‑known illegal terms ("no pets" in Ontario, etc.).  They give instant
answers without touching the vector store while more exotic clauses fall
back to the embedding search.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "faiss.index"
META_FILE = "meta.pkl"

# Province‑specific regex shortcuts → (pattern, legal?, source_id, reason)
_REGEX_RULES: Dict[str, List[Tuple[str, bool, str, str]]] = {
    "ON": [
        (r"\bno\s+pets?\b", False, "ON_RTA_014", "Ontario RTA s.14 voids no‑pets clauses."),
        (
            r"post[\- ]dated\s+cheques?",
            False,
            "ON_RTA_013",
            "Landlords may accept but cannot require post‑dated cheques (s.13).",
        ),
    ],
    "BC": [
        (
            r"\blate\s+fee\b",
            False,
            "BC_PG_37",
            "Late fee > $25 exceeds BC Tenancy Policy Guideline 37.",
        ),
    ],
}


# ──────────────────────────────────────────────────────────────────────
# Vector store helpers
# ──────────────────────────────────────────────────────────────────────
class _VectorStore:
    """Light wrapper around a FAISS flat‑IP index and parallel metadata."""

    def __init__(self, index: faiss.Index, meta: List[dict], model: SentenceTransformer):
        self.index = index
        self.meta = meta
        self.model = model

    @classmethod
    def load(cls, folder: Path, model_name: str = DEFAULT_MODEL) -> "_VectorStore":
        index = faiss.read_index(str(folder / INDEX_FILE))
        with open(folder / META_FILE, "rb") as fh:
            meta: List[dict] = pickle.load(fh)
        model = SentenceTransformer(model_name)
        return cls(index, meta, model)

    # high‑level search
    def search(self, text: str, k: int = 4) -> List[dict]:
        vec = self.model.encode([text]).astype("float32")
        D, I = self.index.search(vec, k)
        hits = []
        for idx, dist in zip(I[0], D[0]):
            if idx == -1:
                continue
            item = dict(self.meta[idx])
            item["score"] = float(dist)
            hits.append(item)
        return hits

    # utility: build a brand‑new index
    @staticmethod
    def build_index(src_dir: Path, dst_folder: Path, model_name: str = DEFAULT_MODEL) -> None:
        """Embed every *.txt file under *src_dir* and write index + meta."""
        model = SentenceTransformer(model_name)
        texts: List[str] = []
        meta: List[dict] = []
        for txt_file in src_dir.rglob("*.txt"):
            province = txt_file.parent.name.upper()
            with open(txt_file, "r", encoding="utf-8") as fh:
                raw = fh.read()
            # naive sentence split – later replace with NLTK or spaCy if needed
            chunks = [raw[i : i + 600] for i in range(0, len(raw), 600)]
            for n, chunk in enumerate(chunks):
                texts.append(chunk)
                meta.append({"id": f"{province}_{txt_file.stem}_{n:03}", "province": province, "text": chunk})

        vecs = model.encode(texts, show_progress_bar=True).astype("float32")
        idx = faiss.IndexFlatIP(vecs.shape[1])
        faiss.normalize_L2(vecs)
        idx.add(vecs)

        dst_folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(idx, str(dst_folder / INDEX_FILE))
        with open(dst_folder / META_FILE, "wb") as fh:
            pickle.dump(meta, fh)


# global lazy‑loaded store
_STORE: _VectorStore | None = None


def _store() -> _VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = _VectorStore.load(Path(__file__).parent)
    return _STORE


# ──────────────────────────────────────────────────────────────────────
# Public API: check_legality
# ──────────────────────────────────────────────────────────────────────

def _regex_first(clause: str, province: str) -> Tuple[bool | None, str | None, str | None]:
    """Quick pattern pass for common clauses."""
    for pattern, legal, sid, reason in _REGEX_RULES.get(province, []):
        if re.search(pattern, clause, flags=re.I):
            return legal, reason, sid
    return None, None, None


def check_legality(clause: str, province: str, *, top_k: int = 4) -> dict:
    """Assess a lease clause.

    Parameters
    ----------
    clause : str
        Raw lease term.
    province : str
        Two‑letter code (ON, BC, …)

    Returns
    -------
    dict with keys ``legal`` (bool), ``reason`` (str), ``source_id`` (str)
    """
    province = province.upper()
    clause = clause.strip()

    # 1) cheap regex shortcuts – no embedding call needed
    legal, reason, sid = _regex_first(clause, province)
    if legal is not None:
        return {"legal": legal, "reason": reason, "source_id": sid}

    # 2) semantic search over statute chunks
    hits = _store().search(f"{province} {clause}", k=top_k)
    if not hits:
        return {
            "legal": False,
            "reason": "No matching statute found – consult your provincial tenancy board.",
            "source_id": "NONE",
        }

    best = hits[0]
    snippet = best["text"]
    sid = best["id"]

    # naive heuristic – refine later with a classifier or more shots
    illegal_markers = ["void", "not enforceable", "illegal", "prohibited"]
    legal_flag = not any(tok in snippet.lower() for tok in illegal_markers)

    return {"legal": legal_flag, "reason": snippet[:400], "source_id": sid}


# ──────────────────────────────────────────────────────────────────────
# Manual rebuild helper (CLI):  python -m leasecheck.tools.legality build data/statutes_clean
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "build":
        src = Path(sys.argv[2])
        dst = Path(__file__).parent
        print(f"[leasecheck] building vector index from {src} → {dst} …")
        _VectorStore.build_index(src, dst)
        print("done.")
    else:
        print("Usage: python -m leasecheck.tools.legality build <statute_txt_dir>")
