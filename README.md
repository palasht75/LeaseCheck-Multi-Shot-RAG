# LeaseCheck-Multi-Shot-RAG

Detect ⚖️ illegal or unfair clauses in Canadian residential leases — powered by Retrieval Augmented Generation (RAG), multi-shot prompting, and a lean Python tool.  
This repo demonstrates end-to-end modern LLM engineering for recruiters.

---

## 🛑 Problem

Canadian tenancy laws vary by province. Leases often include clauses that are void or illegal under local regulations.

> **Examples:**  
> - Ontario prohibits no‑pet clauses under **RTA s.14**  
> - British Columbia caps late fees at **$25**  

Tenants need an easy chat‑based tool to verify clause legality against the current statutes.

---

## 🧩 Solution Overview

**LeaseCheck‑Multi‑Shot‑RAG** combines a small Python tool with GPT‑4o mini to deliver a fast, accurate clause auditor:

1. **Multi‑shot prompting** — teaches GPT‑4o when to call our tool  
2. **Function calling** — invokes `check_legality(clause, province)` automatically  
3. **Retrieval‑Augmented Generation** — uses a FAISS vector database of statutes  
    - Regex shortcuts catch common cases instantly  
    - Semantic search finds the most relevant statute snippet  
4. **Two‑pass loop**  
    - **Pass 1:** GPT‑4o decides whether to call the tool  
    - **Pass 2:** after injecting the tool’s JSON result, GPT‑4o streams a final, cited explanation  

---

## 🗂️ Repo Structure

| Layer             | Path                                         | Description                                 |
|-------------------|----------------------------------------------|---------------------------------------------|
| **UI**            | `leasecheck/ui/streamlit_app.py`             | Streamlit chat app with GPT‑4o mini streaming and function calling  |
| **Tool + RAG**    | `leasecheck/tools/legality.py`               | `check_legality()` tool: regex shortcuts + FAISS lookup              |
| **Ingestion**     | `leasecheck/ingest/make_index.py`            | Scrape HTML/PDF → clean → chunk → embed → build FAISS index         |
| **CI**            | `.github/workflows/rebuild_vectors.yml`      | Weekly GitHub Action to rebuild vector DB and open PR if changed    |
| **Config & Tests**| `pyproject.toml`, `ruff.toml`, `black.toml`  | Poetry setup, linting, formatting; add `tests/` for your unit tests |

---

## 🏗️ Reproduce Locally

```bash
# 1. Clone and install dependencies
poetry install    # sets up Python 3.11 env and installs all packages

# 2. Add your OpenAI key
echo "openai_api_key=sk‑YOUR_KEY" > .env

# 3. Build the vector database (first run ~4 minutes)
poetry run python -m leasecheck.ingest.make_index `
    --raw   data/statutes_raw `
    --clean data/statutes_clean `
    --dst   leasecheck/tools `
    --provinces ON BC

# 4. Launch the Streamlit chat
poetry run streamlit run leasecheck/ui/streamlit_app.py
```

---

## 🛠️ Using `check_legality`

You can call the legality-check tool directly from Python.

**Python example to test `check_legality()`:**

```python
from leasecheck.tools.legality import check_legality
# Check a no‑pets clause in Ontario
result = check_legality(
    clause="The tenant agrees to keep the premises free of pets of any kind.",
    province="ON"
)
print(result)
# => {'legal': False, 'reason': 'Ontario RTA s.14 voids no‑pets clauses.', 'source_id': 'ON_RTA_014'}
```

Paste a clause, select a province, and watch GPT-4o mini stream a verdict with ✅ or ❌ plus citations.

---

## 🚀 Shipping and CI

A weekly GitHub Action downloads updated statutes, rebuilds embeddings, and opens a PR if the vector index changed.

---

## 📈 Why Recruiters Will Notice

Modern stack: GPT-4o function calling, FAISS RAG, Streamlit streaming, Poetry, Ruff, GitHub Actions

---

## 📜 License

MIT for code. Statute texts remain public domain under their respective governments.