# leasecheck/ui/streamlit_app.py
"""Streamlit UI – GPT‑4o mini + local `check_legality` function

* Uses **GPT‑4o mini** via OpenAI SDK (no legacy “tools” param).
* Reads API key from `.env` variable **`openai_api_key`**.
* Streams tokens and logs interactions to `logs/app.log`.

Quick start
===========
```bash
# .env (repo root)
openai_api_key=sk‑your‑real‑key
LOG_LEVEL=INFO  # or DEBUG

poetry run streamlit run leasecheck/ui/streamlit_app.py
```
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import openai
import streamlit as st
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionChunk

from leasecheck.tools.legality import check_legality

# ──────────────────────────────────────────────────────────────────────
# 1. Logging
# ──────────────────────────────────────────────────────────────────────
load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("leasecheck-ui")

# ──────────────────────────────────────────────────────────────────────
# 2. OpenAI client – GPT‑4o mini
# ──────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("openai_api_key") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing openai_api_key in .env")

client = openai.OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"
log.info("Model: %s", MODEL)

# ──────────────────────────────────────────────────────────────────────
# 3. Prompt skeleton
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are LeaseCheck‑Canada, an assistant that flags potentially illegal or unfair clauses in Canadian residential leases. "
    "Always cite the statute or official guideline relied upon, using Markdown links like [Ontario RTA §14](#source_ON_RTA_014). "
    "Provide plain‑language guidance — no binding legal advice."
)

# ──────────────────────────────────────────────────────────────────────
# 4. Function schema (old-style `functions` param for compatibility)
# ──────────────────────────────────────────────────────────────────────
FUNCTION_DEF = {
    "name": "check_legality",
    "description": "Assess a Canadian residential lease clause and return legality details.",
    "parameters": {
        "type": "object",
        "properties": {
            "clause": {"type": "string", "description": "Raw lease clause."},
            "province": {"type": "string", "enum": ["ON", "BC"], "description": "Province code."},
        },
        "required": ["clause", "province"],
    },
}

# ──────────────────────────────────────────────────────────────────────
# 5. Few‑shot (minimal)
# ──────────────────────────────────────────────────────────────────────
SHOT_MESSAGES: List[Dict[str, Any]] = [
    {"role": "user", "content": "Tenant must provide 12 post‑dated cheques."},
    {
        "role": "assistant",
        "content": "According to Ontario RTA s.13, landlords may accept but cannot demand post‑dated cheques. This clause is unenforceable.",
    },
]

# ──────────────────────────────────────────────────────────────────────
# 6. Chat helper – streaming + function call
# ──────────────────────────────────────────────────────────────────────

def stream_chat(messages: List[Dict[str, Any]]):
    """Generator yielding streamed GPT‑4o tokens with function execution."""

    # Pass 1 – let model decide whether to call function
    first = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        functions=[FUNCTION_DEF],
        function_call="auto",
        stream=False,
    )

    msg = first.choices[0].message
    print(f"Model response: {msg.function_call or '<no content>'}")

    if msg.function_call and msg.function_call.name == "check_legality":
        print(f"Function call: {msg.function_call.name} with args {msg.function_call.arguments}")
        args = json.loads(msg.function_call.arguments)
        log.debug("Function args: %s", args)
        result = check_legality(**args)
        log.debug("Function result: %s", result)

        messages.extend([
            {"role": "assistant", "content": None, "function_call": msg.function_call.model_dump()},
            {"role": "function", "name": "check_legality", "content": json.dumps(result)},
        ])
    else:
        # No function call – stream this response
        for tok in (msg.content or "").split():
            yield tok + " "
        return

    # Pass 2 – get final response with streamed output
    stream_iter = client.chat.completions.create(model=MODEL, messages=messages, stream=True)
    for chunk in stream_iter:  # type: ChatCompletionChunk
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token


def build_conversation(clause: str, province: str) -> List[Dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}] + SHOT_MESSAGES + [
        {"role": "user", "content": f"Clause: {clause}\nProvince: {province}"}
    ]

# ──────────────────────────────────────────────────────────────────────
# 7. Streamlit UI
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="LeaseCheck‑Canada", page_icon="📜")
st.title("📜 LeaseCheck‑Canada – AI Lease Clause Auditor (GPT‑4o mini)")

clause_in = st.text_area("Lease clause", placeholder="e.g. Tenant shall not keep pets …", height=140)
province_in = st.selectbox("Province", ["ON", "BC"], index=0)

if st.button("Analyze") and clause_in.strip():
    with st.spinner("Thinking…"):
        convo = build_conversation(clause_in.strip(), province_in)
        print(f"Conversation: {convo}")
        st.write_stream(stream_chat(convo))
