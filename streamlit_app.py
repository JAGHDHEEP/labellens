"""LabelLens hosted demo (Streamlit Community Cloud).

Same pipeline as the full app - entity scoping, hybrid BM25 + dense retrieval,
RRF, cross-encoder rerank, abstention gates and grounding checks - with the
model providers swapped for ones that fit a free CPU host:

* embeddings: snowflake-arctic-embed-s (33M) on CPU via ONNX (full app: bge-m3 on GPU)
* answers:    Groq API (full app: a local model through Ollama)
* index:      compact copy with float16 vectors, downloaded on first start

Secrets (Streamlit "Settings > Secrets"):
    GROQ_API_KEY         required
    LABELLENS_INDEX_URL  URL of labellens-compact.db.gz (a GitHub release asset)
    LABELLENS_INDEX_SHA256  optional checksum of that file
"""

from __future__ import annotations

import gzip
import hashlib
import html
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="LabelLens", page_icon="💊", layout="wide")

# Root-level Streamlit secrets -> environment, where the providers look for them.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str | int | float):
            os.environ.setdefault(_key, str(_value))
except FileNotFoundError:  # no secrets file when run locally without one
    pass

from labellens.config import Settings  # noqa: E402
from labellens.providers import ProviderError  # noqa: E402
from labellens.service import InvalidQuestion  # noqa: E402

INDEX_DIR = Path("data/deploy")
INDEX_NAME = "labellens-compact.db"
QUESTIONS_PER_MINUTE = int(os.environ.get("LABELLENS_RATE_LIMIT_PER_MINUTE", "5"))

EXAMPLES = [
    "What are the contraindications of metformin?",
    "Can warfarin be taken with amiodarone?",
    "What is the boxed warning for clozapine?",
    "How should apixaban be dosed in renal impairment?",
    "What is the retail price of atorvastatin?",
]

STATUS_LABEL = {
    "answered": ("Answered", "green"),
    "abstained_retrieval": ("No evidence found", "orange"),
    "abstained_llm": ("Not in the label", "orange"),
    "blocked_ungrounded": ("Withheld: failed verification", "red"),
}


def hosted_settings() -> Settings:
    defaults = dict(
        data_dir=INDEX_DIR, db_name=INDEX_NAME,
        embed_provider="fastembed", embed_model="snowflake/snowflake-arctic-embed-s",
        embed_query_prefix="Represent this sentence for searching relevant passages: ",
        llm_provider="groq", llm_model="openai/gpt-oss-20b", llm_timeout_s=60.0,
        rerank_n=20,  # CPU budget: 20 cross-encoder pairs instead of 30
    )
    # Environment (LABELLENS_*) still wins over these hosted defaults.
    env = Settings.from_env()
    base = Settings()
    overrides = {k: v for k, v in defaults.items() if getattr(env, k) == getattr(base, k)}
    return env.with_(**overrides)


def ensure_index(settings: Settings) -> None:
    """Download and unpack the compact index once per container."""
    if settings.db_path.exists():
        return
    url = os.environ.get("LABELLENS_INDEX_URL", "")
    if not url:
        st.error("The index is not configured. Set LABELLENS_INDEX_URL in the app's secrets.")
        st.stop()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    gz = settings.db_path.with_name(settings.db_path.name + ".gz.part")
    with st.status("First start: downloading the label index (one time, about a minute)…",
                   expanded=False) as status:
        with urllib.request.urlopen(url, timeout=120) as resp, gz.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        expected = os.environ.get("LABELLENS_INDEX_SHA256", "").strip().lower()
        if expected and hashlib.sha256(gz.read_bytes()).hexdigest() != expected:
            gz.unlink(missing_ok=True)
            status.update(label="Index download failed its checksum.", state="error")
            st.stop()
        tmp = settings.db_path.with_suffix(".db.part")
        with gzip.open(gz, "rb") as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        tmp.replace(settings.db_path)  # atomic: a crash never leaves a half-written index
        gz.unlink(missing_ok=True)
        status.update(label="Index ready.", state="complete")


@st.cache_resource(show_spinner="Loading models and the label index…")
def load_service():
    from labellens.bootstrap import build_service

    settings = hosted_settings()
    ensure_index(settings)
    return build_service(settings)


def allow_question() -> bool:
    """Per-session rate limit: the demo spends a shared, rate-limited API key."""
    now = time.time()
    recent = [t for t in st.session_state.get("asked_at", []) if now - t < 60]
    if len(recent) >= QUESTIONS_PER_MINUTE:
        st.warning(f"Demo limit: {QUESTIONS_PER_MINUTE} questions per minute. "
                   f"Try again in {int(60 - (now - recent[0])) + 1} s.")
        return False
    st.session_state["asked_at"] = [*recent, now]
    return True


def render_answer(text: str) -> str:
    safe = html.escape(text)
    return re.sub(r"\[(\d{1,2})\]", r"<sup><b>[\1]</b></sup>", safe).replace("\n", "<br>")


def render_sources(slot, sources: list[dict], final: bool) -> None:
    with slot.container():
        if not sources:
            st.caption("No passages retrieved.")
            return
        for s in sources:
            mark = "✅ cited · " if final and s["cited"] else ""
            title = f"[{s['n']}] {s['drug']} — {s['section']}"
            with st.expander(title, expanded=final and s["cited"]):
                if s.get("subsection"):
                    st.caption(s["subsection"])
                st.write(s["text"])
                rerank = "–" if s["score"] is None else s["score"]
                st.caption(f"{mark}rerank {rerank} · BM25 #{s['bm25_rank'] or '–'} · "
                           f"dense #{s['dense_rank'] or '–'} · "
                           f"[full label]({s['url']})")


def main() -> None:
    st.title("LabelLens")
    st.markdown("Answers from official **FDA drug labels**, with a citation for every claim "
                "— or an honest *not in the label*.")

    with st.sidebar:
        st.subheader("About this demo")
        st.markdown(
            "Hybrid BM25 + dense retrieval over **1,200 FDA labels (73k passages)**, drug-name "
            "scoping, cross-encoder reranking, a refusal gate, and automatic checks that every "
            "cited number appears in its source.\n\n"
            "This free CPU deployment uses a small embedding model and the Groq API. "
            "The full version (bge-m3 on GPU, local LLM) is in the repository.")
        st.caption("Reference tool on public openFDA data. **Not medical advice.**")

    if "question" not in st.session_state:
        st.session_state["question"] = ""
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            st.session_state["question"] = ex
            st.session_state["submit"] = True

    with st.form("ask", clear_on_submit=False):
        question = st.text_input("Question", key="question", max_chars=600,
                                 placeholder="e.g. What are the contraindications of metformin?")
        submitted = st.form_submit_button("Ask", type="primary")
    submitted = submitted or st.session_state.pop("submit", False)
    if not submitted or not question.strip():
        return
    if not allow_question():
        return

    service = load_service()
    left, right = st.columns([5, 4], gap="large")
    left.subheader("Answer")
    answer_slot = left.empty()
    meta_slot = left.empty()
    right.subheader("Evidence")
    sources_slot = right.empty()

    text, final = "", None
    try:
        with left, st.spinner("Searching labels…"):
            events = service.ask_stream(question)
            for event, payload in events:
                if event == "sources":
                    render_sources(sources_slot, payload, final=False)
                    break
        for event, payload in events:
            if event == "token":
                text += payload
                answer_slot.markdown(render_answer(text), unsafe_allow_html=True)
            elif event == "final":
                final = payload
    except InvalidQuestion as exc:
        answer_slot.warning(str(exc))
        return
    except ProviderError:
        answer_slot.error("The answer model is temporarily unavailable (rate limit or outage). "
                          "Please try again in a minute.")
        sources_slot.empty()
        return

    answer_slot.markdown(render_answer(final.text), unsafe_allow_html=True)
    render_sources(sources_slot, final.sources, final=final.status == "answered")
    label, color = STATUS_LABEL.get(final.status, (final.status, "gray"))
    badges = [f":{color}-badge[{label}]"]
    g = final.grounding
    if final.status == "answered":
        badges.append(":green-badge[Grounding checks passed]" if g.get("ok")
                      else ":red-badge[Grounding check failed]")
        badges.append(f":blue-badge[{round(g.get('citation_coverage', 0) * 100)}% sentences cited]")
        if g.get("unsupported_numbers"):
            badges.append(f":red-badge[Unverified numbers: {', '.join(g['unsupported_numbers'])}]")
        if g.get("truncated"):
            badges.append(":red-badge[Answer cut off]")
    badges += [f":gray-badge[Scope: {d}]" for d in final.scoped_drugs[:3]]
    if final.cached:
        badges.append(":gray-badge[cached]")
    t = final.timings_ms
    timing = " · ".join(f"{k.replace('_', ' ')} {round(t[k])} ms"
                        for k in ("retrieval", "rerank", "first_token", "total") if k in t)
    meta_slot.markdown(" ".join(badges) + f"\n\n<small>{timing} · {final.model}</small>",
                       unsafe_allow_html=True)


main()
