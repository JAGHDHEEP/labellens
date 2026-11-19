"""The hosted (Streamlit) configuration: Groq provider, compact index, rate limiting."""

import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from labellens.api import create_app
from labellens.ingest import embed_missing
from labellens.providers import TRUNCATED, GroqLLM, HashEmbedder, OverlapReranker, ProviderError
from labellens.retrieval import Retriever
from labellens.service import RagService
from labellens.store import Store


def _sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def _groq(handler) -> GroqLLM:
    return GroqLLM(api_key="test-key", transport=httpx.MockTransport(handler))


def test_groq_streams_answer_text_only_and_flags_truncation():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, content=_sse(
            {"choices": [{"delta": {"reasoning": "thinking..."}}]},  # never shown
            {"choices": [{"delta": {"content": "Metformin is "}}]},
            {"choices": [{"delta": {"content": "indicated [1]."}, "finish_reason": "length"}]},
        ))

    pieces = list(_groq(handler).stream("sys", "user", max_tokens=100))
    assert pieces == ["Metformin is ", "indicated [1].", TRUNCATED]
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["reasoning_effort"] == "low" and seen["body"]["max_tokens"] == 400
    assert seen["body"]["temperature"] == 0 and seen["body"]["stream"] is True


@pytest.mark.parametrize("status", [429, 500])
def test_groq_errors_become_provider_errors(status):
    llm = _groq(lambda r: httpx.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(ProviderError):
        list(llm.stream("sys", "user"))


def test_groq_requires_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        GroqLLM()


def test_compact_export_keeps_everything_and_stores_half_precision(indexed, tmp_path, settings):
    store, _ = indexed
    dest = tmp_path / "compact.db"
    store.export_compact(dest)
    compact = Store(dest)
    assert compact.stats()["chunks"] == store.stats()["chunks"]
    assert compact.aliases() == store.aliases()
    assert compact.term_exists("coumadin", exact=True)
    assert compact.vec_dtype == np.float16 and compact.stats()["embeddings"] == 0

    emb = HashEmbedder()
    embed_missing(compact, emb)
    vec = compact._db.execute("SELECT vec FROM embeddings LIMIT 1").fetchone()[0]
    assert len(vec) == emb.dim * 2  # float16 on disk
    dense = compact.load_dense(emb.model)
    assert dense.matrix.dtype == np.float32 and len(dense) == compact.stats()["chunks"]

    r = Retriever(compact, emb, OverlapReranker(), settings)
    res = r.retrieve("What are the contraindications of Coumadin?")
    assert res.scoped_drugs == ["Warfarin Sodium (Coumadin)"] and res.hits


def test_rate_limit_per_client(retriever, settings):
    from labellens.providers import FakeLLM

    svc = RagService(retriever, FakeLLM(), settings.with_(rate_limit_per_minute=2))
    with TestClient(create_app(service=svc)) as c:
        q = {"question": "What is the starting dose of metformin?"}
        a = {"x-forwarded-for": "1.1.1.1"}
        assert c.post("/v1/ask", json=q, headers=a).status_code == 200
        assert c.post("/v1/ask", json=q, headers=a).status_code == 200
        blocked = c.post("/v1/ask", json=q, headers=a)
        assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) > 0
        # another visitor is unaffected, and so is the non-question API
        assert c.post("/v1/ask", json=q, headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200
        assert c.get("/healthz").status_code == 200
