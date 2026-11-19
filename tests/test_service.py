import json

import pytest
from fastapi.testclient import TestClient

from labellens.api import create_app
from labellens.generation import build_user_prompt, is_abstention, verify_grounding
from labellens.providers import FakeLLM
from labellens.service import InvalidQuestion, RagService

Q = "What is the starting dose of metformin for adults?"


def _hits(retriever):
    return retriever.retrieve(Q).hits


# ------------------------------------------------------------------ grounding
def test_grounded_answer_passes(retriever):
    hits = _hits(retriever)
    n = next(i for i, h in enumerate(hits, 1) if "500 mg" in h.chunk.text)
    g = verify_grounding(f"The recommended starting dose is 500 mg orally twice a day. [{n}]", hits)
    assert g.ok and g.cited_sources == [n] and g.citation_coverage == 1.0


def test_hallucinated_number_is_caught(retriever):
    hits = _hits(retriever)
    n = next(i for i, h in enumerate(hits, 1) if "500 mg" in h.chunk.text)
    g = verify_grounding(f"The recommended starting dose is 850 mg orally twice a day. [{n}]", hits)
    assert not g.ok and g.unsupported_numbers == ["850 mg"]


def test_number_must_come_from_the_cited_source_not_any_source(retriever):
    hits = _hits(retriever)
    wrong = next(i for i, h in enumerate(hits, 1) if "2550" not in h.chunk.text)
    g = verify_grounding(f"The maximum dose is 2550 mg per day in adults. [{wrong}]", hits)
    assert "2550 mg" in g.unsupported_numbers


def test_invalid_citation_is_caught(retriever):
    g = verify_grounding("Metformin improves glycemic control in adults. [9]", _hits(retriever))
    assert not g.ok and g.invalid_citations == [9]


def test_uncited_answer_fails_and_coverage_is_reported(retriever):
    g = verify_grounding("Metformin improves glycemic control in adults.", _hits(retriever))
    assert not g.ok and g.citation_coverage == 0.0


def test_partial_number_is_not_a_match(retriever):
    hits = _hits(retriever)
    n = next(i for i, h in enumerate(hits, 1) if "2550" in h.chunk.text)
    g = verify_grounding(f"The maximum daily dose is 255 mg for all adults. [{n}]", hits)
    assert g.unsupported_numbers == ["255 mg"]


def test_abstention_marker_and_prompt_shape(retriever):
    assert is_abstention("INSUFFICIENT_EVIDENCE: the sources do not mention price.")
    assert is_abstention("**INSUFFICIENT_EVIDENCE**: nothing")
    assert not is_abstention("Metformin is indicated for type 2 diabetes. [1]")
    prompt = build_user_prompt(Q, _hits(retriever))
    assert prompt.startswith("<sources>\n[1] Metformin Hydrochloride (Glucophage) | ")
    assert prompt.rstrip().endswith(f"Question: {Q}")


# -------------------------------------------------------------------- service
def test_answer_has_sources_and_passes_checks(service):
    ans = service.ask(Q)
    assert ans.status == "answered"
    assert ans.grounding["ok"] and ans.sources[0]["cited"]
    assert ans.scoped_drugs == ["Metformin Hydrochloride (Glucophage)"]
    assert {"retrieval", "generation", "total"} <= set(ans.timings_ms)


def test_retrieval_gate_refuses_without_calling_the_llm(service):
    ans = service.ask("What is the airspeed velocity of an unladen swallow?")
    assert ans.status == "abstained_retrieval"
    assert service.llm.calls == []  # no generation tokens spent
    assert "generation" not in ans.timings_ms


def test_llm_abstention_is_surfaced(retriever, settings):
    svc = RagService(retriever, FakeLLM("INSUFFICIENT_EVIDENCE: the label does not list a price."), settings)
    ans = svc.ask(Q)
    assert ans.status == "abstained_llm" and "price" in ans.text
    assert not any(s["cited"] for s in ans.sources)


def test_strict_mode_withholds_ungrounded_answers(retriever, settings):
    bad = FakeLLM("The starting dose is 9999 mg once daily. [1]")
    lenient = RagService(retriever, bad, settings).ask(Q)
    assert lenient.status == "answered" and not lenient.grounding["ok"]
    strict = RagService(retriever, bad, settings.with_(strict_grounding=True)).ask(Q)
    assert strict.status == "blocked_ungrounded" and "9999" not in strict.text


def test_cache_hits_and_is_invalidated_by_reindex(service):
    first = service.ask(Q)
    again = service.ask("  " + Q.upper() + " ")
    assert not first.cached and again.cached and len(service.llm.calls) == 1
    service.retriever.store.bump_version()  # what an ingest job does on commit
    assert not service.ask(Q).cached and len(service.llm.calls) == 2


@pytest.mark.parametrize("bad", ["", "  ", "hi", "x" * 601])
def test_invalid_questions_are_rejected(service, bad):
    with pytest.raises(InvalidQuestion):
        service.ask(bad)


# ------------------------------------------------------------------------ api
@pytest.fixture
def client(service):
    with TestClient(create_app(service=service)) as c:
        yield c


def test_health_ready_and_metrics(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").json()["status"] == "ready"
    client.post("/v1/ask", json={"question": Q})
    body = client.get("/metrics").text
    assert 'labellens_requests_total{status="answered"}' in body
    assert "labellens_stage_seconds_bucket" in body


def test_ask_endpoint_and_request_id(client):
    r = client.post("/v1/ask", json={"question": Q}, headers={"x-request-id": "abc123"})
    assert r.status_code == 200 and r.headers["x-request-id"] == "abc123"
    data = r.json()
    assert data["status"] == "answered" and data["sources"]


def test_validation_errors_are_422(client):
    assert client.post("/v1/ask", json={"question": "a"}).status_code == 422
    assert client.post("/v1/ask", json={}).status_code == 422


def test_stream_emits_sources_tokens_then_final(client):
    with client.stream("POST", "/v1/ask/stream", json={"question": Q}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        frames = [f for f in "".join(r.iter_text()).split("\n\n") if f]
    events = [f.split("\n")[0].removeprefix("event: ") for f in frames]
    assert events[0] == "sources" and events[-1] == "final" and "token" in events
    final = json.loads(frames[-1].split("\n")[1].removeprefix("data: "))
    streamed = "".join(json.loads(f.split("\n")[1].removeprefix("data: "))
                       for f, e in zip(frames, events) if e == "token")
    assert final["text"] == streamed


def test_search_endpoint_explains_ranking(client):
    data = client.post("/v1/search", json={"question": "warfarin amiodarone interaction"}).json()
    assert data["scoped_drugs"] == ["Warfarin Sodium (Coumadin)"]
    assert {"bm25_rank", "dense_rank", "rerank_score"} <= set(data["hits"][0])


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "LabelLens" in r.text


def test_unknown_term_is_refused_before_retrieval_can_mislead(service):
    ans = service.ask("What is the recommended starting dose of Zorvanex?")
    assert ans.status == "abstained_retrieval" and '"zorvanex"' in ans.text
    assert service.llm.calls == []


def test_known_drug_with_unusual_word_is_not_refused_as_unknown(retriever):
    res = retriever.retrieve("Is metformin okay for a xylophonist with kidney problems?")
    assert res.unknown_terms == []  # a drug was linked, so the unknown-term gate stays out of it


def test_running_service_picks_up_new_labels_without_restart(service, settings):
    from labellens.ingest import embed_missing, ingest_records

    from .conftest import make_label

    q = "What is examplamine indicated for?"
    assert service.ask(q).status == "abstained_retrieval"  # not indexed yet
    store = service.retriever.store
    ingest_records(store, [make_label(
        "set-new", "examplamine", "Examplo",
        indications_and_usage="1 INDICATIONS AND USAGE Examplamine is indicated for the "
                              "treatment of seasonal allergic rhinitis in adults.")], settings)
    embed_missing(store, service.retriever.embedder)
    ans = service.ask(q)
    assert ans.status == "answered" and ans.scoped_drugs == ["Examplamine (Examplo)"]
