"""Regressions found by running an outside reviewer's question set against the hosted demo."""

from labellens.generation import normalize_citations, verify_grounding
from labellens.ingest import embed_missing, ingest_records
from labellens.providers import FakeLLM
from labellens.retrieval import balance_entities
from labellens.service import RagService

from .conftest import make_label


def test_lenticular_and_fullwidth_citations_are_understood(retriever):
    # gpt-oss-20b cited as 【1】; the checker saw no citations and failed a correct answer.
    assert normalize_citations("Contraindicated in severe renal impairment【1】【2】.") == \
        "Contraindicated in severe renal impairment[1][2]."
    assert normalize_citations("Monitor INR ［1, 2］") == "Monitor INR [1][2]"
    hits = retriever.retrieve("What are the contraindications of metformin?").hits
    n = next(i for i, h in enumerate(hits, 1) if "metabolic acidosis" in h.chunk.text)
    g = verify_grounding(f"- Acute or chronic metabolic acidosis【{n}】", hits)
    assert g.ok and g.cited_sources == [n]


def test_displayed_answer_uses_canonical_citations(retriever, settings):
    svc = RagService(retriever, FakeLLM("Metformin is contraindicated in severe renal impairment【1】."),
                     settings)
    ans = svc.ask("What are the contraindications of metformin?")
    assert "[1]" in ans.text and "【" not in ans.text and ans.sources[0]["cited"]


def test_each_named_drug_gets_evidence_slots(retriever):
    res = retriever.retrieve("Compare the contraindications of metformin and warfarin")
    drugs = {h.chunk.doc_id for h in res.hits}
    assert {"set-metformin", "set-warfarin"} <= drugs


def test_balance_entities_round_robins_and_keeps_rank_order(retriever):
    hits = retriever.retrieve("metformin warfarin contraindications dose").hits
    a = [h for h in hits if h.chunk.doc_id == "set-metformin"]
    b = [h for h in hits if h.chunk.doc_id == "set-warfarin"]
    ranked = a + b  # every metformin passage outranks every warfarin one
    out = balance_entities(ranked, [frozenset({"set-metformin"}), frozenset({"set-warfarin"})], 2)
    assert [h.chunk.doc_id for h in out] == ["set-metformin", "set-warfarin"]
    assert balance_entities(ranked, [frozenset({"set-metformin"})], 2) == ranked[:2]


def test_safety_question_pins_the_boxed_warning(indexed, settings):
    store, embedder = indexed
    ingest_records(store, [make_label(
        "set-clozo", "clozopine", "Clozex",
        boxed_warning="WARNING: SEVERE NEUTROPENIA Clozopine can cause severe neutropenia, "
                      "which can lead to serious and fatal infections.",
        pediatric_use="8.4 Pediatric Use Safety and effectiveness of clozopine in pediatric "
                      "patients have not been established. Is clozopine safe in children? Unknown.",
        geriatric_use="8.5 Geriatric Use Clinical studies did not include sufficient numbers of "
                      "subjects aged 65 and over to determine whether clozopine is safe for them.",
        dosage_and_administration="2 DOSAGE AND ADMINISTRATION Start clozopine at 12.5 mg once "
                                  "or twice daily. Safety monitoring is required.",
    )], settings)
    embed_missing(store, embedder)
    from labellens.providers import OverlapReranker
    from labellens.retrieval import Retriever
    r = Retriever(store, embedder, OverlapReranker(), settings.with_(final_k=2))
    hits = r.retrieve("Is clozopine safe?").hits
    assert any(h.chunk.section_key == "boxed_warning" for h in hits)


def test_question_about_an_unnamed_drug_is_refused_without_calling_the_llm(service):
    ans = service.ask("Can I take this drug with my other medication?")
    assert ans.status == "abstained_retrieval" and "Which drug" in ans.text
    assert service.llm.calls == [] and ans.sources == []


def test_naming_the_drug_lifts_the_unnamed_drug_gate(service):
    ans = service.ask("Can I take metformin with my other medication?")
    assert ans.status == "answered"


def test_unnamed_drug_stream_shows_no_unrelated_evidence(service):
    events = list(service.ask_stream("Can I take this drug with my other medication?"))
    assert events[0] == ("sources", [])
    assert events[-1][1].status == "abstained_retrieval" and events[-1][1].sources == []
