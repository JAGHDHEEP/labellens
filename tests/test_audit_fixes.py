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


def test_restated_highlight_sentence_is_collapsed_for_the_model():
    from labellens.generation import collapse_restated_sentences

    text = ("Metformin extended-release tablets are indicated as an adjunct to diet and exercise "
            "to improve glycemic control in adults with type 2 diabetes mellitus. Metformin "
            "extended-release tablets are a biguanide indicated as an adjunct to diet and exercise "
            "to improve glycemic control in adults with type 2 diabetes mellitus.")
    out = collapse_restated_sentences(text)
    assert out.count("indicated as an adjunct") == 1 and "biguanide" in out


def test_sentences_with_different_numbers_are_never_collapsed():
    from labellens.generation import collapse_restated_sentences

    text = ("The recommended starting dose is 500 mg orally twice a day with meals. "
            "The recommended starting dose is 850 mg orally twice a day with meals and water.")
    assert collapse_restated_sentences(text) == text


def test_service_embeds_questions_on_cpu_but_ingest_keeps_the_gpu(settings):
    import json as _json

    import httpx

    from labellens.providers import build_embedder

    sent = []

    def handler(req):
        sent.append(_json.loads(req.content))
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    for for_queries in (True, False):
        e = build_embedder(settings.with_(embed_provider="ollama"), for_queries=for_queries)
        e._http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")
        e.embed(["what is the dose?"])
    assert sent[0]["options"] == {"num_gpu": 0}  # query path: CPU, the LLM keeps the GPU
    assert "options" not in sent[1]  # ingest path: Ollama's default (GPU)


PARTIAL = ("Metformin is contraindicated in severe renal impairment (eGFR below 30 mL/min/1.73 m2) [1].\n"
           "INSUFFICIENT_EVIDENCE: the sources do not cover use with iodinated contrast.")


def test_cited_answer_followed_by_a_gap_note_is_a_partial_answer_not_a_refusal():
    from labellens.generation import finalize_partial_answer, is_abstention

    assert not is_abstention(PARTIAL)
    out = finalize_partial_answer(PARTIAL)
    assert out.startswith("Metformin is contraindicated") and "INSUFFICIENT_EVIDENCE" not in out
    assert out.endswith("Not covered by the sources: the sources do not cover use with iodinated contrast.")
    assert is_abstention("INSUFFICIENT_EVIDENCE: no price is listed.")
    assert is_abstention("I am sorry.\nINSUFFICIENT_EVIDENCE: no price is listed.")


def test_service_keeps_the_partial_answer_and_checks_it(retriever, settings):
    hits = retriever.retrieve("What are the contraindications of metformin?").hits
    n = next(i for i, h in enumerate(hits, 1) if "severe renal" in h.chunk.text)
    svc = RagService(retriever, FakeLLM(PARTIAL.replace("[1]", f"[{n}]")), settings)
    ans = svc.ask("What are the contraindications of metformin?")
    assert ans.status == "answered" and ans.grounding["ok"]
    assert "Not covered by the sources" in ans.text and "INSUFFICIENT" not in ans.text


def test_stream_never_previews_the_sentinel_line(retriever, settings):
    from labellens.generation import stream_safe_prefix

    assert stream_safe_prefix("Dose is 500 mg [1].\nINSUFF") == "Dose is 500 mg [1].\n"
    assert stream_safe_prefix("Dose is 500 mg [1].\n**INSUFFICIENT_EVIDENCE: x") == "Dose is 500 mg [1].\n"
    assert stream_safe_prefix("Dose is 500 mg [1].\nIt is taken") == "Dose is 500 mg [1].\nIt is taken"
    svc = RagService(retriever, FakeLLM(PARTIAL), settings)
    events = list(svc.ask_stream("What are the contraindications of metformin?"))
    shown = "".join(p for e, p in events if e == "token")
    assert "INSUFFICIENT" not in shown and events[-1][1].status == "answered"
    assert events[-1][1].text == shown


def test_number_glued_to_a_word_in_the_label_still_counts_as_evidence(retriever):
    from labellens.generation import _digits

    assert _digits("not recommended in patients younger than18 years") == \
        "not recommended in patients younger than 18 years"
    assert _digits("inhibitor of CYP2C9 and MATE2-K") == "inhibitor of CYP2C9 and MATE2-K"
