import numpy as np
import pytest

from labellens.normalize import normalize_label
from labellens.retrieval import EntityLinker, drug_aliases, rrf
from labellens.store import fts_query

from .conftest import METFORMIN


def test_rrf_rewards_agreement_between_arms():
    scores = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert ranked[:2] == [1, 3]  # found by both arms beats found by one
    assert scores[2] > scores[4]


def test_fts_query_is_sanitised():
    q = fts_query('metformin" OR text:* NEAR( DROP -- the dose?')
    assert '"metformin"' in q and '"dose"' in q
    assert "*" not in q and "(" not in q and "--" not in q
    assert fts_query("the of and") == ""


def test_aliases_include_brand_and_salt_stripped_generic():
    aliases = drug_aliases(normalize_label(METFORMIN))
    assert {"metformin hydrochloride", "metformin", "glucophage"} <= aliases


def test_entity_linker_longest_match_and_typos():
    linker = EntityLinker({"metformin": {"m"}, "insulin glargine": {"g"}, "insulin": {"i"},
                           "atorvastatin": {"a"}})
    assert linker.link("What is the dose of Metformin?")[0] == {"m"}
    assert linker.link("insulin glargine storage")[0] == {"g"}  # not plain insulin
    assert linker.link("can metformin be used with insulin")[0] == {"m", "i"}
    assert linker.link("atorvastatn side effects")[0] == {"a"}  # typo tolerance
    assert linker.link("what are common side effects of statins")[0] == set()


def test_bm25_respects_document_scope(indexed):
    store, _ = indexed
    assert store.bm25_search("contraindicated", 10)
    scoped = store.bm25_search("contraindicated", 10, doc_ids=["set-warfarin"])
    docs = {store.get_chunks([r])[r][0].doc_id for r, _ in scoped}
    assert docs == {"set-warfarin"}
    assert store.bm25_search("contraindicated", 10, doc_ids=[]) == []


def test_dense_search_is_exact_and_scoped(indexed):
    store, embedder = indexed
    index = store.load_dense(embedder.model)
    assert len(index) == store.stats()["chunks"]
    q = embedder.embed(["venous thrombosis pulmonary embolism prophylaxis"])[0]
    best = index.search(q, 3)
    assert store.get_chunks([best[0][0]])[best[0][0]][0].doc_id == "set-warfarin"
    assert best[0][1] == pytest.approx(float(np.max(index.matrix @ q)))
    scoped = index.search(q, 3, doc_ids=["set-metformin"])
    assert all(store.get_chunks([r])[r][0].doc_id == "set-metformin" for r, _ in scoped)


def test_retriever_scopes_to_the_named_drug(retriever):
    res = retriever.retrieve("What are the contraindications of Coumadin?")
    assert res.scoped_drugs == ["Warfarin Sodium (Coumadin)"]
    assert res.hits and all(h.chunk.doc_id == "set-warfarin" for h in res.hits)
    assert res.hits[0].chunk.section_key == "contraindications"
    assert res.top_score is not None and "rerank" in res.timings_ms


def test_ablation_switches_change_the_pipeline(retriever, settings):
    res = retriever.retrieve("contraindications", settings.with_(use_dense=False, use_rerank=False))
    assert all(h.dense_rank is None and h.rerank_score is None for h in res.hits)
    assert res.top_score is None
    res = retriever.retrieve("contraindications", settings.with_(use_bm25=False))
    assert all(h.bm25_rank is None for h in res.hits)
    unscoped = retriever.retrieve("contraindications of warfarin",
                                  settings.with_(use_entity_scope=False))
    assert unscoped.scoped_drugs == []


def test_near_duplicates_collapse_and_are_recorded(retriever):
    from labellens.retrieval import collapse_near_duplicates

    hits = retriever.retrieve("contraindications of warfarin").hits
    a = hits[0]
    twin = type(a)(rowid=999, chunk=type(a.chunk)(**{**a.chunk.__dict__, "chunk_id": "twin",
                   "text": a.chunk.text + " USP"}), drug="Other Maker")
    out = collapse_near_duplicates([a, twin, *hits[1:]], limit=3)
    assert twin not in out and "twin" in a.dup_ids
    assert len(out) <= 3 and len({h.chunk.chunk_id for h in out}) == len(out)
