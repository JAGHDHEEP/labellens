"""Regression tests for defects found in the independent code review.

Each test names the failure it pins down, so a reader can see what went wrong
before and why the current behaviour is the way it is.
"""

import pytest

from labellens.chunking import chunk_document
from labellens.generation import (
    clean_abstention,
    could_be_abstention,
    is_abstention,
    normalize_citations,
    verify_grounding,
)
from labellens.ingest import embed_missing, ingest_records
from labellens.normalize import _excise_overlaps, normalize_label
from labellens.providers import TRUNCATED, FakeLLM, HashEmbedder, OverlapReranker, ProviderError
from labellens.retrieval import EntityLinker, Retriever, drug_aliases, strip_salt
from labellens.service import RagService
from labellens.store import Store

from .conftest import CORPUS, METFORMIN, make_label

Q = "What is the starting dose of metformin for adults?"


# ------------------------------------------------------------------ ingestion
def test_boxed_warning_survives_when_one_sentence_is_repeated_elsewhere():
    boxed = ("WARNING: RISK OF THYROID C-CELL TUMORS. In rodents the drug causes thyroid C-cell "
             "tumors at clinically relevant exposures. It is unknown whether it causes such "
             "tumors in humans. Contraindicated in patients with a personal or family history "
             "of medullary thyroid carcinoma.")
    wnc = ("5.1 Risk of Thyroid C-cell Tumors. In rodents the drug causes thyroid C-cell "
           "tumors at clinically relevant exposures. Counsel patients regarding the risk. "
           "5.2 Pancreatitis Acute pancreatitis has been observed in patients treated. " * 3)
    out = _excise_overlaps({"boxed_warning": boxed, "warnings_and_cautions": wnc})
    assert out["boxed_warning"] == boxed  # previously deleted by a 200-character probe


def test_short_contraindications_none_is_kept():
    doc = normalize_label(make_label(
        "s-none", "entecavir", "Baraclude",
        indications_and_usage="1 INDICATIONS AND USAGE For chronic hepatitis B.",
        contraindications="4 CONTRAINDICATIONS None."))
    keys = {c.section_key for c in chunk_document(doc)}
    assert "contraindications" in keys  # 16 % of real labels lost this section before


def test_metadata_only_revision_is_reindexed(settings):
    store = Store(":memory:")
    ingest_records(store, [METFORMIN], settings)
    renamed = {**METFORMIN, "version": "2",
               "openfda": {**METFORMIN["openfda"], "brand_name": ["Glumetza"]}}
    rep = ingest_records(store, [renamed], settings)
    assert rep.updated == 1 and "glumetza" in store.aliases()


def test_ingest_without_bump_lets_embedding_finish_first(settings):
    store = Store(":memory:")
    v0 = store.index_version
    ingest_records(store, CORPUS, settings, bump=False)
    assert store.index_version == v0  # a running server must not reload BM25-only data
    embed_missing(store, HashEmbedder())
    assert store.index_version > v0


# ------------------------------------------------------------------- linking
def test_fuzzy_match_does_not_turn_one_drug_into_another():
    aliases = {"epinephrine": {"epi"}, "metformin": {"met"}}
    corpus_words = {"norepinephrine"}  # mentioned in other labels' interaction sections
    linker = EntityLinker(aliases, term_exists=lambda t, exact=False: t in corpus_words)
    assert linker.link("what is the dose of norepinephrine")[0] == set()
    assert linker.link("metformine side effects")[0] == {"met"}  # a real typo still links


@pytest.mark.parametrize("name,base", [
    ("Clopidogrel Bisulfate", "clopidogrel"), ("Ketorolac Tromethamine", "ketorolac"),
    ("Fluticasone Propionate", "fluticasone"), ("adalimumab-bwwd", "adalimumab"),
    ("Olmesartan Medoxomil", "olmesartan"), ("Sodium Chloride", "sodium chloride"),
])
def test_salt_and_biosimilar_suffixes_are_stripped(name, base):
    assert strip_salt(name) == base


def test_common_words_never_become_scoping_aliases():
    for generic in ["dehydrated alcohol", "sodium bicarbonate", "potassium chloride"]:
        doc = normalize_label(make_label(
            "s", generic, generic.title(),
            indications_and_usage="1 INDICATIONS AND USAGE Used for testing purposes."))
        aliases = drug_aliases(doc)
        assert not aliases & {"alcohol", "bicarbonate", "chloride", "sodium", "potassium"}
        assert generic in aliases  # the full name still links


# ---------------------------------------------------------------- grounding
def test_short_bullets_are_still_checked(retriever):
    hits = retriever.retrieve(Q).hits
    g = verify_grounding("Dosing:\n- Start: 9999 mg [1]\n- Hypersensitivity [8]", hits)
    assert "9999 mg" in g.unsupported_numbers and 8 in g.invalid_citations and not g.ok


def test_grouped_citations_are_understood(retriever):
    hits = retriever.retrieve(Q).hits
    n = next(i for i, h in enumerate(hits, 1) if "500 mg" in h.chunk.text)
    for form in (f"[{n}, {n}]", f"[{n},{n}]"):
        g = verify_grounding(
            f"The recommended starting dose is 500 mg orally twice a day {form}.", hits)
        assert g.ok and g.unsupported_numbers == [], form
    assert normalize_citations("x [1-3] y [2, 4]") == "x [1][2][3] y [2][4]"


def test_small_wrong_dose_is_not_validated_by_unrelated_digits(retriever):
    hits = retriever.retrieve(Q).hits
    n = next(i for i, h in enumerate(hits, 1) if "2550" in h.chunk.text)
    g = verify_grounding(f"The maximum dose is 2 mg per day for adults [{n}].", hits)
    assert g.unsupported_numbers == ["2 mg"]  # "type 2" and "1.73 m2" used to validate it


def test_empty_answer_is_not_ok(retriever):
    assert not verify_grounding("", retriever.retrieve(Q).hits).ok


def test_abstention_detected_after_a_preamble_and_extra_text_dropped():
    reply = "I am sorry.\nINSUFFICIENT_EVIDENCE: no price is listed.\nHowever the dose is 9999 mg [1]."
    assert is_abstention(reply)
    assert clean_abstention(reply) == "no price is listed."
    assert could_be_abstention("INSUF") and could_be_abstention("**INSUFFICIENT")
    assert not could_be_abstention("Metformin is")


# ------------------------------------------------------------------- service
class ScriptedLLM(FakeLLM):
    def __init__(self, pieces):
        super().__init__()
        self.pieces = pieces

    def stream(self, system, user, max_tokens=700):
        self.calls.append((system, user))
        yield from self.pieces


def test_empty_completion_raises_and_is_not_cached(retriever, settings):
    svc = RagService(retriever, ScriptedLLM([""]), settings)
    with pytest.raises(ProviderError):
        svc.ask(Q)
    assert len(svc._cache) == 0


def test_truncated_answer_is_flagged_and_not_cached(retriever, settings):
    svc = RagService(retriever, ScriptedLLM(["The starting dose is 500 mg [1]", TRUNCATED]),
                     settings)
    ans = svc.ask(Q)
    assert ans.grounding["truncated"] and not ans.grounding["ok"]
    assert not svc.ask(Q).cached


def test_strict_mode_never_streams_the_withheld_text(retriever, settings):
    svc = RagService(retriever, ScriptedLLM(["The starting dose ", "is 9999 mg ", "[1]."]),
                     settings.with_(strict_grounding=True))
    events = list(svc.ask_stream(Q))
    streamed = "".join(p for e, p in events if e == "token")
    assert "9999" not in streamed and events[-1][1].status == "blocked_ungrounded"


def test_abstention_sentinel_is_never_streamed_raw(retriever, settings):
    svc = RagService(retriever, ScriptedLLM(["INSUFF", "ICIENT_EVIDENCE: no price ", "listed."]),
                     settings)
    events = list(svc.ask_stream(Q))
    streamed = "".join(p for e, p in events if e == "token")
    assert "INSUFFICIENT" not in streamed and events[-1][1].status == "abstained_llm"


def test_retriever_sees_reingested_data_without_restart(indexed, settings):
    store, emb = indexed
    r = Retriever(store, emb, OverlapReranker(), settings)
    ingest_records(store, [make_label(
        "set-x", "examplamine", "Examplo",
        indications_and_usage="1 INDICATIONS AND USAGE Examplamine treats allergic rhinitis.")],
        settings)
    embed_missing(store, emb)
    res = r.retrieve("examplamine indication")  # no explicit reload call
    assert res.scoped_drugs == ["Examplamine (Examplo)"]
    assert res.index_version == store.index_version


def test_stemmed_typo_still_links_and_prune_clears_vocabulary(indexed, settings):
    store, emb = indexed
    # "metformine" stems to "metformin" in FTS, but is not an exact corpus word.
    assert store.term_exists("metformine") and not store.term_exists("metformine", exact=True)
    r = Retriever(store, emb, OverlapReranker(), settings)
    assert r.retrieve("metformine renal dose").scoped_drugs == ["Metformin Hydrochloride (Glucophage)"]
    assert store.term_exists("coumadin", exact=True)
    ingest_records(store, [CORPUS[0]], settings, prune=True)
    assert not store.term_exists("coumadin", exact=True)


def test_typo_inside_multiword_name_and_salt_names_link():
    aliases = {"isosorbide dinitrate": {"iso"}, "sodium bicarbonate": {"bicarb"},
               "calcium chloride": {"cacl"}}
    linker = EntityLinker(aliases, term_exists=lambda t, exact=False: False)
    assert linker.link("daily break for isosoride dinitrate")[0] == {"iso"}
    assert linker.link("initial dose of sodium bicarbnate")[0] == {"bicarb"}
    assert linker.link("calcium chloride infiltration")[0] == {"cacl"}


def test_full_salt_names_are_aliases_but_single_salt_words_are_not():
    doc = normalize_label(make_label(
        "s", "sodium bicarbonate", "Neut",
        indications_and_usage="1 INDICATIONS AND USAGE Used for metabolic acidosis in adults."))
    aliases = drug_aliases(doc)
    assert "sodium bicarbonate" in aliases and "bicarbonate" not in aliases
