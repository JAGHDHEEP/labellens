"""Sentence-level claim verification, with a deterministic stand-in for the NLI
model (no download): a premise "entails" a hypothesis when it contains all of
the hypothesis's content words."""

import re

import numpy as np

from labellens.service import RagService
from labellens.verify import (
    NLIVerifier,
    claim_text,
    qualifier_gaps,
    split_claims,
    verify_answer,
    windows,
)

from .conftest import FakeLLM

_STOP = {"the", "a", "an", "of", "in", "is", "are", "was", "were", "to", "and", "with", "for",
         "by", "be", "at", "or", "on", "as", "it"}


class OverlapNLI:
    def scores(self, premises, hypothesis):
        want = {w for w in re.findall(r"[a-z0-9.]+", hypothesis.lower()) if w not in _STOP}
        rows = []
        for p in premises:
            have = set(re.findall(r"[a-z0-9.]+", p.lower()))
            e = 0.95 if want <= have else 0.0
            c = 0.95 if " not " in f" {p.lower()} " and e == 0 and len(want & have) > 2 else 0.0
            rows.append([e, c])
        return np.array(rows)


P = {1: ("In placebo-controlled trials of 11 AEDs, patients randomized to one of the AEDs had "
         "twice the risk of suicidal thinking. Primidone is indicated for grand mal seizures.",
         "Primidone"),
     2: ("Primidone tablets contain 50 mg or 250 mg of primidone.", "Primidone"),
     3: ("In adults, the dose is 250 mg daily.", "Primidone")}


def _verifier():
    return NLIVerifier(OverlapNLI(), entail_t=0.5, mode="s1")


def test_claims_are_split_and_citations_read():
    claims = split_claims("Dosing:\n- Adults take 250 mg daily [3].\n- Tablets contain 50 mg [2][1].")
    assert [c["cites"] for c in claims] == [[3], [1, 2]]
    assert claim_text(claims[1]["text"]) == "Tablets contain 50 mg."


def test_single_sentence_windows_and_table_rows():
    w = windows("First sentence here. Second sentence here.\nNausea | 6.1 | 15.8", mode="s1")
    assert w == ["First sentence here.", "Second sentence here.", "Nausea | 6.1 | 15.8"]
    assert windows("It is given daily.", mode="drug+s1", drug="Primidone") == ["Primidone: It is given daily."]


def test_supported_claim_passes_and_misattributed_claim_is_flagged():
    v = verify_answer("Patients randomized to one of the AEDs had twice the risk of suicidal thinking [1]. "
                      "Patients randomized to primidone had twice the risk of suicidal thinking [1].",
                      P, _verifier())
    assert [c["verdict"] for c in v.claims] == ["supported", "unsupported"]
    assert v.flagged == 1 and v.citations_added == 0


def test_a_wrong_citation_is_repaired_by_adding_the_right_one():
    v = verify_answer("Primidone tablets contain 50 mg or 250 mg of primidone [3].", P, _verifier())
    assert v.claims[0]["verdict"] == "recited" and v.claims[0]["passage"] == 2
    assert v.text == "Primidone tablets contain 50 mg or 250 mg of primidone [3][2]."
    assert v.citations_added == 1 and v.flagged == 0


def test_a_population_the_supporting_passage_never_names_is_flagged():
    assert qualifier_gaps("In pediatric patients, the dose is 250 mg daily.", P[3][0]) == ["pediatric"]
    assert qualifier_gaps("In adults, the dose is 250 mg daily.", P[3][0]) == []
    assert qualifier_gaps("In the ARISTOTLE trial, bleeding rose.", "In AVERROES, bleeding rose.") == ["ARISTOTLE"]


def test_service_reports_verification_and_checks_the_repaired_answer(retriever, settings):
    q = "What are the contraindications of metformin?"
    hits = retriever.retrieve(q).hits
    n = next(i for i, h in enumerate(hits, 1) if "severe renal" in h.chunk.text)
    sentence = next(s for s in re.split(r"(?<=\.)\s+", hits[n - 1].chunk.text) if "severe renal" in s)
    other = 1 if n != 1 else 2
    svc = RagService(retriever, FakeLLM(f"{sentence.rstrip('.')} [{other}]."), settings,
                     verifier=_verifier())
    ans = svc.ask(q)
    assert ans.status == "answered"
    assert ans.verification["citations_added"] == 1
    assert f"[{n}]" in ans.text and n in ans.grounding["cited_sources"]
    assert "verify" in ans.timings_ms
