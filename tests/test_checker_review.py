"""Regression tests from the independent review of the grounding-checker rules
added after the live check (table "%" captions, label numbers, gap notes)."""

from labellens.generation import finalize_partial_answer, verify_grounding
from labellens.models import Chunk, Hit

AE_TABLE = ("Adverse Reaction | Placebo (N=262) | Drugx 10 mg (N=525) | Drugx 20 mg (N=530)\n"
            "Nausea | 6.1 | 15.8 | 20.3\n"
            "Diarrhea | 3.2 | 7.4 | 9.9\n"
            "Headache | 5.0 | 6.2 | 7.1")


def _hits(text: str, kind: str = "text", caption: str = "") -> list[Hit]:
    c = Chunk(chunk_id="x", doc_id="d", section_key="s", section_title="S", subsection=caption,
              ordinal=0, kind=kind, text=text, header="", text_hash="x", n_words=len(text.split()))
    return [Hit(rowid=0, chunk=c, drug="Drugx")]


def test_percent_from_a_table_caption_supports_only_the_matching_cell():
    hits = _hits(AE_TABLE, kind="table", caption="Table 1: Adverse Reactions (%)")

    def ok(answer: str) -> bool:
        return verify_grounding(answer, hits).ok

    assert ok("Nausea occurred in 15.8% of patients on drugx 10 mg [1].")
    assert ok("Headache: 5% on placebo [1].")  # 5 is the cell value 5.0
    assert not ok("Nausea occurred in 20% of patients taking drugx 20 mg [1].")  # a dose header
    assert not ok("Diarrhea occurred in 15.8% of patients [1].")  # Nausea's value: row swap
    assert not ok("Nausea: 262% [1].")  # N=262 is a count, not a percentage


def test_a_stray_percent_in_prose_does_not_make_numbers_percentages():
    prose = _hits("CrCl 30 to 50: 250 mg q12h. Dilute in 0.9% Sodium Chloride.")
    assert not verify_grounding("For CrCl 30 to 50, reduce the dose by 50% [1].", prose).ok


def test_trailing_zero_matches_and_dose_units_after_day_words_are_kept():
    assert verify_grounding("Half-life is 47.8 (13.7) hours [1].",
                            _hits("t ½ (h) | 47.8 (13.70)", kind="table")).ok
    src = _hits("Take 500 mcg once a day. The recommended dose is 4 mg once daily.")
    assert verify_grounding("Take once a day 500 mg [1].", src).unsupported_numbers == ["500 mg"]
    assert verify_grounding("Maximum per day 4 g [1].", src).unsupported_numbers == ["4 g"]
    weeks = _hits("Week 0: 100 mg | Week 2: 50 mg", kind="table")
    assert verify_grounding("Week 2 dose: 50 mg [1].", weeks).ok


def test_gap_note_removal_keeps_list_shape():
    numbered = ("Contraindications:\n1. Severe renal impairment [1].\n2. Metabolic acidosis [1].\n"
                "3. The sources do not mention use with alcohol.")
    assert finalize_partial_answer(numbered) == (
        "Contraindications:\n1. Severe renal impairment [1].\n2. Metabolic acidosis [1].")


def test_gap_note_removal_keeps_lines_that_carry_content():
    listed = "- Nausea [1]\n- Diarrhea [1]\n- Lactic acidosis (no information on incidence)"
    assert finalize_partial_answer(listed) == listed
    caution = ("Dose: 500 mg twice daily [1].\n"
               "There are insufficient data over 80 years; monitor renal function.")
    assert finalize_partial_answer(caution) == caution


def test_sentinel_line_of_a_partial_answer_is_dropped():
    assert finalize_partial_answer("Dose is 20 mg [1].\nINSUFFICIENT_EVIDENCE: no dialysis data.") \
        == "Dose is 20 mg [1]."
