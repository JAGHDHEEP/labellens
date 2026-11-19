import math

import pytest

from labellens.evalkit.dataset import lexical_overlap, misspell, split_of, unanswerable
from labellens.evalkit.metrics import (
    auroc,
    bootstrap_ci,
    hit_at_k,
    ndcg_at_k,
    percentile,
    reciprocal_rank,
)
from labellens.evalkit.runner import _as_bool


def test_ranking_metrics():
    ranked = ["a", "b", "c", "d"]
    assert hit_at_k(ranked, {"c"}, 2) == 0.0 and hit_at_k(ranked, {"c"}, 3) == 1.0
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0
    assert ndcg_at_k(ranked, {"a"}) == 1.0
    assert ndcg_at_k(ranked, {"b"}) == pytest.approx(1 / math.log2(3))


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5 and percentile([], 50) == 0.0


def test_auroc_and_ties():
    assert auroc([3, 4], [1, 2]) == 1.0
    assert auroc([1, 2], [3, 4]) == 0.0
    assert auroc([1], [1]) == 0.5


def test_bootstrap_ci_brackets_the_mean():
    vals = [1.0] * 70 + [0.0] * 30
    lo, hi = bootstrap_ci(vals)
    assert lo < 0.7 < hi and hi - lo < 0.25


def test_split_is_deterministic_and_roughly_balanced():
    ids = [f"a{i:04d}" for i in range(400)]
    splits = [split_of(i) for i in ids]
    assert splits == [split_of(i) for i in ids]
    assert 150 < splits.count("test") < 250


def test_misspell_changes_exactly_one_character_position():
    for i in range(6):
        typo = misspell("atorvastatin", i)
        assert typo != "atorvastatin" and abs(len(typo) - len("atorvastatin")) == 1


def test_overlap_and_judge_bool_parsing():
    assert lexical_overlap("metformin renal dose", "Metformin dose in renal impairment") == 1.0
    assert _as_bool(True) is True and _as_bool("false") is False and _as_bool("maybe") is None


def test_unanswerable_items_do_not_name_indexed_drugs(indexed):
    store, _ = indexed
    items = unanswerable(store, per_kind=6)
    fictional = [i for i in items if i["type"] == "unknown_drug"]
    assert fictional and all("metformin" not in i["question"].lower() for i in fictional)
    assert {i["split"] for i in items} <= {"calib", "test"}
