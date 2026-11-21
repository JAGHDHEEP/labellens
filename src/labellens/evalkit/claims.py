"""Sentence-level claim verification: data collection and scoring.

The deterministic grounding checks verify citations and numbers. They cannot
see a sentence that attaches a real fact from the passages to the wrong
population, trial or drug ("misattribution"), which a strong judge found in
about 1 answer in 10. This module builds the data to measure verifiers that
try to catch it, sentence by sentence:

* ``collect``: answer golden-set questions with the full system and store, for
  every answered question, the answer split into sentences, the passage
  numbers each sentence cites, and the text of every passage.
* gold labels per sentence come from a strong model (kept in
  ``eval/claims/gold_<split>.json``); see ``docs/DECISIONS.md``.
* ``score``: precision / recall / F1 of a verifier's "flag" decisions against
  those labels.

Verifiers are tuned on the ``calib`` split and reported once on ``test``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..bootstrap import build_service
from ..config import Settings
from ..generation import normalize_citations
from ..verify import split_claims as _split
from . import dataset


def split_claims(answer: str) -> list[dict]:
    return _split(normalize_citations(answer))


def collect(split: str, n: int, out: Path, settings: Settings, seed: int = 5) -> dict:
    items = [i for i in dataset.read_jsonl(Path("eval/golden/golden_v1.jsonl"))
             if i.get("split") == split and i["type"] == "answerable"]
    random.Random(seed).shuffle(items)
    svc = build_service(settings)
    rows = []
    for it in items[:n]:
        ans = svc.ask(it["question"])
        if ans.status != "answered":
            continue
        rows.append({
            "id": it["id"], "question": it["question"], "reference": it["answer"],
            "answer": ans.text,
            "claims": split_claims(ans.text),
            "passages": [{"n": s["n"], "drug": s["drug"], "section": s["section"],
                          "subsection": s["subsection"], "text": s["text"]}
                         for s in ans.sources],
        })
    data = {"split": split, "generator": settings.llm_model, "n_questions": min(n, len(items)),
            "n_answered": len(rows), "n_claims": sum(len(r["claims"]) for r in rows),
            "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return {k: v for k, v in data.items() if k != "rows"}


def score(gold: dict[str, bool], flagged: dict[str, bool]) -> dict:
    """``gold[key]`` is True for a bad claim (should be flagged)."""
    keys = [k for k in gold if k in flagged]
    tp = sum(gold[k] and flagged[k] for k in keys)
    fp = sum((not gold[k]) and flagged[k] for k in keys)
    fn = sum(gold[k] and not flagged[k] for k in keys)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(keys), "bad": tp + fn, "flagged": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


BAD = frozenset({"misattributed", "contradicted", "unsupported"})


def load(split: str) -> tuple[list[dict], dict[str, dict]]:
    rows = json.loads(Path(f"eval/claims/answers_{split}.json").read_text(encoding="utf-8"))["rows"]
    gold = json.loads(Path(f"eval/claims/gold_{split}.json").read_text(encoding="utf-8"))["labels"]
    return rows, gold


def nli_scores(rows: list[dict], model_repo: str, mode: str = "s1") -> dict[str, dict]:
    """Raw NLI scores per claim and passage, computed once; thresholds are
    then swept offline. -> {key: {"cites", "per": {n: [entail, contra]}, "drug"}}"""
    import time

    from ..verify import NLIModel, NLIVerifier, claim_text

    v = NLIVerifier(NLIModel(model_repo), mode=mode)
    out, t0, n = {}, time.perf_counter(), 0
    for r in rows:
        passages = {p["n"]: p["text"] for p in r["passages"]}
        drugs = {p["n"]: p["drug"] for p in r["passages"]}
        for c in r["claims"]:
            hyp = claim_text(c["text"])
            per = {}
            for k, text in passages.items():
                e, con, _ = v._best(hyp, text, drugs[k])
                per[k] = [round(e, 4), round(con, 4)]
            n += 1
            out[f'{r["id"]}#{c["i"]}'] = {"cites": c["cites"], "per": per}
    ms = (time.perf_counter() - t0) * 1000 / max(1, n)
    return {"model": model_repo, "mode": mode, "ms_per_claim_all_passages": round(ms, 1), "claims": out}


def nli_verdict(rec: dict, entail_t: float, contra_t: float) -> str:
    cited = [n for n in rec["cites"] if str(n) in rec["per"] or n in rec["per"]]
    per = {int(k): v for k, v in rec["per"].items()}
    if any(per[n][0] >= entail_t for n in cited):
        return "supported"
    if any(v[0] >= entail_t for n, v in per.items() if n not in cited):
        return "recited"
    if any(per[n][1] >= contra_t for n in cited):
        return "contradicted"
    return "unsupported"


def evaluate(verdicts: dict[str, str], gold: dict[str, dict]) -> dict:
    """Bad-claim detection and citation repair against the gold labels."""
    keys = [k for k in verdicts if k in gold and gold[k]["label"] != "not_a_claim"]
    bad = {k: gold[k]["label"] in BAD for k in keys}
    flagged = {k: verdicts[k] in ("contradicted", "unsupported") for k in keys}
    det = score(bad, flagged)
    sup = [k for k in keys if gold[k]["label"] == "supported"]
    det["false_alarm_rate"] = round(sum(flagged[k] for k in sup) / max(1, len(sup)), 3)
    wc = [k for k in keys if gold[k]["label"] == "wrong_citation"]
    rec = [k for k in keys if verdicts[k] == "recited"]
    det["repair_recall"] = round(sum(verdicts[k] == "recited" for k in wc) / max(1, len(wc)), 3)
    det["repair_precision"] = round(sum(gold[k]["label"] == "wrong_citation" for k in rec) / max(1, len(rec)), 3)
    det["n_wrong_citation"] = len(wc)
    return det


MAX_FALSE_ALARMS = 0.05  # share of true claims a verifier may flag; fixed before tuning


def _with_qualifiers(rows: list[dict], scores: dict, verdicts: dict[str, str],
                     entail_t: float) -> dict[str, str]:
    """A claim the NLI model accepts is still flagged if it names a population,
    species or trial that its supporting passage never mentions."""
    from ..verify import qualifier_gaps

    out = dict(verdicts)
    for r in rows:
        texts = {p["n"]: (p["text"], p["drug"]) for p in r["passages"]}
        for c in r["claims"]:
            k = f'{r["id"]}#{c["i"]}'
            if out.get(k) not in ("supported", "recited"):
                continue
            per = {int(n): v for n, v in scores[k]["per"].items()}
            pool = [n for n in c["cites"] if n in per] if out[k] == "supported" else list(per)
            best = max(pool, key=lambda n: per[n][0])
            if per[best][0] >= entail_t and qualifier_gaps(c["text"], *texts[best]):
                out[k] = "unsupported"
    return out


def report(verifiers: dict[str, dict], out: Path) -> dict:
    """Tune each NLI verifier's threshold on calib, then score once on test.

    ``verifiers``: name -> {"calib": scores, "test": scores, "ms": ...}."""
    rows = {s: load(s)[0] for s in ("calib", "test")}
    gold = {s: load(s)[1] for s in ("calib", "test")}
    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    result: dict = {"rule": f"maximise recall of false claims on calib with at most "
                            f"{MAX_FALSE_ALARMS:.0%} of true claims flagged; report on test once",
                    "verifiers": {}}
    for name, v in verifiers.items():
        for qual in (False, True):
            def run(split, t, qual=qual, v=v):
                verdicts = {k: nli_verdict(rec, t, 0.9) for k, rec in v[split]["claims"].items()}
                if qual:
                    verdicts = _with_qualifiers(rows[split], v[split]["claims"], verdicts, t)
                return evaluate(verdicts, gold[split]), verdicts
            sweep = [(t, run("calib", t)[0]) for t in grid]
            ok = [(t, m) for t, m in sweep if m["false_alarm_rate"] <= MAX_FALSE_ALARMS]
            t, dev = max(ok, key=lambda x: (x[1]["recall"], x[1]["precision"])) if ok else (None, None)
            test = run("test", t)[0] if t is not None else None
            result["verifiers"][name + (" + qualifiers" if qual else "")] = {
                "threshold": t, "calib": dev, "test": test,
                "ms_per_claim_all_passages": v["test"].get("ms_per_claim_all_passages"),
                "calib_sweep": [{"t": t_, **{k: m[k] for k in ("recall", "precision", "false_alarm_rate")}}
                                for t_, m in sweep]}
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    import sys

    split, n = sys.argv[1], int(sys.argv[2])
    print(json.dumps(collect(split, n, Path(f"eval/claims/answers_{split}.json"),
                             Settings.from_env()), indent=2))
