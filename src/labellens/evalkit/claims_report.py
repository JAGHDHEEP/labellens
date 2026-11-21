"""Build eval/results/claims.json: every verifier variant, threshold chosen on
calib by the pre-set rule, then scored once on test.

    python -m labellens.evalkit.claims_report
"""

from __future__ import annotations

import json
from pathlib import Path

from .claims import MAX_FALSE_ALARMS, _with_qualifiers, evaluate, load, nli_verdict

GRID = [0.002, 0.005, 0.01, 0.02, 0.03] + [round(0.05 * i, 2) for i in range(1, 20)]
VARIANTS = {"small, single sentences": "small_s1", "small, 1-2 sentence windows": "small_s1p2",
            "small, drug name + sentence": "small_drugps1", "base, single sentences": "base_s1",
            "base, drug name + sentence": "base_drugps1"}
KEYS = ("tp", "bad", "recall", "precision", "false_alarm_rate", "repair_recall",
        "repair_precision", "n_wrong_citation")


def main() -> dict:
    rows = {s: load(s)[0] for s in ("calib", "test")}
    gold = {s: load(s)[1] for s in ("calib", "test")}
    out: dict = {"rule": f"for each variant, the threshold with the highest recall of false claims "
                         f"on calib while flagging at most {MAX_FALSE_ALARMS:.0%} of true claims; "
                         f"then scored once on test",
                 "gold": "sentence labels by a stronger model against the passages the system was "
                         "given; false = misattributed, contradicted or unsupported",
                 "variants": {}}
    for name, tag in VARIANTS.items():
        files = {s: Path(f"eval/claims/nli_{tag}_{s}.json") for s in ("calib", "test")}
        if not all(f.exists() for f in files.values()):
            continue
        sc = {s: json.loads(f.read_text(encoding="utf-8")) for s, f in files.items()}
        for qual in (False, True):
            def run(split, t, sc=sc, qual=qual):
                v = {k: nli_verdict(r, t, 0.9) for k, r in sc[split]["claims"].items()}
                if qual:
                    v = _with_qualifiers(rows[split], sc[split]["claims"], v, t)
                return evaluate(v, gold[split])
            dev = [(t, run("calib", t)) for t in GRID]
            ok = [x for x in dev if x[1]["false_alarm_rate"] <= MAX_FALSE_ALARMS]
            if not ok:
                continue
            t, m = max(ok, key=lambda x: (x[1]["recall"], x[1]["precision"]))
            test = run("test", t)
            out["variants"][name + (" + qualifier check" if qual else "")] = {
                "threshold": t, "calib": {k: m[k] for k in KEYS}, "test": {k: test[k] for k in KEYS},
                "ms_per_claim_all_6_passages": sc["test"]["ms_per_claim_all_passages"]}
    q = {s: json.loads(Path(f"eval/claims/quote_{s}.json").read_text(encoding="utf-8"))
         for s in ("calib", "test")}
    out["llm_quote_verifier"] = {
        "model": q["test"]["model"], "ms_per_answer": q["test"]["ms_per_answer"],
        **{s: {k: v for k, v in evaluate({c: x["verdict"] for c, x in q[s]["claims"].items()},
                                         gold[s]).items() if k in KEYS} for s in ("calib", "test")}}
    out["shipped"] = json.loads(Path("eval/claims/selection.json").read_text(encoding="utf-8"))
    Path("eval/results/claims.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = main()
    for name, v in r["variants"].items():
        print(f"{name:45s} t={v['threshold']:<6} dev R {v['calib']['recall']:.2f} FA "
              f"{v['calib']['false_alarm_rate']:.3f} | test R {v['test']['recall']:.2f} FA "
              f"{v['test']['false_alarm_rate']:.3f} repair {v['test']['repair_recall']:.2f}")
    print("LLM quote verifier test:", r["llm_quote_verifier"]["test"])
