"""Re-score stored answers with the current grounding checker.

Grounding checks are deterministic functions of (answer text, retrieved
passages). When the checker changes, the stored answers can be re-scored
without regenerating them: retrieval is re-run with the recorded settings
(deterministic for a fixed index) and ``verify_grounding`` is applied again.
LLM-judge scores are untouched, since the answers are the same.

    python -m labellens.evalkit.rescore eval/results/generation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..bootstrap import build_retriever
from ..config import Settings
from ..generation import verify_grounding
from .metrics import mean


def rescore(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    recorded = summary["settings"]
    settings = Settings.from_env(
        abstain_threshold=float(recorded["abstain_threshold"]),
        final_k=int(recorded["final_k"]), rerank_n=int(recorded["rerank_n"]))
    retriever = build_retriever(settings)

    answered = [r for r in data["rows"] if r["type"] == "answerable" and r["status"] == "answered"]
    before = summary["answerable"]["of_answered"].copy()
    for row in answered:
        hits = retriever.retrieve(row["question"]).hits
        truncated = row["grounding"].get("truncated", False)
        g = verify_grounding(row["answer"], hits)
        if truncated:
            g.ok = False
        row["grounding"] = {**g.to_dict(), "truncated": truncated}

    oa = summary["answerable"]["of_answered"]
    oa["grounding_checks_pass"] = round(mean([float(r["grounding"]["ok"]) for r in answered]), 4)
    oa["has_unsupported_number"] = round(mean([float(bool(r["grounding"]["unsupported_numbers"]))
                                               for r in answered]), 4)
    oa["has_invalid_citation"] = round(mean([float(bool(r["grounding"]["invalid_citations"]))
                                             for r in answered]), 4)
    oa["mean_citation_coverage"] = round(mean([r["grounding"]["citation_coverage"]
                                               for r in answered]), 4)
    summary["grounding_rescored"] = {
        "reason": "number checker accepts units after a spread (47.8 (13.7) hours), unit "
                  "synonyms (µg/mcg), units in table column headers, and numbers in the drug name",
        "before": {k: before[k] for k in ("grounding_checks_pass", "has_unsupported_number")},
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"before": summary["grounding_rescored"]["before"],
            "after": {k: oa[k] for k in ("grounding_checks_pass", "has_unsupported_number",
                                         "has_invalid_citation", "mean_citation_coverage")}}


if __name__ == "__main__":
    print(json.dumps(rescore(Path(sys.argv[1] if len(sys.argv) > 1
                                  else "eval/results/generation.json")), indent=2))
