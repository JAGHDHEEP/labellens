"""External benchmark: FDARxBench (Apache-2.0, github.com/xiongbetty/FDARxBench).

17,207 expert-reviewed questions over 685 FDA labels: factual (one section),
multi-hop (two sections of one label) and refusal (the label does not contain
the answer). Each question names its label by ``set_id``.

LabelLens is run end to end, unchanged, on a separate index built from those
labels as served today by openFDA (fetched by ``set_id``). The paper's own
setting gives a model the whole label in context; this is a retrieval setting,
and label text may have been revised since the benchmark's snapshot - both
are reported with the results.

Grading follows the benchmark: each answer is CORRECT / INCORRECT /
NOT_ATTEMPTED against the reference; a refusal item is correct only if the
system declines.

    python -m labellens.evalkit.fdarx fetch     # labels -> data/fdarx/labels.jsonl.gz
    python -m labellens.evalkit.fdarx sample    # stratified sample
    python -m labellens.evalkit.fdarx answer    # run LabelLens on the sample
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

import httpx

from ..fetch import _get, _slim, _usable

SRC = Path("data/external/fdarxbench")
OUT = Path("data/fdarx")
SAMPLE = Path("eval/fdarx/sample.json")
SIZES = {"factual": 200, "multihop": 100, "refusal": 100}


def read_qa() -> list[dict]:
    return [json.loads(x) for x in (SRC / "qa.jsonl").read_text(encoding="utf-8").splitlines()]


def fetch_labels() -> dict:
    ids = sorted({q["set_id"] for q in read_qa()})
    OUT.mkdir(parents=True, exist_ok=True)
    found, missing = 0, []
    with httpx.Client(timeout=60) as client, gzip.open(OUT / "labels.jsonl.gz", "wt",
                                                       encoding="utf-8") as f:
        for sid in ids:
            res = _get(client, f'set_id:"{sid}"', 1).get("results", [])
            if res and _usable(res[0]):
                f.write(json.dumps(_slim(res[0])) + "\n")
                found += 1
            else:
                missing.append(sid)
    (OUT / "missing_set_ids.json").write_text(json.dumps(missing, indent=1), encoding="utf-8")
    return {"set_ids": len(ids), "fetched": found, "missing": len(missing)}


# Numbered PLR sections -> the openFDA field LabelLens normally reads.
_PLR = {1: "indications_and_usage", 2: "dosage_and_administration", 3: "dosage_forms_and_strengths",
        4: "contraindications", 5: "warnings_and_cautions", 6: "adverse_reactions",
        7: "drug_interactions", 8: "use_in_specific_populations", 9: "drug_abuse_and_dependence",
        10: "overdosage", 11: "description", 12: "clinical_pharmacology",
        13: "nonclinical_toxicology", 14: "clinical_studies", 15: "references",
        16: "how_supplied", 17: "information_for_patients"}
# Unnumbered headings (older and OTC labels).
_NAMED = [(r"^(boxed warning|warning:)", "boxed_warning"), (r"^uses?\b|^indications", "indications_and_usage"),
          (r"^directions|^dosage", "dosage_and_administration"), (r"^do not use", "do_not_use"),
          (r"^ask a doctor or pharmacist", "ask_doctor_or_pharmacist"), (r"^ask a doctor", "ask_doctor"),
          (r"^stop use", "stop_use"), (r"^when using", "when_using"),
          (r"^keep out of reach", "keep_out_of_reach_of_children"), (r"^warnings?\b", "warnings"),
          (r"^contraindications", "contraindications"), (r"^adverse reactions", "adverse_reactions"),
          (r"^precautions", "precautions"), (r"^overdosage", "overdosage"),
          (r"^clinical pharmacology", "clinical_pharmacology"), (r"^description", "description"),
          (r"^how supplied", "how_supplied"), (r"^active ingredient", "active_ingredient"),
          (r"^inactive ingredient", "inactive_ingredient"), (r"^purpose", "purpose"),
          (r"^other information|^storage", "storage_and_handling")]


def _field_for(heading: str, current: str | None) -> str | None:
    import re

    m = re.match(r"^(\d{1,2})(?:\.\d{1,2})?\s+[A-Za-z]", heading)
    if m and int(m.group(1)) in _PLR:
        return _PLR[int(m.group(1))]
    low = heading.lower().strip()
    for pat, key in _NAMED:
        if re.match(pat, low):
            return key
    return current  # a sub-heading, table row or caption stays in its section


def build_records() -> dict:
    """The benchmark's own label text -> openFDA-shaped records, so the
    unchanged LabelLens pipeline indexes exactly the text the references were
    written from. Names and brands come from openFDA where the label is still
    served; otherwise the benchmark's drug name is used."""
    import re

    meta = {}
    if (OUT / "labels.jsonl.gz").exists():
        with gzip.open(OUT / "labels.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                meta[rec["set_id"]] = rec.get("openfda", {})
    n, sections = 0, 0
    with gzip.open(OUT / "bench_labels.jsonl.gz", "wt", encoding="utf-8") as out:
        for line in (SRC / "labels.jsonl").read_text(encoding="utf-8").splitlines():
            lab = json.loads(line)
            fields: dict[str, list[str]] = {}
            current = None
            for block in re.split(r"\|\|PASSAGE_\d+\|\|\n", lab["label_raw"]):
                block = block.strip()
                if not block:
                    continue
                head = block.split("\n", 1)[0].strip()
                current = _field_for(head, current) or "spl_unclassified_section_text"
                fields.setdefault(current, []).append(block.replace("\n", " "))
            fda = meta.get(lab["set_id"]) or {}
            rec = {"set_id": lab["set_id"], "version": "1", "effective_time": "",
                   "openfda": {"generic_name": fda.get("generic_name") or [lab["drug_name"]],
                               "brand_name": fda.get("brand_name") or [lab["drug_name"]],
                               "manufacturer_name": fda.get("manufacturer_name") or [""],
                               "route": fda.get("route") or [],
                               "substance_name": fda.get("substance_name") or [],
                               "pharm_class_epc": fda.get("pharm_class_epc") or []}}
            for key, parts in fields.items():
                rec[key] = [" ".join(parts)]
            out.write(json.dumps(rec) + "\n")
            n += 1
            sections += len(fields)
    return {"labels": n, "sections": sections, "with_openfda_names": sum(1 for _ in meta)}


def sample(seed: int = 13) -> dict:
    """Stratified by task; only questions whose label is indexed; one
    question per label per task where possible, so a few labels do not dominate."""
    have = set()
    with gzip.open(OUT / "bench_labels.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            have.add(json.loads(line)["set_id"])
    rng = random.Random(seed)
    qa = [q for q in read_qa() if q["set_id"] in have]
    picked = []
    for task, n in SIZES.items():
        pool = [q for q in qa if q["task"] == task]
        rng.shuffle(pool)
        seen, take = set(), []
        for q in pool:  # first pass: distinct labels
            if q["set_id"] not in seen:
                take.append(q)
                seen.add(q["set_id"])
            if len(take) == n:
                break
        picked += take
    rows = [{k: q[k] for k in ("qid", "task", "set_id", "drug_name", "question", "answer",
                               "references")} for q in picked]
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE.write_text(json.dumps({"seed": seed, "sizes": SIZES, "rows": rows}, indent=1,
                                 ensure_ascii=False), encoding="utf-8")
    return {t: sum(r["task"] == t for r in rows) for t in SIZES}


def answer() -> dict:
    from ..bootstrap import build_service
    from ..config import Settings

    settings = Settings.from_env(data_dir=OUT, db_name="labellens.db")
    svc = build_service(settings)
    rows = json.loads(SAMPLE.read_text(encoding="utf-8"))["rows"]
    out = []
    for r in rows:
        ans = svc.ask(r["question"])
        out.append({**r, "status": ans.status, "system_answer": ans.text,
                    "scoped_drugs": ans.scoped_drugs, "grounding_ok": ans.grounding.get("ok"),
                    "cited": [{"n": s["n"], "drug": s["drug"], "section": s["section"],
                               "text": s["text"]} for s in ans.sources if s["cited"]],
                    "total_ms": ans.timings_ms.get("total")})
    path = Path("eval/fdarx/answers.json")
    path.write_text(json.dumps({"generator": settings.llm_model, "rows": out}, indent=1,
                               ensure_ascii=False), encoding="utf-8")
    return {"answered": sum(o["status"] == "answered" for o in out), "n": len(out)}


if __name__ == "__main__":
    print(json.dumps({"fetch": fetch_labels, "build": build_records, "sample": sample,
                      "answer": answer}[sys.argv[1]](),
                     indent=2))
