"""Evaluation tasks: ``labellens eval synth|retrieval|gate|generation``.

Retrieval and generation are scored separately on purpose. A single
end-to-end number cannot tell you whether a bad answer came from missing
evidence or from a model that ignored good evidence - and those have
completely different fixes.

Every published number is computed on the held-out ``test`` split. The
abstention threshold is chosen on the ``calib`` split only.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from dataclasses import asdict
from pathlib import Path

from ..bootstrap import build_retriever
from ..config import Settings
from ..providers import CrossEncoderReranker, OllamaLLM, build_embedder, build_llm
from ..retrieval import Retriever, strip_salt
from ..service import RagService, should_refuse
from ..store import Store
from . import dataset
from .metrics import auroc, bootstrap_ci, hit_at_k, mean, ndcg_at_k, percentile, reciprocal_rank

log = logging.getLogger(__name__)

# name -> (index variant, setting overrides). Ordered from naive to full.
CONFIGS: dict[str, tuple[str, dict]] = {
    "naive: dense, raw chunks": ("raw", dict(use_bm25=False, use_rerank=False, use_entity_scope=False)),
    "bm25, raw chunks": ("raw", dict(use_dense=False, use_rerank=False, use_entity_scope=False)),
    "dense + context header": ("context", dict(use_bm25=False, use_rerank=False, use_entity_scope=False)),
    "bm25 + context header": ("context", dict(use_dense=False, use_rerank=False, use_entity_scope=False)),
    "hybrid RRF": ("context", dict(use_rerank=False, use_entity_scope=False)),
    "hybrid RRF + rerank": ("context", dict(use_entity_scope=False)),
    "hybrid RRF + entity scope": ("context", dict(use_rerank=False)),
    "full: hybrid + scope + rerank": ("context", dict()),
    "full, raw chunks (no header)": ("raw", dict()),
}
FULL = "full: hybrid + scope + rerank"


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path)


def _base(display: str) -> str:
    return strip_salt(display.split(" (")[0])


def _test(items: list[dict]) -> list[dict]:
    return [i for i in items if i.get("split") == "test"]


# ------------------------------------------------------------------- synth
def task_synth(args, settings: Settings) -> int:
    store = Store(settings.db_path)
    writer = OllamaLLM(settings.ollama_url, args.gen_model, think=False, fmt="json")
    items, card = dataset.synthesize(store, writer, args.n)
    items += dataset.unanswerable(store)
    golden = Path(args.golden)
    dataset.write_jsonl(golden, items)
    card.update({
        "writer_model": args.gen_model,
        "answerable": sum(i["type"] == "answerable" for i in items),
        "unanswerable": sum(i["type"] != "answerable" for i in items),
        "by_split": {s: sum(i["split"] == s for i in items) for s in ("calib", "test")},
        "by_name_style": {s: sum(i.get("name_style") == s for i in items)
                          for s in sorted({i.get("name_style") for i in items if i.get("name_style")})},
    })
    _write(golden.with_suffix(".card.json"), card)
    print(json.dumps(card, indent=2))
    return 0


# --------------------------------------------------------------- retrieval
def _score_config(retriever: Retriever, cfg: Settings, items: list[dict],
                  served_k: int | None = None) -> dict:
    """Score one configuration.

    The ranking is scored to depth 10 so MRR@10 and Recall@10 are defined, but
    the product puts only ``served_k`` passages in the prompt (6 locally, 4 in
    the hosted demo). ``recall@served`` is the share of questions whose gold
    passage the model actually sees; it is measured with ``final_k=served_k``
    exactly as served, because per-drug balancing depends on the slot count.
    """
    served_k = served_k or cfg.final_k
    retriever._qcache.clear()  # every config pays for its own query embeddings
    m: dict[str, list[float]] = {k: [] for k in (
        "hit@1", "recall@5", "recall@10", "mrr@10", "ndcg@10", "strict_recall@5",
        "section_recall@5", "right_drug@5", "scope_linked", "recall@served",
        "strict_recall@served")}
    lat: list[float] = []
    for it in items:
        t = time.perf_counter()
        res = retriever.retrieve(it["question"], cfg)
        lat.append((time.perf_counter() - t) * 1000)
        gold = {it["chunk_id"]}
        gold_base = _base(it["drug"])
        strict = [h.chunk.chunk_id for h in res.hits]
        # A surviving hit that absorbed the gold chunk as a near-duplicate is
        # credited only when it is the same molecule (another manufacturer's
        # copy of the same paragraph). A different drug's look-alike class
        # warning is NOT credited. Strict numbers are reported alongside.
        credited = [it["chunk_id"] if (it["chunk_id"] in h.dup_ids and _base(h.drug) == gold_base)
                    else h.chunk.chunk_id for h in res.hits]
        m["hit@1"].append(hit_at_k(credited, gold, 1))
        m["recall@5"].append(hit_at_k(credited, gold, 5))
        m["recall@10"].append(hit_at_k(credited, gold, 10))
        m["mrr@10"].append(reciprocal_rank(credited, gold, 10))
        m["ndcg@10"].append(ndcg_at_k(credited, gold, 10))
        m["strict_recall@5"].append(hit_at_k(strict, gold, 5))
        m["section_recall@5"].append(float(any(
            _base(h.drug) == gold_base and h.chunk.section_key == it["section_key"]
            for h in res.hits[:5])))
        m["right_drug@5"].append(float(any(_base(h.drug) == gold_base for h in res.hits[:5])))
        m["scope_linked"].append(float(it["doc_id"] in res.scoped_doc_ids))
        served = retriever.retrieve(it["question"], cfg.with_(final_k=served_k)).hits
        m["recall@served"].append(float(any(
            h.chunk.chunk_id == it["chunk_id"]
            or (it["chunk_id"] in h.dup_ids and _base(h.drug) == gold_base) for h in served)))
        m["strict_recall@served"].append(float(any(h.chunk.chunk_id == it["chunk_id"] for h in served)))
    out = {k: round(mean(v), 4) for k, v in m.items()}
    out["served_k"] = served_k
    lo, hi = bootstrap_ci(m["recall@5"])
    out["recall@5_ci95"] = [round(lo, 4), round(hi, 4)]
    out["latency_ms_p50"] = round(percentile(lat, 50), 1)
    out["latency_ms_p95"] = round(percentile(lat, 95), 1)
    out["_per_item"] = m
    return out


def task_retrieval(args, settings: Settings) -> int:
    items = [i for i in _test(dataset.read_jsonl(Path(args.golden))) if i["type"] == "answerable"]
    store = Store(settings.db_path)
    embedder = build_embedder(settings)
    reranker = CrossEncoderReranker(settings.rerank_model)
    served_k = settings.final_k  # passages the product puts in the prompt
    base = settings.with_(final_k=10)
    retrievers: dict[str, Retriever] = {}
    results: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    configs = {FULL: CONFIGS[FULL]} if args.only_full else CONFIGS
    index_name = settings.embed_index_name or embedder.model
    for name, (variant, overrides) in configs.items():
        have, total = store.dense_coverage(index_name, variant)
        if have < total:  # a partial index would silently understate a baseline
            skipped[name] = f"'{variant}' embeddings cover {have}/{total} chunks"
            log.warning("skipping %r: %s (run: labellens ingest --variant %s)",
                        name, skipped[name], variant)
            continue
        if variant not in retrievers:
            retrievers[variant] = Retriever(store, embedder, reranker, base, variant=variant)
        results[name] = _score_config(retrievers[variant], base.with_(**overrides), items, served_k)
        r = results[name]
        log.info("%-34s R@5 %.3f  MRR %.3f  p95 %.0f ms", name, r["recall@5"], r["mrr@10"],
                 r["latency_ms_p95"])

    slices: dict = {}
    if FULL in results:
        per = results[FULL]["_per_item"]
        hit5, linked = per["recall@5"], per["scope_linked"]
        cut = sorted(i["lexical_overlap"] for i in items)[len(items) // 2]
        slices["by_lexical_overlap"] = {
            "median_overlap": cut,
            "recall@5_low_overlap": round(mean([h for h, i in zip(hit5, items) if i["lexical_overlap"] < cut]), 4),
            "recall@5_high_overlap": round(mean([h for h, i in zip(hit5, items) if i["lexical_overlap"] >= cut]), 4),
        }
        styles: dict[str, list[tuple[float, float]]] = {}
        for h, lk, i in zip(hit5, linked, items):
            styles.setdefault(i["name_style"], []).append((h, lk))
        slices["by_name_style"] = {
            k: {"n": len(v), "recall@5": round(mean([a for a, _ in v]), 4),
                "scope_link_rate": round(mean([b for _, b in v]), 4)} for k, v in sorted(styles.items())}
        slices["by_scope_link"] = {
            "linked": {"n": int(sum(linked)), "recall@5": round(mean([h for h, lk in zip(hit5, linked) if lk]), 4)},
            "not_linked": {"n": int(len(linked) - sum(linked)),
                           "recall@5": round(mean([h for h, lk in zip(hit5, linked) if not lk]), 4)},
        }
        slices["full_pipeline_misses"] = [
            {"id": i["id"], "question": i["question"], "drug": i["drug"], "section": i["section_key"]}
            for h, i in zip(hit5, items) if not h][:40]
    for r in results.values():
        r.pop("_per_item", None)

    report = {"split": "test", "n_questions": len(items), "corpus": store.stats(),
              "embedder": embedder.model, "reranker": settings.rerank_model,
              # the exact knobs, so a result can always be tied to the system it describes
              "settings": {"db": str(settings.db_path.name), "embed_provider": settings.embed_provider,
                           "embed_model": settings.embed_model,
                           "embed_query_prefix": bool(settings.embed_query_prefix),
                           "bm25_k": settings.bm25_k, "dense_k": settings.dense_k,
                           "rerank_n": settings.rerank_n, "served_final_k": served_k,
                           "scored_depth": 10},
              "configs": results, "skipped": skipped, "slices": slices}
    _write(Path(args.out) / f"retrieval{args.tag}.json", report)
    _print_table(results)
    if args.tag:  # a tagged run (e.g. the hosted index) is reported, not gated
        return 0
    return _regression_gate(results, Path(args.baseline), update=args.update_baseline)


def _print_table(results: dict[str, dict]) -> None:
    cols = ["hit@1", "recall@5", "strict_recall@5", "mrr@10", "ndcg@10", "right_drug@5",
            "latency_ms_p95"]
    print(f"\n{'config':36s}" + "".join(f"{c:>16s}" for c in cols))
    for name, m in results.items():
        print(f"{name:36s}" + "".join(f"{m[c]:>16.3f}" for c in cols))


def _regression_gate(results: dict, baseline_path: Path, update: bool, tol: float = 0.02) -> int:
    """Fail (exit 1) if the full pipeline regressed against the committed baseline."""
    if FULL not in results:
        return 0
    current = {k: results[FULL][k] for k in ("recall@5", "strict_recall@5", "mrr@10", "hit@1")}
    if update or not baseline_path.exists():
        _write(baseline_path, current)
        return 0
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failed = [k for k, v in baseline.items() if current.get(k, 0) < v - tol]
    for k in baseline:
        print(f"gate {k}: baseline {baseline[k]:.3f} -> current {current.get(k, 0):.3f} "
              f"{'REGRESSED' if k in failed else 'ok'}")
    return 1 if failed else 0


# -------------------------------------------------------------------- gate
def task_gate(args, settings: Settings) -> int:
    """Calibrate the reranker-score abstention threshold on calib; report on test.

    The service gate refuses on (a) no hits, (b) unknown terms, (c) top reranker
    score < threshold. (a) and (b) need no tuning, so the threshold is fitted
    only on calib items that reach (c). Reported refusal rates apply the *whole*
    gate (via ``should_refuse``) to the test split.
    """
    items = dataset.read_jsonl(Path(args.golden))
    retriever = build_retriever(settings)
    rows = []
    for it in items:
        res = retriever.retrieve(it["question"])
        answerable = it["type"] == "answerable"
        rows.append({
            "id": it["id"], "split": it["split"], "type": it["type"], "res": res,
            "answerable": answerable,
            # an answerable question whose gold chunk was not retrieved is, from
            # the generator's point of view, unanswerable
            "gold_retrieved": answerable and any(
                it["chunk_id"] in (h.chunk.chunk_id, *h.dup_ids) for h in res.hits),
            "pre_gated": not res.hits or bool(res.unknown_terms) or res.unnamed_drug,
            "top": res.top_score if res.top_score is not None else -math.inf,
        })

    def refused(r: dict, thr: float) -> bool:
        return should_refuse(r["res"], thr)

    calib = [r for r in rows if r["split"] == "calib" and not r["pre_gated"]]
    pos = [r for r in calib if r["gold_retrieved"]]
    neg = [r for r in calib if not r["answerable"]]
    grid = [x / 2 for x in range(-24, 13)]
    curve = [{"threshold": t,
              "unanswerable_refused": round(mean([float(refused(r, t)) for r in neg]), 4),
              "answerable_wrongly_refused": round(mean([float(refused(r, t)) for r in pos]), 4)}
             for t in grid]
    # Operating point: the highest threshold that wrongly refuses NO answerable
    # calib question. A wrongly refused valid question is the costlier error:
    # an unanswerable question that slips past this gate still meets the
    # model's own INSUFFICIENT_EVIDENCE check. (A first version allowed 2 %
    # false refusals on calib; with ~30 calib negatives that choice was set by
    # two items and became 8 % on test - overfitting a small sample.)
    ok = [c for c in curve if c["answerable_wrongly_refused"] == 0.0]
    chosen = max(ok, key=lambda c: c["threshold"]) if ok else curve[0]
    thr = chosen["threshold"]

    test = [r for r in rows if r["split"] == "test"]
    t_ans = [r for r in test if r["answerable"]]
    t_unans = [r for r in test if not r["answerable"]]
    by_type: dict[str, list[float]] = {}
    for r in t_unans:
        by_type.setdefault(r["type"], []).append(float(refused(r, thr)))
    pre_by_type: dict[str, list[float]] = {}
    for r in t_unans:
        pre_by_type.setdefault(r["type"], []).append(float(r["pre_gated"]))

    def score(r: dict) -> float:
        return -99.0 if r["pre_gated"] else r["top"]

    report = {
        "threshold_chosen_on": "calib", "chosen": chosen,
        "test": {
            "n_answerable": len(t_ans), "n_unanswerable": len(t_unans),
            "answerable_wrongly_refused": round(mean([float(refused(r, thr)) for r in t_ans]), 4),
            "answerable_with_gold_wrongly_refused": round(mean(
                [float(refused(r, thr)) for r in t_ans if r["gold_retrieved"]]), 4),
            "unanswerable_refused": round(mean([float(refused(r, thr)) for r in t_unans]), 4),
            "unanswerable_refused_by_type": {k: round(mean(v), 4) for k, v in sorted(by_type.items())},
            "share_caught_before_reranker_by_type": {k: round(mean(v), 4) for k, v in sorted(pre_by_type.items())},
            "auroc_gold_retrieved_vs_unanswerable": round(auroc(
                [score(r) for r in t_ans if r["gold_retrieved"]], [score(r) for r in t_unans]), 4),
            "auroc_all_answerable_vs_unanswerable": round(auroc(
                [score(r) for r in t_ans], [score(r) for r in t_unans]), 4),
        },
        "calib_curve": curve,
        "test_curve": [{"threshold": t,
                        "unanswerable_refused": round(mean([float(refused(r, t)) for r in t_unans]), 4),
                        "answerable_wrongly_refused": round(mean([float(refused(r, t)) for r in t_ans]), 4)}
                       for t in grid],
        "scores": [{"id": r["id"], "split": r["split"], "type": r["type"],
                    "top_score": None if math.isinf(r["top"]) else round(r["top"], 3),
                    "pre_gated": r["pre_gated"], "gold_retrieved": r["gold_retrieved"]} for r in rows],
    }
    _write(Path(args.out) / "gate.json", report)
    print(json.dumps({"chosen": chosen, "test": report["test"]}, indent=2))
    return 0


# -------------------------------------------------------------- generation
JUDGE_SYSTEM = ("You are a strict evaluator of a drug-information assistant. "
                "Reply with a single JSON object and nothing else.")
JUDGE_PROMPT = """Question: {question}

Reference answer (written from the official label): {reference}

Label excerpts the assistant was allowed to use:
{sources}

Assistant's answer: {answer}

Judge two things independently.
1. "correct": true if the assistant's answer conveys the same key fact as the reference answer (wording may differ; extra supported detail is fine). false if it contradicts the reference, misses its key fact, or answers a different question.
2. "faithful": true only if EVERY factual claim in the assistant's answer is directly supported by the label excerpts above. false if any claim is absent from, or contradicts, the excerpts.

Return JSON: {{"correct": true|false, "faithful": true|false, "reason": "<one sentence>"}}"""


def _as_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in {"true", "false"}:
        return v.strip().lower() == "true"
    return None


def _judge(judge: OllamaLLM, item: dict, ans) -> dict:
    sources = "\n".join(f"[{s['n']}] {s['drug']} | {s['section']}: {s['text']}" for s in ans.sources)
    raw = judge.generate(JUDGE_SYSTEM, JUDGE_PROMPT.format(
        question=item["question"], reference=item["answer"], sources=sources,
        answer=re.sub(r"\s+", " ", ans.text)), max_tokens=200)
    m = re.search(r"\{.*\}", raw, flags=re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        out = {}
    correct, faithful = _as_bool(out.get("correct")), _as_bool(out.get("faithful"))
    return {"correct": correct, "faithful": faithful, "reason": str(out.get("reason", ""))[:300],
            "parsed": correct is not None and faithful is not None}


def _rate(rows: list[dict], key: str) -> float:
    """Mean of a judge field over rows the judge actually scored (parse failures excluded)."""
    vals = [r["judge"][key] for r in rows if r["judge"]["parsed"]]
    return round(mean([float(v) for v in vals]), 4)


def task_generation(args, settings: Settings) -> int:
    items = _test(dataset.read_jsonl(Path(args.golden)))
    rng = random.Random(21)
    answerable = [i for i in items if i["type"] == "answerable"]
    rng.shuffle(answerable)
    answerable = answerable[: args.n]
    unanswerable = [i for i in items if i["type"] != "answerable"]

    gate_file = Path(args.out) / "gate.json"
    if gate_file.exists():  # use the threshold calibrated on the calib split
        settings = settings.with_(abstain_threshold=json.loads(
            gate_file.read_text(encoding="utf-8"))["chosen"]["threshold"])
    settings = settings.with_(cache_size=0)
    retriever = build_retriever(settings)
    llm = build_llm(settings.llm_provider, settings.llm_model, settings.ollama_url,
                    settings.llm_timeout_s)
    service = RagService(retriever, llm, settings)
    judge = OllamaLLM(settings.ollama_url, args.judge_model, think=False, fmt="json")

    # Pass 1: generate every answer. Pass 2: judge them. Interleaving the two
    # would swap generator and judge in and out of GPU memory on every item.
    rows, to_judge = [], []
    for k, it in enumerate(answerable + unanswerable, start=1):
        ans = service.ask(it["question"])
        row = {"id": it["id"], "type": it["type"], "question": it["question"],
               "status": ans.status, "answer": ans.text, "top_score": ans.top_score,
               "grounding": ans.grounding, "timings_ms": ans.timings_ms}
        if it["type"] == "answerable":
            row["reference"] = it["answer"]
            gold_sources = [s for s in ans.sources
                            if it["chunk_id"] in (s["chunk_id"], *s["dup_chunk_ids"])]
            row["gold_in_context"] = bool(gold_sources)
            row["gold_cited"] = any(s["cited"] for s in gold_sources)
            if ans.status == "answered":
                to_judge.append((row, it, ans))
        rows.append(row)
        if k % 10 == 0:
            log.info("generation eval %d/%d", k, len(answerable) + len(unanswerable))
    for k, (row, it, ans) in enumerate(to_judge, start=1):
        row["judge"] = _judge(judge, it, ans)
        if k % 20 == 0:
            log.info("judged %d/%d", k, len(to_judge))

    a = [r for r in rows if r["type"] == "answerable"]
    answered = [r for r in a if r["status"] == "answered"]
    u = [r for r in rows if r["type"] != "answerable"]
    lat = [r["timings_ms"].get("total", 0) for r in answered]
    ttft = [r["timings_ms"].get("retrieval", 0) + r["timings_ms"].get("first_token", 0)
            for r in answered]
    summary = {
        "split": "test", "generator": llm.model, "judge": args.judge_model,
        "abstain_threshold": settings.abstain_threshold,
        "answerable": {
            "n": len(a),
            "answered_rate": round(len(answered) / max(1, len(a)), 4),
            "refused_by_retrieval_gate": round(mean([float(r["status"] == "abstained_retrieval") for r in a]), 4),
            "refused_by_llm": round(mean([float(r["status"] == "abstained_llm") for r in a]), 4),
            "gold_chunk_in_context": round(mean([float(r["gold_in_context"]) for r in a]), 4),
            "end_to_end_correct": round(sum(bool(r.get("judge", {}).get("correct")) for r in a) / max(1, len(a)), 4),
            "of_answered": {
                "n": len(answered),
                "judge_correct": _rate(answered, "correct"),
                "judge_faithful": _rate(answered, "faithful"),
                "judge_parse_failures": sum(not r["judge"]["parsed"] for r in answered),
                "grounding_checks_pass": round(mean([float(r["grounding"]["ok"]) for r in answered]), 4),
                "has_unsupported_number": round(mean([float(bool(r["grounding"]["unsupported_numbers"])) for r in answered]), 4),
                "has_invalid_citation": round(mean([float(bool(r["grounding"]["invalid_citations"])) for r in answered]), 4),
                "truncated": round(mean([float(bool(r["grounding"].get("truncated"))) for r in answered]), 4),
                "mean_citation_coverage": round(mean([r["grounding"]["citation_coverage"] for r in answered]), 4),
                "cites_gold_chunk": round(mean([float(r["gold_cited"]) for r in answered]), 4),
            },
        },
        "unanswerable": {
            "n": len(u),
            "refused_total": round(mean([float(r["status"] != "answered") for r in u]), 4),
            "refused_by_retrieval_gate": round(mean([float(r["status"] == "abstained_retrieval") for r in u]), 4),
            "refused_by_llm": round(mean([float(r["status"] == "abstained_llm") for r in u]), 4),
            "by_type": {t: round(mean([float(r["status"] != "answered") for r in u if r["type"] == t]), 4)
                        for t in sorted({r["type"] for r in u})},
        },
        "latency_ms_answered": {
            "p50_total": round(percentile(lat, 50)), "p95_total": round(percentile(lat, 95)),
            "p50_time_to_first_token": round(percentile(ttft, 50)),
            "p95_time_to_first_token": round(percentile(ttft, 95))},
        "settings": {k: str(v) for k, v in asdict(settings).items()},
    }
    _write(Path(args.out) / "generation.json", {"summary": summary, "rows": rows})
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))
    return 0


def main(args, settings: Settings) -> int:
    return {"synth": task_synth, "retrieval": task_retrieval, "gate": task_gate,
            "generation": task_generation}[args.task](args, settings)
