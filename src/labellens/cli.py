"""Command line interface: ``labellens <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .observability import configure_logging

log = logging.getLogger("labellens")


def _raw_path(settings: Settings) -> Path:
    return settings.data_dir / "raw" / "labels.jsonl.gz"


def cmd_fetch(args, settings: Settings) -> int:
    from .fetch import build_corpus

    n = build_corpus(_raw_path(settings), target_docs=args.docs, pages=args.pages)
    print(f"wrote {n} labels to {_raw_path(settings)}")
    return 0


def cmd_ingest(args, settings: Settings) -> int:
    from .fetch import read_corpus
    from .ingest import embed_missing, ingest_records
    from .providers import OllamaEmbedder
    from .store import Store

    store = Store(settings.db_path)
    report = ingest_records(store, read_corpus(Path(args.corpus or _raw_path(settings))),
                            settings, prune=args.prune, bump=args.no_embed)
    if not args.no_embed:
        embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model, settings.embed_batch)
        variants = [args.variant]
        # Keep every embedding variant that is already in use complete, so the
        # header-less ablation index never silently falls behind the main one.
        other = "raw" if args.variant == "context" else "context"
        if store.dense_coverage(embedder.model, other)[0] > 0:
            variants.append(other)
        for variant in variants:
            report.embedded += embed_missing(store, embedder, variant=variant)
        if report.embedded == 0 and (report.added or report.updated or report.removed):
            store.bump_version()
    print(json.dumps({**asdict(report), **store.stats()}, indent=2))
    return 0


def cmd_stats(args, settings: Settings) -> int:
    from .store import Store

    print(json.dumps(Store(settings.db_path).stats(), indent=2))
    return 0


def cmd_search(args, settings: Settings) -> int:
    from .bootstrap import build_retriever

    res = build_retriever(settings).retrieve(args.question)
    print(f"scope: {res.scoped_drugs or 'none'}   top_score: {res.top_score}   {res.timings_ms}")
    for i, h in enumerate(res.hits, start=1):
        score = "" if h.rerank_score is None else f"{h.rerank_score:6.2f}"
        print(f"\n[{i}] {score}  bm25#{h.bm25_rank} dense#{h.dense_rank}  {h.chunk.header}")
        print("    " + h.chunk.text[:300].replace("\n", " "))
    return 0


def cmd_ask(args, settings: Settings) -> int:
    from .bootstrap import build_service

    ans = build_service(settings).ask(args.question)
    if args.json:
        print(json.dumps(asdict(ans), indent=2, ensure_ascii=False))
        return 0
    print(f"\n{ans.text}\n")
    print(f"status={ans.status}  grounding_ok={ans.grounding.get('ok')}  "
          f"top_score={ans.top_score}  scope={ans.scoped_drugs}")
    for s in ans.sources:
        mark = "*" if s["cited"] else " "
        print(f" {mark}[{s['n']}] {s['drug']} | {s['section']} {s['subsection']}")
    print(f"timings_ms={ans.timings_ms}")
    return 0


def cmd_serve(args, settings: Settings) -> int:
    import uvicorn

    from .api import create_app

    configure_logging(json_logs=args.json_logs)
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port, log_config=None)
    return 0


def cmd_eval(args, settings: Settings) -> int:
    from .evalkit import runner

    return runner.main(args, settings)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="labellens", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download labels from openFDA")
    f.add_argument("--docs", type=int, default=1200)
    f.add_argument("--pages", type=int, default=70)
    f.set_defaults(fn=cmd_fetch)

    i = sub.add_parser("ingest", help="chunk, index and embed (incremental)")
    i.add_argument("--corpus")
    i.add_argument("--prune", action="store_true", help="remove labels absent from the corpus")
    i.add_argument("--no-embed", action="store_true")
    i.add_argument("--variant", default="context", choices=["context", "raw"])
    i.set_defaults(fn=cmd_ingest)

    sub.add_parser("stats", help="index statistics").set_defaults(fn=cmd_stats)

    s = sub.add_parser("search", help="retrieval only (debug)")
    s.add_argument("question")
    s.set_defaults(fn=cmd_search)

    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_ask)

    sv = sub.add_parser("serve", help="run the HTTP API and web UI")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--json-logs", action="store_true")
    sv.set_defaults(fn=cmd_serve)

    e = sub.add_parser("eval", help="evaluation harness")
    e.add_argument("task", choices=["synth", "retrieval", "generation", "gate"])
    e.add_argument("--n", type=int, default=300)
    e.add_argument("--golden", default="eval/golden/golden_v1.jsonl")
    e.add_argument("--out", default="eval/results")
    e.add_argument("--judge-model", default="qwen3:8b")
    e.add_argument("--gen-model", default="qwen3:8b")
    e.add_argument("--baseline", default="eval/baseline.json")
    e.add_argument("--update-baseline", action="store_true")
    e.set_defaults(fn=cmd_eval)

    # Label text contains symbols (arrows, Greek letters) that legacy Windows
    # consoles cannot encode; never let printing a result crash a command.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = p.parse_args(argv)
    if args.cmd != "serve":
        configure_logging()
    return args.fn(args, Settings.from_env())


if __name__ == "__main__":
    sys.exit(main())
