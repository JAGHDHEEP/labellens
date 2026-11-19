"""Structured logging and Prometheus metrics.

Per-stage latency histograms matter more than a single request histogram: when
p95 regresses, the first question is always *which stage* (embedding, rerank,
or generation), and it has to be answerable from a dashboard, not a debugger.
"""

from __future__ import annotations

import json
import logging
import sys
import time

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("labellens_requests_total", "Questions handled, by outcome", ["status"])
STAGE_SECONDS = Histogram(
    "labellens_stage_seconds", "Latency per pipeline stage", ["stage"],
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
GROUNDING_FAILURES = Counter(
    "labellens_grounding_failures_total", "Answers that failed a post-generation check", ["kind"])
CACHE_HITS = Counter("labellens_cache_hits_total", "Answers served from the response cache")
INDEX_CHUNKS = Gauge("labellens_index_chunks", "Chunks in the loaded index")
INDEX_VERSION = Gauge("labellens_index_version", "Version of the loaded index")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(json_logs: bool = False, level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
