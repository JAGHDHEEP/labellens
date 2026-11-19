"""The question-answering service: retrieve -> gate -> generate -> verify.

The abstention gate is the most important product decision in the system.
A RAG system that always answers is confidently wrong whenever retrieval
misses, and a user cannot tell that case apart from a good answer. So:

* Gate 1 (before the LLM): if the best reranker score is below a threshold
  calibrated on the golden set, refuse immediately. This is also the cheapest
  possible path - no generation tokens are spent on unanswerable questions.
* Gate 2 (inside the LLM): the prompt gives the model an explicit, parseable
  way to say the sources do not contain the answer.
* Gate 3 (after the LLM): deterministic grounding checks; in strict mode an
  answer with unsupported numbers or invalid citations is withheld.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import observability as obs
from .config import Settings
from .generation import (
    SYSTEM_PROMPT,
    Grounding,
    build_user_prompt,
    clean_abstention,
    could_be_abstention,
    is_abstention,
    normalize_citations,
    verify_grounding,
)
from .models import Hit, RetrievalResult
from .providers import LLM, TRUNCATED, ProviderError
from .retrieval import Retriever

log = logging.getLogger(__name__)

NO_EVIDENCE_MSG = (
    "I can't answer this from the indexed FDA labels: no passage was relevant enough to "
    "support an answer. Try naming the drug explicitly, or rephrase the question."
)
UNNAMED_DRUG_MSG = (
    "Which drug do you mean? The question refers to a drug without naming it, and labels "
    "differ by drug. Ask again with the drug names, for example: \"Can warfarin be taken "
    "with amiodarone?\""
)
UNGROUNDED_MSG = (
    "An answer was generated but withheld because it contained details that could not be "
    "verified against the cited label text. The closest passages are listed below."
)
MAX_QUESTION_CHARS = 600


class InvalidQuestion(ValueError):
    pass


@dataclass
class Answer:
    question: str
    text: str
    status: str  # answered | abstained_retrieval | abstained_llm | blocked_ungrounded
    sources: list[dict] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    scoped_drugs: list[str] = field(default_factory=list)
    top_score: float | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    model: str = ""
    index_version: int = 0
    cached: bool = False
    request_id: str = ""


def _source_dict(n: int, h: Hit, cited: bool) -> dict:
    return {
        "n": n, "drug": h.drug, "section": h.chunk.section_title, "subsection": h.chunk.subsection,
        "text": h.chunk.text, "chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id,
        "dup_chunk_ids": list(h.dup_ids),
        "score": None if h.rerank_score is None else round(h.rerank_score, 3),
        "bm25_rank": h.bm25_rank, "dense_rank": h.dense_rank, "cited": cited,
        "url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={h.chunk.doc_id}",
    }


def should_refuse(res: RetrievalResult, threshold: float) -> bool:
    """The pre-generation gate: True when retrieval is too weak to justify an LLM call.

    A module-level function (not a method) so the offline evaluation calibrates
    and scores exactly the decision the service makes.
    """
    if not res.hits or res.unknown_terms or res.unnamed_drug:
        return True
    return res.top_score is not None and not math.isinf(threshold) and res.top_score < threshold


def validate_question(question: str) -> str:
    q = " ".join((question or "").split())
    if len(q) < 3:
        raise InvalidQuestion("question is too short")
    if len(q) > MAX_QUESTION_CHARS:
        raise InvalidQuestion(f"question exceeds {MAX_QUESTION_CHARS} characters")
    return q


class RagService:
    def __init__(self, retriever: Retriever, llm: LLM, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings
        self._cache: OrderedDict[tuple, Answer] = OrderedDict()
        self._lock = threading.Lock()
        obs.INDEX_CHUNKS.set(len(retriever.dense))
        obs.INDEX_VERSION.set(retriever.index_version)

    # ----------------------------------------------------------------- cache
    # The index version is part of the key, so a re-ingest invalidates every
    # cached answer without an explicit flush. The version is captured once per
    # request and used for both lookup and store, so an answer computed against
    # version N can never be filed under version N+1.
    def _key(self, q: str, version: int) -> tuple:
        return (q.lower(), version, self.llm.model)

    def _cache_get(self, key: tuple) -> Answer | None:
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
        return hit

    def _cache_put(self, key: tuple, ans: Answer) -> None:
        # A truncated answer must not be replayed to the next user.
        if self.settings.cache_size <= 0 or ans.grounding.get("truncated"):
            return
        with self._lock:
            self._cache[key] = ans
            while len(self._cache) > self.settings.cache_size:
                self._cache.popitem(last=False)

    # ------------------------------------------------------------------ core
    def _gate(self, res: RetrievalResult) -> bool:
        return should_refuse(res, self.settings.abstain_threshold)

    def _finish(self, q: str, res: RetrievalResult, raw: str, timings: dict, rid: str,
                truncated: bool = False) -> Answer:
        hits = res.hits
        if is_abstention(raw):
            status, text, g = "abstained_llm", clean_abstention(raw), Grounding()
        else:
            g = verify_grounding(raw, hits)
            status, text = "answered", normalize_citations(raw).strip()
            if g.invalid_citations:
                obs.GROUNDING_FAILURES.labels("invalid_citation").inc()
            if g.unsupported_numbers:
                obs.GROUNDING_FAILURES.labels("unsupported_number").inc()
            if truncated:
                g.ok = False
                obs.GROUNDING_FAILURES.labels("truncated").inc()
            if not g.ok and self.settings.strict_grounding:
                status, text = "blocked_ungrounded", UNGROUNDED_MSG
        cited = set(g.cited_sources)
        grounding = {**g.to_dict(), "truncated": truncated}
        return Answer(
            question=q, text=text, status=status,
            sources=[_source_dict(i, h, i in cited) for i, h in enumerate(hits, start=1)],
            grounding=grounding, scoped_drugs=res.scoped_drugs, top_score=res.top_score,
            timings_ms=timings, model=self.llm.model, index_version=res.index_version,
            request_id=rid,
        )

    def _refuse(self, q: str, res: RetrievalResult, timings: dict, rid: str) -> Answer:
        text = NO_EVIDENCE_MSG
        if res.unnamed_drug:
            text = UNNAMED_DRUG_MSG
        elif res.unknown_terms:
            text = ("I can't answer this: " + ", ".join(f'"{t}"' for t in res.unknown_terms[:3])
                    + " does not appear in any indexed FDA label. Check the spelling, or the "
                    "drug may not be in this index.")
        return Answer(
            question=q, text=text, status="abstained_retrieval",
            sources=[_source_dict(i, h, False) for i, h in enumerate(res.hits, start=1)],
            grounding=Grounding().to_dict(), scoped_drugs=res.scoped_drugs,
            top_score=res.top_score, timings_ms=timings, model=self.llm.model,
            index_version=res.index_version, request_id=rid,
        )

    def _record(self, ans: Answer) -> None:
        obs.REQUESTS.labels(ans.status).inc()
        obs.INDEX_CHUNKS.set(len(self.retriever.dense))
        obs.INDEX_VERSION.set(self.retriever.index_version)
        for stage, ms in ans.timings_ms.items():
            obs.STAGE_SECONDS.labels(stage).observe(ms / 1000)
        log.info("answered", extra={"fields": {
            "request_id": ans.request_id, "status": ans.status, "top_score": ans.top_score,
            "scoped": ans.scoped_drugs, "grounding_ok": ans.grounding.get("ok"),
            "timings_ms": ans.timings_ms, "cached": ans.cached}})

    def ask(self, question: str) -> Answer:
        final: Answer | None = None
        for event, payload in self.ask_stream(question):
            if event == "final":
                final = payload
        assert final is not None
        return final

    def ask_stream(self, question: str) -> Iterator[tuple[str, object]]:
        """Yield ("sources", list) -> ("token", str)* -> ("final", Answer).

        Token events are a live preview; the "final" event carries the answer of
        record. Two cases are never previewed, because showing text and then
        retracting it is worse than a short pause: output that may still turn
        out to be the INSUFFICIENT_EVIDENCE sentinel, and - in strict mode - any
        answer before it has passed the grounding checks.
        """
        q = validate_question(question)
        rid = uuid.uuid4().hex[:12]
        if self.retriever.refresh_if_stale():
            log.info("index reloaded",
                     extra={"fields": {"index_version": self.retriever.index_version}})
        key = self._key(q, self.retriever.index_version)

        cached = self._cache_get(key)
        if cached is not None:
            obs.CACHE_HITS.inc()
            ans = Answer(**{**cached.__dict__, "cached": True, "request_id": rid,
                            "timings_ms": {"total": 0.0}})
            yield "sources", ans.sources
            yield "token", ans.text
            self._record(ans)
            yield "final", ans
            return

        t0 = time.perf_counter()
        res = self.retriever.retrieve(q)
        timings = {k: v for k, v in res.timings_ms.items() if k != "total"}
        timings["retrieval"] = res.timings_ms.get("total", 0.0)
        yield "sources", [_source_dict(i, h, False) for i, h in enumerate(res.hits, start=1)]

        if self._gate(res):
            timings["total"] = round((time.perf_counter() - t0) * 1000, 2)
            ans = self._refuse(q, res, timings, rid)
            yield "token", ans.text
        else:
            t_gen = time.perf_counter()
            parts: list[str] = []
            first_token_at: float | None = None
            truncated = False
            released = 0  # characters already sent as preview
            hold_all = self.settings.strict_grounding
            for piece in self.llm.stream(SYSTEM_PROMPT, build_user_prompt(q, res.hits),
                                         self.settings.llm_max_tokens):
                if piece == TRUNCATED:
                    truncated = True
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                parts.append(piece)
                text = "".join(parts)
                if hold_all or could_be_abstention(text):
                    continue
                yield "token", text[released:]
                released = len(text)
            raw = "".join(parts)
            if not raw.strip():
                # An empty completion is a backend failure, not an answer.
                raise ProviderError("model returned an empty completion")
            now = time.perf_counter()
            timings["first_token"] = round(((first_token_at or now) - t_gen) * 1000, 2)
            timings["generation"] = round((now - t_gen) * 1000, 2)
            timings["total"] = round((now - t0) * 1000, 2)
            ans = self._finish(q, res, raw, timings, rid, truncated)
            if released == 0:
                yield "token", ans.text

        self._cache_put(key, ans)
        self._record(ans)
        yield "final", ans
