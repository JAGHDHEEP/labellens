"""HTTP API (FastAPI).

Endpoints are synchronous ``def`` handlers on purpose: the pipeline is
CPU/IO-bound blocking code (SQLite, ONNX, HTTP to the model server), so
FastAPI runs each request in its thread pool instead of blocking the event loop.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .providers import ProviderError
from .service import MAX_QUESTION_CHARS, InvalidQuestion, RagService, validate_question

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(service: RagService | None = None, settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            from .bootstrap import build_service  # heavy imports only when serving for real

            app.state.service = build_service(settings or Settings.from_env())
        yield

    app = FastAPI(title="LabelLens", version=__version__, lifespan=lifespan)
    app.state.service = service

    def svc() -> RagService:
        return app.state.service

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    @app.exception_handler(InvalidQuestion)
    async def bad_question(_: Request, exc: InvalidQuestion):
        return JSONResponse(status_code=422, content={"error": str(exc)})

    @app.exception_handler(ProviderError)
    async def provider_down(_: Request, exc: ProviderError):
        # A model backend is unavailable: fail loudly and retryably rather than
        # answer without retrieval or without verification.
        log.error("provider error: %s", exc)
        return JSONResponse(status_code=503, content={"error": "model backend unavailable"},
                            headers={"Retry-After": "5"})

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    def readyz():
        svc().retriever.refresh_if_stale()
        n = len(svc().retriever.dense)
        if n == 0:
            return JSONResponse(status_code=503, content={"status": "index empty"})
        return {"status": "ready", "chunks": n, "index_version": svc().retriever.index_version}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/stats")
    def stats():
        s = svc()
        return {**s.retriever.store.stats(), "llm": s.llm.model,
                "embedder": s.retriever.embedder.model,
                "abstain_threshold": s.settings.abstain_threshold}

    @app.post("/v1/search")
    def search(req: AskRequest):
        """Retrieval only - the debugging endpoint: why did this chunk rank here?"""
        res = svc().retriever.retrieve(validate_question(req.question))
        return {
            "scoped_drugs": res.scoped_drugs, "top_score": res.top_score,
            "timings_ms": res.timings_ms,
            "hits": [{"drug": h.drug, "section": h.chunk.section_title,
                      "subsection": h.chunk.subsection, "bm25_rank": h.bm25_rank,
                      "dense_rank": h.dense_rank, "rrf_score": round(h.rrf_score, 5),
                      "rerank_score": h.rerank_score, "text": h.chunk.text} for h in res.hits],
        }

    @app.post("/v1/ask")
    def ask(req: AskRequest):
        return asdict(svc().ask(req.question))

    @app.post("/v1/ask/stream")
    def ask_stream(req: AskRequest):
        question = validate_question(req.question)  # fail before the stream opens

        def events():
            try:
                for event, payload in svc().ask_stream(question):
                    yield _sse(event, asdict(payload) if event == "final" else payload)
            except ProviderError as exc:
                log.error("provider error mid-stream: %s", exc)
                yield _sse("error", {"error": "model backend unavailable"})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
