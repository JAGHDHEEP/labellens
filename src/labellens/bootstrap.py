"""Composition root: the only place concrete providers are wired together."""

from __future__ import annotations

from .config import Settings
from .providers import CrossEncoderReranker, OllamaEmbedder, build_llm
from .retrieval import Retriever
from .service import RagService
from .store import Store


def build_retriever(settings: Settings, store: Store | None = None,
                    variant: str = "context") -> Retriever:
    store = store or Store(settings.db_path)
    embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model, settings.embed_batch)
    reranker = CrossEncoderReranker(settings.rerank_model) if settings.use_rerank else None
    return Retriever(store, embedder, reranker, settings, variant=variant)


def build_service(settings: Settings) -> RagService:
    retriever = build_retriever(settings)
    llm = build_llm(settings.llm_provider, settings.llm_model, settings.ollama_url,
                    settings.llm_timeout_s)
    return RagService(retriever, llm, settings)
