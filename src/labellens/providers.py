"""Model providers behind small protocols.

The pipeline depends on three capabilities - embed, rerank, generate - never on
a vendor. Each has a production implementation and a deterministic fake, so the
whole system (including the HTTP API) is unit-testable with no GPU, no network
and no API key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterator
from typing import Protocol

import httpx
import numpy as np

log = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """A model backend failed after retries."""


# Yielded as the last stream item when the model hit its output-token limit, so
# the service can flag a cut-off answer instead of presenting it as complete.
TRUNCATED = "\x00TRUNCATED"


# ------------------------------------------------------------------ embedders
class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.clip(norms, 1e-12, None)).astype(np.float32)


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, batch: int = 32, timeout: float = 120.0):
        self.model = model
        self.batch = batch
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch):
            out.append(self._embed_batch(texts[i: i + self.batch]))
        return _unit(np.vstack(out)) if out else np.zeros((0, 1), np.float32)

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        for attempt in range(4):
            try:
                r = self._http.post("/api/embed", json={"model": self.model, "input": batch})
                r.raise_for_status()
                return np.asarray(r.json()["embeddings"], dtype=np.float32)
            except (httpx.HTTPError, KeyError) as exc:
                if attempt == 3:
                    raise ProviderError(f"embedding failed: {exc}") from exc
                time.sleep(1.5**attempt)
        raise ProviderError("unreachable")


class FastEmbedEmbedder:
    """ONNX embeddings on CPU via fastembed: no model server, ~70 MB for bge-small.

    Used by the hosted demo, which has no GPU and about 1 GB of memory.
    """

    def __init__(self, model: str, batch: int = 64, parallel: int | None = None):
        from fastembed import TextEmbedding

        self.model = model
        self.batch = batch
        self.parallel = parallel  # 0 = all CPU cores (bulk indexing); None = in-process
        self._enc = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), np.float32)
        vecs = self._enc.embed(texts, batch_size=self.batch, parallel=self.parallel)
        return _unit(np.asarray(list(vecs), dtype=np.float32))


class StoredAs:
    """Wrap an embedder so its vectors are stored under another model name."""

    def __init__(self, inner: Embedder, name: str):
        self.inner = inner
        self.model = name

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.inner.embed(texts)


def build_embedder(settings) -> Embedder:
    if settings.embed_provider == "fastembed":
        return FastEmbedEmbedder(settings.embed_model, settings.embed_batch)
    if settings.embed_provider == "ollama":
        return OllamaEmbedder(settings.ollama_url, settings.embed_model, settings.embed_batch)
    raise ValueError(f"unknown embed provider: {settings.embed_provider}")


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests and CI."""

    def __init__(self, dim: int = 256):
        self.model = f"hash-{dim}"
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                m[i, h % self.dim] += 1.0
        return _unit(m)


# ------------------------------------------------------------------ rerankers
class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """ONNX cross-encoder via fastembed: no PyTorch in the serving image."""

    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = model
        self._enc = TextCrossEncoder(model_name=model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._enc.rerank(query, passages)]


class OverlapReranker:
    """Test double: scores by query-term overlap, scaled like a logit."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        from .store import STOPWORDS

        q = set(re.findall(r"[a-z0-9]+", query.lower())) - STOPWORDS
        out = []
        for p in passages:
            toks = set(re.findall(r"[a-z0-9]+", p.lower()))
            out.append(10.0 * len(q & toks) / max(1, len(q)) - 5.0)
        return out


# ----------------------------------------------------------------------- LLMs
class LLM(Protocol):
    model: str

    def generate(self, system: str, user: str, max_tokens: int = 700) -> str: ...

    def stream(self, system: str, user: str, max_tokens: int = 700) -> Iterator[str]: ...


class OllamaLLM:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0,
                 think: bool | None = None, num_ctx: int = 8192, fmt: str | None = None):
        self.model = model
        self.think = think
        self.fmt = fmt  # "json" constrains output to valid JSON (used by the eval judge)
        self.num_ctx = num_ctx
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def _payload(self, system: str, user: str, max_tokens: int, stream: bool) -> dict:
        body = {
            "model": self.model,
            "stream": stream,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            # temperature 0 + fixed seed: answers must be reproducible to be evaluable.
            "options": {"temperature": 0, "seed": 7, "num_ctx": self.num_ctx,
                        "num_predict": max_tokens},
        }
        if self.think is not None:
            body["think"] = self.think
        if self.fmt:
            body["format"] = self.fmt
        return body

    def generate(self, system: str, user: str, max_tokens: int = 700) -> str:
        for attempt in range(3):
            try:
                r = self._http.post("/api/chat", json=self._payload(system, user, max_tokens, False))
                r.raise_for_status()
                return r.json()["message"]["content"].strip()
            except (httpx.HTTPError, KeyError) as exc:
                if attempt == 2:
                    raise ProviderError(f"generation failed: {exc}") from exc
                time.sleep(2.0**attempt)
        raise ProviderError("unreachable")

    def stream(self, system: str, user: str, max_tokens: int = 700) -> Iterator[str]:
        try:
            with self._http.stream("POST", "/api/chat",
                                   json=self._payload(system, user, max_tokens, True)) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    msg = json.loads(line)
                    if msg.get("error"):  # Ollama reports mid-stream failures in-band
                        raise ProviderError(f"generation failed: {msg['error']}")
                    piece = msg.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if msg.get("done") and msg.get("done_reason") == "length":
                        yield TRUNCATED
        except httpx.HTTPError as exc:
            raise ProviderError(f"generation failed: {exc}") from exc


class AnthropicLLM:
    """Claude via the official SDK (optional extra: ``pip install labellens[anthropic]``).

    Credentials are resolved by the SDK (ANTHROPIC_API_KEY or an ``ant auth``
    profile). Sampling parameters are not sent: current Claude models reject
    ``temperature`` and reason adaptively; ``effort`` is the cost/latency knob.
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", timeout: float = 120.0):
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self.effort = effort
        self._client = anthropic.Anthropic(timeout=timeout)

    def _open(self, system: str, user: str, max_tokens: int):
        # max_tokens also covers reasoning tokens, so leave headroom above the answer budget.
        return self._client.messages.stream(
            model=self.model,
            max_tokens=max(4096, max_tokens * 4),
            system=system,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": user}],
        )

    def generate(self, system: str, user: str, max_tokens: int = 700) -> str:
        return "".join(self.stream(system, user, max_tokens)).strip()

    def stream(self, system: str, user: str, max_tokens: int = 700) -> Iterator[str]:
        a = self._anthropic
        try:
            with self._open(system, user, max_tokens) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
            if final.stop_reason == "refusal":
                raise ProviderError("model declined the request")
            if final.stop_reason == "max_tokens":
                yield TRUNCATED
        except a.RateLimitError as exc:
            raise ProviderError(f"rate limited: {exc}") from exc
        except a.APIStatusError as exc:
            raise ProviderError(f"API error {exc.status_code}: {exc}") from exc
        except a.APIConnectionError as exc:
            raise ProviderError(f"connection error: {exc}") from exc


class FakeLLM:
    """Test double: returns a canned reply, or echoes the first source with a citation."""

    def __init__(self, reply: str | None = None):
        self.model = "fake"
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str, max_tokens: int = 700) -> str:
        self.calls.append((system, user))
        if self.reply is not None:
            return self.reply
        m = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\n|\Z)", user, flags=re.S)
        first = (m.group(1).strip().split(". ")[0] if m else "No sources").rstrip(".")
        return f"{first}. [1]"

    def stream(self, system: str, user: str, max_tokens: int = 700) -> Iterator[str]:
        text = self.generate(system, user, max_tokens)
        for i in range(0, len(text), 12):
            yield text[i: i + 12]


class GroqLLM:
    """Groq's OpenAI-compatible chat API (hosted demo; free tier available).

    The key is read from ``GROQ_API_KEY`` and never logged. Reasoning models
    (gpt-oss) return their reasoning in a separate field, so only the answer
    text is streamed; their output budget includes reasoning, hence the
    larger ``max_tokens`` sent to the API.
    """

    BASE_URL = "https://api.groq.com/openai/v1"
    MAX_WAIT_S = 12.0  # wait out a short rate-limit window rather than fail the user

    def __init__(self, model: str = "openai/gpt-oss-20b", timeout: float = 60.0,
                 api_key: str | None = None, transport: httpx.BaseTransport | None = None,
                 fallbacks: list[str] | None = None, sleep=time.sleep):
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ProviderError("GROQ_API_KEY is not set")
        self.model = model
        # Groq's free tier meters tokens per minute *per model*, so a second
        # model is a second budget: on 429 the request moves to the next one.
        self.models = [model, *(fallbacks or [])]
        self._sleep = sleep
        self._http = httpx.Client(base_url=self.BASE_URL, timeout=timeout, transport=transport,
                                  headers={"Authorization": f"Bearer {key}"})

    @staticmethod
    def _payload(model: str, system: str, user: str, max_tokens: int) -> dict:
        reasoning = "gpt-oss" in model
        body = {
            "model": model, "stream": True, "temperature": 0, "seed": 7,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens * 4 if reasoning else max_tokens,
        }
        if reasoning:
            body["reasoning_effort"] = "low"
        return body

    def generate(self, system: str, user: str, max_tokens: int = 700) -> str:
        return "".join(p for p in self.stream(system, user, max_tokens) if p != TRUNCATED).strip()

    def stream(self, system: str, user: str, max_tokens: int = 700) -> Iterator[str]:
        """Try each model in turn on a rate limit; then wait once if the wait is short."""
        waits: list[float] = []
        for attempt in range(2):
            for model in self.models:
                retry_after = yield from self._stream_one(model, system, user, max_tokens)
                if retry_after is None:
                    return
                waits.append(retry_after)
            wait = min(waits)
            if attempt == 0 and wait <= self.MAX_WAIT_S:
                self._sleep(wait)
                waits.clear()
                continue
            break
        raise ProviderError("rate limited by Groq; try again shortly")

    def _stream_one(self, model: str, system: str, user: str, max_tokens: int):
        """Stream one model's answer. Returns None when done, or the
        Retry-After seconds when the model was rate-limited before any output."""
        try:
            with self._http.stream("POST", "/chat/completions",
                                   json=self._payload(model, system, user, max_tokens)) as r:
                if r.status_code == 429:
                    try:
                        return float(r.headers.get("retry-after", "30"))
                    except ValueError:
                        return 30.0
                if r.status_code >= 400:
                    r.read()
                    raise ProviderError(f"Groq API error {r.status_code}: {r.text[:200]}")
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    choice = (json.loads(data).get("choices") or [{}])[0]
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        yield piece
                    if choice.get("finish_reason") == "length":
                        yield TRUNCATED
        except httpx.HTTPError as exc:
            raise ProviderError(f"generation failed: {exc}") from exc
        return None


def build_llm(provider: str, model: str, ollama_url: str, timeout: float = 120.0,
              fallbacks: list[str] | None = None) -> LLM:
    if provider == "groq":
        return GroqLLM(model=model, timeout=timeout, fallbacks=fallbacks)
    if provider == "anthropic":
        return AnthropicLLM(model=model, timeout=timeout)
    if provider == "ollama":
        return OllamaLLM(ollama_url, model, timeout=timeout)
    if provider == "fake":
        return FakeLLM()
    raise ValueError(f"unknown llm provider: {provider}")
