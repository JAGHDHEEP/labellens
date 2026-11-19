"""Runtime configuration.

Every knob is an environment variable (LABELLENS_<FIELD>), so the same image
runs locally, in CI and in production with no code changes (12-factor).
Defaults are the values the evaluation in eval/results was run with.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

_PREFIX = "LABELLENS_"


@dataclass(frozen=True)
class Settings:
    # --- storage -----------------------------------------------------------
    data_dir: Path = Path("data")
    db_name: str = "labellens.db"

    # --- model providers ---------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    embed_batch: int = 32
    llm_provider: str = "ollama"  # "ollama" | "anthropic"
    llm_model: str = "llama3.1:8b"
    llm_timeout_s: float = 120.0
    llm_max_tokens: int = 700
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- chunking ----------------------------------------------------------
    chunk_max_words: int = 220
    chunk_min_words: int = 25

    # --- retrieval ---------------------------------------------------------
    bm25_k: int = 50  # candidates from the lexical arm
    dense_k: int = 50  # candidates from the dense arm
    rrf_k: int = 60  # RRF damping constant (Cormack et al. 2009 default)
    rerank_n: int = 30  # fused candidates sent to the cross-encoder
    final_k: int = 6  # chunks placed in the prompt
    use_bm25: bool = True
    use_dense: bool = True
    use_rerank: bool = True
    use_entity_scope: bool = True
    # Reranker logit below which we refuse *before* spending an LLM call.
    # Calibrated on the golden set (see eval/results); -inf disables the gate.
    abstain_threshold: float = -4.0

    # --- serving -----------------------------------------------------------
    cache_size: int = 512
    strict_grounding: bool = False  # True: replace ungrounded answers with a refusal

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_name

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        values: dict = {}
        for f in fields(cls):
            raw = os.environ.get(_PREFIX + f.name.upper())
            if raw is None:
                continue
            default = getattr(cls, f.name)
            if isinstance(default, bool):
                values[f.name] = raw.strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(default, int):
                values[f.name] = int(raw)
            elif isinstance(default, float):
                values[f.name] = float(raw)
            elif isinstance(default, Path):
                values[f.name] = Path(raw)
            else:
                values[f.name] = raw
        values.update(overrides)
        return cls(**values)

    def with_(self, **changes) -> Settings:
        return replace(self, **changes)
