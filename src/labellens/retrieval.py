"""Query understanding + hybrid retrieval + reranking.

Pipeline:  scope -> (BM25 || dense) -> RRF -> cross-encoder -> collapse -> top-k

* **Entity scoping.** Every label has a "Contraindications" section, so the
  dominant failure in this corpus is *right section, wrong drug*. Drug names in
  the query are linked against an alias table built at ingest time and used as
  a hard metadata filter. No model call, sub-millisecond, and the single
  largest quality lever in the ablation.
* **Two arms.** BM25 catches exact identifiers (drug names, enzyme names such
  as "CYP3A4", lab values) that dense models blur; dense catches paraphrase
  ("can I drink alcohol" -> "ethanol potentiates ..."). They fail on different
  queries, which is why fusing them beats either.
* **RRF.** Fusion is rank-based, so BM25 scores and cosine similarities - which
  live on unrelated scales - never need to be normalised or tuned.
* **Retrieve wide, rerank narrow.** The cross-encoder reads query and passage
  together, so it is far more accurate than either first-stage arm but ~1000x
  more expensive per document. It only ever sees the fused top ``rerank_n``.
"""

from __future__ import annotations

import difflib
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .models import Document, Hit, RetrievalResult
from .providers import Embedder, Reranker
from .store import DenseIndex, Store

_WORD = re.compile(r"[a-z0-9]+")
# Salt / ester words. Stripped only from the END of a generic name, so
# "metformin hydrochloride" -> "metformin" but "sodium chloride" stays whole.
_SALTS = frozenset(
    ["hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium", "sulfate", "tartrate", "succinate", "maleate", "mesylate", "besylate", "phosphate", "acetate", "citrate", "fumarate", "bromide", "hydrobromide", "nitrate", "dihydrate", "monohydrate", "anhydrous", "disodium", "dipotassium", "hemihydrate", "trihydrate", "bisulfate", "tromethamine", "propionate", "dipropionate", "dihydrochloride", "medoxomil", "mofetil", "axetil", "etexilate", "bitartrate", "oxalate", "hyclate", "gluconate", "acetonide", "cypionate", "valerate", "furoate", "butyrate", "decanoate", "enanthate", "pamoate", "tosylate", "lactate", "stearate", "palmitate", "chloride", "lysine", "hydrate", "sesquihydrate", "pentahydrate", "dimeglumine", "meglumine", "benzoate", "marboxil", "alafenamide", "disoproxil", "pivoxil", "proxetil", "cilexetil"]
)
# Aliases that are also ordinary words would hijack unrelated questions
# ("can I drink alcohol on antibiotics" must not scope to Dehydrated Alcohol).
_ALIAS_BLOCKLIST = frozenset(
    ["water", "oxygen", "air", "sodium", "potassium", "calcium", "magnesium", "iron", "zinc", "pain", "relief", "control", "cold", "allergy", "sleep", "care", "plus", "extra", "daily", "kit", "tablets", "tablet", "capsules", "injection", "solution", "alcohol", "dextrose", "glucose", "chloride", "bicarbonate", "iodide", "fluoride", "acid", "oil", "nitrogen", "helium", "urea", "sulfur", "charcoal", "glycerin", "sterile", "saline", "lubricant", "antacid", "fiber", "vitamin", "vitamins", "nicotine", "caffeine", "salt", "sugar"]
)
_BIOSIMILAR_SUFFIX = re.compile(r"-[a-z]{4}$")  # adalimumab-bwwd, insulin glargine-aglr
MAX_ALIAS_WORDS = 6


def normalize_name(name: str) -> str:
    return " ".join(_WORD.findall(name.lower()))


def strip_salt(name: str) -> str:
    """Base molecule name: drop a biosimilar suffix, then trailing salt words."""
    base = _BIOSIMILAR_SUFFIX.sub("", name.strip().lower())
    words = normalize_name(base).split()
    full = " ".join(words)
    while len(words) > 1 and words[-1] in _SALTS:
        words.pop()
    stripped = " ".join(words)
    # For electrolytes the salt *is* the drug: "sodium chloride" must not become "sodium".
    if stripped in _SALTS or stripped in _ALIAS_BLOCKLIST:
        return full
    return stripped


def _usable_alias(alias: str) -> bool:
    # Block single generic words ("chloride", "alcohol"), never a full name made
    # of them: "sodium bicarbonate" and "calcium chloride" are real drugs people ask about.
    words = alias.split()
    if len(words) == 1 and (alias in _SALTS or alias in _ALIAS_BLOCKLIST):
        return False
    return len(alias) >= 4 and 0 < len(words) <= MAX_ALIAS_WORDS


def drug_aliases(doc: Document) -> set[str]:
    """Names a user might type for this label: generic, brand, and base molecule."""
    out: set[str] = set()
    for name in [doc.generic_name, *doc.brand_names]:
        out.add(normalize_name(name))
        out.add(strip_salt(name))
    return {a for a in out if _usable_alias(a)}


class EntityLinker:
    def __init__(self, aliases: dict[str, set[str]],
                 term_exists: Callable[[str], bool] | None = None):
        self.aliases = aliases
        self._term_exists = term_exists
        # Every word that occurs in some drug name: the spelling-correction lexicon.
        self._lexicon = sorted({w for a in aliases for w in a.split() if len(w) >= 6})

    def link(self, query: str) -> tuple[set[str], list[str]]:
        """Return (doc_ids, matched_aliases) using greedy longest-match n-grams.

        Misspelled words are corrected against the drug-name lexicon *before*
        matching, so a typo inside a multi-word name ("isosoride dinitrate")
        still resolves to the full alias ("isosorbide dinitrate").
        """
        tokens = [self.correct(t) for t in _WORD.findall(query.lower())]
        doc_ids: set[str] = set()
        matched: list[str] = []
        i = 0
        while i < len(tokens):
            for n in range(min(MAX_ALIAS_WORDS, len(tokens) - i), 0, -1):
                gram = " ".join(tokens[i: i + n])
                if gram in self.aliases:
                    doc_ids |= self.aliases[gram]
                    matched.append(gram)
                    i += n
                    break
            else:
                i += 1
        return doc_ids, matched

    def correct(self, tok: str) -> str:
        """Typo tolerance ("metformine", "bicarbnate"), deliberately narrow.

        Many *different* drugs are one edit apart (norepinephrine/epinephrine,
        prednisolone/prednisone), and silently scoping to the wrong drug is the
        worst thing this module can do. So a correction is made only if the
        token is not itself a word anywhere in the corpus (a real drug named in
        another label's interaction section is not a typo), the replacement
        starts with the same letter, and the lengths differ by at most one.
        """
        if len(tok) < 6 or tok in self.aliases or (
                self._term_exists and self._term_exists(tok, exact=True)):
            return tok
        close = difflib.get_close_matches(tok, self._lexicon, n=1, cutoff=0.85)
        if close and close[0][0] == tok[0] and abs(len(close[0]) - len(tok)) <= 1:
            return close[0]
        return tok


def rrf(rankings: Iterable[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


DUPLICATE_JACCARD = 0.85


def collapse_near_duplicates(hits: list[Hit], limit: int) -> list[Hit]:
    """Keep the best-ranked copy of each near-identical passage.

    The same molecule is sold by many manufacturers whose labels share almost
    all of their text, and class-wide warnings are repeated verbatim across
    drugs. Without this, the top-k is one paragraph six times and the model
    sees no second piece of evidence. Suppressed chunk ids are recorded on the
    surviving hit so nothing is silently lost.
    """
    kept: list[tuple[Hit, set[str]]] = []
    for h in hits:
        toks = set(_WORD.findall(h.chunk.text.lower()))
        for other, other_toks in kept:
            union = len(toks | other_toks)
            if union and len(toks & other_toks) / union >= DUPLICATE_JACCARD:
                other.dup_ids.append(h.chunk.chunk_id)
                break
        else:
            if len(kept) < limit:
                kept.append((h, toks))
    return [h for h, _ in kept]


@dataclass(frozen=True)
class _Snapshot:
    """Everything derived from one index version, swapped in as a unit so a
    request never mixes the dense matrix of one version with the aliases of another."""

    dense: DenseIndex
    linker: EntityLinker
    doc_names: dict[str, str]
    version: int


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, reranker: Reranker | None,
                 settings: Settings, variant: str = "context"):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings
        self.variant = variant  # "context" (header + text) or "raw" (ablation)
        self._qcache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._reload_lock = threading.Lock()
        self.reload()

    # ------------------------------------------------------------ index state
    def reload(self) -> None:
        version = self.store.index_version  # read first: never label new data with an old version
        self._snap = _Snapshot(
            dense=self.store.load_dense(self.settings.embed_index_name or self.embedder.model,
                                        self.variant),
            linker=EntityLinker(self.store.aliases(), self.store.term_exists),
            doc_names=self.store.doc_names(),
            version=version,
        )

    def refresh_if_stale(self) -> bool:
        """Pick up a re-ingest without a restart (one metadata read per call)."""
        if self.store.index_version == self._snap.version:
            return False
        with self._reload_lock:  # one reload, not one per concurrent request
            if self.store.index_version != self._snap.version:
                self.reload()
                return True
        return False

    @property
    def dense(self) -> DenseIndex:
        return self._snap.dense

    @property
    def linker(self) -> EntityLinker:
        return self._snap.linker

    @property
    def index_version(self) -> int:
        return self._snap.version

    # ---------------------------------------------------------------- search
    def _embed_query(self, query: str) -> np.ndarray:
        cached = self._qcache.get(query)
        if cached is not None:
            return cached
        vec = self.embedder.embed([self.settings.embed_query_prefix + query])[0]
        self._qcache[query] = vec
        while len(self._qcache) > 2048:
            self._qcache.popitem(last=False)
        return vec

    def retrieve(self, query: str, settings: Settings | None = None) -> RetrievalResult:
        cfg = settings or self.settings
        self.refresh_if_stale()
        snap = self._snap  # one consistent view for the whole request
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        scope: set[str] | None = None
        unknown: list[str] = []
        if cfg.use_entity_scope:
            ids, _ = snap.linker.link(query)
            scope = ids or None
            if scope is None:
                # No known drug in the question. If it also contains a word that
                # occurs nowhere in the corpus, the question is almost certainly
                # about something that is not indexed (or is misspelled beyond
                # repair). First-stage retrieval would still return 50
                # relevant-looking dosing passages for *other* drugs, so this is
                # the only place the problem is visible.
                unknown = [t for t in dict.fromkeys(_WORD.findall(query.lower()))
                           if len(t) >= 5 and not t.isdigit() and not self.store.term_exists(t)
                           and snap.linker.correct(t) == t]
        timings["scope"] = (time.perf_counter() - t0) * 1000

        bm25: list[tuple[int, float]] = []
        dense: list[tuple[int, float]] = []
        if cfg.use_bm25:
            t = time.perf_counter()
            header_weight = 2.0 if self.variant == "context" else 0.0
            bm25 = self.store.bm25_search(query, cfg.bm25_k, scope, header_weight)
            timings["bm25"] = (time.perf_counter() - t) * 1000
        if cfg.use_dense and len(snap.dense):
            t = time.perf_counter()
            qvec = self._embed_query(query)
            timings["embed_query"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            dense = snap.dense.search(qvec, cfg.dense_k, scope)
            timings["dense"] = (time.perf_counter() - t) * 1000

        bm25_rank = {rid: i for i, (rid, _) in enumerate(bm25, start=1)}
        dense_rank = {rid: i for i, (rid, _) in enumerate(dense, start=1)}
        fused = rrf([[r for r, _ in bm25], [r for r, _ in dense]], cfg.rrf_k)
        ordered = sorted(fused, key=lambda r: fused[r], reverse=True)

        # Every configuration ranks the same pool depth, so ablations differ only
        # in *how* the pool is ordered, never in how much of it they see.
        pool = ordered[: cfg.rerank_n]
        chunks = self.store.get_chunks(pool)
        hits = [
            Hit(rowid=r, chunk=chunks[r][0], drug=chunks[r][1], bm25_rank=bm25_rank.get(r),
                dense_rank=dense_rank.get(r), rrf_score=fused[r])
            for r in pool if r in chunks
        ]

        top_score: float | None = None
        if cfg.use_rerank and self.reranker and hits:
            t = time.perf_counter()
            passages = [h.chunk.embed_text if self.variant == "context" else h.chunk.text
                        for h in hits]
            for h, s in zip(hits, self.reranker.score(query, passages)):
                h.rerank_score = s
            hits.sort(key=lambda h: h.rerank_score, reverse=True)
            timings["rerank"] = (time.perf_counter() - t) * 1000
            top_score = hits[0].rerank_score

        unique = collapse_near_duplicates(hits, cfg.final_k)
        timings["total"] = (time.perf_counter() - t0) * 1000
        return RetrievalResult(
            query=query, hits=unique,
            scoped_drugs=sorted({snap.doc_names.get(d, d) for d in (scope or ())}),
            scoped_doc_ids=sorted(scope or ()),
            timings_ms={k: round(v, 2) for k, v in timings.items()}, top_score=top_score,
            unknown_terms=unknown, index_version=snap.version,
        )
