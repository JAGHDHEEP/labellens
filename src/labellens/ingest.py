"""Incremental, idempotent ingestion.

Labels are revised constantly (new boxed warnings, new interactions). A
production index must follow the source without a full rebuild, so ingest is a
*diff* against what is already indexed:

* unchanged label (same content hash)  -> skipped, zero work
* revised label                         -> chunks/FTS/aliases replaced atomically
* label missing from the new snapshot   -> removed (with ``prune=True``)
* embeddings are computed only for chunk texts never seen before, because the
  embedding table is keyed by the hash of the embedded text.

Running the command twice is a no-op; a crash mid-run is safe to re-run.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

from .chunking import chunk_document
from .config import Settings
from .normalize import normalize_label
from .providers import Embedder
from .retrieval import drug_aliases
from .store import Store

log = logging.getLogger(__name__)

# Bump when cleaning or chunking logic changes. It is folded into each stored
# document fingerprint, so a code change re-indexes every label on the next
# ingest instead of silently leaving old-style chunks in the index.
PIPELINE_VERSION = "4"


@dataclass
class IngestReport:
    seen: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    removed: int = 0
    chunks_written: int = 0
    embedded: int = 0
    seconds: float = 0.0


def ingest_records(store: Store, records: Iterable[dict], settings: Settings,
                   prune: bool = False, bump: bool = True) -> IngestReport:
    """Diff ``records`` against the index. Pass ``bump=False`` when embedding
    follows immediately, so a running server reloads once, after the vectors
    for the new chunks exist, instead of serving them BM25-only in between."""
    rep = IngestReport()
    t0 = time.perf_counter()
    known = store.doc_hashes()
    present: set[str] = set()
    for rec in records:
        rep.seen += 1
        doc = normalize_label(rec)
        if doc is None:
            rep.skipped += 1
            continue
        present.add(doc.doc_id)
        doc.content_hash = (f"{PIPELINE_VERSION}:{settings.chunk_max_words}:"
                            f"{settings.chunk_min_words}:{doc.content_hash}")
        if known.get(doc.doc_id) == doc.content_hash:
            rep.unchanged += 1
            continue
        chunks = chunk_document(doc, settings.chunk_max_words, settings.chunk_min_words)
        if not chunks:
            rep.skipped += 1
            continue
        store.upsert_document(doc, chunks, drug_aliases(doc))
        rep.chunks_written += len(chunks)
        if doc.doc_id in known:
            rep.updated += 1
        else:
            rep.added += 1
    if prune:
        for doc_id in set(known) - present:
            store.delete_document(doc_id)
            rep.removed += 1
    # The vocabulary is append-only during upserts; rebuild it when labels were
    # removed (or for an index created before it existed) so it never vouches
    # for a drug that is no longer indexed.
    if rep.removed or (store.vocab_size() == 0 and store.stats()["chunks"]):
        store.rebuild_vocab()
    if bump and (rep.added or rep.updated or rep.removed):
        store.bump_version()
    rep.seconds = round(time.perf_counter() - t0, 2)
    return rep


def embed_missing(store: Store, embedder: Embedder, variant: str = "context",
                  batch: int = 64) -> int:
    """Embed every chunk text that has no vector yet. Resumable: commits per batch."""
    todo = store.missing_embeddings(embedder.model, variant)
    if not todo:
        return 0
    t0 = time.perf_counter()
    for i in range(0, len(todo), batch):
        part = todo[i: i + batch]
        vecs = embedder.embed([t for _, t in part])
        store.put_embeddings(embedder.model, [(h, v) for (h, _), v in zip(part, vecs)])
        done = i + len(part)
        if (i // batch) % 20 == 0 or done == len(todo):
            rate = done / max(1e-6, time.perf_counter() - t0)
            log.info("embedded %d/%d (%s) %.0f chunks/s", done, len(todo), variant, rate)
    store.gc_embeddings()
    store.bump_version()
    return len(todo)
