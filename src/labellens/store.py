"""Single-file index: documents, chunks, BM25 (FTS5) and embeddings in SQLite.

Why one transactional store instead of "a vector DB + a search engine":

* The two classic production failures of hybrid RAG are *index drift* (the
  vector index and the keyword index disagree about which chunks exist after a
  partial re-ingest) and *orphaned vectors* after a document is updated.
  Here a label revision replaces its chunks, FTS rows and aliases in one
  transaction, so both arms always see the same corpus.
* At this corpus size (tens of thousands of chunks) exact brute-force cosine
  over a contiguous float32 matrix takes a few milliseconds and has 100 %
  recall. An ANN index would add a recall/latency trade-off and an operational
  dependency while being *slower* end to end. The dense index sits behind a
  two-method interface (``DenseIndex.search``) so it can be swapped for
  pgvector/Qdrant when the corpus outgrows RAM - see docs/DECISIONS.md.

Embeddings are keyed by the hash of the exact text that was embedded, which
gives idempotent re-ingestion for free: unchanged chunks are never re-embedded.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .models import Chunk, Document

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
    doc_id TEXT PRIMARY KEY, version INTEGER, effective_date TEXT,
    generic_name TEXT, display_name TEXT, brand_names TEXT, manufacturer TEXT,
    route TEXT, pharm_class TEXT, content_hash TEXT, n_chunks INTEGER, ingested_at TEXT);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, doc_id TEXT NOT NULL,
    section_key TEXT, section_title TEXT, subsection TEXT, ordinal INTEGER,
    kind TEXT, text TEXT, header TEXT, text_hash TEXT, raw_hash TEXT, n_words INTEGER);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id, ordinal);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    header, text, tokenize='porter unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS embeddings(
    text_hash TEXT, model TEXT, vec BLOB, PRIMARY KEY(text_hash, model)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS aliases(
    alias TEXT, doc_id TEXT, PRIMARY KEY(alias, doc_id)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS vocab(term TEXT PRIMARY KEY) WITHOUT ROWID;
"""

_TOKEN = re.compile(r"[a-z0-9]+")
# Function words carry no BM25 signal and only widen the OR query.
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have", "how", "i", "if", "in", "is", "it", "its", "my", "of", "on", "or", "should", "that", "the", "their", "there", "this", "to", "was", "what", "when", "which", "who", "will", "with", "you", "your", "about", "any", "been", "did", "during", "get", "into", "may", "me", "much", "not", "than", "then", "these", "they", "those", "use", "used", "using", "were", "would", "could", "take", "taking", "taken", "give", "given", "patient", "patients"]
)


def fts_query(text: str) -> str:
    """Build a safe FTS5 OR-query. User text never reaches the FTS parser raw."""
    tokens = [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]
    seen: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.append(t)
    return " OR ".join(f'"{t}"' for t in seen[:32])


class DenseIndex:
    """Exact cosine search over an in-memory matrix of unit vectors."""

    def __init__(self, rowids: np.ndarray, doc_codes: np.ndarray, matrix: np.ndarray,
                 doc_index: dict[str, int]):
        self.rowids = rowids
        self.doc_codes = doc_codes
        self.matrix = matrix
        self.doc_index = doc_index

    def __len__(self) -> int:
        return len(self.rowids)

    def search(self, qvec: np.ndarray, k: int, doc_ids: Iterable[str] | None = None) -> list[tuple[int, float]]:
        if len(self.rowids) == 0:
            return []
        if doc_ids is not None:
            codes = [self.doc_index[d] for d in doc_ids if d in self.doc_index]
            idx = np.flatnonzero(np.isin(self.doc_codes, codes))
            if len(idx) == 0:
                return []
            scores = self.matrix[idx] @ qvec
        else:
            idx = None
            scores = self.matrix @ qvec
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        rows = self.rowids[idx[top]] if idx is not None else self.rowids[top]
        return [(int(r), float(s)) for r, s in zip(rows, scores[top])]


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # One connection guarded by a lock: the API serves reads from a thread
        # pool, and SQLite serialises writers anyway.
        # A generous busy timeout: ingest and embedding jobs may write while the
        # API or an evaluation reads.
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, value))
            self._db.commit()

    @property
    def index_version(self) -> int:
        return int(self.get_meta("index_version", "0"))

    def bump_version(self) -> int:
        v = self.index_version + 1
        self.set_meta("index_version", str(v))
        return v

    # ------------------------------------------------------------- documents
    def doc_hashes(self) -> dict[str, str]:
        with self._lock:
            rows = self._db.execute("SELECT doc_id, content_hash FROM documents").fetchall()
        return {r["doc_id"]: r["content_hash"] for r in rows}

    def _delete_doc(self, doc_id: str) -> None:
        self._db.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id=?)", (doc_id,))
        self._db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self._db.execute("DELETE FROM aliases WHERE doc_id=?", (doc_id,))
        self._db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))

    def delete_document(self, doc_id: str) -> None:
        with self._lock, self._db:
            self._delete_doc(doc_id)

    def upsert_document(self, doc: Document, chunks: list[Chunk], aliases: Iterable[str]) -> None:
        """Atomically replace a document and everything derived from it."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, self._db:  # one transaction: all arms change together
            self._delete_doc(doc.doc_id)
            self._db.execute(
                "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (doc.doc_id, doc.version, doc.effective_date, doc.generic_name, doc.display_name,
                 json.dumps(doc.brand_names), doc.manufacturer, doc.route, doc.pharm_class,
                 doc.content_hash, len(chunks), now))
            for c in chunks:
                raw_hash = hashlib.sha256(c.text.encode("utf-8")).hexdigest()
                cur = self._db.execute(
                    "INSERT INTO chunks(chunk_id, doc_id, section_key, section_title, subsection,"
                    " ordinal, kind, text, header, text_hash, raw_hash, n_words)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (c.chunk_id, c.doc_id, c.section_key, c.section_title, c.subsection, c.ordinal,
                     c.kind, c.text, c.header, c.text_hash, raw_hash, c.n_words))
                self._db.execute("INSERT INTO chunks_fts(rowid, header, text) VALUES(?,?,?)",
                                 (cur.lastrowid, c.header, c.text))
            self._db.executemany("INSERT OR IGNORE INTO vocab VALUES(?)", [
                (t,) for t in {t for c in chunks for t in _TOKEN.findall(c.embed_text.lower())}])
            self._db.executemany("INSERT OR IGNORE INTO aliases VALUES(?,?)",
                                 [(a, doc.doc_id) for a in aliases])

    def aliases(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        with self._lock:
            for r in self._db.execute("SELECT alias, doc_id FROM aliases"):
                out.setdefault(r["alias"], set()).add(r["doc_id"])
        return out

    def doc_names(self) -> dict[str, str]:
        with self._lock:
            rows = self._db.execute("SELECT doc_id, display_name FROM documents").fetchall()
        return {r["doc_id"]: r["display_name"] for r in rows}

    # ------------------------------------------------------------ embeddings
    def missing_embeddings(self, model: str, variant: str = "context") -> list[tuple[str, str]]:
        """(hash, text) pairs that still need a vector for ``model``."""
        col, expr = (("text_hash", "header || char(10) || text") if variant == "context"
                     else ("raw_hash", "text"))
        with self._lock:
            rows = self._db.execute(
                f"SELECT DISTINCT c.{col} AS h, {expr} AS t FROM chunks c "
                f"LEFT JOIN embeddings e ON e.text_hash = c.{col} AND e.model = ? "
                "WHERE e.text_hash IS NULL", (model,)).fetchall()
        return [(r["h"], r["t"]) for r in rows]

    def put_embeddings(self, model: str, items: list[tuple[str, np.ndarray]]) -> None:
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES(?,?,?)",
                [(h, model, np.asarray(v, dtype=np.float32).tobytes()) for h, v in items])

    def gc_embeddings(self) -> int:
        """Drop vectors no chunk references any more (after label revisions)."""
        with self._lock, self._db:
            cur = self._db.execute(
                "DELETE FROM embeddings WHERE text_hash NOT IN (SELECT text_hash FROM chunks)"
                " AND text_hash NOT IN (SELECT raw_hash FROM chunks)")
            return cur.rowcount

    def load_dense(self, model: str, variant: str = "context") -> DenseIndex:
        col = "text_hash" if variant == "context" else "raw_hash"
        with self._lock:
            rows = self._db.execute(
                f"SELECT c.id, c.doc_id, e.vec FROM chunks c JOIN embeddings e "
                f"ON e.text_hash = c.{col} AND e.model = ? ORDER BY c.id", (model,)).fetchall()
        doc_index: dict[str, int] = {}
        if not rows:
            return DenseIndex(np.zeros(0, np.int64), np.zeros(0, np.int32),
                              np.zeros((0, 1), np.float32), doc_index)
        dim = len(rows[0]["vec"]) // 4
        matrix = np.empty((len(rows), dim), dtype=np.float32)
        rowids = np.empty(len(rows), dtype=np.int64)
        codes = np.empty(len(rows), dtype=np.int32)
        for i, r in enumerate(rows):
            matrix[i] = np.frombuffer(r["vec"], dtype=np.float32)
            rowids[i] = r["id"]
            codes[i] = doc_index.setdefault(r["doc_id"], len(doc_index))
        return DenseIndex(rowids, codes, matrix, doc_index)

    # ---------------------------------------------------------------- search
    def bm25_search(self, query: str, k: int, doc_ids: Iterable[str] | None = None,
                    header_weight: float = 2.0) -> list[tuple[int, float]]:
        match = fts_query(query)
        if not match:
            return []
        if header_weight <= 0:
            # Header-less ablation: a zero weight alone would still let a chunk
            # *match* through its header; restrict matching to the text column.
            match = f"text : ({match})"
        sql = ("SELECT chunks_fts.rowid AS id, bm25(chunks_fts, ?, 1.0) AS s FROM chunks_fts "
               "{join} WHERE chunks_fts MATCH ? {where} ORDER BY s LIMIT ?")
        params: list = [header_weight]
        if doc_ids is not None:
            ids = list(doc_ids)
            if not ids:
                return []
            sql = sql.format(join="JOIN chunks c ON c.id = chunks_fts.rowid",
                             where=f"AND c.doc_id IN ({','.join('?' * len(ids))})")
            params += [match, *ids, k]
        else:
            sql = sql.format(join="", where="")
            params += [match, k]
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        # FTS5 bm25() is "lower is better"; flip the sign so higher = better.
        return [(r["id"], -float(r["s"])) for r in rows]

    def term_exists(self, token: str, exact: bool = False) -> bool:
        """Does ``token`` occur anywhere in the corpus? One indexed lookup.

        ``exact=False`` matches through the Porter stemmer, so "kidneys" counts
        as present when only "kidney" occurs - right for deciding whether a
        question is about something outside the corpus. ``exact=True`` uses the
        unstemmed vocabulary - right for typo detection, where the stemmer would
        turn "metformine" into the real word "metformin" and hide the typo.
        """
        if not _TOKEN.fullmatch(token):
            return True
        with self._lock:
            if exact:
                row = self._db.execute("SELECT 1 FROM vocab WHERE term = ?", (token,)).fetchone()
            else:
                row = self._db.execute("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1",
                                       (f'"{token}"',)).fetchone()
        return row is not None

    def rebuild_vocab(self, batch: int = 2000) -> int:
        """Recompute the exact vocabulary from current chunks (after pruning, or
        for an index built before the vocab table existed). Commits per batch so
        concurrent writers are never blocked for long."""
        with self._lock, self._db:
            self._db.execute("DELETE FROM vocab")
        last = 0
        while True:
            with self._lock:
                rows = self._db.execute(
                    "SELECT id, header, text FROM chunks WHERE id > ? ORDER BY id LIMIT ?",
                    (last, batch)).fetchall()
            if not rows:
                break
            terms = {t for r in rows for t in _TOKEN.findall(f"{r['header']} {r['text']}".lower())}
            with self._lock, self._db:
                self._db.executemany("INSERT OR IGNORE INTO vocab VALUES(?)", [(t,) for t in terms])
            last = rows[-1]["id"]
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM vocab").fetchone()[0]

    def vocab_size(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM vocab").fetchone()[0]

    def get_chunks(self, rowids: Iterable[int]) -> dict[int, tuple[Chunk, str]]:
        ids = list(rowids)
        if not ids:
            return {}
        with self._lock:
            rows = self._db.execute(
                "SELECT c.*, d.display_name FROM chunks c JOIN documents d USING(doc_id) "
                f"WHERE c.id IN ({','.join('?' * len(ids))})", ids).fetchall()
        return {r["id"]: (_row_to_chunk(r), r["display_name"]) for r in rows}

    def iter_chunks(self) -> list[tuple[int, Chunk, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT c.*, d.display_name FROM chunks c JOIN documents d USING(doc_id) "
                "ORDER BY c.id").fetchall()
        return [(r["id"], _row_to_chunk(r), r["display_name"]) for r in rows]

    def dense_coverage(self, model: str, variant: str = "context") -> tuple[int, int]:
        """(chunks that have a vector, total chunks) for one embedding variant."""
        col = "text_hash" if variant == "context" else "raw_hash"
        with self._lock:
            total = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            have = self._db.execute(
                f"SELECT COUNT(*) FROM chunks c JOIN embeddings e ON e.text_hash = c.{col} "
                "AND e.model = ?", (model,)).fetchone()[0]
        return have, total

    def stats(self) -> dict:
        with self._lock:
            docs = self._db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = self._db.execute("SELECT COUNT(*), COALESCE(SUM(n_words),0) FROM chunks").fetchone()
            embs = self._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {"documents": docs, "chunks": chunks[0], "words": chunks[1],
                "embeddings": embs, "index_version": self.index_version}


def _row_to_chunk(r: sqlite3.Row) -> Chunk:
    return Chunk(chunk_id=r["chunk_id"], doc_id=r["doc_id"], section_key=r["section_key"],
                 section_title=r["section_title"], subsection=r["subsection"], ordinal=r["ordinal"],
                 kind=r["kind"], text=r["text"], header=r["header"], text_hash=r["text_hash"],
                 n_words=r["n_words"])
