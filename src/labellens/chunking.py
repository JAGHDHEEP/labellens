"""Structure-aware chunking for drug labels.

Design constraints that drove this (rather than a fixed-size splitter):

* A label's unit of meaning is the numbered subsection ("5.2 Hepatotoxicity").
  Chunks never cross a section or subsection boundary, so a chunk is always
  *about one thing* and its metadata is exact.
* Every label has the same section names. A chunk that says "Do not use in
  patients with severe renal impairment" is useless without knowing *which
  drug*. Each chunk therefore carries a deterministic contextual header
  ("Metformin Hydrochloride (Glucophage) | Contraindications") that is embedded
  and lexically indexed together with the text. This is the cheap, LLM-free
  variant of "contextual retrieval"; its effect is measured in the ablation.
* The cross-encoder reranker has a 512-token window, so chunks are capped at
  ~220 words to keep query + header + passage inside it without truncation.
"""

from __future__ import annotations

import hashlib
import re

from .models import Chunk, Document, Section

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[0-9])")
_ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|vs|approx|Fig|No|Dr|St|etc|U\.S|al)\.)$")
_SECTION_HEAD = re.compile(r"^(\d{1,2})\s+([A-Z][A-Z\s,&/()\-:]{3,}?)\s+(?=[A-Z][a-z]|\d|\(|\[)")
_SUBSECTION = re.compile(r"(?:^|(?<=\s))(\d{1,2})\.(\d{1,2})\s+(?=[A-Z][A-Za-z])")
_CONNECTORS = {"and", "or", "of", "in", "with", "the", "to", "for", "on", "a", "an", "by", "from"}


def split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Re-join when the split happened after an abbreviation ("e.g. Aspirin").
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _subsection_title(words: list[str]) -> str:
    title: list[str] = []
    for w in words[:9]:
        bare = w.strip(",;:()[]")
        if bare[:1].isupper() or bare.lower() in _CONNECTORS:
            title.append(w.rstrip(",;:"))
        else:
            # The flattened text has no delimiter between the heading and the
            # sentence that follows, so the boundary is inferred:
            #   "Renal Impairment In patients ..."  -> capitalised connector is
            #       the sentence start; the title is everything before it.
            #   "Hepatotoxicity Cases of ..."       -> a lowercase connector means
            #       the sentence began one word earlier; give that word back.
            #   "Serious Skin Reactions Serious skin ..." -> same, no connector.
            sentence_started_at_connector = False
            while title and title[-1].lower() in _CONNECTORS:
                sentence_started_at_connector = title.pop()[:1].isupper()
            if len(title) >= 2 and not sentence_started_at_connector:
                title.pop()
            break
    while title and title[-1].lower() in _CONNECTORS:
        title.pop()
    return " ".join(title[:7])


def split_subsections(text: str) -> list[tuple[str, str]]:
    """Split section text into (subsection_label, body) at numbered headings.

    A candidate "N.M Title" is accepted only if N matches the section's own
    number and M increases monotonically - this rejects doses like "2.5 Mg".
    """
    major: str | None = None
    m = _SECTION_HEAD.match(text)
    if m:
        major = m.group(1)
        text = text[m.end():]
    else:
        lead = _SUBSECTION.match(text)
        if lead:
            major = lead.group(1)

    if major is None:
        return [("", text)]

    cuts: list[tuple[int, int, str]] = []  # (start, body_start, label)
    last_minor = 0
    for sm in _SUBSECTION.finditer(text):
        if sm.group(1) != major:
            continue
        minor = int(sm.group(2))
        if minor <= last_minor or minor > last_minor + 3:
            continue
        last_minor = minor
        title = _subsection_title(text[sm.end(): sm.end() + 160].split())
        cuts.append((sm.start(), sm.end(), f"{major}.{minor} {title}".strip()))

    if not cuts:
        return [("", text)]

    parts: list[tuple[str, str]] = []
    if cuts[0][0] > 0 and text[: cuts[0][0]].strip():
        parts.append(("", text[: cuts[0][0]].strip()))
    for i, (_, body_start, label) in enumerate(cuts):
        end = cuts[i + 1][0] if i + 1 < len(cuts) else len(text)
        body = text[body_start:end].strip()
        if body:
            parts.append((label, body))
    return parts


def pack(units: list[str], max_words: int, min_words: int, joiner: str = " ") -> list[str]:
    """Greedily pack sentences/rows into chunks of at most ``max_words``."""
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_words = 0
    for unit in units:
        words = unit.split()
        # A single oversized unit (run-on list, giant table row) is hard-split.
        while len(words) > max_words:
            if cur:
                chunks.append(cur)
                cur, cur_words = [], 0
            chunks.append([" ".join(words[:max_words])])
            words = words[max_words:]
        if not words:
            continue
        if cur and cur_words + len(words) > max_words:
            chunks.append(cur)
            cur, cur_words = [], 0
        cur.append(" ".join(words))
        cur_words += len(words)
    if cur:
        chunks.append(cur)
    # Fold a tiny trailing chunk into its predecessor instead of indexing a fragment.
    if len(chunks) >= 2 and sum(len(u.split()) for u in chunks[-1]) < min_words:
        tail = chunks.pop()
        chunks[-1].extend(tail)
    return [joiner.join(c) for c in chunks]


def _make_chunk(doc: Document, sec: Section, sub: str, ordinal: int, text: str) -> Chunk:
    header = f"{doc.display_name} | {sec.title}"
    if sub:
        header += f" | {sub}"
    embed_text = f"{header}\n{text}"
    text_hash = hashlib.sha256(embed_text.encode("utf-8")).hexdigest()
    chunk_id = hashlib.sha1(f"{doc.doc_id}|{sec.key}|{ordinal}|{text_hash}".encode()).hexdigest()[:16]
    return Chunk(
        chunk_id=chunk_id, doc_id=doc.doc_id, section_key=sec.key, section_title=sec.title,
        subsection=sub, ordinal=ordinal, kind=sec.kind, text=text, header=header,
        text_hash=text_hash, n_words=len(text.split()),
    )


def chunk_document(doc: Document, max_words: int = 220, min_words: int = 25) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in doc.sections:
        if sec.kind == "table":
            rows = [r for r in sec.text.split("\n") if r.strip()]
            pieces = [("", p) for p in pack(rows, max_words, min_words, joiner="\n")]
        else:
            pieces = []
            for sub, body in split_subsections(sec.text):
                for p in pack(split_sentences(body), max_words, min_words):
                    pieces.append((sub, p))
        for sub, text in pieces:
            # Keep very short chunks ("None.") - they are real answers - but not
            # empty ones or a bare heading with no body.
            if not re.search(r"[a-z0-9]", text):
                continue
            chunks.append(_make_chunk(doc, sec, sub, len(chunks), text))
    return chunks
