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
# A numbered heading _SECTION_HEAD cannot strip safely because what follows it
# is not ordinary prose: "2 DOSAGE AND ADMINISTRATION * if ulcer present ...",
# "5 WARNINGS AND PRECAUTIONS • Symptomatic ...", "4 CONTRAINDICATIONS WINREVAIR
# is ...". The number still names the section, so subsections are found; the
# heading is left in the text rather than risk cutting a drug name off it.
# Also "2. DOSAGE AND ...", "2 Dosage and Administration ..." and "1INDICATIONS".
_SECTION_NUMBER = re.compile(r"^(\d{1,2})\.?\s*[A-Z][A-Za-z&]+\b")
# A table caption: "Table 2", "TABLE 3.", "Table 2a:", "Table IV".
_TABLE_CAPTION = re.compile(r"^table\s+(?:\d+[a-z]?|[ivx]+)\b", re.I)
_CAPTION_END = re.compile(r"(?<=[a-z)])\s+(?:\d{1,2}[.)]|[*†‡§∗])\s*(?=[A-Z])")
MAX_TITLE_WORDS = 40
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
        lead = _SUBSECTION.match(text) or _SECTION_NUMBER.match(text)
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
        if not body:
            continue
        # "1.4 Diabetic Retinopathy (DR)" with nothing under it (indications
        # listed as bare headings) is not a chunk of its own: a 3-word chunk
        # repeats the heading and takes a retrieval slot.
        if parts and len(body.split()) <= len(label.split()) + 1:
            prev_label, prev_body = parts[-1]
            parts[-1] = (prev_label, f"{prev_body} {label.split(' ', 1)[0]} {body}")
            continue
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


def _make_chunk(doc: Document, sec: Section, sub: str, ordinal: int, text: str,
                repeat: int = 0) -> Chunk:
    header = f"{doc.display_name} | {sec.title}"
    if sub:
        header += f" | {sub}"
    embed_text = f"{header}\n{text}"
    text_hash = hashlib.sha256(embed_text.encode("utf-8")).hexdigest()
    # The id hashes the section and the exact embedded text (plus a counter for
    # an identical chunk repeated in one section), not the document-wide
    # position: adding a chunk early in a label must not rename every later
    # chunk, or every stored reference to one (the golden set) breaks.
    chunk_id = hashlib.sha1(f"{doc.doc_id}|{sec.key}|{repeat}|{text_hash}".encode()).hexdigest()[:16]
    return Chunk(
        chunk_id=chunk_id, doc_id=doc.doc_id, section_key=sec.key, section_title=sec.title,
        subsection=sub, ordinal=ordinal, kind=sec.kind, text=text, header=header,
        text_hash=text_hash, n_words=len(text.split()),
    )


def table_title(rows: list[str]) -> str:
    """The caption of one linearised table, if its first line is one.

    "Table 2 Recommended Dosage Regimen of Omeprazole in Pediatric Patients by
    Indication" is a title; a data row ("Indication | Dose"), a labelled cell
    ("Age: 1 to 16 years") or a footnote ("* if ulcer present ...") is not.

    Only a numbered caption is trusted. A first line that is not one may be a
    group row ("Ear and Labyrinth Disorders") or a footnote that the label puts
    before the body ("1 Includes abdominal pain"); stamped on every chunk of
    the table it would mislabel them, which is worse than no label.
    """
    if len(rows) < 2:
        return ""
    first = rows[0].strip()
    if " | " in first or not _TABLE_CAPTION.match(first):
        return ""
    # Labels often run the first footnote into the caption line:
    # "... in Pediatric Patients by Indication 1. The efficacy of ...".
    head = _TABLE_CAPTION.match(first).end()
    first = first[:head] + _CAPTION_END.split(first[head:], maxsplit=1)[0]
    return " ".join(first.split()[:MAX_TITLE_WORDS])


def table_pieces(text: str, max_words: int, min_words: int) -> list[tuple[str, str]]:
    """Chunk a table section one table at a time (tables are separated by a
    blank line), labelling every chunk of a table with that table's caption.

    Without this, a chunk could start with the last rows of an adult dosing
    table and continue into a pediatric one, and later chunks of the pediatric
    table lost the only line that said who the doses are for. The 8B model
    then reported "5 mg once daily for 5 to less than 10 kg" as the dose.
    """
    blocks: list[list[str]] = []
    for block in re.split(r"\n\s*\n", text):
        rows = [r for r in block.split("\n") if r.strip()]
        if not rows:
            continue
        small = sum(len(r.split()) for r in rows) < min_words
        # A tiny untitled table ("Figure 1", a two-cell unit table) joins the
        # table before it rather than becoming a fragment chunk.
        if blocks and small and not table_title(rows):
            blocks[-1] += rows
        else:
            blocks.append(rows)
    pieces: list[tuple[str, str]] = []
    for rows in blocks:
        title = table_title(rows)
        pieces += [(title, p) for p in pack(rows, max_words, min_words, joiner="\n")]
    return pieces


def chunk_document(doc: Document, max_words: int = 220, min_words: int = 25) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in doc.sections:
        if sec.kind == "table":
            pieces = table_pieces(sec.text, max_words, min_words)
        else:
            pieces = []
            for sub, body in split_subsections(sec.text):
                for p in pack(split_sentences(body), max_words, min_words):
                    pieces.append((sub, p))
        seen: dict[tuple[str, str], int] = {}
        for sub, text in pieces:
            # Keep very short chunks ("None.") - they are real answers - but not
            # empty ones or a bare heading with no body.
            if not re.search(r"[a-z0-9]", text):
                continue
            repeat = seen.get((sub, text), 0)
            seen[(sub, text)] = repeat + 1
            chunks.append(_make_chunk(doc, sec, sub, len(chunks), text, repeat))
    return chunks
