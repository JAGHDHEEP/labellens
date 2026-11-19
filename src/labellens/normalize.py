"""Turn a raw openFDA label record into a clean Document.

Three data-quality problems in the source are handled here, each of which
measurably pollutes retrieval if ignored:

1. *Overlapping fields.* openFDA emits both a parent section and its children
   (``use_in_specific_populations`` contains the full text of ``pregnancy``).
   Indexing both yields duplicate chunks that crowd out distinct evidence.
   We keep the child (more specific metadata) and excise its text from the parent.
2. *Duplicated highlights.* The flattened text repeats sentences (the
   "Highlights" bullet plus the full paragraph). We drop exact repeats.
3. *Tables as HTML.* ``*_table`` fields are raw HTML. We linearise each row as
   "header: value" pairs so a row is retrievable on its own.
"""

from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser

from .models import Document, Section

# Fields that are not clinical prose or duplicate it in patient language.
SKIP_FIELDS = {
    "spl_product_data_elements", "package_label_principal_display_panel",
    "spl_unclassified_section", "recent_major_changes", "recent_major_changes_table",
    "spl_patient_package_insert", "spl_patient_package_insert_table", "spl_medguide",
    "spl_medguide_table", "instructions_for_use", "instructions_for_use_table",
    "references", "set_id", "id", "version", "effective_time", "openfda",
    "information_for_patients_table", "patient_medication_information",
    "patient_medication_information_table",
}

TITLE_OVERRIDES = {
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "dosage_forms_and_strengths": "Dosage Forms and Strengths",
    "warnings_and_cautions": "Warnings and Precautions",
    "boxed_warning": "Boxed Warning",
    "use_in_specific_populations": "Use in Specific Populations",
    "carcinogenesis_and_mutagenesis_and_impairment_of_fertility":
        "Carcinogenesis, Mutagenesis, Impairment of Fertility",
    "how_supplied": "How Supplied / Storage",
    "storage_and_handling": "Storage and Handling",
    "information_for_patients": "Patient Counseling Information",
}

_WS = re.compile(r"\s+")
# openFDA renders cross references as "( 5.1 )" / "( 2.1 , 2.3 )" with inner spaces.
_XREF = re.compile(r"\(\s+\d[\d.,\s]*\s+\)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def title_for(key: str) -> str:
    base = key[:-6] if key.endswith("_table") else key
    return TITLE_OVERRIDES.get(base) or base.replace("_", " ").title()


def clean_text(text: str) -> str:
    text = _XREF.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return _drop_repeated_sentences(text)


def _drop_repeated_sentences(text: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for sent in _SENT_SPLIT.split(text):
        norm = sent.strip().lower()
        if len(norm) > 40:  # only dedupe real sentences, never short fragments
            if norm in seen:
                continue
            seen.add(norm)
        out.append(sent.strip())
    return " ".join(s for s in out if s)


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.caption = ""
        self._cell: list[str] | None = None
        self._row: list[str] | None = None
        self._in_caption = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
        elif tag == "caption":
            self._in_caption = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(_WS.sub(" ", " ".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None
        elif tag == "caption":
            self._in_caption = False

    def handle_data(self, data):
        if self._in_caption:
            self.caption += data
        elif self._cell is not None:
            self._cell.append(data)


def table_to_text(html: str) -> str:
    """Linearise an HTML table: one line per row, cells labelled by header."""
    p = _TableParser()
    p.feed(html)
    if not p.rows:
        return ""
    lines: list[str] = []
    caption = _WS.sub(" ", p.caption).strip()
    if caption:
        lines.append(caption)
    header = p.rows[0]
    body = p.rows[1:] if len(p.rows) > 1 else p.rows
    for row in body:
        if len(row) == len(header) and len(header) > 1 and row is not header:
            cells = [f"{h}: {c}" if h else c for h, c in zip(header, row) if c]
        else:
            cells = [c for c in row if c]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _excise_overlaps(sections: dict[str, str]) -> dict[str, str]:
    """Remove child-section text from any longer section that contains it.

    Only an *exact* containment is acted on. An earlier version also dropped a
    section when a 200-character probe of it appeared in a longer one; review
    showed that deleted 16 boxed warnings (one sentence of a boxed warning is
    routinely repeated under Warnings and Precautions). Losing a boxed warning
    is far worse than indexing a duplicated paragraph, and duplicates are
    handled downstream by near-duplicate collapsing anyway.
    """
    order = sorted(sections, key=lambda k: len(sections[k]))
    shortened: set[str] = set()
    for i, small in enumerate(order):
        s_text = sections[small]
        if len(s_text) < 80:
            continue
        for big in order[i + 1:]:
            b_text = sections[big]
            if len(b_text) > len(s_text) and s_text in b_text:
                sections[big] = _WS.sub(" ", b_text.replace(s_text, " ")).strip()
                shortened.add(big)
    # A parent emptied by excision is just a bare heading; drop it. Sections that
    # were short to begin with are kept: "4 CONTRAINDICATIONS None." is an answer.
    return {k: v for k, v in sections.items()
            if v and not (k in shortened and len(v) < 40)}


def normalize_label(rec: dict) -> Document | None:
    fda = rec.get("openfda", {})
    generic = (fda.get("generic_name") or [""])[0].strip()
    if not rec.get("set_id") or not generic:
        return None

    texts: dict[str, str] = {}
    tables: dict[str, str] = {}
    for key, value in rec.items():
        if key in SKIP_FIELDS or not isinstance(value, list) or not value:
            continue
        if not isinstance(value[0], str):
            continue
        if key.endswith("_table"):
            rendered = "\n".join(t for t in (table_to_text(v) for v in value) if t)
            if rendered:
                tables[key] = rendered
        else:
            cleaned = clean_text(" ".join(value))
            if cleaned:
                texts[key] = cleaned

    texts = _excise_overlaps(texts)
    sections = [Section(k, title_for(k), v, "text") for k, v in texts.items()]
    sections += [Section(k, title_for(k) + " (table)", v, "table") for k, v in tables.items()]
    if not sections:
        return None

    # Metadata is part of the fingerprint: a revision that only renames the
    # brand must still refresh aliases and chunk headers.
    digest = hashlib.sha256(repr((
        rec.get("version"), rec.get("effective_time"), fda.get("generic_name"),
        fda.get("brand_name"), fda.get("manufacturer_name"), fda.get("route"))).encode())
    for s in sorted(sections, key=lambda s: s.key):
        digest.update(s.key.encode())
        digest.update(s.text.encode())

    return Document(
        doc_id=rec["set_id"],
        version=int(rec.get("version") or 0),
        effective_date=str(rec.get("effective_time") or ""),
        generic_name=generic.title() if generic.isupper() else generic,
        brand_names=[b.title() if b.isupper() else b for b in fda.get("brand_name", [])][:3],
        manufacturer=(fda.get("manufacturer_name") or [""])[0],
        route=", ".join(fda.get("route", [])).lower(),
        pharm_class="; ".join(fda.get("pharm_class_epc", [])),
        sections=sections,
        content_hash=digest.hexdigest(),
    )
