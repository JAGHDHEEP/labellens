"""Core domain types shared by ingestion, retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    key: str  # openFDA field name, e.g. "drug_interactions"
    title: str  # human title, e.g. "Drug Interactions"
    text: str
    kind: str = "text"  # "text" | "table"


@dataclass
class Document:
    doc_id: str  # SPL set id: stable across label revisions
    version: int
    effective_date: str  # YYYYMMDD
    generic_name: str
    brand_names: list[str]
    manufacturer: str
    route: str
    pharm_class: str
    sections: list[Section]
    content_hash: str = ""

    @property
    def display_name(self) -> str:
        brand = next((b for b in self.brand_names if b.lower() != self.generic_name.lower()), "")
        return f"{self.generic_name} ({brand})" if brand else self.generic_name


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_key: str
    section_title: str
    subsection: str
    ordinal: int  # position within the document, for stitching neighbours
    kind: str
    text: str
    header: str  # contextual header: drug + section path
    text_hash: str  # sha256 of the exact string that is embedded
    n_words: int

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}"


@dataclass
class Hit:
    """A retrieved chunk plus the provenance of *why* it was retrieved."""

    rowid: int
    chunk: Chunk
    drug: str
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    # chunk ids of lower-ranked near-duplicates this hit stands in for
    dup_ids: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    scoped_drugs: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    top_score: float | None = None
    # query terms that occur nowhere in the corpus (only computed when no drug was linked)
    unknown_terms: list[str] = field(default_factory=list)
    scoped_doc_ids: list[str] = field(default_factory=list)
    index_version: int = 0
    # the question points at a drug ("this drug", "my medication") without naming one
    unnamed_drug: bool = False
