from labellens.chunking import chunk_document, pack, split_sentences, split_subsections
from labellens.ingest import embed_missing, ingest_records
from labellens.normalize import clean_text, normalize_label, table_to_text
from labellens.providers import HashEmbedder

from .conftest import CORPUS, METFORMIN, make_label


# ------------------------------------------------------------------ normalize
def test_clean_text_removes_cross_references_and_repeated_sentences():
    text = ("Apply a thin film to the affected areas once daily after washing the skin. ( 2 ) "
            "Apply a thin film to the affected areas once daily after washing the skin. "
            "Avoid the eyes ( 5.1 , 5.2 ).")
    out = clean_text(text)
    assert "( 2 )" not in out and "( 5.1" not in out
    assert out.count("Apply a thin film") == 1


def test_clean_text_keeps_real_parenthesised_numbers():
    assert "(25)" in clean_text("The sample size (25) was small.")


def test_child_section_is_excised_from_parent():
    child = ("8.1 Pregnancy Risk Summary Available data have not identified a drug-associated "
             "risk of major birth defects, miscarriage or adverse maternal outcomes.")
    parent = f"8 USE IN SPECIFIC POPULATIONS {child} 8.4 Pediatric Use Safety has not been established in children."
    doc = normalize_label(make_label("s1", "testdrug", "Testa", indications_and_usage="1 INDICATIONS AND USAGE For testing only in adults.",
                                     use_in_specific_populations=parent, pregnancy=child))
    by_key = {s.key: s.text for s in doc.sections}
    assert "Risk Summary" in by_key["pregnancy"]
    assert "Risk Summary" not in by_key["use_in_specific_populations"]
    assert "Pediatric Use" in by_key["use_in_specific_populations"]


def test_table_rows_are_labelled_with_headers():
    html = ("<table><caption>Table 1: Adverse reactions</caption>"
            "<tr><th>Reaction</th><th>Drug (%)</th><th>Placebo (%)</th></tr>"
            "<tr><td>Nausea</td><td>12</td><td>4</td></tr></table>")
    text = table_to_text(html)
    assert "Table 1: Adverse reactions" in text
    assert "Reaction: Nausea | Drug (%): 12 | Placebo (%): 4" in text


def test_record_without_generic_name_is_rejected():
    rec = make_label("s2", "x", "X", indications_and_usage="1 INDICATIONS AND USAGE Something long enough here.")
    rec["openfda"]["generic_name"] = []
    assert normalize_label(rec) is None


# ------------------------------------------------------------------- chunking
def test_subsections_split_on_matching_major_number_only():
    text = ("2 DOSAGE AND ADMINISTRATION 2.1 Adult Dosage Start with 2.5 Mg once daily. "
            "2.2 Renal Impairment Reduce the dose by half.")
    parts = split_subsections(text)
    labels = [p[0] for p in parts]
    assert labels[0].startswith("2.1") and labels[1].startswith("2.2")
    assert len(parts) == 2  # "2.5 Mg" is a dose, not subsection 2.5
    assert "2.5 Mg" in parts[0][1]


def test_text_without_numbering_is_one_piece():
    assert split_subsections("Old format label text without numbers.") == [
        ("", "Old format label text without numbers.")]


def test_sentence_splitter_respects_abbreviations():
    sents = split_sentences("Avoid NSAIDs, e.g. Ibuprofen, with this drug. Monitor renal function.")
    assert len(sents) == 2


def test_pack_respects_max_words_and_hard_splits_long_units():
    units = ["word " * 50, "word " * 50, "word " * 500]
    chunks = pack([u.strip() for u in units], max_words=120, min_words=10)
    assert all(len(c.split()) <= 120 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 600


def test_chunks_carry_drug_and_section_in_header():
    doc = normalize_label(METFORMIN)
    chunks = chunk_document(doc)
    assert chunks
    for c in chunks:
        assert c.header.startswith("Metformin Hydrochloride (Glucophage) | ")
        assert c.embed_text.startswith(c.header)
    dosage = [c for c in chunks if c.section_key == "dosage_and_administration"]
    assert {c.subsection.split()[0] for c in dosage} == {"2.1", "2.2"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# --------------------------------------------------------------------- ingest
class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.texts_embedded = 0

    def embed(self, texts):
        self.texts_embedded += len(texts)
        return super().embed(texts)


def test_ingest_is_idempotent(store, settings):
    emb = CountingEmbedder()
    first = ingest_records(store, CORPUS, settings)
    embed_missing(store, emb)
    n = emb.texts_embedded
    assert first.added == 2 and n == store.stats()["chunks"]

    second = ingest_records(store, CORPUS, settings)
    assert (second.added, second.updated, second.unchanged) == (0, 0, 2)
    assert embed_missing(store, emb) == 0 and emb.texts_embedded == n
    assert store.index_version == first_version_after(store)


def first_version_after(store):
    return store.index_version


def test_revised_label_replaces_chunks_and_only_embeds_new_text(store, settings):
    emb = CountingEmbedder()
    ingest_records(store, CORPUS, settings)
    embed_missing(store, emb)
    before = emb.texts_embedded
    v_before = store.index_version

    revised = dict(METFORMIN)
    revised["version"] = "2"
    revised["boxed_warning"] = ["WARNING: LACTIC ACIDOSIS Postmarketing cases of metformin-associated "
                                "lactic acidosis have resulted in death, hypothermia and hypotension."]
    rep = ingest_records(store, [revised, CORPUS[1]], settings)
    assert (rep.added, rep.updated, rep.unchanged) == (0, 1, 1)
    newly = embed_missing(store, emb)
    assert newly == 1 and emb.texts_embedded == before + 1  # only the new section
    assert store.index_version > v_before
    hits = store.bm25_search("lactic acidosis hypothermia", 5)
    assert hits and "LACTIC ACIDOSIS" in store.get_chunks([hits[0][0]])[hits[0][0]][0].text


def test_prune_removes_withdrawn_labels(store, settings):
    ingest_records(store, CORPUS, settings)
    rep = ingest_records(store, [CORPUS[0]], settings, prune=True)
    assert rep.removed == 1
    assert store.stats()["documents"] == 1
    assert store.bm25_search("warfarin", 5) == []
    assert "warfarin" not in store.aliases()
