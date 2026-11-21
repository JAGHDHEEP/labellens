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


# ------------------------------------------- population context (live audit, omeprazole)
def test_subsections_found_when_heading_is_followed_by_a_symbol_or_capitals():
    for head in ("2 DOSAGE AND ADMINISTRATION * if ulcer present, continue 20 mg. ",
                 "2 DOSAGE AND ADMINISTRATION • For most patients 50 mg. ",
                 "2 DOSAGE & ADMINISTRATION OMEPRAZOLE is taken before meals. "):
        text = (head + "2.1 Recommended Adult Dosage Take 20 mg once daily. "
                "2.2 Recommended Pediatric Dosage Regimen by Indication 5 to less than 10 kg: 5 mg.")
        labels = [lbl for lbl, _ in split_subsections(text)]
        assert labels[-2].startswith("2.1") and labels[-1].startswith("2.2 Recommended Pediatric"), head
        # The heading stays in the text: nothing after it (a drug name) is cut off.
        assert "".join(b for _, b in split_subsections(text)).startswith(head.strip()[:20])


def test_each_table_is_chunked_on_its_own_and_keeps_its_caption():
    from labellens.chunking import table_pieces

    adult = table_to_text("<table><caption>Table 1 Recommended Dosage in Adults</caption>"
                          "<tr><th>Indication</th><th>Dose</th></tr>"
                          "<tr><td>GERD</td><td>20 mg once daily</td></tr></table>")
    rows = "".join(f"<tr><td>{w} to less than {w + 5} kg</td><td>{w} mg once daily</td></tr>"
                   for w in range(5, 400, 5))
    peds = table_to_text("<table><caption>Table 2 Recommended Dosage in Pediatric Patients</caption>"
                         f"<tr><th>Weight</th><th>Dose</th></tr>{rows}</table>")
    rec = make_label("s-ome", "omeprazole", "Prilosec",
                     dosage_and_administration="2 DOSAGE AND ADMINISTRATION Take before meals.")
    rec["dosage_and_administration_table"] = [
        "<table><caption>Table 1 Recommended Dosage in Adults</caption>"
        "<tr><th>Indication</th><th>Dose</th></tr><tr><td>GERD</td><td>20 mg once daily</td></tr></table>",
        "<table><caption>Table 2 Recommended Dosage in Pediatric Patients</caption>"
        f"<tr><th>Weight</th><th>Dose</th></tr>{rows}</table>"]
    doc = normalize_label(rec)
    table = next(s for s in doc.sections if s.kind == "table")
    assert table.text == adult + "\n\n" + peds
    pieces = table_pieces(table.text, max_words=60, min_words=5)
    peds_pieces = [t for lbl, t in pieces if lbl.endswith("Pediatric Patients")]
    assert len(peds_pieces) > 1  # every chunk of the long table is labelled, not just the first
    assert all("GERD" not in t for t in peds_pieces)  # the adult row never shares a chunk
    chunks = [c for c in chunk_document(doc, 60, 5) if c.kind == "table"]
    assert all("Pediatric Patients" in c.header for c in chunks if "kg" in c.text)


def test_only_numbered_captions_label_a_table():
    from labellens.chunking import table_title

    assert table_title(["* if ulcer present, continue 20 mg", "GERD | 20 mg"]) == ""
    assert table_title(["1 Includes abdominal pain upper", "Nausea | 12"]) == ""  # footnote first
    assert table_title(["Ear and Labyrinth Disorders", "Vertigo | 3"]) == ""  # a group row
    assert table_title(["Indication | Dose", "GERD | 20 mg"]) == ""
    assert table_title(["Table 9: Type 1 Diabetes Mellitus – Adults", "x | y"]).startswith("Table 9")
    assert table_title(["TABLE IV Dosing in Renal Impairment", "x | y"]).startswith("TABLE IV")
    long = "Table 5: Adverse Reactions " + "in the High Dose Group " * 6 + "in Pediatric Patients"
    assert table_title([long, "x | y"]).startswith("Table 5")  # long captions are cut, not dropped
    assert table_title(["Table 1 only"]) == ""  # a caption with no rows labels nothing


def test_caption_in_a_one_cell_first_row_is_kept():
    html = ("<table><tr><td colspan='2'>Table 2. Recommended dose for Pediatric Patients</td></tr>"
            "<tr><th>Infection</th><th>Dose</th></tr>"
            "<tr><td>Acute otitis media</td><td>30 mg/kg as a single dose</td></tr></table>")
    text = table_to_text(html)
    assert text.split(chr(10))[0] == "Table 2. Recommended dose for Pediatric Patients"
    assert "Infection: Acute otitis media | Dose: 30 mg/kg as a single dose" in text


def test_numbered_heading_variants_are_split():
    for head in ("2. DOSAGE AND ADMINISTRATION For dialysis only. ",
                 "2 Dosage and Administration Read first. ", "2DOSAGE AND ADMINISTRATION x. "):
        text = head + "2.1 Basic Dosing Use 5 mg daily. 2.2 Dosing Regimen Use the lowest dose."
        assert [lbl.split()[0] for lbl, _ in split_subsections(text) if lbl] == ["2.1", "2.2"], head


def test_heading_only_subsections_are_not_their_own_chunks():
    text = ("1 INDICATIONS AND USAGE Pavblu is indicated for: 1.1 Neovascular Age-Related Macular "
            "Degeneration (nAMD) Treat monthly for three months then every 8 weeks. "
            "1.2 Diabetic Retinopathy (DR)")
    parts = split_subsections(text)
    assert all(len(body.split()) > len(lbl.split()) + 1 for lbl, body in parts)
    assert "Diabetic Retinopathy (DR)" in parts[-1][1]


def test_chunk_ids_do_not_shift_when_an_earlier_section_gains_chunks():
    doc = normalize_label(METFORMIN)
    before = {c.text: c.chunk_id for c in chunk_document(doc) if c.section_key == "contraindications"}
    rec = dict(METFORMIN, indications_and_usage=[METFORMIN["indications_and_usage"][0] + " Extra. " * 120])
    after = {c.text: c.chunk_id for c in chunk_document(normalize_label(rec))
             if c.section_key == "contraindications"}
    assert before == after


def test_sibling_tables_of_one_label_are_not_collapsed_as_duplicates():
    from labellens.models import Chunk, Hit
    from labellens.retrieval import collapse_near_duplicates

    rows = "Aprepitant | Day 1: 125 mg | Day 2: 80 mg | Day 3: 80 mg" + chr(10) + "Dexamethasone | Day 1: 12 mg"

    def hit(cid, doc, sub):
        c = Chunk(chunk_id=cid, doc_id=doc, section_key="dosage_and_administration_table",
                  section_title="Dosage (table)", subsection=sub, ordinal=0, kind="table",
                  text=rows, header="", text_hash=cid, n_words=len(rows.split()))
        return Hit(rowid=0, chunk=c, drug="Aprepitant")

    hec, mec = hit("a", "d1", "Table 1 HEC regimen"), hit("b", "d1", "Table 2 MEC regimen")
    other_maker = hit("c", "d2", "Table 1 HEC regimen")
    kept = collapse_near_duplicates([hec, mec, other_maker], 5)
    assert [h.chunk.chunk_id for h in kept] == ["a", "b"]  # siblings kept, copy from d2 collapsed


def test_caption_stops_where_a_footnote_starts_but_keeps_its_own_number():
    from labellens.chunking import table_title

    run_on = "Table 2 Dosage in Pediatric Patients by Indication 1. The efficacy beyond 8 weeks"
    assert table_title([run_on, "x | y"]) == "Table 2 Dosage in Pediatric Patients by Indication"
    assert table_title(["Table 12. Response to Ezetimibe (Mean % Change)", "x | y"]) == \
        "Table 12. Response to Ezetimibe (Mean % Change)"
