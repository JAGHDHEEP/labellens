# Design decisions

Each entry: the decision, the constraint that forced it, what was rejected, and when it should be revisited. Numbers quoted here come from `eval/results/` and are reproduced in the README.

## 1. One SQLite file instead of a vector database plus a search engine

**Decision.** Documents, chunks, the BM25 index (FTS5), embeddings and the drug-alias table live in one SQLite database. Dense search is exact brute-force cosine over a float32 matrix loaded into memory at start-up.

**Constraint.** Labels are revised continuously, so the index is updated incrementally forever. With two stores, every update is a dual write, and a crash between the writes leaves the keyword arm and the vector arm disagreeing about which chunks exist. That failure is silent: retrieval quality just degrades.

**Why this is enough.** 72,879 chunks × 1024 dims = 285 MB. A full scan is one matrix-vector product and takes single-digit milliseconds, with 100 % recall by construction. An ANN index (HNSW) would add build time, tuning parameters and a recall loss in exchange for a latency win on a stage that is already under 1 % of request time; the reranker and the LLM dominate.

**Rejected.** Qdrant / pgvector / Elasticsearch. All good, none needed at this size, each one more thing to deploy, back up and keep consistent.

**Revisit when** the matrix no longer fits comfortably in RAM (roughly a few million chunks) or more than one writer is needed. `DenseIndex.search(qvec, k, doc_ids)` and `Store.bm25_search(...)` are the only two calls the retriever makes, so the swap is contained. pgvector would be the first choice because it keeps the single-transaction property.

## 2. Structure-aware chunking with a deterministic contextual header

**Decision.** Split on the label's own structure (section, then numbered subsection, then sentences packed to at most 220 words). Prepend `Drug (Brand) | Section | Subsection` to every chunk before embedding and BM25 indexing.

**Constraint.** Every label has the same section names and much of the same boilerplate. "Contraindicated in patients with severe renal impairment" appears in hundreds of labels; without the drug name attached, the chunk is unattributable and the embedding of one is indistinguishable from the embedding of another.

**Rejected.** Fixed-size windows with overlap (cuts dose tables and subsection logic in half, and metadata becomes approximate). LLM-written chunk context à la "contextual retrieval" (one LLM call per chunk; at 73k chunks on a laptop GPU that is hours, and the label's structure already provides the context for free).

**Size limit.** 220 words keeps query + header + passage inside the cross-encoder's 512-token window. Larger chunks would be silently truncated by the reranker, i.e. ranked on their first half.

## 3. Entity scoping as a hard metadata filter

**Decision.** Drug names in the question are linked against an alias table built at ingest time (generic name, brand names, generic without the salt). If any drug is linked, both retrieval arms are restricted to those labels.

**Constraint.** The dominant failure in this corpus is *right section, wrong drug*. Neither BM25 nor a dense model reliably prefers the metformin contraindications over the 1,199 other contraindications sections.

**Why a dictionary and not an LLM/NER call.** The entity inventory is closed and known at ingest time. A dictionary lookup is exact, costs microseconds, cannot hallucinate a drug, and is trivially testable. Typos are handled by a fuzzy fallback that only runs when nothing matched exactly.

**Trade-off accepted.** A hard filter cannot recover from a wrong link. Aliases that are also ordinary words are blocked, and aliases shorter than four characters are dropped. A soft boost would be more forgiving but would let other drugs' boilerplate back into the top-k.

## 4. Hybrid retrieval fused with RRF

**Decision.** BM25 top-50 and dense top-50 are fused by Reciprocal Rank Fusion (k = 60).

**Constraint.** The two arms fail on different queries: BM25 on paraphrase ("can I drink alcohol"), dense on exact identifiers ("CYP2C9", "eGFR below 30"). Their scores are on unrelated scales, so any weighted sum needs per-corpus normalisation and tuning that silently goes stale. RRF uses ranks only, so there is nothing to tune and nothing to drift.

## 5. Retrieve wide, rerank narrow, with an ONNX cross-encoder

**Decision.** The fused top-30 is re-scored by `ms-marco-MiniLM-L-6-v2` running on ONNX Runtime (CPU); the top 6 go to the model.

**Constraint.** First-stage retrievers encode query and passage separately; a cross-encoder reads them together and is much more accurate, but costs a forward pass per pair. Thirty pairs is the budget that keeps reranking under the generation time.

**Why ONNX.** No PyTorch in the serving image: smaller image, faster cold start, no GPU contention with the embedding and generation models.

**Why six chunks.** More context is not free: it raises latency and cost linearly and makes it more likely the model blends facts from the wrong passage.

## 6. Near-duplicate collapsing

**Decision.** After reranking, passages whose token sets overlap ≥ 0.85 (Jaccard) are collapsed to the best-ranked copy; the ids of the suppressed copies are kept on the survivor.

**Constraint.** About 13 % of the labels are the same molecule from another manufacturer with almost identical text. Without collapsing, the top-6 is one paragraph six times. De-duplicating at the source instead would be wrong: metoprolol succinate and metoprolol tartrate are different products with different dosing.

## 7. Three abstention gates

**Decision.** Refuse (a) before generation if the question contains a term that occurs nowhere in the corpus and no drug was linked, or if the best reranker score is under a threshold calibrated on the golden set; (b) inside generation via an explicit `INSUFFICIENT_EVIDENCE` protocol; (c) after generation, optionally, if deterministic grounding checks fail.

**Constraint.** In this domain a confident wrong answer is worse than no answer, and users cannot tell the two apart. The pre-generation gates also make unanswerable questions the *cheapest* path rather than the most expensive.

**Why the unknown-term gate exists.** Measured, not assumed: for "What is the starting dose of Zorvanex?" the reranker happily scores other drugs' dosing sections as relevant, because they *are* relevant to "starting dose". Only the absence of the word from 10 M words of label text reveals that the subject is not in the index.

## 8. Deterministic grounding checks, with the LLM judge kept offline

**Decision.** At request time, answers are checked by code: citations must reference supplied sources, sentences should carry citations, and every number in a sentence must appear verbatim in the sources that sentence cites. An LLM judge is used only in the offline evaluation.

**Constraint.** A second LLM call per request doubles latency and cost and is itself fallible. The checks chosen are the ones that are exact and cheap, and the number check targets the single most dangerous error in this domain: a wrong dose.

**Known limit.** These checks cannot detect a fluent sentence that misstates its source without changing a number. That residual risk is what the offline faithfulness metric measures.

## 9. Providers behind protocols, with deterministic fakes

**Decision.** `Embedder`, `Reranker` and `LLM` are three small protocols. Each has a real implementation and a deterministic fake.

**Constraint.** A RAG system whose tests need a GPU and an API key does not get tested. With the fakes, the full pipeline including the HTTP API runs in CI in seconds, and switching the generator from a local model to Claude is a configuration change.

## 10. Reproducible generation

**Decision.** Temperature 0 and a fixed seed for the local generator; the response cache is keyed by (question, index version, model).

**Constraint.** An answer that changes between runs cannot be evaluated, debugged or regression-tested. Including the index version in the cache key means a re-ingest invalidates every cached answer with no explicit flush.

## 11. Two vocabularies: stemmed for "is it in the corpus?", exact for "is it a typo?"

**Decision.** The out-of-corpus gate asks the stemmed FTS index whether a word occurs anywhere. The typo matcher asks a separate, unstemmed vocabulary table.

**Constraint.** Each question needs the opposite behaviour. "Kidneys" must count as present when only "kidney" occurs, or a normal question would be refused. But the Porter stemmer reduces the typo "metformine" to the real word "metformin", which made the typo matcher conclude it was not a typo. The bug appeared as a side effect of fixing the previous one (fuzzy matching had mapped *norepinephrine* onto the *epinephrine* label), and a regression test caught it.

## 12. Independent review before publishing numbers

**Decision.** Before any metric went into the README, the codebase went through an independent review in three areas: ingestion and retrieval correctness, service and security, and evaluation methodology. Every finding then went through an adversarial verification pass that tried to refute it.

**What it found.** 19 confirmed defects. Five were high severity: 16 boxed warnings silently deleted by an over-eager de-duplication heuristic, wrong-drug typo matching, short bullets bypassing the number check, empty completions reported as grounded answers, and duplicate-crediting that could inflate recall. Each one is fixed and pinned by a test in `tests/test_review_regressions.py`. Five claims were refuted and left as they were.

**Why it is in this document.** The evaluation-methodology findings changed what the published numbers mean: a held-out split, the gate scored exactly as the service applies it, and strict recall next to credited recall. A metric is only as trustworthy as the harness that produced it.

## 13. Refusal threshold: zero false refusals on calibration, judged on held-out data

**Decision.** The reranker-score threshold is the highest one that wrongly refuses *no* answerable question in the calibration split. Every reported refusal rate comes from the test split.

**What went wrong first.** The first rule allowed up to 2 % false refusals on calibration. It picked a threshold that looked good there (82.8 % of unanswerable questions refused, 1.6 % of valid ones wrongly refused), but on the held-out split it wrongly refused 8.3 % of valid questions. Only about 30 calibration negatives reach this gate, so the "optimal" threshold was decided by two borderline items. The held-out split exists to catch exactly this.

**Why zero.** The two errors don't cost the same. A valid question refused is a user turned away. An unanswerable question that passes this gate still meets the model's explicit `INSUFFICIENT_EVIDENCE` check. So the gate should be conservative, and the model handles the rest.

**Found along the way.** The wrongly refused questions were almost all misspelled drug names. Two linker defects were behind them: generic-word blocking also blocked full names such as *sodium bicarbonate*, and typo correction only compared against one-word names, so *isosoride dinitrate* never matched. Typos are now corrected word by word against the drug-name lexicon before exact matching. Misspelled questions linked to the correct label: 74 of 79.

## 14. Tables are chunked one at a time, and every chunk carries its caption

**Constraint.** A dose without its population is a wrong answer. Labels state pediatric doses in weight-band tables ("5 to less than 10 kg: 5 mg"), and the only words saying who those doses are for are often in the table caption ("Table 2 … in Pediatric Patients").

**What went wrong.** A live check against current labels asked "What is the dose of omeprazole for GERD?". The answer listed the pediatric weight bands as the dose, 3 runs out of 3, although the adult dose was among the passages. Two chunking defects were behind it. The dosage section began `2 DOSAGE AND ADMINISTRATION *`, and the subsection splitter required a letter or digit after the heading, so the "2.2 Recommended Pediatric Dosage" title never reached a chunk header. And all tables of a field were packed into one stream, so a chunk could hold the end of the adult table and the start of the pediatric one, while later chunks lost the caption entirely.

**Decision.** More heading forms are recognised (`2.`, title case, a symbol after the title, no space). Each table is packed on its own, and every chunk of a table carries the table's numbered caption as its subsection, so it is embedded with the chunk and shown to the model in the source header. Only "Table N" captions are trusted. A first row that is a group label ("Ear and Labyrinth Disorders") or a footnote would mislabel every chunk of the table, which is worse than no label. A caption stored as a one-cell first row is kept rather than consumed as the column header.

**Rejected.** A prompt rule ("name the population of every dose") did not fix omeprazole and made two other dosing answers worse. The missing information had to be put back into the evidence, not demanded from the model.

**Consequences, found by an independent review of the fix.** Chunk ids hashed the chunk's position in the whole document, so adding chunks early in a label renamed every later chunk, including 94 gold passages. Ids now hash the section and the exact text, the golden set was re-mapped, and the eval refuses to run if a gold id is missing. Near-duplicate collapse compared text only, so two tables of one label that differ only by caption (the HEC and MEC regimens) collapsed into one; same-label chunks with different captions are no longer duplicates. Retrieval on the golden set moved from 90.6% to 90.1% Recall@5, one question, which the golden set cannot reward because its gold passages are never tables.

## 15. The model may not assert an absence

**Constraint.** Every claim shown must be one the system can check. A sentence that cites nothing and says "the sources do not cover X" cannot be checked: it claims an absence.

**What went wrong.** Re-judging all answers with a much stronger model found that 9 of 15 unfaithful answers had a false closing note, and 8 of those were otherwise correct answers ("The sources do not cover breastfeeding", with the lactation section among the passages). The partial-answer fix had made these notes visible. In a blind A/B, 16 of 27 such notes were false under the old prompt and 10 of 18 under a stricter rule.

**Decision.** Rule 4 now tells the model to check every source before saying anything is missing, and to say nothing when the question is answered. Uncited closing sentences that assert an absence are removed before checking and display, and a partial answer's `INSUFFICIENT_EVIDENCE` line is dropped rather than shown. A cited sentence such as "efficacy beyond 8 weeks has not been established [5]" is label content and stays.

**Cost.** True notes go too (8 of 18 in the A/B). A user is no longer told that part of a question was not covered; they see only cited facts. That trade is deliberate: an unverifiable note that is false more often than true makes the answer worse.

## 16. Two judges, and the stronger one is the headline

**Constraint.** An 8B judge grading an 8B generator can share its blind spots.

**Decision.** The offline judge (`qwen3:8b`) stays, because it is free, local and reproducible, and it compares versions well. Every published answer-quality number is also re-judged by a much stronger model that sees the passages and the reference but not the 8B verdicts. Where the two disagree, the stronger judge's figure is the one quoted: 91.7% correct end to end, against 95.8% from the 8B judge.

## 17. A sentence verifier that warns and repairs, but never deletes

**Constraint.** About 1 answer in 10 attaches a real fact to the wrong trial, population or drug. Every number in such a sentence is in the source, so the deterministic checks pass it.

**Decision.** Each sentence is checked against the passages by a small ONNX NLI model (`cross-encoder/nli-deberta-v3-small`) on single sentences prefixed with the drug name, plus a deterministic check that every population, species or trial the sentence names appears in its supporting passage. A sentence supported only by a passage it does not cite gets that citation added. A sentence nothing supports is marked "not verified" in the UI and kept.

**How it was chosen.** Ten variants (two model sizes, three ways of cutting passages, with and without the qualifier check) and an LLM quote-checker were scored against sentence-level gold labels on calibration answers. The rule was fixed before any result: the highest recall of false sentences with at most 5% of true ones flagged. The winner was recorded with a timestamp before its test score existed. Test: 32% of false sentences caught, 4.3% of true ones flagged, 88% of wrong citations repaired.

**Rejected.** Deleting flagged sentences: at 4% false alarms, a correct sentence would be removed from roughly one answer in seven. The LLM quote-checker: 23% false alarms and 6.5 s per answer, because an 8B model paraphrases what it is asked to quote. Two-sentence premises: the NLI model scored a verbatim sentence 0.986 alone and 0.003 with a neighbour appended.

**Cost.** About 1 s per answer on CPU and a second model in memory, so it is off on the 1 GB hosted demo.

## 18. An external benchmark, run on its own label text

**Constraint.** Every earlier number came from questions generated from this project's own chunks.

**Decision.** LabelLens was run unchanged on a stratified 400-question sample of FDARxBench (expert-reviewed, written by others). Its own label text was converted to the openFDA format and indexed with the normal pipeline, because 289 of its 685 labels are no longer served under the same id and the rest may have been revised. Every answer was graded by hand with the benchmark's CORRECT / INCORRECT / NOT_ATTEMPTED scheme.

**Result.** Factual 89.5%, multi-hop 58.0%, refusal F1 0.98. It is not directly comparable with the paper's figures (different grader, sample, setting), and the README says so. It found the next problem: two-part questions, where the 8B model answers one half.
