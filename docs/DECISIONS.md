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
