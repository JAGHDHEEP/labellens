# LabelLens

**Evidence-grounded question answering over FDA drug labels.** Every claim cites the label passage it came from, every number is checked against that passage, and when the labels don't contain the answer, the system says so instead of guessing.

> Reference tool built on public openFDA data. Not medical advice.

---

## The problem

A prescription drug label (the "package insert") is the legal source of truth for dosing, contraindications, interactions and warnings. It runs 30–100 pages, and every label uses the same section names. Clinicians and pharmacists look things up in these labels constantly.

A generic RAG chatbot fails here in three specific ways:

1. **Right section, wrong drug.** "What are the contraindications of metformin?" matches roughly 1,200 *Contraindications* sections about equally well.
2. **Near-duplicates.** The same molecule is sold by many manufacturers with almost identical text, so the top results end up as one paragraph repeated six times.
3. **Confident wrong numbers.** A hallucinated dose is the most dangerous possible failure, and a fluent answer gives no hint that it's wrong.

LabelLens is built around those three failure modes, and measures how well it handles each one.

## What it does

```
question ─► entity scope ─► BM25 (FTS5) ─┐
            (alias table)                 ├─► RRF ─► cross-encoder ─► near-dup ─► gate ─► LLM ─► grounding
                          dense (bge-m3) ─┘          rerank (ONNX)     collapse   (refuse?)        checks
```

| Stage | What it does | Why |
|---|---|---|
| **Structure-aware chunking** | Splits on the label's own sections and numbered subsections, and chunks each table on its own. Prepends `Drug (Brand) \| Section \| Subsection or table caption` to every chunk. | A chunk saying "contraindicated in severe renal impairment" is useless without the drug name attached, and "5 mg once daily for 5 to <10 kg" is dangerous without "Pediatric Patients". |
| **Entity scoping** | Links drug names (generic, brand, salt-less, biosimilar, typo-tolerant) to an alias table built at ingest, then hard-filters both retrieval arms. | Fixes "right section, wrong drug" with a microsecond dictionary lookup instead of an LLM call. |
| **Hybrid retrieval + RRF** | BM25 top-50 and exact dense top-50, fused by rank. | BM25 catches `CYP2C9` and `eGFR below 30`; dense catches paraphrase. Rank fusion needs no score tuning. |
| **Cross-encoder rerank** | Re-scores the fused top-30 with an ONNX `ms-marco-MiniLM` model. | Retrieve wide, rerank narrow. No PyTorch in the serving image. |
| **Near-duplicate collapse** | Keeps the best copy of passages with ≥ 0.85 Jaccard overlap, and records what it suppressed. | Six distinct pieces of evidence, not one paragraph six times. |
| **Abstention gates** | Refuses before generation on (a) a word found nowhere in the corpus or (b) a reranker score below a threshold calibrated on held-out data. The model can also refuse with `INSUFFICIENT_EVIDENCE`. | A refusal is the cheapest path, not the most expensive one. |
| **Sentence-level verification** | Each answer sentence is checked against the passages by a small ONNX entailment model. A sentence that only an uncited passage supports gets that citation added; a sentence no passage supports, or whose supporting passage never names the population or trial it states, is marked "not verified". | The checks below verify citations and numbers. This is the layer that looks at what a sentence *says*. It repairs 88% of wrong citations; it catches about a third of false sentences (measured below). |
| **Grounding checks** | Citations must point at supplied sources. Every number must appear in the passage it cites, with its unit when it has one ("2 days" is not supported by "2 weeks"). Uncited closing claims that "the sources do not cover X" are removed. | Catches a wrong dose deterministically, with no second LLM call. It cannot catch a sentence that reverses or misattributes its source without changing a number; that is measured offline below, not blocked. |

## Results

All numbers come from the **held-out test split** of a 480-question golden set (400 answerable, 80 unanswerable), over 1,200 labels / 78,852 passages. Thresholds were tuned on a separate calibration split. Raw outputs are in [`eval/results/`](eval/results), and the method and its known biases are described in [`evalkit/dataset.py`](src/labellens/evalkit/dataset.py).

### Retrieval: what each component is worth

192 answerable test questions, each written from one label passage (its *gold* passage). *Recall@5* = the gold passage is in the top 5. *In prompt* = it is among the 6 passages actually given to the model.

| Configuration | Hit@1 | Recall@5 | In prompt (6) | MRR@10 | Right drug in top 5 | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| Naive: dense only, plain chunks | 42.2% | 65.6% | 68.2% | 0.524 | 90.6% | 65 ms |
| BM25 only, plain chunks | 28.6% | 55.7% | 58.9% | 0.402 | 86.5% | 44 ms |
| Dense + contextual header | 51.6% | 82.8% | 84.4% | 0.647 | 97.9% | 54 ms |
| Hybrid (BM25 + dense, RRF) | 47.9% | 78.1% | 81.2% | 0.612 | 93.2% | 111 ms |
| Hybrid + cross-encoder rerank | 53.1% | 85.4% | 86.5% | 0.660 | 99.0% | 1.2 s |
| Hybrid + entity scoping | 55.2% | 88.5% | 89.6% | 0.683 | 97.9% | 190 ms |
| **Full: hybrid + scoping + rerank** | **58.3%** | **90.1%** | **91.1%** | **0.708** | **99.0%** | 1.2 s |
| Full, without contextual header | 52.1% | 81.8% | 83.3% | 0.644 | 99.0% | 1.2 s |
| Scoping + rerank, dense only (no BM25) | 58.3% | 89.6% | 90.6% | 0.709 | 99.0% | – |
| Scoping + rerank, BM25 only (no dense) | 58.3% | 89.1% | 89.6% | 0.703 | 98.4% | – |

Full pipeline Recall@5: 95% bootstrap CI **85.9–94.3%**. Strict (exact-chunk) Recall@5: 85.9%. The last two rows were measured later on a slower machine state, so their latency is not comparable and is omitted. The credited figure also counts another manufacturer's identical copy of the same paragraph. Every number was re-derived by an independent script that shares no code with the evaluation harness, and they match; the exact settings of each run are stored in the result file.

What this does *not* measure: gold passages are prose only (tables and passages under 60 words are never gold), and each question has exactly one gold passage, so a correct answer found in a different passage counts as a miss.

**What the ablation shows**

- **The contextual header is the largest single gain:** +17.2 points for dense retrieval, and +8.3 points inside the full pipeline. Without it, "contraindicated in severe renal impairment" could be about any of 1,200 drugs.
- **Naive hybrid search hurt** (78.1% vs 82.8% for dense alone). Every label shares the same section vocabulary, so BM25 alone is weak (67.2% even with the contextual header) and drags the fusion down.
- **Entity scoping is the cheapest lever:** 88.5% with no reranker, at about 200 ms instead of 1.2 s.
- **Once scoped and reranked, either arm alone is almost as good.** Dropping BM25 costs 0.5 points (paired 95% CI −1.6 to +3.1) and dropping dense costs 1.0 point (−1.0 to +3.1); neither difference is significant on 192 questions. BM25 stays because it is free (the FTS5 index lives in the same file), it matches exact strings such as `CYP2C9` or a brand name the embedder has never seen, and it keeps working while new passages are still waiting for their vectors. That is a robustness argument, not a measured accuracy gain.
- **Once scoped, the embedding model barely matters.** A label has a median of 56 passages, so BM25-50 plus dense-50 already cover nearly all of it, and the reranker decides. The hosted demo's 33M-parameter CPU model scores the same Recall@5 (90.1%) as `bge-m3` (568M) on GPU. The hosted demo gives the model only 4 passages, though, so its gold-in-prompt rate is 88.0%.
- **Bias check:** Recall@5 is 84.3% on questions with below-median word overlap with their source, and 95.2% above it. Misspelled drug names: 93.0% (n = 43).
- **Re-chunking cost nothing measurable.** Labelling table chunks with their captions and splitting about 540 more sections into subsections (73,470 → 78,852 passages) moved Recall@5 from 90.6% to 90.1%, one question. The gold passages were re-mapped to the new chunks first (391 of 400 by text overlap, 9 split ones by their reference answer), and the eval now refuses to run if a gold id is missing from the index. The benefit of the change does not show here at all, because gold passages are never tables; it shows in the live check below.

### Refusing when the labels don't have the answer

Gate threshold chosen on the calibration split, then measured on 192 answerable and 34 unanswerable test questions.

| | Test split |
|---|---:|
| Unanswerable questions refused *before* any LLM call | **94.1%** |
| Questions about invented drugs refused | 94.4% |
| Real-drug questions labels never answer (price, patents…) refused | 93.8% |
| Answerable questions wrongly refused | **0.5%** (1 of 192) |
| AUROC, answerable vs unanswerable | 0.997 |

A first threshold rule allowed 2% false refusals on calibration data. It wrongly refused **8.3%** on the test split, because about 30 calibration negatives had let two borderline items set the threshold. The rule now requires zero false refusals on calibration data, and the model's own `INSUFFICIENT_EVIDENCE` check backs it up. The full story is in [DECISIONS §13](docs/DECISIONS.md#13-refusal-threshold-zero-false-refusals-on-calibration-judged-on-held-out-data).

### Answer quality

The local configuration was asked 154 test questions (120 answerable, 34 unanswerable): generator `llama3.1:8b`, judge `qwen3:8b`, a different model so nothing grades its own work.

| Answerable questions (n = 120) | 8B judge | Stronger judge |
|---|---:|---:|
| **Correct end to end** (answered, and judged correct) | 95.8% | **91.7%** |
| Gold passage among the 6 the model saw | 94.2% | |
| Model chose `INSUFFICIENT_EVIDENCE` | 0% | |
| Refused by the retrieval gate | 0.8% | |

| Of the 119 answers given | 8B judge | Stronger judge |
|---|---:|---:|
| Judged correct against the reference | 96.6% | **92.4%** |
| Judged faithful: every claim supported by the passages | 100% | **90.8%** |
| States something a passage contradicts or reverses | | 3.4% (4) |
| Cites a passage that does not support the claim | | 10.1% |
| **Deterministic grounding checks passed** | **97.5%** | |
| Sentences carrying a citation | 95.0% | |
| Citations pointing at a passage that wasn't supplied | 0% | |
| Cut off by the token limit | 0% | |

**Why two judges.** The generator and the original judge are both 8B models, and a small judge can fail along with the model it grades. So every answer was re-judged by a much stronger model (Claude Opus 5.5, four independent batches, shown the passages and the reference but not the 8B judge's verdicts), with a rubric that checks each number, population, route, indication and direction. The two agree on correctness for 92% of answers. Of the 9 disagreements, the stronger judge was stricter in 7. Its verdicts are in [`eval/results/strong_judge.json`](eval/results/strong_judge.json). Treat **91.7% correct end to end** as the headline, not 95.8%.

| Unanswerable questions (n = 34) | |
|---|---:|
| **Refused** | **100%** (94.1% before the LLM, 5.9% by the LLM) |

Latency with the local GPU model (RTX 4060 laptop): p50 4.3–5.5 s / p95 9.3–10.1 s end to end, and p50 about 2 s until the first streamed token (retrieval included). The range is between runs on the same machine: a back-to-back test of the old and new index showed the slower runs came from the laptop, not the code. The hosted Groq model generates in under 1 s.

**What the stronger judge found, and what was fixed.** Its first pass found 15 unfaithful answers, and 9 of them had the same cause. The model closed a correct answer with a sentence such as "The sources do not cover breastfeeding" while a supplied passage covered exactly that. In a blind A/B on the 35 answers that had such a note, 16 of 27 notes were false under the old prompt and 10 of 18 under a stricter one, while all 35 answers were judged correct under both. A note asserts an absence, cites nothing, and cannot be checked, so the system no longer shows one: rule 4 was tightened, and an uncited closing "not covered" sentence is removed before the answer is checked and displayed. Faithfulness under the stronger judge went from 87.4% to 90.8%, and citation coverage from 87.9% to 95.0%.

**What remains** is mostly misattribution. A real fact from the passages gets pinned to the wrong population, trial or drug: a pooled 11-drug result described as primidone's, later-onset SMA figures given for infantile-onset, a nitroglycerin study attributed to isosorbide. Every number in such a sentence appears in the source, so the deterministic check passes. The rest: one answer reverses its source (it says multiple small injections cause atrophy; the label recommends them to minimise it), one gives a wrong standard deviation, and one makes a false "the labels give no comparison" claim in the middle of an answer, where the closing-note filter does not look. This is the main open problem (see Limitations).

**Earlier changes**, between the first run and the version above (same 120 questions, same models and seed):

- **Partial answers are kept.** On 8% of answerable questions, the model had written a full cited answer and then closed with an `INSUFFICIENT_EVIDENCE` line for the part the sources don't cover. The service treated the whole reply as a refusal and threw the answer away. Such replies are now kept and checked. (The uncovered-part line was at first shown as a note; it is now dropped, see above and [DECISIONS §15](docs/DECISIONS.md#15-the-model-may-not-assert-an-absence).) All 10 recovered answers were judged correct and faithful, and unanswerable questions are still refused 100% of the time.
- **Faster.** On Windows, `localhost` tried IPv6 first and cost about 2 s per new connection to Ollama; the client now uses `127.0.0.1`. The question embedding also runs on the CPU (about 80 ms median), so the chat model keeps the 8 GB GPU instead of being swapped out on every question.

**Reading these numbers honestly**

- **Two failure modes are left, about equally common.** Retrieval: the gold passage is missing from the prompt for 5.8% of questions, and no model can answer from missing evidence. Generation: misattribution within the passages it was given (above).
- **3 answers were flagged by the number check; 2 flags are deserved.** Apixaban's ARISTOTLE bleeding rates and an azithromycin standard deviation are cited to passages that do not contain them. The third is a false positive: relugolix's 35% is in the cited passage, but that table was flattened into prose with its "%" stated far from the numbers, and the checker now accepts a table-wide "%" only for a real cell in a matching row. The stronger judge found 12 answers with a wrong citation, so the check catches the numeric ones and misses the rest.
- **The checker has been corrected three times, never to hide a failure.** Each time, the stored answers were re-scored deterministically (`python -m labellens.evalkit.rescore`) without regenerating them. The re-scorer now appends each correction, with its reason and before and after figures, to a history in `generation.json`; earlier corrections are in the git history of that file. The first correction covered real label formats (`47.8 (13.7) hours`, `µg` for `mcg`, the `%` in a table header). The second covered numbers glued to a word in the label text ("younger than18 years"). The third covered a number used as a label ("Week 2 dose" is not a claim about "2 doses"), trailing zeros ("13.7" in a "13.70" cell) and a "%" stated once in a table's caption or header row. An independent review of that third change showed the first version was too loose (it accepted "20%" read off a "20 mg" column header, and a mg dose after "per day" with the wrong unit); the version shipped accepts only a cell value in the row the sentence names, and never drops a mass, volume or percentage unit.
- An 8B model is a lenient judge. Treat judge scores as relative signals for comparing versions, not absolute truth.

### Sentence-level claim verification

The grounding checks catch a wrong number. They cannot catch a sentence that attaches a real fact to the wrong trial, population or drug, which the stronger judge found in about 1 answer in 10. This layer checks each sentence against the passages with a small natural-language-inference model (`cross-encoder/nli-deberta-v3-small`, ONNX, CPU, no PyTorch), plus a deterministic check that every population, species or trial name a sentence states appears in the passage that supports it.

**How it was measured.** Two sets of real answers were labelled sentence by sentence by a stronger model, against the exact passages the system had been given: 118 answers / 461 sentences from the *calibration* questions (for choosing) and 119 answers / 428 sentences from the *test* questions (for reporting). Each sentence is *supported*, *wrong citation* (true, but the cited passage does not say it), or *false* (misattributed, contradicted or unsupported). Ten variants were compared. The selection rule was written down before any result was seen: the highest recall of false sentences on calibration, with at most 5% of true sentences flagged. The chosen configuration was saved with a timestamp ([`eval/claims/selection.json`](eval/claims/selection.json)) before its test score was computed.

| Shipped: small NLI model + drug name + qualifier check | Calibration | **Test** |
|---|---:|---:|
| False sentences caught | 48% (11/23) | **32% (6/19)** |
| True sentences wrongly flagged | 4.1% | **4.3%** |
| Wrong citations repaired by adding the right passage | 77% | **88% (22/25)** |

What the comparison showed ([`eval/results/claims.json`](eval/results/claims.json)):

- **The verifier's input mattered more than its size.** NLI models are trained on sentence pairs. A sentence copied word for word from a passage scored 0.986 entailment on its own and 0.003 with the next sentence appended, so the first version, which compared claims with two-sentence windows, flagged 8 to 10% of true sentences at any threshold. Comparing single sentences, prefixed with the drug name because passages say "it" or the brand, fixed most of that. The larger base model did not do better.
- **An LLM verifier was worse and slower.** Asking `llama3.1:8b` to quote the supporting sentence (with the quote then checked to exist) caught 53% of false sentences but flagged 23% of true ones, at 6.5 s per answer. The NLI verifier takes about 1 s per answer on CPU.
- **Test differs from calibration, and that is reported as is.** Another variant (1 to 2 sentence windows) caught 42% of false sentences on test, more than the shipped one, but less on calibration, where the choice was made. With 19 false sentences in the test set, these differences are within noise.
- **What it still misses.** Most misattributions read as entailed: "patients randomized to one of the AEDs" and "patients randomized to primidone" differ by one noun phrase. It catches about a third of false sentences, so it is a warning layer, not a filter. Flagged sentences are underlined in the UI and kept, not deleted.

### External benchmark: FDARxBench

[FDARxBench](https://github.com/xiongbetty/FDARxBench) (2026, Apache-2.0) is an expert-reviewed set of 17,207 questions over 685 FDA labels, written by others, with factual, multi-hop (two sections of one label) and refusal (the label does not contain it) questions. LabelLens was run **unchanged** on a stratified sample of 400 (200 factual, 100 multi-hop, 100 refusal; seed 13; one question per label where possible). The index was built from **the benchmark's own label text**, converted to the openFDA format, so answers and references come from the same label versions. Every answer was graded by hand against the reference with the benchmark's scheme: CORRECT / INCORRECT / NOT_ATTEMPTED. A multi-hop answer counts as correct only if both parts are right. Grades and reasons: [`eval/results/fdarxbench.json`](eval/results/fdarxbench.json).

| FDARxBench (sample of 400) | LabelLens (local, 8B) | 95% CI |
|---|---:|---:|
| Factual accuracy (n = 200) | **89.5%** | 84.5 - 93.0% |
| Multi-hop accuracy (n = 100) | **58.0%** | 48.2 - 67.2% |
| Refusal questions declined (n = 100) | **97%** | 91.5 - 99.0% |
| Refusal F1 / answerable questions wrongly refused | **0.98** / 0.3% | |

For context, the paper reports its best models at about 0.56 factual and 0.47 multi-hop accuracy and 0.71 to 0.80 refusal F1, with the whole label in the prompt and an LLM grader. **These numbers are not directly comparable**: a sample of 400, a different grader (by hand here), and a retrieval setting rather than the whole label. They do show that the system holds up on questions it was not built around, written by people who never saw it.

What the benchmark found:

- **Multi-hop is the weak spot.** Of the 41 wrong multi-hop answers, 28 answered one half of a two-part question and dropped the other ("What is the dose, and what did the carcinogenicity studies show?"). Of the rest, 3 wrongly said the label does not cover the second part, 1 invented a fact, and the others missed the key detail. Retrieval returns six passages ranked for the question as a whole, and an 8B model tends to answer the first part. Splitting compound questions and retrieving for each part is the obvious next step.
- **4 of 21 wrong factual answers have defective references** (a reference that is only section numbers, or a lactation note for a dosing question). They were still counted as wrong.
- **Refusals transfer.** The gate and the model declined 97 of 100 questions the labels cannot answer, and wrongly refused 1 of 300 answerable ones. The three misses answered with nearby label text (thyroid replacement, liver monitoring) instead of declining.

### Live check against today's FDA labels

Fourteen new, hand-written everyday questions (maximum acetaminophen dose, ibuprofen in the third trimester, warfarin's boxed warning, omeprazole for GERD, an Ozempic price question, "can I double my pills"…) were answered locally. Every cited passage and every number was then compared with the **current** label fetched from the openFDA API ([`eval/results/livecheck.json`](eval/results/livecheck.json)). This is a spot check: 14 questions cannot estimate an error rate.

- No answer contained a number missing from the live label. 20 of 23 cited passages are still word for word in today's label; 3 labels have been revised since the index was built, so a production deployment needs a scheduled re-ingest (it is incremental).
- **It found a real safety bug.** "What is the dose of omeprazole for GERD?" was answered with the pediatric weight-band table (5 mg for 5 to <10 kg, 10 mg for 10 to <20 kg…), with nothing saying it was pediatric, 3 runs out of 3. The adult dose was among the passages. The cause was chunking, not the model: the section began `2 DOSAGE AND ADMINISTRATION *`, a heading form the subsection splitter did not recognise, and all of a field's tables were packed together, so a chunk could hold the end of the adult table and the start of the pediatric one while later chunks lost the "Pediatric Patients" caption. A prompt rule ("name the population") did not help and made two other answers worse, so it was dropped.
- **Fix:** more heading forms are recognised, each table is chunked on its own, and every chunk of a table carries its numbered caption in the header the model sees. An independent review of that fix found ten edge cases (captions stored in a table's first row and dropped, group rows and footnotes taken as captions, sibling tables merged as near-duplicates, gold ids shifted by a position-based chunk id…), all fixed and pinned by tests. Omeprazole now gets "20 mg once daily for up to 4 weeks", 3 runs out of 3, and seven other dosing questions name the population for every weight- or age-banded dose.
- One answer is still wrong after the fix: atorvastatin's discontinuation rates are listed as the "≥2%" adverse reactions. Both sentences are in one passage, and the 8B model merged them.

### Outside spot check: 16 adversarial questions

An outside reviewer wrote a test plan covering interactions, multi-drug comparisons, missing information, patient-specific questions, prompt injection and ambiguous questions. I ran 16 of its questions on the hosted configuration and audited every answer against its cited passages ([`eval/results/spotcheck.json`](eval/results/spotcheck.json)). Five failures turned up, and each fix is pinned by a test in `tests/test_audit_fixes.py`:

| Failure found | Fix |
|---|---|
| `gpt-oss` cited as `【1】`, so a correct answer failed the grounding check | Lenticular and full-width brackets are normalised before checking and before display |
| "Compare metformin and apixaban": 3 of 4 passages were apixaban | When a question names several drugs, evidence slots are dealt round-robin per drug |
| "Is metformin safe?" answered with pediatric and pregnancy notes and missed the boxed warning | For safety questions, the drug's boxed warning is pinned into the evidence |
| "Can warfarin be taken with amiodarone?" began "Warfarin can be taken…", from one label only | The prompt forbids clearance statements and asks what each label says |
| "Can I take this drug with my other medication?" was refused in one run but answered from four unrelated labels in another | A question that points at a drug without naming one is refused before the LLM, deterministically |

After the fixes: 6 of 16 questions correctly refused (cost, probability, patient-specific dose, fictional drug, prompt injection, unnamed drug); 8 correct; 1 partially correct (the metformin/apixaban comparison still omits metformin's eGFR 30–45 rule); and the manufacturer-comparison question refused because that feature isn't supported. All 9 answered questions passed the grounding checks. Retrieval Recall@5 on the test split was unchanged at 90.6% at the time (90.1% after the later re-chunking).

## Engineering for production

- **One transactional store.** Documents, chunks, the BM25 index, embeddings and aliases live in one SQLite file. A label revision replaces its chunks, keyword index and aliases in one transaction. Vectors for new text follow when embedding finishes, and the index version is bumped only then, so a running server never loads a half-embedded index (but see Limitations for what it serves meanwhile). ([why](docs/DECISIONS.md#1-one-sqlite-file-instead-of-a-vector-database-plus-a-search-engine))
- **Incremental, idempotent ingest.** Unchanged labels are skipped by content hash. Embeddings are keyed by the hash of the embedded text, so a re-run embeds only new text. A pipeline version in the fingerprint re-indexes everything when cleaning logic changes.
- **Hot reload.** A running server picks up a re-ingest without a restart and swaps in the new index as one immutable snapshot. The response cache is keyed by index version, so it invalidates itself.
- **Streaming that never retracts.** Server-sent events stream the answer, but text that may turn out to be a refusal, and every answer in strict mode, is held back until it has been verified.
- **Observability.** Prometheus metrics with per-stage latency histograms (scope, BM25, embed, dense, rerank, first token, generation), grounding-failure counters, structured JSON logs with request ids, and `/healthz` and `/readyz`.
- **Failure handling.** Retries with backoff for model backends, a `503` with `Retry-After` when a backend is down, and empty or truncated completions flagged and never cached.
- **Swappable providers.** Embedder, reranker and LLM are three small protocols. The generator runs locally on Ollama or on Claude (`LABELLENS_LLM_PROVIDER=anthropic`) with no code change.
- **Tests that need no GPU.** 136 tests use deterministic fakes for every model, so CI runs the whole pipeline, including the HTTP API, in about a second.

## Quick start

Requirements: Python 3.10+, [Ollama](https://ollama.com) with `bge-m3` and a chat model (default `llama3.1:8b`).

```bash
ollama pull bge-m3
```

```bash
ollama pull llama3.1:8b
```

```bash
pip install -e ".[dev]"
```

```bash
labellens fetch
```

```bash
labellens ingest
```

```bash
labellens serve
```

Then open http://127.0.0.1:8000. `fetch` downloads about 1,200 labels (~25 MB). `ingest` chunks and embeds them; on an RTX 4060 laptop GPU that takes about 35 minutes the first time and seconds on later runs.

Other commands:

```bash
labellens ask "Can warfarin be taken with amiodarone?"
```

```bash
labellens search "metformin contraindications"
```

```bash
pytest
```

With Docker: `docker compose up --build`. The first-run commands are listed at the top of `docker-compose.yml`.

### API

| Endpoint | Purpose |
|---|---|
| `POST /v1/ask` | `{"question": "..."}` → answer, status, cited sources, grounding report, timings |
| `POST /v1/ask/stream` | Same answer as server-sent events: `sources` → `token`* → `final` |
| `POST /v1/search` | Retrieval only, with each hit's BM25 rank, dense rank, RRF and rerank scores. For debugging. |
| `GET /v1/stats`, `/healthz`, `/readyz`, `/metrics` | Operations |

### Reproducing the evaluation

```bash
labellens eval synth --n 400
```

```bash
bash run_eval.sh
```

## Hosted demo (Streamlit Community Cloud)

The free hosted version runs the same pipeline (entity scoping, hybrid retrieval, reranking, gates and grounding checks) on a CPU-only host with about 1 GB of memory. The model providers are swapped, and the CPU budget sets a smaller rerank and prompt:

| | Full (local) | Hosted (free) |
|---|---|---|
| Passage embeddings | `bge-m3`, 1024-d, GPU | `snowflake-arctic-embed-s`, 384-d. The index is embedded on GPU with Ollama and queried on CPU with ONNX; same weights, verified identical vectors. |
| Index | about 1.1 GB, float32 (two embedding variants: with and without the contextual header, for the ablation) | Compact export with float16 vectors, downloaded on first start |
| Answer model | `llama3.1:8b` on Ollama | `gpt-oss-20b` on Groq, falling back to `gpt-oss-120b` on rate limits |
| Candidates reranked | 30 | 20 |
| Passages in prompt | 6 | 4 (Groq's free tier allows 8k tokens per minute per model) |
| Protection | none | 5 questions per minute per session, short rate-limit waits instead of errors |
| **Retrieval Recall@5 (test split)** | **90.1%** | **90.1%** (Hit@1 58.3%) |
| **Gold passage in the prompt** | **91.1%** of 6 passages | **88.0%** of 4 passages |

Deploy your own:

1. `labellens export` builds `data/deploy/labellens-compact.db.gz`. Upload it as a GitHub release asset.
2. On [share.streamlit.io](https://share.streamlit.io), create an app from this repository with main file `streamlit_app.py` and Python 3.12.
3. In the app's **Secrets**, set `GROQ_API_KEY`, `LABELLENS_INDEX_URL` (the release asset URL) and, optionally, `LABELLENS_INDEX_SHA256`. See `.streamlit/secrets.toml.example`.

## Repository map

```
src/labellens/
  fetch.py         openFDA client, de-duplicated corpus builder
  normalize.py     label cleaning: overlapping fields, repeated sentences, HTML tables
  chunking.py      section/subsection-aware chunking with contextual headers
  store.py         SQLite: documents, chunks, FTS5, embeddings, aliases, vocabulary
  ingest.py        incremental, idempotent ingestion and embedding
  retrieval.py     entity linking, hybrid search, RRF, rerank, near-dup collapse
  generation.py    prompt, citation and number grounding checks
  verify.py        sentence-level claim verification (NLI + qualifier check)
  service.py       gates, streaming, cache, hot reload
  api.py           FastAPI app;  static/index.html  web UI
streamlit_app.py   free hosted demo (CPU embeddings, Groq answers)
  evalkit/         golden-set synthesis, ablation, gate calibration, LLM judge,
                   claim-verifier study (claims.py), FDARxBench runner (fdarx.py)
docs/DECISIONS.md  every design decision, the constraint behind it, what was rejected
tests/             136 tests, including regression tests for every review and audit finding
```

## Limitations

- **Synthetic questions.** The golden set is generated by an 8B model from label passages, so it inherits the chunker's view of a label: questions come from prose passages the chunker produced, and real users ask differently. The 16 outside questions and the 14 live-check questions are hand-written, but 30 questions is a spot check. A few hundred clinician-written questions would be the real test.
- **The answer judges.** The 8B judge scored 95.8% correct; a stronger blind judge scored 91.7%. Neither is a pharmacist. A human review of a sample is the next step before trusting either number.
- **Misattribution is only partly caught.** The sentence verifier catches about a third of false sentences on held-out data (with 4% of true sentences flagged) and repairs most wrong citations, but a sentence that swaps one noun phrase ("one of the AEDs" for "primidone") usually reads as entailed. A larger generator is the next lever. The verifier needs a second model in memory, so it is off on the 1 GB hosted demo.
- **Two-part questions.** On FDARxBench, multi-hop accuracy was 58% against 89.5% for single-part questions: the model often answers one half. Question decomposition is the planned fix.
- **Ranking, not just recall.** The right passage is in the top 5 for 90.1% of questions but ranked first for only 58.3%. The model sees six passages (91.1% contain the gold one), so this costs little in answers, but a section-aware query router would raise it.
- **The refusal gate is tuned on easy negatives.** Its threshold allows zero false refusals on calibration data, and AUROC is 0.997, but the unanswerable questions are invented drugs or facts no label states (price, patents). A question with a false premise about a real drug ("why is X contraindicated in asthma?" when it is not) would score as answerable, and only the model and the checks stand behind it.
- **Re-ingesting under a running server.** During an ingest, chunks are replaced immediately but their vectors arrive only when embedding finishes (minutes for a full rebuild). Until the index version is bumped, a running server keeps its old vector snapshot, whose row ids no longer exist, so dense search returns nothing for re-ingested labels and answers come from BM25 alone. Restart-free updates should build into a second file and switch. The hosted demo already works that way (it downloads a finished index).
- **Unlinked drug names.** When no drug name in the question can be linked (3 of 192 test questions, mostly 5-ingredient combination products), retrieval runs unscoped and misses.
- **Paraphrase errors.** The deterministic checks can't catch a sentence that misstates its source without changing a number. That residual risk is measured offline (judge faithfulness) but not blocked at request time.
- **Corpus size.** The index holds about 1,200 of the ~37,000 prescription labels in openFDA (the most recent revision of each distinct drug, plus about 100 commonly prescribed drugs). Nothing here shows the numbers hold at 37,000. Scoping should carry over, since a question about one drug still searches only that drug's labels, but unscoped questions, BM25 noise, the alias table's ambiguity and an exact in-memory dense scan (about 2.4 million passages: 1.9 GB of vectors at 384-d float16, 10 GB at 1024-d float32, past the point where an approximate index is worth it) all get harder. Measuring that is the next piece of work, not a solved one.
- **Label text only.** No clinical guidelines and no patient-specific reasoning. LabelLens reports what the labels say. It supports a professional's own judgment and doesn't replace a pharmacist or a clinical decision-support system such as Lexicomp or Micromedex. A label that doesn't mention an interaction is not evidence that none exists.
- **One label revision per drug.** The index keeps the latest revision of each distinct label, so it can't compare manufacturers' versions or show what changed between revisions. Sources link to the exact label on DailyMed by set id.
- **Attribution in multi-drug answers.** Every fact cites a passage, and the UI shows which drug's label each passage comes from. But a sentence like "the warfarin label says…" can still name the wrong label while citing the right passage. The checks don't catch that.

## License

MIT. Label data comes from [openFDA](https://open.fda.gov), which describes its data as unvalidated and not intended for clinical decisions.
