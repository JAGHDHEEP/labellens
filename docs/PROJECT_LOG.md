# LabelLens: project log, start to finish

Every phase of the project: what was built, what went wrong, why it went wrong, how it was fixed, and what was measured afterwards. Dates are from the git history (all commits by JAGHDHEEP). Numbers come from the files in `eval/results/` at the time.

---

## At a glance

| | |
|---|---|
| What | Question answering over official FDA drug labels: every sentence cited, every number checked, refuses when the labels don't say |
| Corpus | 1,200 openFDA labels → 78,852 passages (10.8 M words) |
| Stack | Python, SQLite (FTS5 + vectors in one file), bge-m3 via Ollama, ONNX cross-encoder reranker, llama3.1:8b, FastAPI + web UI, Streamlit + Groq for the hosted demo |
| Timeline | 11 commits, 19–21 Nov 2026 |
| Tests | 0 → 104 → 113 → 125 → 130 → 136, all running in CI with no GPU and no API key |
| Final results | Retrieval Recall@5 90.1%; unanswerable questions refused 100%; answers correct end to end 91.7% (strong blind judge); FDARxBench factual 89.5%, multi-hop 58.0%, refusal F1 0.98 |

---

## Phase 0: The idea and the design (before the first commit)

**Goal.** A RAG project that is not "chat with PDFs": a domain where a wrong answer is dangerous, so the system has to prove its answers.

**Why FDA drug labels.**
- They are the legal source of truth for dosing, warnings and interactions.
- They are public (openFDA).
- They break naive RAG in three specific ways:
  1. **Right section, wrong drug.** All 1,200 labels share the same section names, so "What are the contraindications of metformin?" matches every Contraindications section about equally.
  2. **Near-duplicates.** The same molecule is sold by many manufacturers with almost identical text.
  3. **Confident wrong numbers.** A hallucinated dose looks exactly like a real one.

**Key design decisions** (full reasoning in `docs/DECISIONS.md` §1–11):

| Decision | Why |
|---|---|
| One SQLite file (FTS5 + vectors + aliases) instead of a vector DB plus a search engine | One transaction per label update, so the keyword and vector indexes can't drift apart. Exact search over about 285 MB of vectors is fast enough. |
| Chunk by the label's own sections and subsections (max 220 words), with a `Drug \| Section \| Subsection` header on every chunk | A chunk saying "contraindicated in severe renal impairment" is useless without the drug name. 220 words fits the reranker's 512-token window. |
| Entity scoping: a dictionary lookup of drug names, then search only those labels | Fixes "right section, wrong drug" in microseconds with no LLM call. |
| Hybrid BM25 + dense, fused with RRF | Each finds what the other misses; RRF needs no score tuning. |
| ONNX cross-encoder rerank of the top 30, keep 6 | Much more precise; no PyTorch in the serving image. |
| Near-duplicate collapse (Jaccard ≥ 0.85) | Six distinct pieces of evidence instead of one paragraph six times. |
| Three refusal gates | Refusing is the cheapest path, not the most expensive. |
| Deterministic grounding checks (citations + numbers) at request time; LLM judge only offline | Catches a wrong dose with code, with no second LLM call. |
| Fakes for every model | Tests run in CI without a GPU or API key. |

**Problems found while building:**
- **The unknown-drug problem.** For "What is the starting dose of Zorvanex?" (an invented drug), the reranker happily scored other drugs' dosing sections as relevant. *Fix:* the unknown-term gate refuses when a word appears nowhere in 10 M words of label text and no drug was linked.
- **The stemmer bug.** The out-of-corpus check needs stemming ("kidneys" must match "kidney"), but the Porter stemmer turned the typo "metformine" into the real word "metformin", so the typo matcher decided it wasn't a typo. It appeared as a side effect of another fix (fuzzy matching had mapped *norepinephrine* onto the *epinephrine* label). *Fix:* two vocabularies, a stemmed one for "is it in the corpus?" and an exact one for "is it a typo?". A regression test caught it.

**The independent review before publishing any number** (§12). The codebase was reviewed in three areas: retrieval correctness, service and security, and evaluation method. Every finding then had to survive an attempt to refute it. That confirmed **19 defects, 5 of them high severity**:
1. **16 boxed warnings silently deleted** by an over-eager de-duplication heuristic: one sentence of a boxed warning is often repeated under Warnings, and the heuristic treated the whole boxed warning as a duplicate. *Fix:* only exact containment is excised.
2. **Typo matching linked the wrong drug.**
3. **Short bullets bypassed the number check** ("Max 2550 mg [1]" is exactly where a wrong dose would hide).
4. **Empty model completions were reported as grounded answers.**
5. **Duplicate-crediting could inflate recall.**

All fixed and pinned in `tests/test_review_regressions.py`. The evaluation findings changed the method: a held-out test split, the gate scored exactly as the service applies it, and strict recall reported next to credited recall.

---

## Phase 1: First version (commit `177458f`, 19 Nov 11:42)

**Built.** The full pipeline (hybrid retrieval, scoping, rerank, de-dup, gates, grounding checks), a FastAPI service with a streaming web UI, the evaluation harness, a 480-question golden set, the decision log, Docker and CI.

**The golden set.** 480 questions generated by `qwen3:8b` from real passages (so each question's gold passage is known), including misspellings and brand names, plus 80 unanswerable ones (invented drugs, and real drugs asked about things labels never contain, such as price). They're split into *calibration* (for tuning) and *test* (for reporting only).

---

## Phase 2: Free hosted demo (commits `74d541d`, `9e9398c`, `689d0f5`, 19 Nov)

**Goal.** A public demo that costs nothing to run.

**Built.**
- A Streamlit app on the same pipeline.
- Groq (`gpt-oss-20b`, falling back to `gpt-oss-120b`) as the model.
- A small CPU embedding model (`snowflake-arctic-embed-s`).
- `labellens export`, which builds a compact float16 index (130 MB compressed) published as a GitHub release asset and checked by SHA-256 on download.

**Problems and fixes:**
- **CI failed.** The virtual environment was set up wrongly in GitHub Actions. *Fix:* let `setup-uv` create it (`9e9398c`).
- **Groq free-tier rate limits.** *Fix:* switch to a fallback model with its own budget and wait out short limits; send 4 passages instead of 6 to fit 8k tokens per minute.
- **Low citation coverage on gpt-oss** (60% of sentences cited). *Fix:* a short "closing reminder" after the sources repeating the two most-broken rules. Coverage went from 60% to 72% in an A/B test, because models attend most to the end of a long prompt.
- **Refusal threshold overfit** (§13). The first rule allowed 2% false refusals on calibration and looked fine there, but on the test split it **wrongly refused 8.3%** of valid questions: only about 30 calibration negatives existed, so two borderline items decided the threshold. *Fix:* zero false refusals on calibration. Result on test: 0.5%.
  - **Found along the way:** most wrongly refused questions were misspelled drug names. Generic-word blocking also blocked "sodium bicarbonate", and typo correction only compared one-word names ("isosoride dinitrate" never matched). *Fix:* word-by-word typo correction. 74 of 79 misspelled questions now link correctly.

---

## Phase 3: Publishing results (commit `7db3d04`, 19 Nov 13:36)

**Built.**
- A 9-way retrieval ablation, gate results and answer quality in the README.
- `evalkit.rescore`, which re-scores stored answers when the checker changes, without regenerating them.
- A CI regression gate on retrieval metrics.

**Problem: the number checker rejected real label formats.** Correct answers were failing because labels write numbers in ways the checker didn't expect:
- `47.8 (13.7) hours`, where the unit comes after a spread
- `µg` in the label but `mcg` in the answer
- the `%` only in a table's column header
- the number inside the drug's own name (e.g. "Clobetasol 0.05%")

*Fix:* accept each format, with a test for each, while prose stays strict ("type 2 diabetes … 500 mg" must never validate "2 mg").

**Results published:**
- Recall@5 rose from 66.1% (naive) to **90.6%** (full pipeline).
- **Biggest single lever: the contextual header (+17 points).**
- **Surprise:** naive hybrid search was *worse* than dense alone (79.7% vs 83.3%), because BM25 on identical section vocabulary adds noise. Hybrid only pays off after scoping.

---

## Phase 4: Outside adversarial audit (commits `4bf6c6e`, `d5a5025`, 19 Nov afternoon)

**What happened.** An outside reviewer wrote a test plan covering interactions, comparisons, missing information, patient-specific questions, prompt injection and ambiguous questions. I ran 16 of the questions on the hosted configuration and audited every answer against its cited passages.

**5 failures found, all fixed with tests (`tests/test_audit_fixes.py`):**

| Failure | Cause | Fix |
|---|---|---|
| A correct answer failed the grounding check | gpt-oss cites as `【1】` | Normalise lenticular and full-width brackets |
| "Compare metformin and apixaban": 3 of 4 passages were apixaban | Top-k ranked for the whole question | Deal evidence slots round-robin per drug |
| "Is metformin safe?" missed the boxed warning | The warning wasn't in the top passages | Pin the boxed warning for safety questions |
| "Can warfarin be taken with amiodarone?" began "Warfarin can be taken…" | The model gave clearance | Prompt rule: never say whether to take it; say what each label states |
| "Can I take this drug with my medication?" answered from 4 random labels | No drug named, so unscoped search | Refuse an unnamed-drug question before the LLM and show no unrelated evidence (`d5a5025`) |

---

## Phase 5: Auditing the measurement itself (commit `d9f278c`, 19 Nov 16:59)

**What the line-by-line audit found.**
- **Recall@5 was not what the user experiences.** The model sees 6 passages locally but 4 on the hosted demo. *Fix:* a new metric, "gold passage in the prompt", scored at the number of passages actually served: **91.7% local, 88.5% hosted**.
- **Result files didn't say which settings produced them.** *Fix:* every run now records its exact settings.
- **Every headline number was recomputed by an independent script** sharing no code with the harness. They matched: Recall@5 90.62%, strict 86.46%, right drug 98.96%.

---

## Phase 6: The bug hidden inside a metric, and a 2-second delay (commit `dc52272`, 19 Nov 17:59)

**The bug.** The eval said the model "declined" 8.3% of answerable questions, which I first read as over-caution. Then a user's screenshot showed a simple question refused. Reading the raw output showed the model had written a **full, correct, cited answer**, then a closing `INSUFFICIENT_EVIDENCE` line about one detail. The code treated that keyword on *any* line as a refusal and threw the whole answer away.

*Fix:* a cited answer followed by the sentinel line is a partial answer, so it's kept and checked, and the streamed preview never shows the sentinel. **Same 120 questions, same seed: correct end to end went from 88.3% to 96.7%**, model declines from 8.3% to 0%, and unanswerable questions were still refused 100% of the time.

**The 2-second delay.**
- **Cause 1:** on Windows, `localhost` tried IPv6 first, adding about 2 s to every new connection to Ollama. *Fix:* use `127.0.0.1` (2,090 ms → 45 ms).
- **Cause 2:** the embedding model and the chat model were swapping on the 8 GB GPU for every question. *Fix:* embed questions on the CPU (about 60–80 ms) so the chat model stays loaded.
- **Result:** p50 latency 5.9 s → 4.3 s.

**Also:** numbers glued to words in the label text ("younger than18 years") failed the number check. *Fix:* re-split them before checking.

---

## Phase 7: Housekeeping (19–20 Nov, no code commits)

- **Disk space.** The index, models and caches outgrew the system drive. The project (11,723 files, each verified by hash) moved to a second drive, and the editable Python install was re-pointed to the new path. Tests, git and a live answer were re-checked from there.
- **Secrets.** The Groq key lives only in the git-ignored `.streamlit/secrets.toml`; the hosted app reads it from Streamlit's secrets store.

---

## Phase 8: Real-life check against today's FDA labels (20–21 Nov)

**What was done.** 14 new, hand-written everyday questions were answered locally, and every cited passage and number was compared with the **current** label fetched from the openFDA API.

**Results.**
- No number in any answer was missing from the live label.
- 19 of 21 cited passages were still word for word in today's label. The others had been revised, so a real deployment needs scheduled re-ingests.
- **A real safety bug:** "What is the dose of omeprazole for GERD?" was answered with **pediatric weight-band doses** (5 mg for 5 to <10 kg…) with nothing saying they were pediatric, 3 runs out of 3. The adult dose was among the passages.

**Root cause: chunking, not the model.**
1. The section began `2 DOSAGE AND ADMINISTRATION *`, a heading format the subsection splitter didn't recognise, so "2.2 Recommended Pediatric Dosage" never reached a chunk header.
2. All tables of a field were packed together, so one chunk mixed the end of the adult table with the start of the pediatric one, and later chunks lost the "Pediatric Patients" caption.

**A prompt rule ("name the population") was tried and rejected.** It didn't fix omeprazole and made two other answers worse. The missing information had to go back into the evidence.

**Fix (commit `f14a6af`):**
- Recognise more heading formats.
- Chunk each table on its own, with its "Table N" caption in every chunk's header.

**An independent review of that fix found 10 more problems**, all fixed:
- captions stored in a table's first row were thrown away (azithromycin's pediatric table)
- group rows ("Ear and Labyrinth Disorders") and footnotes were taken as captions
- captions over 30 words were rejected
- tiny fragment chunks
- two tables of one label that differ only by caption (high- vs moderate-emetogenic regimen) were merged as "duplicates"
- **chunk IDs were based on position in the document, so adding chunks renamed 94 gold passages and would have silently counted them as misses.** IDs are now hashed from section and text, the golden set was re-mapped (400/400), and the eval now refuses to run if a gold ID is missing.

**Result.** Omeprazole gives "20 mg once daily for up to 4 weeks" 3 out of 3 times. Recall@5 went from 90.6% to 90.1%, a one-question change; the golden set can't reward this fix because its gold passages are never tables. The index was rebuilt (78,852 passages), and the hosted index was republished as release `index-v2`.

---

## Phase 9: An outside critique, and the strong judge (21 Nov)

**A pasted critique listed 10 weaknesses. Each was checked, not just accepted or dismissed:**

| Claim | Verdict | What was done |
|---|---|---|
| Only 1,200 labels | Valid | Documented: not tested at 37,000 labels (would need about 1.9–10 GB of vectors) |
| An 8B model judging an 8B model | **Valid and it mattered** | Re-judged everything with a much stronger blind judge: **91.7% correct end to end, not 95.8%** |
| Synthetic gold set | Valid | Documented; external benchmark added later (Phase 11) |
| The number check misses unit-free errors | Partly wrong | "2 days" vs "2 weeks" *is* caught. Reversed meaning is not, which led to Phase 10 |
| Gate tuned to zero false refusals | Fair | Documented: easy negatives, no false-premise questions |
| SQLite during re-ingest | **Valid, worse than stated** | A live server loses dense search for re-ingested labels until it reloads. README corrected |
| 14 questions is small | Valid | Labelled a spot check |
| CPU query embedding | Minor | About 80 ms; the reranker is the real cost |
| Is BM25 needed? | **Valid** | Measured: after scoping and rerank, dropping BM25 costs 0.5 points (not significant). Kept as robustness insurance and said so |

**What the strong judge revealed:** 9 of 15 "unfaithful" answers ended with a **false "the sources do not cover X" note**, created by the Phase 6 partial-answer fix. A blind A/B found 16 of 27 such notes false. *Fix:* a stricter prompt rule, and uncited "not covered" sentences are now removed, because the system can't verify a claim of absence. Faithfulness went from 87.4% to 90.8%, and citation coverage from 87.9% to 95.0%.

**The checker's third correction**, also reviewed independently: the first version of a table-"%" rule was too loose (it accepted "20%" read off a "20 mg" column header). The shipped version accepts only a real cell in the row the sentence names.

**A latency scare that wasn't the code.** p50 jumped from 4.3 s to 7.7 s. Timing the old and new index back to back showed both equally slow, so the cause was the laptop's state, not the change.

---

## Phase 10: Sentence-level claim verifier (commit `c49353d`, 21 Nov)

**Problem.** About 1 answer in 10 attaches a real fact to the wrong trial or population, for example a pooled 11-drug result described as primidone's. Every number is correct, so the number check passes it.

**Method.**
- 237 answers (889 sentences) labelled sentence by sentence: supported / wrong citation / misattributed / contradicted / unsupported.
- Split into calibration answers (for choosing) and test answers (for reporting).
- 10 verifier variants plus an LLM quote-checker compared.
- **The selection rule was written down first**, and the chosen configuration was saved with a timestamp before its test score existed.

**What was learned.**
- **The input format mattered more than model size.** The NLI model scored a verbatim sentence 0.986 on its own and 0.003 with one extra sentence attached. Single sentences, prefixed with the drug name, fixed most false alarms.
- **The LLM checker was worse:** 23% false alarms and 6.5 s per answer, because an 8B model paraphrases what it's asked to quote.

**Shipped:** a small NLI model (ONNX, CPU) plus a check that every patient group or trial a sentence names appears in its passage.
- **Test results:** catches 32% of false sentences, flags 4.3% of true ones, and **repairs 88% of wrong citations**.
- **In the UI:** flagged sentences are underlined, not deleted; a missing citation is added; badges show "N sentences checked" and "+1 citation added".
- Off on the 1 GB hosted demo.

**Honest note.** On test, another variant beat the shipped one (42% vs 32%) after losing on calibration. With 19 false sentences, that's noise, and switching would have meant tuning on the test set.

---

## Phase 11: External benchmark: FDARxBench (21 Nov)

**What.** An expert-reviewed benchmark by other researchers (17,207 questions, 685 labels, Apache-2.0).

**Problems setting it up:**
- 289 of its 685 labels are no longer served by openFDA under the same ID, and the rest may have been revised. *Fix:* convert the benchmark's **own label text** to the openFDA format and index it with the unchanged pipeline.
- The data files are stored in Git LFS (the LFS media URL returned 404). *Fix:* download from the raw URL.

**Run.** A stratified sample of 400 (200 factual, 100 multi-hop, 100 refusal). Every written answer (305) was graded by hand against the reference; the other 95 were declined answers, scored mechanically.

| | Result | 95% CI |
|---|---:|---:|
| Factual | **89.5%** | 84.5–93.0% |
| Multi-hop | **58.0%** | 48.2–67.2% |
| Refusal F1 | **0.98** | |
| Answerable questions wrongly refused | 0.3% | |

The paper reports its best models at about 0.56 / 0.47 / 0.71–0.80, but with a different grader, sample and setting, so they aren't directly comparable.

**Found:** **28 of 41 multi-hop misses answered only half of a two-part question.** 4 of the 21 factual misses have defective references (e.g. a reference that is only section numbers).

---

## Open problems (where it stands)

1. **Two-part questions.** Multi-hop accuracy is 58%. Next: split compound questions and retrieve for each part.
2. **Misattribution.** Caught about a third of the time. Next lever: a larger generator.
3. **Scale.** Untested beyond 1,200 labels.
4. **Re-ingest under a live server.** Should build into a second file and switch.
5. **Human evaluation.** Both judges are models; a pharmacist-reviewed sample is the real test.

## Lessons

1. **Read raw outputs before you interpret a rate.** The "8.3% over-caution" was a parsing bug.
2. **A metric is only as good as the harness.** Position-based chunk IDs would have silently turned 94 gold passages into misses.
3. **Fix the evidence, not the prompt.** The omeprazole prompt rule failed; the chunking fix worked.
4. **Choose on calibration, report on test, once.** The overfit threshold (0.5% → 8.3%) and the verifier's test result proved why.
5. **Quote the stricter judge.** 95.8% → 91.7% is the honest number.
6. **Every fix gets reviewed.** The table-caption fix and the checker's "%" rule each had real bugs that an independent review caught.
