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
| **Structure-aware chunking** | Splits on the label's own sections and numbered subsections. Prepends `Drug (Brand) \| Section \| Subsection` to every chunk. | A chunk saying "contraindicated in severe renal impairment" is useless without the drug name attached. |
| **Entity scoping** | Links drug names (generic, brand, salt-less, biosimilar, typo-tolerant) to an alias table built at ingest, then hard-filters both retrieval arms. | Fixes "right section, wrong drug" with a microsecond dictionary lookup instead of an LLM call. |
| **Hybrid retrieval + RRF** | BM25 top-50 and exact dense top-50, fused by rank. | BM25 catches `CYP2C9` and `eGFR below 30`; dense catches paraphrase. Rank fusion needs no score tuning. |
| **Cross-encoder rerank** | Re-scores the fused top-30 with an ONNX `ms-marco-MiniLM` model. | Retrieve wide, rerank narrow. No PyTorch in the serving image. |
| **Near-duplicate collapse** | Keeps the best copy of passages with ≥ 0.85 Jaccard overlap, and records what it suppressed. | Six distinct pieces of evidence, not one paragraph six times. |
| **Abstention gates** | Refuses before generation on (a) a word found nowhere in the corpus or (b) a reranker score below a threshold calibrated on held-out data. The model can also refuse with `INSUFFICIENT_EVIDENCE`. | A refusal is the cheapest path, not the most expensive one. |
| **Grounding checks** | Citations must point at supplied sources. Every number must appear, with its unit, in the passage it cites. | Catches a wrong dose deterministically, with no second LLM call. |

## Results

All numbers come from the **held-out test split** of a 480-question golden set (400 answerable, 80 unanswerable), over 1,200 labels / 73,470 passages. Thresholds were tuned on a separate calibration split. Raw outputs are in [`eval/results/`](eval/results), and the method and its known biases are described in [`evalkit/dataset.py`](src/labellens/evalkit/dataset.py).

### Retrieval: what each component is worth

192 answerable test questions. *Recall@5* = the passage the question was written from is in the top 5.

| Configuration | Hit@1 | Recall@5 | MRR@10 | Right drug in top 5 | p95 latency |
|---|---:|---:|---:|---:|---:|
| Naive: dense only, plain chunks | 43.2% | 66.1% | 0.533 | 90.1% | 50 ms |
| BM25 only, plain chunks | 27.1% | 56.2% | 0.392 | 85.9% | 80 ms |
| Dense + contextual header | 53.6% | 83.3% | 0.663 | 97.9% | 62 ms |
| Hybrid (BM25 + dense, RRF) | 50.5% | 79.7% | 0.633 | 94.3% | 115 ms |
| Hybrid + cross-encoder rerank | 53.1% | 87.0% | 0.667 | 98.4% | 1.5 s |
| Hybrid + entity scoping | 56.8% | 89.6% | 0.699 | 98.4% | 179 ms |
| **Full: hybrid + scoping + rerank** | **59.4%** | **90.6%** | **0.720** | **99.0%** | 1.6 s |
| Full, without contextual header | 53.1% | 81.2% | 0.651 | 99.0% | 1.5 s |

Full pipeline Recall@5: 95% bootstrap CI **86.5–94.8%**. Strict (exact-chunk) Recall@5: 86.5%. The credited figure also counts another manufacturer's identical copy of the same paragraph.

**What the ablation shows**

- **The contextual header is the largest single gain:** +17.2 points for dense retrieval, and +9.4 points inside the full pipeline. Without it, "contraindicated in severe renal impairment" could be about any of 1,200 drugs.
- **Naive hybrid search hurt** (79.7% vs 83.3% for dense alone). Every label shares the same section vocabulary, so BM25 alone is weak (56.2%) and drags the fusion down. Hybrid pays off only once entity scoping removes the other labels.
- **Entity scoping is the cheapest lever:** 89.6% with no reranker, at 179 ms instead of 1.5 s.
- **Once scoped, the embedding model barely matters.** A label has a median of 53 passages, so BM25-50 plus dense-50 already cover nearly all of it, and the reranker decides. The hosted demo's 33M-parameter CPU model scores the same Recall@5 (90.6%) as `bge-m3` (568M) on GPU.
- **Bias check:** Recall@5 is 85.4% on questions with below-median word overlap with their source, and 95.2% above it. Misspelled drug names: 95.3% (n = 43).

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

The local configuration answered 154 test questions (120 answerable, 34 unanswerable): generator `llama3.1:8b`, judge `qwen3:8b`, a different model so nothing grades its own work.

| Answerable questions (n = 120) | |
|---|---:|
| **Correct end to end** (answered, and judged correct) | **88.3%** |
| Gold passage among the 6 the model saw | 94.2% |
| Model chose `INSUFFICIENT_EVIDENCE` | 8.3% |

| Of the 109 answers given | |
|---|---:|
| Judged correct against the reference | 97.3% |
| Judged faithful: every claim supported by the passages | 99.1% |
| **Deterministic grounding checks passed** | **97.2%** |
| Sentences carrying a citation | 90.3% |
| Citations pointing at a passage that wasn't supplied | 0% |
| Cut off by the token limit | 0% |

| Unanswerable questions (n = 34) | |
|---|---:|
| **Refused** | **100%** (94.1% before the LLM, 5.9% by the LLM) |

Latency with the local GPU model: p50 5.9 s / p95 10.3 s end to end, and 3.0 s to the first streamed token. The hosted Groq model generates in under 1 s.

**Reading these numbers honestly**

- The main failure mode is **over-caution, not hallucination.** The model declined 8.3% of answerable questions, against 0.9% of answers judged unfaithful.
- **3 answers were flagged by the number check.** In 2, the model cited the wrong passage for a correct number. The third is a known gap in the checker: in a long flattened table, the "%" is too far from its number.
- A first version of the checker over-flagged (6.4% of answers) on real label formats: `47.8 (13.7) hours`, `µg` vs `mcg`, and the `%` in a table header. The fix is pinned by tests. Stored answers were re-scored deterministically (`python -m labellens.evalkit.rescore`) without regenerating them, and the before and after figures are kept in `generation.json`.
- An 8B model is a lenient judge. Treat judge scores as relative signals for comparing versions, not absolute truth.

### Outside spot check: 16 adversarial questions

An outside reviewer wrote a test plan covering interactions, multi-drug comparisons, missing information, patient-specific questions, prompt injection and ambiguous questions. I ran 16 of its questions on the hosted configuration and audited every answer against its cited passages ([`eval/results/spotcheck.json`](eval/results/spotcheck.json)). Five failures turned up, and each fix is pinned by a test in `tests/test_audit_fixes.py`:

| Failure found | Fix |
|---|---|
| `gpt-oss` cited as `【1】`, so a correct answer failed the grounding check | Lenticular and full-width brackets are normalised before checking and before display |
| "Compare metformin and apixaban": 3 of 4 passages were apixaban | When a question names several drugs, evidence slots are dealt round-robin per drug |
| "Is metformin safe?" answered with pediatric and pregnancy notes and missed the boxed warning | For safety questions, the drug's boxed warning is pinned into the evidence |
| "Can warfarin be taken with amiodarone?" began "Warfarin can be taken…", from one label only | The prompt forbids clearance statements and asks what each label says |
| "Can I take this drug with my other medication?" was refused in one run but answered from four unrelated labels in another | A question that points at a drug without naming one is refused before the LLM, deterministically |

After the fixes: 6 of 16 questions correctly refused (cost, probability, patient-specific dose, fictional drug, prompt injection, unnamed drug); 8 correct; 1 partially correct (the metformin/apixaban comparison still omits metformin's eGFR 30–45 rule); and the manufacturer-comparison question refused because that feature isn't supported. All 9 answered questions passed the grounding checks. Retrieval Recall@5 on the test split is unchanged at 90.6%.

## Engineering for production

- **One transactional store.** Documents, chunks, the BM25 index, embeddings and aliases live in one SQLite file. A label revision replaces all of them atomically, so the keyword and vector indexes can never drift apart. ([why](docs/DECISIONS.md#1-one-sqlite-file-instead-of-a-vector-database-plus-a-search-engine))
- **Incremental, idempotent ingest.** Unchanged labels are skipped by content hash. Embeddings are keyed by the hash of the embedded text, so a re-run embeds only new text. A pipeline version in the fingerprint re-indexes everything when cleaning logic changes.
- **Hot reload.** A running server picks up a re-ingest without a restart and swaps in the new index as one immutable snapshot. The response cache is keyed by index version, so it invalidates itself.
- **Streaming that never retracts.** Server-sent events stream the answer, but text that may turn out to be a refusal, and every answer in strict mode, is held back until it has been verified.
- **Observability.** Prometheus metrics with per-stage latency histograms (scope, BM25, embed, dense, rerank, first token, generation), grounding-failure counters, structured JSON logs with request ids, and `/healthz` and `/readyz`.
- **Failure handling.** Retries with backoff for model backends, a `503` with `Retry-After` when a backend is down, and empty or truncated completions flagged and never cached.
- **Swappable providers.** Embedder, reranker and LLM are three small protocols. The generator runs locally on Ollama or on Claude (`LABELLENS_LLM_PROVIDER=anthropic`) with no code change.
- **Tests that need no GPU.** 105 tests use deterministic fakes for every model, so CI runs the whole pipeline, including the HTTP API, in about a second.

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

The free hosted version runs the same pipeline (entity scoping, hybrid retrieval, reranking, gates and grounding checks) on a CPU-only host with about 1 GB of memory. Only the model providers are swapped:

| | Full (local) | Hosted (free) |
|---|---|---|
| Passage embeddings | `bge-m3`, 1024-d, GPU | `snowflake-arctic-embed-s`, 384-d. The index is embedded on GPU with Ollama and queried on CPU with ONNX; same weights, verified identical vectors. |
| Index | 713 MB, float32 | Compact export with float16 vectors, downloaded on first start |
| Answer model | `llama3.1:8b` on Ollama | `gpt-oss-20b` on Groq, falling back to `gpt-oss-120b` on rate limits |
| Passages in prompt | 6 | 4 (Groq's free tier allows 8k tokens per minute per model) |
| Protection | none | 5 questions per minute per session, short rate-limit waits instead of errors |
| **Retrieval Recall@5 (test split)** | **90.6%** | **90.6%** (Hit@1 59.4%, p95 1.8 s on CPU) |

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
  service.py       gates, streaming, cache, hot reload
  api.py           FastAPI app;  static/index.html  web UI
streamlit_app.py   free hosted demo (CPU embeddings, Groq answers)
  evalkit/         golden-set synthesis, ablation, gate calibration, LLM judge
docs/DECISIONS.md  every design decision, the constraint behind it, what was rejected
tests/             105 tests, including regression tests for every review and audit finding
```

## Limitations

- **Synthetic questions.** The golden set is LLM-generated from label passages. Its biases and mitigations are described in `src/labellens/evalkit/dataset.py`. Clinician-written questions would be a stronger test.
- **Ranking, not just recall.** The right passage is in the top 5 for 90.6% of questions but ranked first for only 59.4%. The model sees all six passages, so this costs little in answers, but a section-aware query router would raise it.
- **Unlinked drug names.** When no drug name in the question can be linked (3 of 192 test questions, mostly 5-ingredient combination products), retrieval runs unscoped and misses.
- **Paraphrase errors.** The deterministic checks can't catch a sentence that misstates its source without changing a number. That residual risk is measured offline (judge faithfulness) but not blocked at request time.
- **Corpus size.** The index holds about 1,200 of the ~37,000 prescription labels in openFDA (the most recent revision of each distinct drug, plus about 100 commonly prescribed drugs). The design targets the full set; the remaining work is ingest time.
- **Label text only.** No clinical guidelines and no patient-specific reasoning. LabelLens reports what the labels say. It supports a professional's own judgment and doesn't replace a pharmacist or a clinical decision-support system such as Lexicomp or Micromedex. A label that doesn't mention an interaction is not evidence that none exists.
- **One label revision per drug.** The index keeps the latest revision of each distinct label, so it can't compare manufacturers' versions or show what changed between revisions. Sources link to the exact label on DailyMed by set id.
- **Attribution in multi-drug answers.** Every fact cites a passage, and the UI shows which drug's label each passage comes from. But a sentence like "the warfarin label says…" can still name the wrong label while citing the right passage. The checks don't catch that.

## License

MIT. Label data comes from [openFDA](https://open.fda.gov), which describes its data as unvalidated and not intended for clinical decisions.
