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

<!-- RESULTS -->

## Engineering for production

- **One transactional store.** Documents, chunks, the BM25 index, embeddings and aliases live in one SQLite file. A label revision replaces all of them atomically, so the keyword and vector indexes can never drift apart. ([why](docs/DECISIONS.md#1-one-sqlite-file-instead-of-a-vector-database-plus-a-search-engine))
- **Incremental, idempotent ingest.** Unchanged labels are skipped by content hash. Embeddings are keyed by the hash of the embedded text, so a re-run embeds only new text. A pipeline version in the fingerprint re-indexes everything when cleaning logic changes.
- **Hot reload.** A running server picks up a re-ingest without a restart and swaps in the new index as one immutable snapshot. The response cache is keyed by index version, so it invalidates itself.
- **Streaming that never retracts.** Server-sent events stream the answer, but text that may turn out to be a refusal, and every answer in strict mode, is held back until it has been verified.
- **Observability.** Prometheus metrics with per-stage latency histograms (scope, BM25, embed, dense, rerank, first token, generation), grounding-failure counters, structured JSON logs with request ids, and `/healthz` and `/readyz`.
- **Failure handling.** Retries with backoff for model backends, a `503` with `Retry-After` when a backend is down, and empty or truncated completions flagged and never cached.
- **Swappable providers.** Embedder, reranker and LLM are three small protocols. The generator runs locally on Ollama or on Claude (`LABELLENS_LLM_PROVIDER=anthropic`) with no code change.
- **Tests that need no GPU.** 88 tests use deterministic fakes for every model, so CI runs the whole pipeline, including the HTTP API, in about a second.

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
tests/             88 tests, including regression tests for every review finding
```

## Limitations

- **Synthetic questions.** The golden set is LLM-generated from label passages. Its biases and mitigations are described in `src/labellens/evalkit/dataset.py`. Clinician-written questions would be a stronger test.
- **Paraphrase errors.** The deterministic checks can't catch a sentence that misstates its source without changing a number. That residual risk is measured offline (judge faithfulness) but not blocked at request time.
- **Corpus size.** The index holds about 1,200 of the ~37,000 prescription labels in openFDA (the most recent revision of each distinct drug, plus about 100 commonly prescribed drugs). The design targets the full set; the remaining work is ingest time.
- **Label text only.** No clinical guidelines and no patient-specific reasoning.

## License

MIT. Label data comes from [openFDA](https://open.fda.gov), which describes its data as unvalidated and not intended for clinical decisions.
