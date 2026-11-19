"""Golden-set construction.

Answerable items are synthesised: an LLM reads one sampled chunk and writes a
question that chunk answers, so the gold chunk is known by construction and
retrieval can be scored without human labelling.

Known biases, stated rather than hidden, and what is done about each:

* **Vocabulary leakage.** A question written *from* a passage shares words with
  it, which flatters lexical retrieval. The writer is told to paraphrase, is
  not shown the section title (which is part of the indexed chunk header), and
  the overlap of every question with its gold chunk (header + text) is
  recorded so results can be sliced by it.
* **Linker-friendly names.** If every question used an exact alias, entity
  scoping would be evaluated only on queries it resolves perfectly. The drug
  is therefore named in rotating styles - generic, generic without the salt,
  brand, and a deliberately misspelled name - and the link rate is reported.
* **Selection.** Questions without any drug name are dropped (in a corpus of
  1,200 labels "what is the maximum dose?" has no single right answer); the
  number dropped is written to the dataset card.

Unanswerable items are generated deterministically from templates and exist to
measure the one thing answerable items cannot: whether the system refuses.

Every item carries a ``split`` (calib / test) derived from a hash of its id.
Thresholds are tuned on calib and every published number is computed on test.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from pathlib import Path

from ..providers import LLM
from ..retrieval import EntityLinker, normalize_name, strip_salt
from ..store import STOPWORDS, Store

log = logging.getLogger(__name__)

SECTIONS = [
    "indications_and_usage", "dosage_and_administration", "contraindications",
    "warnings_and_cautions", "warnings", "boxed_warning", "drug_interactions",
    "adverse_reactions", "pregnancy", "use_in_specific_populations", "overdosage",
    "mechanism_of_action", "pharmacokinetics", "how_supplied", "storage_and_handling",
    "pediatric_use", "geriatric_use",
]

WRITER_SYSTEM = (
    "You write evaluation questions for a drug-label search system. "
    "Reply with a single JSON object and nothing else."
)
WRITER_PROMPT = """Passage from the FDA label of {name}:
\"\"\"{text}\"\"\"

Write ONE question a pharmacist or clinician would realistically ask that this passage answers.
- Refer to the drug exactly as "{name}".
- Ask about one specific fact that the passage states (a dose, a condition, a risk, an instruction).
- Use your own words: do not copy phrases from the passage; prefer synonyms and everyday clinical wording.
- Do not name the label section the passage comes from.
- At most 25 words. Do not mention "the passage" or "the label".
Then answer it in one or two sentences using only the passage.

Return JSON: {{"question": "...", "answer": "..."}}
If the passage contains no specific answerable fact, return {{"question": "", "answer": ""}}."""

FICTIONAL = [
    "Zorvanex", "Trelimab", "Quorvidol", "Brenzetide", "Maxulitran", "Voxapirene", "Dalquinox",
    "Fenbritumab", "Oprelvastin", "Sytramidol", "Kelvoxetine", "Namprizole", "Ulvetrigan",
    "Corzalimus", "Pryndavir", "Hexolurate", "Javrelisib", "Lumoctadine", "Wexaprofen", "Ybrutamide",
]
FICTIONAL_TEMPLATES = [
    "What is the recommended starting dose of {d}?",
    "What are the contraindications of {d}?",
    "Does {d} interact with warfarin?",
    "Is {d} safe to use during pregnancy?",
    "What is the boxed warning for {d}?",
    "How should {d} be stored?",
]
OUT_OF_LABEL_TEMPLATES = [
    "What is the average retail price of {d} per tablet in the United States?",
    "Which insurance plans cover {d} without prior authorization?",
    "How many prescriptions for {d} were written worldwide last year?",
    "Who was the scientist that first synthesized {d}, and in what year?",
    "What is the current stock price of the company that manufactures {d}?",
    "When does the patent on {d} expire?",
]


def split_of(item_id: str) -> str:
    """Deterministic 50/50 split that does not depend on item order."""
    return "calib" if hashlib.sha1(item_id.encode()).digest()[0] % 2 == 0 else "test"


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def lexical_overlap(question: str, passage: str) -> float:
    q = _content_words(question)
    return round(len(q & _content_words(passage)) / len(q), 3) if q else 0.0


def misspell(name: str, i: int) -> str:
    """A realistic one-character typo: drop or double a letter inside the longest word."""
    words = name.split()
    w = max(range(len(words)), key=lambda j: len(words[j]))
    word = words[w]
    pos = 2 + (i % max(1, len(word) - 4))
    words[w] = word[:pos] + word[pos + 1:] if i % 2 == 0 else word[:pos] + word[pos] + word[pos:]
    return " ".join(words)


def _name_for(display: str, i: int) -> tuple[str, str]:
    """Rotate how the drug is named so entity linking is exercised honestly."""
    m = re.match(r"^(.*?)(?: \((.*)\))?$", display)
    generic, brand = m.group(1), m.group(2)
    base = strip_salt(generic)
    style = ("generic", "generic_no_salt", "brand", "generic_no_salt", "typo")[i % 5]
    if style == "brand" and brand:
        return brand, "brand"
    if style == "generic_no_salt" and base != normalize_name(generic):
        return base, "generic_no_salt"
    if style == "typo" and max(len(w) for w in base.split()) >= 7:
        return base, "typo"  # the typo is injected after the question is written
    return generic.lower(), "generic"


def _parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, flags=re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def synthesize(store: Store, llm: LLM, n: int, seed: int = 13) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    by_doc: dict[str, list] = {}
    for rowid, chunk, drug in store.iter_chunks():
        if chunk.kind == "text" and chunk.section_key in SECTIONS and 60 <= chunk.n_words <= 220:
            by_doc.setdefault(chunk.doc_id, []).append((rowid, chunk, drug))
    docs = sorted(by_doc)
    rng.shuffle(docs)

    items: list[dict] = []
    card = {"labels_sampled": 0, "writer_declined": 0, "dropped_no_drug_name": 0,
            "dropped_too_long": 0}
    for i, doc_id in enumerate(docs):
        if len(items) >= n:
            break
        card["labels_sampled"] += 1
        _, chunk, drug = rng.choice(by_doc[doc_id])  # one question per label: breadth over depth
        name, style = _name_for(drug, i)
        out = _parse(llm.generate(WRITER_SYSTEM, WRITER_PROMPT.format(name=name, text=chunk.text),
                                  max_tokens=220))
        q, a = str(out.get("question", "")).strip(), str(out.get("answer", "")).strip()
        if not q or not a:
            card["writer_declined"] += 1
            continue
        if name.lower() not in q.lower():
            card["dropped_no_drug_name"] += 1
            continue
        if len(q.split()) > 32:
            card["dropped_too_long"] += 1
            continue
        if style == "typo":
            q = re.sub(re.escape(name), misspell(name, i), q, flags=re.I)
        item_id = f"a{len(items):04d}"
        items.append({
            "id": item_id, "split": split_of(item_id), "type": "answerable", "question": q,
            "answer": a, "doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id,
            "section_key": chunk.section_key, "drug": drug, "name_style": style,
            "lexical_overlap": lexical_overlap(q, chunk.embed_text),
        })
        if len(items) % 25 == 0:
            log.info("synthesised %d/%d questions", len(items), n)
    return items, card


def unanswerable(store: Store, seed: int = 13, per_kind: int = 40) -> list[dict]:
    rng = random.Random(seed)
    linker = EntityLinker(store.aliases(), store.term_exists)
    items: list[dict] = []

    def add(kind: str, q: str) -> None:
        item_id = f"u{len(items):04d}"
        items.append({"id": item_id, "split": split_of(item_id), "type": kind, "question": q})

    fictional = [d for d in FICTIONAL if not linker.link(d)[0]]  # must not alias a real drug
    for i in range(per_kind):
        d = fictional[i % len(fictional)]
        add("unknown_drug",
            FICTIONAL_TEMPLATES[(i // len(fictional) + i) % len(FICTIONAL_TEMPLATES)].format(d=d))

    # Real, indexed drugs + questions labels never answer. These cannot be caught by
    # the unknown-term gate; they test the reranker threshold and the model itself.
    names = sorted({strip_salt(n.split(" (")[0]) for n in store.doc_names().values()
                    if 5 <= len(n.split(" (")[0]) <= 24 and " and " not in n.lower()})
    rng.shuffle(names)
    for i in range(per_kind if names else 0):
        add("out_of_label", OUT_OF_LABEL_TEMPLATES[i % len(OUT_OF_LABEL_TEMPLATES)].format(
            d=names[i % len(names)]))
    return items


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
