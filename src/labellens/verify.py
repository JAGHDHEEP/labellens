"""Sentence-level claim verification.

The grounding checks in ``generation.py`` verify that citations exist and that
every number appears in the cited passage. They cannot see a sentence that
attaches a real fact to the wrong population, trial or drug, or reverses it.
This module checks each answer sentence ("claim") against the passages:

* **NLIVerifier** - an ONNX natural-language-inference cross-encoder
  (``cross-encoder/nli-deberta-v3-*``) scores whether a window of a passage
  entails or contradicts the claim. No PyTorch; runs on the CPU.
* **qualifier_gaps** - deterministic: a population, species or trial name the
  claim states must appear in the passage that supports it.
* **QuoteVerifier** - an LLM must quote the passage sentence that supports
  each claim; the quote is then checked to exist in that passage, so the
  verifier cannot invent its own evidence.

Every verifier returns one ``ClaimVerdict`` per claim:

* ``supported``     - a cited passage supports it
* ``recited``       - no cited passage supports it, another supplied one does
                      (the citation can be repaired)
* ``contradicted``  - a cited passage states the opposite
* ``unsupported``   - nothing supplied supports it

Which verifier ships, and with what thresholds, is decided on the calibration
split and reported once on the test split (``evalkit/claims.py``).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .chunking import split_sentences

_CITE_MARK = re.compile(r"\s*\[\d{1,2}\]")


@dataclass
class ClaimVerdict:
    verdict: str  # supported | recited | contradicted | unsupported
    passage: int | None = None  # the passage that supports (or contradicts) it
    entail: float = 0.0
    contra: float = 0.0
    evidence: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return self.verdict in ("supported", "recited")


def claim_text(text: str) -> str:
    """The claim without its citation markers, bullet or "Note:" prefix."""
    text = _CITE_MARK.sub("", text)
    text = re.sub(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


WINDOW_MODES = ("s1", "s1+2", "drug+s1")


def windows(passage: str, mode: str = "s1", drug: str = "", max_words: int = 120) -> list[str]:
    """Premises for the NLI model: single sentences (table rows for tables).

    The NLI models are trained on sentence pairs and are easily thrown by a
    longer premise: a verbatim sentence scored 0.986 entailment alone and
    0.003 with its neighbour appended. ``s1+2`` adds two-sentence windows for
    claims that span sentences; ``drug+s1`` prefixes the drug name, since a
    passage often says "it" or the brand while the claim names the generic.
    Which mode ships is decided on the calibration split."""
    units: list[str] = []
    for line in passage.split("\n"):
        line = line.strip()
        if not line:
            continue
        units += [line] if " | " in line else split_sentences(line)
    if not units:
        units = [passage[:1500]]
    out = list(units)
    if mode == "s1+2":
        out += [" ".join(units[i: i + 2]) for i in range(len(units) - 1)]
    if mode == "drug+s1" and drug:
        out = [f"{drug}: {u}" for u in units]
    return list(dict.fromkeys(" ".join(u.split()[:max_words]) for u in out))


# --------------------------------------------------------------------- NLI
class NLIModel:
    """ONNX NLI cross-encoder. Labels from the model config (for the
    cross-encoder/nli-deberta-v3 family: contradiction, entailment, neutral)."""

    def __init__(self, repo: str = "cross-encoder/nli-deberta-v3-small",
                 cache_dir: str | Path = "data/models", max_len: int = 320, threads: int = 4):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.repo = repo
        files = {f: hf_hub_download(repo, f, cache_dir=str(cache_dir))
                 for f in ("onnx/model.onnx", "tokenizer.json", "config.json")}
        cfg = json.loads(Path(files["config.json"]).read_text(encoding="utf-8"))
        labels = {int(k): v.lower() for k, v in cfg["id2label"].items()}
        self._ent = next(k for k, v in labels.items() if v.startswith("entail"))
        self._con = next(k for k, v in labels.items() if v.startswith("contra"))
        self.tok = Tokenizer.from_file(files["tokenizer.json"])
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(files["onnx/model.onnx"], opts,
                                         providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self.sess.get_inputs()}

    def scores(self, premises: list[str], hypothesis: str, batch: int = 16) -> np.ndarray:
        """-> array (n, 2): P(entailment), P(contradiction) for each premise."""
        out = []
        for i in range(0, len(premises), batch):
            enc = self.tok.encode_batch([(p, hypothesis) for p in premises[i: i + batch]])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            logits = self.sess.run(None, feed)[0]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            out.append(np.stack([p[:, self._ent], p[:, self._con]], axis=1))
        return np.concatenate(out) if out else np.zeros((0, 2))


class NLIVerifier:
    def __init__(self, model: NLIModel, entail_t: float = 0.5, contra_t: float = 0.9,
                 mode: str = "s1"):
        self.model, self.entail_t, self.contra_t, self.mode = model, entail_t, contra_t, mode
        self._cache: dict[tuple[str, str], tuple[float, float, str]] = {}
        self.drugs: dict[int, str] = {}  # passage number -> drug, for "drug+s1"

    def _best(self, claim: str, passage: str, drug: str = "") -> tuple[float, float, str]:
        key = (claim, passage)
        if key not in self._cache:
            wins = windows(passage, self.mode, drug)
            s = self.model.scores(wins, claim)
            i = int(s[:, 0].argmax())
            self._cache[key] = (float(s[i, 0]), float(s[:, 1].max()), wins[i])
        return self._cache[key]

    def check(self, claim: str, cites: list[int], passages: dict[int, str]) -> ClaimVerdict:
        hyp = claim_text(claim)
        cited = [n for n in cites if n in passages]
        best: tuple[float, float, str, int | None] = (0.0, 0.0, "", None)
        for n in cited:
            e, c, w = self._best(hyp, passages[n], self.drugs.get(n, ""))
            if e > best[0]:
                best = (e, max(c, best[1]), w, n)
            else:
                best = (best[0], max(c, best[1]), best[2], best[3])
        if best[0] >= self.entail_t:
            return ClaimVerdict("supported", best[3], best[0], best[1], best[2])
        # Not supported by what it cites: does any other supplied passage support it?
        for n in sorted(set(passages) - set(cited)):
            e, c, w = self._best(hyp, passages[n], self.drugs.get(n, ""))
            if e >= self.entail_t:
                return ClaimVerdict("recited", n, e, best[1], w)
        if cited and best[1] >= self.contra_t:
            return ClaimVerdict("contradicted", best[3], best[0], best[1], best[2])
        return ClaimVerdict("unsupported", best[3], best[0], best[1], best[2])


# --------------------------------------------------------- qualifier check
_POPULATION = re.compile(
    r"\b(pediatric|paediatric|child(?:ren)?|infant(?:s|ile)?|neonat\w*|newborn\w*|adolescent\w*|"
    r"adults?|elderly|geriatric|pregnan\w+|lactat\w+|breastfe\w+|renal|hepatic|dialysis|"
    r"(?:mice|mouse|rats?|rabbits?|monkeys?|dogs?)|"
    r"later-onset|infantile-onset|men|women|males?|females?)\b", re.I)
# A trial or study name: ARISTOTLE, AURA3, SHIFT, CheckMate 067, Study 1.
_TRIAL = re.compile(r"\b(?:[A-Z][A-Z0-9-]{3,}\d*|Study\s+\d+[A-Za-z]?|CheckMate[- ]\d+)\b")
_NOT_TRIAL = frozenset({"FDA", "INR", "ALT", "AST", "ULN", "HIV", "COPD", "GERD", "CYP3A4", "CYP2D6",
                        "CYP2C9", "CYP2C19", "MRHD", "AUC", "NSAID", "NSAIDS", "MAOI", "MAOIS",
                        "SSRI", "SSRIS", "QTC", "ECG", "EKG", "BMI", "TSH", "WBC", "DRESS", "LVOTO"})


def qualifier_gaps(claim: str, passage: str, drug: str = "") -> list[str]:
    """Qualifiers the claim states that the supporting passage never mentions.

    "...in pediatric patients" supported by a passage that says only "adults"
    is a misattribution even when every number matches."""
    hay = f"{drug} {passage}".lower()
    gaps = []
    for m in _POPULATION.finditer(claim):
        w = m.group(1).lower()
        stem = w[:5] if len(w) > 5 else w.rstrip("s")
        if stem not in hay:
            gaps.append(w)
    for m in _TRIAL.finditer(claim):
        t = m.group(0)
        if t.upper() in _NOT_TRIAL or t.lower() in drug.lower():
            continue
        if t.lower() not in hay:
            gaps.append(t)
    return sorted(set(gaps))


# ------------------------------------------------------- LLM quote checker
QUOTE_SYSTEM = """You check an answer about an FDA drug label against numbered passages, one claim at a time.
For each claim:
1. Find the passage sentence that states it. Copy that sentence EXACTLY, word for word, as "quote", and give its passage number as "passage". If no passage states it, use "" and 0.
2. "same_scope": true only if the quoted sentence is about the same drug, population (adult, pediatric, age, weight, renal, pregnant...), trial or study, dose and condition as the claim.
3. "verdict": "supported" if the quote states the claim with the same scope; "contradicted" if a passage states the opposite or a different value; otherwise "unsupported".
Use only the passages. Reply with JSON: {"claims": [{"key": ..., "quote": ..., "passage": ..., "same_scope": ..., "verdict": ...}]}"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class QuoteVerifier:
    def __init__(self, llm, max_tokens: int = 900):
        self.llm, self.max_tokens = llm, max_tokens

    def check_answer(self, claims: list[dict], passages: dict[int, str]) -> dict[int, ClaimVerdict]:
        src = "\n\n".join(f"[{n}] {t}" for n, t in sorted(passages.items()))
        items = "\n".join(f'{{"key": {c["i"]}, "claim": {json.dumps(claim_text(c["text"]))}, '
                          f'"cites": {c["cites"]}}}' for c in claims)
        raw = self.llm.generate(QUOTE_SYSTEM, f"<passages>\n{src}\n</passages>\n\nClaims:\n{items}",
                                max_tokens=self.max_tokens)
        try:
            got = {int(x["key"]): x for x in json.loads(raw[raw.find("{"): raw.rfind("}") + 1])["claims"]}
        except (ValueError, KeyError, TypeError):
            got = {}
        out: dict[int, ClaimVerdict] = {}
        for c in claims:
            x = got.get(c["i"])
            if not x:
                out[c["i"]] = ClaimVerdict("unsupported", notes=["verifier gave no verdict"])
                continue
            n = int(x.get("passage") or 0)
            quote = str(x.get("quote") or "")
            # The quote must really be in that passage: the verifier may not
            # invent its own evidence.
            real = bool(quote) and n in passages and _norm(quote)[:120] in _norm(passages[n])
            verdict = str(x.get("verdict", "unsupported"))
            if verdict == "supported" and not (real and x.get("same_scope")):
                verdict = "unsupported"
            if verdict == "supported" and n not in c["cites"]:
                verdict = "recited"
            out[c["i"]] = ClaimVerdict(verdict, n or None, evidence=quote,
                                       notes=[] if real or not quote else ["quote not found in passage"])
        return out


# ------------------------------------------------------- answer-level check
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(*•-])|\n+")
_BULLET_MARK = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")
_CITE_ID = re.compile(r"\[(\d{1,2})\]")


def split_claims(answer: str) -> list[dict]:
    """Answer -> [{"i", "text", "cites"}]. Citation markers stay in ``text``;
    fragments under four words ("Dosing:") are not claims."""
    out = []
    for raw in _SENT_SPLIT.split(answer):
        text = _BULLET_MARK.sub("", raw).strip()
        if len(_CITE_MARK.sub("", text).split()) < 4:
            continue
        out.append({"i": len(out), "text": text,
                    "cites": sorted({int(x) for x in _CITE_ID.findall(text)})})
    return out


def _add_cite(sentence: str, n: int) -> str:
    """Append [n] to a sentence's citations, or before its final punctuation."""
    marks = list(_CITE_ID.finditer(sentence))
    if marks:
        end = marks[-1].end()
        return f"{sentence[:end]}[{n}]{sentence[end:]}"
    m = re.search(r"[.!?]\s*$", sentence)
    return f"{sentence[:m.start()]} [{n}]{sentence[m.start():]}" if m else f"{sentence} [{n}]"


@dataclass
class Verification:
    text: str
    claims: list[dict]
    flagged: int = 0
    citations_added: int = 0

    def to_dict(self) -> dict:
        return {"claims": self.claims, "flagged": self.flagged,
                "citations_added": self.citations_added}


def verify_answer(text: str, passages: dict[int, tuple[str, str]],
                  verifier: NLIVerifier) -> Verification:
    """Check every claim of an answer against its passages ({n: (text, drug)}).

    A claim supported only by a passage it does not cite gets that citation
    *added* (the original stays: if it was right, nothing is lost). A claim no
    passage supports, or whose supporting passage never mentions a population
    or trial the claim names, is reported as not verified; it is not removed.
    On the held-out test split this caught 32% of false sentences with 4.3% of
    true sentences flagged, and repaired 88% of wrong citations
    (eval/claims/selection.json)."""
    verifier.drugs = {n: d for n, (_, d) in passages.items()}
    texts = {n: t for n, (t, _) in passages.items()}
    out, claims, flagged, added = text, [], 0, 0
    for c in split_claims(text):
        v = verifier.check(c["text"], c["cites"], texts)
        if v.ok and v.passage is not None:
            gaps = qualifier_gaps(c["text"], texts[v.passage], verifier.drugs.get(v.passage, ""))
            if gaps:
                v = ClaimVerdict("unsupported", v.passage, v.entail, v.contra, v.evidence,
                                 [f"not in the supporting passage: {', '.join(gaps)}"])
        new = c["text"]
        if v.verdict == "recited" and v.passage is not None and new in out:
            new = _add_cite(c["text"], v.passage)
            out = out.replace(c["text"], new, 1)
            added += 1
        flagged += not v.ok
        claims.append({"text": new, "verdict": v.verdict, "passage": v.passage,
                       "entail": round(v.entail, 3), "notes": v.notes})
    return Verification(out, claims, flagged, added)


def build_verifier(settings) -> NLIVerifier | None:
    if not settings.verify_claims:
        return None
    return NLIVerifier(NLIModel(settings.verify_model, cache_dir=settings.data_dir / "models"),
                       entail_t=settings.verify_entail_t, mode=settings.verify_mode)
