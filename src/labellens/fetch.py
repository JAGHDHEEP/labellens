"""openFDA drug-label client.

Builds a raw corpus file (gzipped JSONL) of prescription drug labels.
The corpus is deliberately *de-duplicated by generic name*: openFDA holds many
near-identical labels per molecule (one per manufacturer), and indexing all of
them would flood every result list with duplicates of the same paragraph.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://api.fda.gov/drug/label.json"
RX_FILTER = 'openfda.product_type:"HUMAN PRESCRIPTION DRUG"'
MIN_LABEL_CHARS = 4000  # skip stub labels (kits, diluents, placeholders)

# Widely prescribed molecules fetched by name so the demo corpus always
# contains the drugs people actually ask about, independent of recency.
SEED_DRUGS = [
    "metformin", "atorvastatin", "lisinopril", "amlodipine", "metoprolol", "omeprazole",
    "losartan", "simvastatin", "levothyroxine", "gabapentin", "sertraline", "warfarin",
    "apixaban", "rivaroxaban", "clopidogrel", "furosemide", "hydrochlorothiazide",
    "prednisone", "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
    "fluoxetine", "escitalopram", "bupropion", "trazodone", "alprazolam", "lorazepam",
    "zolpidem", "tramadol", "oxycodone", "morphine", "ibuprofen", "naproxen", "celecoxib",
    "insulin glargine", "semaglutide", "liraglutide", "empagliflozin", "sitagliptin",
    "glipizide", "pioglitazone", "rosuvastatin", "pravastatin", "ezetimibe", "carvedilol",
    "spironolactone", "digoxin", "amiodarone", "diltiazem", "valsartan", "montelukast",
    "albuterol", "fluticasone", "budesonide", "tiotropium", "pantoprazole", "famotidine",
    "ondansetron", "methotrexate", "adalimumab", "allopurinol", "colchicine", "lamotrigine",
    "levetiracetam", "topiramate", "valproic acid", "carbamazepine", "phenytoin",
    "lithium", "quetiapine", "olanzapine", "aripiprazole", "risperidone", "clozapine",
    "donepezil", "sumatriptan", "tamsulosin", "finasteride", "sildenafil", "isotretinoin",
    "acyclovir", "valacyclovir", "oseltamivir", "fluconazole", "nitrofurantoin",
    "cephalexin", "vancomycin", "tacrolimus", "cyclosporine", "tamoxifen", "letrozole",
    "methylphenidate", "varenicline", "naloxone", "buprenorphine", "ketorolac",
]


def _get(client: httpx.Client, search: str, limit: int, skip: int = 0) -> dict:
    # openFDA wants '+' between boolean terms; build the query by hand so the
    # encoding is exactly what the API documents.
    encoded = quote(search, safe=":()[]").replace("%20", "+")
    url = f"{BASE_URL}?search={encoded}&sort=effective_time:desc&limit={limit}&skip={skip}"
    for attempt in range(5):
        try:
            r = client.get(url)
            if r.status_code == 404:  # openFDA returns 404 for "no matches"
                return {"results": []}
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if attempt == 4:
                raise
            wait = 2.0**attempt
            log.warning("openFDA request failed (%s); retry in %.0fs", exc, wait)
            time.sleep(wait)
    return {"results": []}


def _slim(rec: dict) -> dict:
    """Keep only label text fields and the identifiers we use."""
    out = {k: v for k, v in rec.items() if isinstance(v, list) and v and isinstance(v[0], str)}
    out["set_id"] = rec.get("set_id")
    out["version"] = rec.get("version")
    out["effective_time"] = rec.get("effective_time")
    fda = rec.get("openfda", {})
    out["openfda"] = {
        k: fda.get(k, [])
        for k in ("generic_name", "brand_name", "manufacturer_name", "route",
                  "substance_name", "pharm_class_epc")
    }
    return out


def _usable(rec: dict) -> bool:
    fda = rec.get("openfda", {})
    if not rec.get("set_id") or not fda.get("generic_name"):
        return False
    if "indications_and_usage" not in rec:
        return False
    size = sum(len(x) for v in rec.values() if isinstance(v, list) for x in v if isinstance(x, str))
    return size >= MIN_LABEL_CHARS


def _key(rec: dict) -> str:
    return " ".join(rec["openfda"]["generic_name"][0].lower().split())


def iter_recent(client: httpx.Client, pages: int, page_size: int = 100) -> Iterator[dict]:
    search = f"{RX_FILTER} AND _exists_:openfda.generic_name"
    for page in range(pages):
        data = _get(client, search, page_size, skip=page * page_size)
        results = data.get("results", [])
        if not results:
            return
        yield from results


def fetch_seed(client: httpx.Client, name: str) -> dict | None:
    data = _get(client, f'{RX_FILTER} AND openfda.generic_name:"{name}"', limit=10)
    candidates = [r for r in data.get("results", []) if _usable(r)]
    # Prefer the single-ingredient product over combination products.
    candidates.sort(key=lambda r: len(r["openfda"].get("substance_name", [])) or 9)
    return candidates[0] if candidates else None


def build_corpus(out_path: Path, target_docs: int = 1200, pages: int = 60,
                 seeds: list[str] | None = None) -> int:
    """Download labels and write a de-duplicated raw corpus. Returns doc count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[str, dict] = {}
    with httpx.Client(timeout=120, headers={"User-Agent": "labellens/0.1"}) as client:
        for name in seeds if seeds is not None else SEED_DRUGS:
            rec = fetch_seed(client, name)
            if rec is None:
                log.warning("seed drug not found: %s", name)
                continue
            seen.setdefault(_key(rec), _slim(rec))
        log.info("seed drugs fetched: %d", len(seen))
        for i, rec in enumerate(iter_recent(client, pages)):
            if len(seen) >= target_docs:
                break
            if _usable(rec):
                # Results are sorted newest-first, so first-seen = latest revision.
                seen.setdefault(_key(rec), _slim(rec))
            if i % 500 == 499:
                log.info("scanned %d labels, kept %d unique drugs", i + 1, len(seen))
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for rec in seen.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(seen)


def read_corpus(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
