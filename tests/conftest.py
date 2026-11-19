"""Shared fixtures: a tiny synthetic corpus and fully offline providers."""

from __future__ import annotations

import pytest

from labellens.config import Settings
from labellens.ingest import embed_missing, ingest_records
from labellens.providers import FakeLLM, HashEmbedder, OverlapReranker
from labellens.retrieval import Retriever
from labellens.service import RagService
from labellens.store import Store


def make_label(set_id: str, generic: str, brand: str, **sections: str) -> dict:
    rec = {
        "set_id": set_id, "version": "1", "effective_time": "20260101",
        "openfda": {"generic_name": [generic.upper()], "brand_name": [brand],
                    "manufacturer_name": ["Acme Pharma"], "route": ["ORAL"],
                    "substance_name": [generic.upper()], "pharm_class_epc": []},
    }
    for key, text in sections.items():
        rec[key] = [text]
    return rec


METFORMIN = make_label(
    "set-metformin", "metformin hydrochloride", "Glucophage",
    indications_and_usage=(
        "1 INDICATIONS AND USAGE Metformin hydrochloride tablets are indicated as an adjunct to "
        "diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus."),
    contraindications=(
        "4 CONTRAINDICATIONS Metformin is contraindicated in patients with severe renal "
        "impairment (eGFR below 30 mL/min/1.73 m2). It is also contraindicated in patients with "
        "acute or chronic metabolic acidosis, including diabetic ketoacidosis."),
    dosage_and_administration=(
        "2 DOSAGE AND ADMINISTRATION 2.1 Adult Dosage The recommended starting dose is 500 mg "
        "orally twice a day with meals. Increase the dose in increments of 500 mg weekly up to a "
        "maximum of 2550 mg per day. 2.2 Pediatric Dosage The recommended starting dose for "
        "pediatric patients 10 years of age and older is 500 mg orally twice a day."),
)

WARFARIN = make_label(
    "set-warfarin", "warfarin sodium", "Coumadin",
    indications_and_usage=(
        "1 INDICATIONS AND USAGE Warfarin sodium is indicated for the prophylaxis and treatment "
        "of venous thrombosis and pulmonary embolism."),
    contraindications=(
        "4 CONTRAINDICATIONS Warfarin is contraindicated in pregnancy, except in pregnant women "
        "with mechanical heart valves. It is contraindicated in patients with hemorrhagic "
        "tendencies or blood dyscrasias."),
    drug_interactions=(
        "7 DRUG INTERACTIONS Inhibitors of CYP2C9 such as amiodarone and fluconazole increase "
        "the exposure of warfarin and raise the risk of bleeding. Monitor INR more frequently."),
)

CORPUS = [METFORMIN, WARFARIN]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, abstain_threshold=-2.0, cache_size=8, rerank_n=10)


@pytest.fixture
def store() -> Store:
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def indexed(store, settings):
    embedder = HashEmbedder()
    ingest_records(store, CORPUS, settings)
    embed_missing(store, embedder)
    return store, embedder


@pytest.fixture
def retriever(indexed, settings) -> Retriever:
    store, embedder = indexed
    return Retriever(store, embedder, OverlapReranker(), settings)


@pytest.fixture
def service(retriever, settings) -> RagService:
    return RagService(retriever, FakeLLM(), settings)
