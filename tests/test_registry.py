from __future__ import annotations

from unpriced.ingest.atus import ingest
from unpriced.registry import ensure_registry
from unpriced.storage import read_parquet


def test_registry_records_sample_ingest(project_paths):
    ensure_registry(project_paths)
    ingest(project_paths, sample=True)
    registry = read_parquet(project_paths.registry)
    assert len(registry) == 1
    assert registry.iloc[0]["source_name"] == "atus"
    assert bool(registry.iloc[0]["sample_mode"]) is True
