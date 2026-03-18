from __future__ import annotations

import subprocess
from pathlib import Path

from unpaidwork.ingest.provenance import (
    read_provenance_sidecar,
    resolve_git_commit_hash,
    sidecar_path,
    write_provenance_sidecar,
)


def test_sidecar_path_derivation() -> None:
    dataset = Path("/tmp/childcare_state_year_panel.parquet")
    expected = Path("/tmp/childcare_state_year_panel.parquet.provenance.json")
    assert sidecar_path(dataset) == expected


def test_write_read_provenance_roundtrip(tmp_path: Path) -> None:
    dataset = tmp_path / "data" / "interim" / "ndcp" / "ndcp_segment_prices.parquet"
    source_a = tmp_path / "data" / "raw" / "ndcp" / "ndcp.xlsx"
    source_b = tmp_path / "data" / "raw" / "qcew" / "qcew.csv"

    sidecar = write_provenance_sidecar(
        dataset_path=dataset,
        source_files=[source_a, str(source_b)],
        source_releases={
            str(source_a): "2024-01-01",
            str(source_b): "2024-02-15",
        },
        parameters={"geography_mode": "state-year", "sample_mode": True},
        config={"extension": "segmented_solver"},
        generated_at="2026-03-17T00:00:00+00:00",
        git_commit_hash="abc123",
    )
    loaded = read_provenance_sidecar(dataset)

    assert sidecar == sidecar_path(dataset)
    assert loaded["dataset_path"] == str(dataset)
    assert loaded["generated_at"] == "2026-03-17T00:00:00+00:00"
    assert loaded["git_commit_hash"] == "abc123"
    assert loaded["source_files"] == [str(source_a), str(source_b)]
    assert loaded["source_releases"][str(source_a)] == "2024-01-01"
    assert loaded["source_releases"][str(source_b)] == "2024-02-15"
    assert loaded["parameters"]["geography_mode"] == "state-year"
    assert loaded["parameters"]["sample_mode"] is True
    assert loaded["config"]["extension"] == "segmented_solver"


def test_write_provenance_handles_missing_git_hash(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "data" / "interim" / "panel.parquet"

    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(subprocess, "run", _raise)
    sidecar = write_provenance_sidecar(
        dataset_path=dataset,
        source_files=[],
        parameters={"alpha_grid": [0.1, 0.5, 1.0]},
    )
    loaded = read_provenance_sidecar(dataset)

    assert sidecar.exists()
    assert loaded["git_commit_hash"] is None
    assert loaded["parameters"]["alpha_grid"] == [0.1, 0.5, 1.0]


def test_resolve_git_commit_hash_returns_none_outside_repo(tmp_path: Path) -> None:
    assert resolve_git_commit_hash(repo_root=tmp_path) is None
