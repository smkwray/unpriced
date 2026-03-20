from __future__ import annotations

from pathlib import Path

import pytest

from unpriced.config import ProjectPaths, ensure_project_dirs


@pytest.fixture
def project_paths(tmp_path: Path) -> ProjectPaths:
    paths = ProjectPaths(
        root=tmp_path,
        raw=tmp_path / "data" / "raw",
        interim=tmp_path / "data" / "interim",
        processed=tmp_path / "data" / "processed",
        registry=tmp_path / "data" / "registry" / "source_registry.parquet",
        outputs_tables=tmp_path / "outputs" / "tables",
        outputs_figures=tmp_path / "outputs" / "figures",
        outputs_reports=tmp_path / "outputs" / "reports",
    )
    ensure_project_dirs(paths)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "project.yaml").write_text(
        "\n".join(
            [
                "alpha_grid:",
                "  - 0.10",
                "  - 0.25",
                "  - 0.50",
                "  - 1.00",
                "paths:",
                "  raw: data/raw",
                "  interim: data/interim",
                "  processed: data/processed",
                "  registry: data/registry/source_registry.parquet",
            ]
        ),
        encoding="utf-8",
    )
    return paths
