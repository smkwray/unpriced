from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    interim: Path
    processed: Path
    registry: Path
    outputs_tables: Path
    outputs_figures: Path
    outputs_reports: Path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def extension_config_dir(root: Path) -> Path:
    return root / "configs" / "extensions"


def resolve_extension_config_path(root: Path, config_name_or_path: str | Path) -> Path:
    candidate = Path(config_name_or_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    if candidate.suffix:
        return extension_config_dir(root) / candidate.name
    return extension_config_dir(root) / f"{candidate.name}.yaml"


def load_extension_config(root: Path, config_name_or_path: str | Path) -> dict[str, Any]:
    path = resolve_extension_config_path(root, config_name_or_path)
    return load_yaml(path)


def load_project_paths(root: Path) -> ProjectPaths:
    config = load_yaml(root / "configs" / "project.yaml")
    raw = root / config["paths"]["raw"]
    interim = root / config["paths"]["interim"]
    processed = root / config["paths"]["processed"]
    registry = root / config["paths"]["registry"]
    return ProjectPaths(
        root=root,
        raw=raw,
        interim=interim,
        processed=processed,
        registry=registry,
        outputs_tables=root / "outputs" / "tables",
        outputs_figures=root / "outputs" / "figures",
        outputs_reports=root / "outputs" / "reports",
    )


def ensure_project_dirs(paths: ProjectPaths) -> None:
    for directory in (
        paths.raw,
        paths.interim,
        paths.processed,
        paths.registry.parent,
        paths.outputs_tables,
        paths.outputs_figures,
        paths.outputs_reports,
    ):
        directory.mkdir(parents=True, exist_ok=True)
