from __future__ import annotations

from datetime import datetime, timezone
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from unpriced.storage import read_json, write_json


def sidecar_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(f"{dataset_path.suffix}.provenance.json")


def resolve_git_commit_hash(repo_root: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
        return None
    commit = result.stdout.strip()
    return commit or None


def write_provenance_sidecar(
    dataset_path: Path,
    source_files: Sequence[Path | str],
    source_releases: Mapping[str, str] | None = None,
    parameters: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
    git_commit_hash: str | None = None,
    repo_root: Path | None = None,
) -> Path:
    generated = generated_at or datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "generated_at": generated,
        "git_commit_hash": git_commit_hash
        if git_commit_hash is not None
        else resolve_git_commit_hash(repo_root=repo_root),
        "source_files": [str(path) for path in source_files],
        "source_releases": dict(source_releases or {}),
        "parameters": dict(parameters or {}),
        "config": dict(config or {}),
    }
    return write_json(payload, sidecar_path(dataset_path))


def read_provenance_sidecar(dataset_path: Path) -> dict[str, Any]:
    return read_json(sidecar_path(dataset_path))
