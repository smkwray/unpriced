from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from unpaidwork.config import ProjectPaths
from unpaidwork.storage import read_parquet, write_parquet

REGISTRY_COLUMNS = [
    "source_name",
    "license",
    "retrieval_method",
    "checksum",
    "last_fetched",
    "citation",
    "raw_path",
    "normalized_path",
    "sample_mode",
]


@dataclass(frozen=True)
class RegistryRecord:
    source_name: str
    license: str
    retrieval_method: str
    checksum: str
    last_fetched: str
    citation: str
    raw_path: str
    normalized_path: str
    sample_mode: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_registry(paths: ProjectPaths) -> Path:
    if paths.registry.exists():
        return paths.registry
    write_parquet(pd.DataFrame(columns=REGISTRY_COLUMNS), paths.registry)
    return paths.registry


def append_registry(paths: ProjectPaths, record: RegistryRecord) -> Path:
    ensure_registry(paths)
    frame = read_parquet(paths.registry)
    updated = pd.concat([frame, pd.DataFrame([asdict(record)])], ignore_index=True)
    updated = updated.drop_duplicates(
        subset=["source_name", "normalized_path", "sample_mode"], keep="last"
    )
    return write_parquet(updated[REGISTRY_COLUMNS], paths.registry)


def build_record(
    source_name: str,
    raw_path: Path,
    normalized_path: Path,
    license_name: str,
    retrieval_method: str,
    citation: str,
    sample_mode: bool,
) -> RegistryRecord:
    return RegistryRecord(
        source_name=source_name,
        license=license_name,
        retrieval_method=retrieval_method,
        checksum=sha256_file(raw_path),
        last_fetched=datetime.now(timezone.utc).isoformat(),
        citation=citation,
        raw_path=str(raw_path),
        normalized_path=str(normalized_path),
        sample_mode=sample_mode,
    )
