from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests

from unpriced.config import ProjectPaths
from unpriced.errors import SourceAccessError
from unpriced.registry import append_registry, build_record
from unpriced.storage import write_parquet


@dataclass(frozen=True)
class SourceSpec:
    name: str
    citation: str
    license_name: str
    retrieval_method: str
    landing_page: str


@dataclass(frozen=True)
class IngestResult:
    source_name: str
    raw_path: Path
    normalized_path: Path
    sample_mode: bool
    dry_run: bool = False
    skipped: bool = False
    detail: str = ""


def _result(
    spec: SourceSpec,
    raw_path: Path,
    normalized_path: Path,
    sample_mode: bool,
    dry_run: bool = False,
    skipped: bool = False,
    detail: str = "",
) -> IngestResult:
    return IngestResult(
        source_name=spec.name,
        raw_path=raw_path,
        normalized_path=normalized_path,
        sample_mode=sample_mode,
        dry_run=dry_run,
        skipped=skipped,
        detail=detail,
    )


def _raw_path_for_url(paths: ProjectPaths, spec: SourceSpec, url: str) -> Path:
    parsed = urlparse(url)
    filename = Path(parsed.path).name or f"{spec.name}.dat"
    return paths.raw / spec.name / filename


def _download(url: str) -> bytes:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "unpriced/0.1 (+research repo)"},
            timeout=120,
        )
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to fetch {url}: {exc}") from exc
    if response.status_code >= 400:
        raise SourceAccessError(
            f"source blocked or unavailable for {url} (HTTP {response.status_code})"
        )
    content_type = (response.headers.get("content-type") or "").lower()
    body = response.content
    if "text/html" in content_type:
        head = body[:1500].decode("utf-8", errors="ignore")
        if "Challenge Validation" in head or "<!DOCTYPE html" in head:
            raise SourceAccessError(
                f"source returned a challenge page instead of data for {url}; this source likely needs a manual browser download"
            )
    return body


def _zip_manifest(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        rows = [
            {
                "entry_name": info.filename,
                "entry_type": "zip_member",
                "file_size": info.file_size,
                "compressed_size": info.compress_size,
            }
            for info in archive.infolist()
        ]
    return pd.DataFrame(rows)


def _xlsx_manifest(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/workbook.xml"))
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = []
    for sheet in root.findall(".//main:sheets/main:sheet", ns):
        rows.append(
            {
                "entry_name": sheet.attrib.get("name", ""),
                "entry_type": "worksheet",
                "sheet_id": sheet.attrib.get("sheetId", ""),
            }
        )
    return pd.DataFrame(rows)


def manifest_from_path(path: Path, url: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        frame = _xlsx_manifest(path)
    elif zipfile.is_zipfile(path):
        frame = _zip_manifest(path)
    else:
        frame = pd.DataFrame(
            [
                {
                    "entry_name": path.name,
                    "entry_type": "file",
                    "file_size": path.stat().st_size,
                }
            ]
        )
    frame["source_url"] = url
    frame["raw_filename"] = path.name
    return frame


def ingest_sample(
    paths: ProjectPaths,
    spec: SourceSpec,
    frame_factory: Callable[[], pd.DataFrame],
    refresh: bool = False,
    dry_run: bool = False,
) -> IngestResult:
    raw_path = paths.raw / spec.name / f"{spec.name}_sample.json"
    normalized_path = paths.interim / spec.name / f"{spec.name}.parquet"
    if dry_run:
        return _result(spec, raw_path, normalized_path, True, dry_run=True, detail="dry-run")
    if (
        raw_path.exists()
        and normalized_path.exists()
        and not refresh
        and normalized_path.stat().st_mtime >= raw_path.stat().st_mtime
    ):
        return _result(spec, raw_path, normalized_path, True, skipped=True, detail="cached")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame_factory()
    raw_payload = {"source": spec.name, "rows": frame.to_dict(orient="records")}
    raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    write_parquet(frame, normalized_path)
    append_registry(
        paths,
        build_record(
            source_name=spec.name,
            raw_path=raw_path,
            normalized_path=normalized_path,
            license_name=spec.license_name,
            retrieval_method=f"{spec.retrieval_method}:sample",
            citation=spec.citation,
            sample_mode=True,
        ),
    )
    return _result(spec, raw_path, normalized_path, True)


def ingest_remote_manifest(
    paths: ProjectPaths,
    spec: SourceSpec,
    url: str,
    refresh: bool = False,
    dry_run: bool = False,
    parser: Callable[[Path, str], pd.DataFrame] | None = None,
) -> IngestResult:
    raw_path = _raw_path_for_url(paths, spec, url)
    normalized_path = paths.interim / spec.name / f"{spec.name}.parquet"
    if dry_run:
        return _result(spec, raw_path, normalized_path, False, dry_run=True, detail=url)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        raw_path.write_bytes(_download(url))
    if (
        normalized_path.exists()
        and not refresh
        and normalized_path.stat().st_mtime >= raw_path.stat().st_mtime
    ):
        return _result(spec, raw_path, normalized_path, False, skipped=True, detail="cached")
    frame = parser(raw_path, url) if parser else manifest_from_path(raw_path, url)
    write_parquet(frame, normalized_path)
    append_registry(
        paths,
        build_record(
            source_name=spec.name,
            raw_path=raw_path,
            normalized_path=normalized_path,
            license_name=spec.license_name,
            retrieval_method=spec.retrieval_method,
            citation=spec.citation,
            sample_mode=False,
        ),
    )
    return _result(spec, raw_path, normalized_path, False)


def ingest_remote_csv(
    paths: ProjectPaths,
    spec: SourceSpec,
    url: str,
    parser: Callable[[pd.DataFrame], pd.DataFrame],
    refresh: bool = False,
    dry_run: bool = False,
) -> IngestResult:
    raw_path = _raw_path_for_url(paths, spec, url)
    normalized_path = paths.interim / spec.name / f"{spec.name}.parquet"
    if dry_run:
        return _result(spec, raw_path, normalized_path, False, dry_run=True, detail=url)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        raw_path.write_bytes(_download(url))
    if (
        normalized_path.exists()
        and not refresh
        and normalized_path.stat().st_mtime >= raw_path.stat().st_mtime
    ):
        return _result(spec, raw_path, normalized_path, False, skipped=True, detail="cached")
    raw_frame = pd.read_csv(raw_path)
    normalized = parser(raw_frame)
    write_parquet(normalized, normalized_path)
    append_registry(
        paths,
        build_record(
            source_name=spec.name,
            raw_path=raw_path,
            normalized_path=normalized_path,
            license_name=spec.license_name,
            retrieval_method=spec.retrieval_method,
            citation=spec.citation,
            sample_mode=False,
        ),
    )
    return _result(spec, raw_path, normalized_path, False)


def require_manual_source_path(raw_path: Path, landing_page: str, source_name: str) -> Path:
    if raw_path.exists():
        return raw_path
    raise SourceAccessError(
        f"missing manual source input for {source_name}: expected local path at {raw_path}; "
        f"download it from {landing_page} and place it there before running the real ingest"
    )


def ingest_placeholder(paths: ProjectPaths, spec: SourceSpec) -> IngestResult:
    raw_path = paths.raw / spec.name / f"{spec.name}_placeholder.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        f"Placeholder for {spec.name}. See {spec.landing_page}.\n",
        encoding="utf-8",
    )
    normalized_path = paths.interim / spec.name / f"{spec.name}.parquet"
    write_parquet(
        pd.DataFrame([{"source": spec.name, "status": "placeholder"}]),
        normalized_path,
    )
    append_registry(
        paths,
        build_record(
            source_name=spec.name,
            raw_path=raw_path,
            normalized_path=normalized_path,
            license_name=spec.license_name,
            retrieval_method=f"{spec.retrieval_method}:placeholder",
            citation=spec.citation,
            sample_mode=False,
        ),
    )
    return _result(spec, raw_path, normalized_path, False)
