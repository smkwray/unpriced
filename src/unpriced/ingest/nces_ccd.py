from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import requests

from unpriced.config import ProjectPaths
from unpriced.errors import SourceAccessError
from unpriced.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpriced.registry import append_registry, build_record
from unpriced.sample_data import nces_ccd as sample_nces_ccd
from unpriced.storage import write_parquet

SPEC = SourceSpec(
    name="nces_ccd",
    citation="https://nces.ed.gov/ccd/ccddata.asp",
    license_name="Public data",
    retrieval_method="download",
    landing_page="https://nces.ed.gov/ccd/ccddata.asp",
)

CCD_URL = (
    "https://prod-ies-dm-migration.s3.us-gov-west-1.amazonaws.com/nces/"
    "asset_builder_data/2025/08/2025046%20Preliminary%20Data%20Release%20CCD%20Nonfiscal_0.zip"
)


def _download_ccd_zip(url: str) -> bytes:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "unpriced/0.1 (+research repo)"},
            timeout=180,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to fetch {url}: {exc}") from exc
    return response.content


def _school_csv_name(names: list[str]) -> str:
    for name in names:
        lower = name.lower()
        if lower.endswith(".csv") and "ccd_sch_" in lower:
            return name
    raise ValueError("CCD zip did not contain the expected school directory CSV")


def _to_flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().eq("YES")


def _normalize_ccd_zip(path: Path, url: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        school_name = _school_csv_name(archive.namelist())
        with archive.open(school_name) as handle:
            schools = pd.read_csv(handle, dtype=str)

    active = schools.loc[schools["SY_STATUS"].astype(str).str.strip().eq("1")].copy()
    active["state_fips"] = active["FIPST"].astype(str).str.zfill(2)
    active["prek_offered"] = _to_flag(active["G_PK_OFFERED"])
    active["kg_offered"] = _to_flag(active["G_KG_OFFERED"])
    active["pk_or_kg_offered"] = active["prek_offered"] | active["kg_offered"]

    grouped = (
        active.groupby("state_fips", as_index=False)
        .agg(
            prek_schools=("prek_offered", "sum"),
            kg_schools=("kg_offered", "sum"),
            pk_or_kg_schools=("pk_or_kg_offered", "sum"),
            operational_schools=("NCESSCH", "nunique"),
        )
        .sort_values("state_fips")
        .reset_index(drop=True)
    )
    grouped["public_school_option_index"] = grouped["pk_or_kg_schools"].div(
        grouped["operational_schools"].replace({0: pd.NA})
    )
    grouped["source_url"] = url
    return grouped


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_nces_ccd, refresh=refresh, dry_run=dry_run)

    raw_path = paths.raw / SPEC.name / Path(CCD_URL).name
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"
    if dry_run:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=CCD_URL)
    if raw_path.exists() and normalized_path.exists() and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(_download_ccd_zip(CCD_URL))
    normalized = _normalize_ccd_zip(raw_path, CCD_URL)
    write_parquet(normalized, normalized_path)
    append_registry(
        paths,
        build_record(
            source_name=SPEC.name,
            raw_path=raw_path,
            normalized_path=normalized_path,
            license_name=SPEC.license_name,
            retrieval_method=SPEC.retrieval_method,
            citation=SPEC.citation,
            sample_mode=False,
        ),
    )
    return IngestResult(SPEC.name, raw_path, normalized_path, False)
