from __future__ import annotations

from pathlib import Path

import pandas as pd

from unpaidwork.config import ProjectPaths
from unpaidwork.ingest.common import IngestResult, SourceSpec, ingest_placeholder, ingest_sample
from unpaidwork.registry import append_registry, build_record
from unpaidwork.sample_data import licensing_supply_shocks as sample_licensing_supply_shocks
from unpaidwork.storage import write_parquet

SPEC = SourceSpec(
    name="licensing",
    citation="https://licensingregulations.acf.hhs.gov/",
    license_name="Public data / manual curation required",
    retrieval_method="manual-curation",
    landing_page="https://licensingregulations.acf.hhs.gov/",
)

RAW_FILENAME = "licensing_supply_shocks.csv"
REQUIRED_COLUMNS = {"state_fips", "year"}


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "state_fips" in data.columns:
        data["state_fips"] = data["state_fips"].astype(str).str.zfill(2)
    if "year" in data.columns:
        data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    for column in (
        "center_labor_intensity_index",
        "center_infant_ratio",
        "center_toddler_ratio",
        "center_infant_group_size",
        "center_toddler_group_size",
    ):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    keep = [
        column
        for column in (
            "state_fips",
            "year",
            "center_labor_intensity_index",
            "center_infant_ratio",
            "center_toddler_ratio",
            "center_infant_group_size",
            "center_toddler_group_size",
            "shock_label",
            "effective_date",
            "source_url",
            "source_note",
        )
        if column in data.columns
    ]
    normalized = data[keep].dropna(subset=["state_fips", "year"]).drop_duplicates(["state_fips", "year"])
    return normalized.sort_values(["state_fips", "year"]).reset_index(drop=True)


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_licensing_supply_shocks, refresh=refresh, dry_run=dry_run)

    raw_path = paths.raw / SPEC.name / RAW_FILENAME
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}_supply_shocks.parquet"
    if dry_run:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=str(raw_path))
    if not raw_path.exists():
        return ingest_placeholder(paths, SPEC)
    if normalized_path.exists() and not refresh and normalized_path.stat().st_mtime >= raw_path.stat().st_mtime:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")

    frame = pd.read_csv(raw_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Licensing shock CSV missing required columns: {', '.join(sorted(missing))}")
    normalized = _normalize(frame)
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
