from __future__ import annotations

import pandas as pd
import requests

from unpaidwork.config import ProjectPaths
from unpaidwork.ingest.common import IngestResult, SourceSpec, ingest_remote_csv, ingest_sample
from unpaidwork.logging import get_logger
from unpaidwork.registry import append_registry, build_record
from unpaidwork.sample_data import qcew as sample_qcew
from unpaidwork.storage import read_parquet, write_parquet

LOGGER = get_logger()

SPEC = SourceSpec(
    name="qcew",
    citation="https://www.bls.gov/cew/additional-resources/open-data/",
    license_name="Public data",
    retrieval_method="download",
    landing_page="https://www.bls.gov/cew/additional-resources/open-data/",
)

# QCEW Open Data API serves county-level NAICS 624410 from 2014 onward.
QCEW_FIRST_AVAILABLE_YEAR = 2014


def _normalize_qcew(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["area_fips"] = data["area_fips"].astype(str).str.zfill(5)
    data["agglvl_code"] = data["agglvl_code"].astype(str)
    data["own_code"] = data["own_code"].astype(str)
    data = data.loc[data["own_code"].eq("5") & data["agglvl_code"].eq("78")].copy()
    data["county_fips"] = data["area_fips"]
    data["state_fips"] = data["area_fips"].str[:2]
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    data["childcare_worker_wage"] = pd.to_numeric(
        data["annual_avg_wkly_wage"], errors="coerce"
    ) / 40.0
    data["outside_option_wage"] = pd.NA
    data["employment"] = pd.to_numeric(data["annual_avg_emplvl"], errors="coerce")
    keep = [
        "county_fips",
        "state_fips",
        "year",
        "industry_code",
        "childcare_worker_wage",
        "outside_option_wage",
        "employment",
    ]
    return data[keep].dropna(subset=["county_fips", "state_fips", "year"])


def _fetch_qcew_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    import io
    return pd.read_csv(io.StringIO(response.text))


def _merge_existing_years(normalized_path, new_frame: pd.DataFrame) -> pd.DataFrame:
    """Merge *new_frame* into existing parquet, replacing overlapping years."""
    if not normalized_path.exists():
        return new_frame
    existing = read_parquet(normalized_path)
    if existing.empty:
        return new_frame
    new_years = set(new_frame["year"].dropna().astype(int).tolist())
    keep_existing = existing.loc[~existing["year"].isin(sorted(new_years))].copy()
    combined = pd.concat([keep_existing, new_frame], ignore_index=True)
    combined = combined.sort_values(["state_fips", "county_fips", "year"], kind="stable").reset_index(drop=True)
    return combined


def existing_years(paths: ProjectPaths) -> set[int]:
    """Return the set of years already in the normalized QCEW parquet."""
    normalized_path = paths.interim / "qcew" / "qcew.parquet"
    if not normalized_path.exists():
        return set()
    try:
        frame = read_parquet(normalized_path)
        return set(frame["year"].dropna().astype(int).tolist())
    except Exception:
        return set()


def ingest_year_range(
    paths: ProjectPaths,
    start_year: int,
    end_year: int,
    refresh: bool = False,
) -> IngestResult:
    """Fetch QCEW for every year in [start_year, end_year] not already present."""
    normalized_path = paths.interim / "qcew" / "qcew.parquet"
    have = existing_years(paths) if not refresh else set()
    need = [y for y in range(max(start_year, QCEW_FIRST_AVAILABLE_YEAR), end_year + 1) if y not in have]
    if not need:
        return IngestResult(SPEC.name, normalized_path, normalized_path, False, skipped=True, detail="all years cached")
    frames = []
    for yr in need:
        url = f"https://data.bls.gov/cew/data/api/{yr}/a/industry/624410.csv"
        LOGGER.info("fetching QCEW for %s", yr)
        try:
            raw = _fetch_qcew_csv(url)
            normalized = _normalize_qcew(raw)
            frames.append(normalized)
            raw_path = paths.raw / SPEC.name / f"qcew_{yr}.csv"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw.to_csv(raw_path, index=False)
        except Exception as exc:
            LOGGER.warning("QCEW year %s unavailable: %s", yr, exc)
    if not frames:
        return IngestResult(SPEC.name, normalized_path, normalized_path, False, skipped=True, detail="no years fetched")
    new_frame = pd.concat(frames, ignore_index=True)
    combined = _merge_existing_years(normalized_path, new_frame)
    write_parquet(combined, normalized_path)
    LOGGER.info("QCEW parquet now covers years %s", sorted(combined["year"].unique().tolist()))
    return IngestResult(SPEC.name, normalized_path, normalized_path, False, detail=f"fetched {len(frames)} years")


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_qcew, refresh=refresh, dry_run=dry_run)
    target_year = year or 2024
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"
    raw_path = paths.raw / SPEC.name / f"qcew_{target_year}.csv"
    if dry_run:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=str(target_year))
    have = existing_years(paths)
    if target_year in have and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")
    url = f"https://data.bls.gov/cew/data/api/{target_year}/a/industry/624410.csv"
    raw = _fetch_qcew_csv(url)
    normalized = _normalize_qcew(raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    combined = _merge_existing_years(normalized_path, normalized)
    write_parquet(combined, normalized_path)
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
