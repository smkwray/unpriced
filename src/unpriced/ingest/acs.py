from __future__ import annotations

import os

import pandas as pd
import requests

from unpriced.config import ProjectPaths
from unpriced.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpriced.logging import get_logger
from unpriced.registry import append_registry, build_record
from unpriced.sample_data import acs as sample_acs
from unpriced.storage import read_parquet, write_parquet

LOGGER = get_logger()

SPEC = SourceSpec(
    name="acs",
    citation="https://www.census.gov/data/developers/data-sets/acs-5year.html",
    license_name="Public data",
    retrieval_method="api",
    landing_page="https://www.census.gov/data/developers/data-sets/acs-5year.html",
)

REQUIRED_COLUMNS = {
    "county_fips",
    "state_fips",
    "year",
    "under5_population",
    "under5_male_population",
    "under5_female_population",
    "under6_population",
    "median_income",
    "rent_index",
    "single_parent_share",
    "parent_employment_rate",
}


def _has_required_schema(path) -> bool:
    if not path.exists():
        return False
    try:
        columns = set(read_parquet(path).columns)
    except Exception:
        return False
    return REQUIRED_COLUMNS.issubset(columns)


def _fetch_acs(year: int) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": (
            "NAME,"
            "B01001_003E,"
            "B01001_027E,"
            "B19113_001E,"
            "B25064_001E,"
            "B23008_002E,"
            "B23008_004E,"
            "B23008_005E,"
            "B23008_006E,"
            "B23008_008E,"
            "B23008_010E,"
            "B23008_013E"
        ),
        "for": "county:*",
    }
    api_key = os.getenv("CENSUS_API_KEY")
    if api_key:
        params["key"] = api_key
    response = requests.get(base, params=params, timeout=60)
    response.raise_for_status()
    rows = response.json()
    frame = pd.DataFrame(rows[1:], columns=rows[0])
    frame["state_fips"] = frame["state"]
    frame["county_fips"] = frame["state"] + frame["county"]
    frame["year"] = year
    frame = frame.rename(
        columns={
            "B01001_003E": "under5_male_population",
            "B01001_027E": "under5_female_population",
            "B19113_001E": "median_income",
            "B25064_001E": "rent_index",
            "B23008_002E": "under6_population",
            "B23008_008E": "under6_one_parent",
        }
    )
    numeric_columns = [
        "under5_male_population",
        "under5_female_population",
        "median_income",
        "rent_index",
        "under6_population",
        "under6_one_parent",
        "B23008_004E",
        "B23008_005E",
        "B23008_006E",
        "B23008_010E",
        "B23008_013E",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["under5_population"] = (
        frame["under5_male_population"].fillna(0.0) + frame["under5_female_population"].fillna(0.0)
    )

    in_labor_force = (
        frame["B23008_004E"].fillna(0.0)
        + frame["B23008_005E"].fillna(0.0)
        + frame["B23008_006E"].fillna(0.0)
        + frame["B23008_010E"].fillna(0.0)
        + frame["B23008_013E"].fillna(0.0)
    )
    denom = frame["under6_population"].replace({0: pd.NA})
    frame["single_parent_share"] = frame["under6_one_parent"].div(denom)
    frame["parent_employment_rate"] = in_labor_force.div(denom)

    keep = [
        "county_fips",
        "state_fips",
        "year",
        "under5_population",
        "under5_male_population",
        "under5_female_population",
        "under6_population",
        "median_income",
        "rent_index",
        "single_parent_share",
        "parent_employment_rate",
    ]
    result = frame[keep].copy()
    for column in [
        "year",
        "under5_population",
        "under5_male_population",
        "under5_female_population",
        "under6_population",
        "median_income",
        "rent_index",
        "single_parent_share",
        "parent_employment_rate",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


# ACS 5-year estimates are available from 2009 onward.
ACS_FIRST_AVAILABLE_YEAR = 2009


def _merge_existing_years(normalized_path, new_frame: pd.DataFrame) -> pd.DataFrame:
    """Merge *new_frame* into an existing parquet, replacing overlapping years."""
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
    """Return the set of years already present in the normalized ACS parquet."""
    normalized_path = paths.interim / "acs" / "acs.parquet"
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
    """Fetch ACS for every year in [start_year, end_year] not already present."""
    normalized_path = paths.interim / "acs" / "acs.parquet"
    have = existing_years(paths) if not refresh else set()
    need = [y for y in range(max(start_year, ACS_FIRST_AVAILABLE_YEAR), end_year + 1) if y not in have]
    if not need:
        return IngestResult(SPEC.name, normalized_path, normalized_path, False, skipped=True, detail="all years cached")
    frames = []
    for yr in need:
        LOGGER.info("fetching ACS 5-year for %s", yr)
        frames.append(_fetch_acs(yr))
        raw_path = paths.raw / SPEC.name / f"acs_{yr}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(frames[-1].to_json(orient="records", indent=2), encoding="utf-8")
    new_frame = pd.concat(frames, ignore_index=True)
    combined = _merge_existing_years(normalized_path, new_frame)
    write_parquet(combined, normalized_path)
    LOGGER.info("ACS parquet now covers years %s", sorted(combined["year"].unique().tolist()))
    return IngestResult(SPEC.name, normalized_path, normalized_path, False, detail=f"fetched {len(need)} years")


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    return ingest_with_options(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
    )


def ingest_with_options(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_acs, refresh=refresh, dry_run=dry_run)
    target_year = year or 2022
    raw_path = paths.raw / SPEC.name / f"acs_{target_year}.json"
    normalized_path = paths.interim / SPEC.name / "acs.parquet"
    if dry_run:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=str(target_year))
    have = existing_years(paths)
    if target_year in have and _has_required_schema(normalized_path) and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")
    frame = _fetch_acs(target_year)
    raw_path = paths.raw / SPEC.name / f"acs_{target_year}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    combined = _merge_existing_years(normalized_path, frame)
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
