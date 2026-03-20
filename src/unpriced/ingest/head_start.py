from __future__ import annotations

import io
import os
import re
from pathlib import Path

import pandas as pd
import requests

from unpriced.config import ProjectPaths
from unpriced.errors import SourceAccessError
from unpriced.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpriced.registry import append_registry, build_record
from unpriced.sample_data import head_start as sample_head_start
from unpriced.storage import write_parquet

SPEC = SourceSpec(
    name="head_start",
    citation="https://headstart.gov/about-us/article/head-start-service-location-datasets",
    license_name="Public data",
    retrieval_method="download",
    landing_page="https://headstart.gov/about-us/article/head-start-service-location-datasets",
)

HEAD_START_URL = "https://s3foa.s3.us-east-1.amazonaws.com/HS_Service_Locations.csv"
COUNTY_CROSSWALK_URL = "https://api.census.gov/data/{year}/acs/acs5"
STATE_ABBREV_TO_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}


def _download_bytes(url: str) -> bytes:
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


def _normalize_county_name(name: str) -> str:
    value = str(name or "").strip().lower()
    value = re.sub(r"[.'`]", "", value)
    value = value.replace("&", "and")
    value = re.sub(r"\bsaint\b", "st", value)
    value = re.sub(r"\bste\b", "st", value)
    value = re.sub(
        r"\b(county|parish|borough|census area|municipality|city and borough|city)\b",
        "",
        value,
    )
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _fetch_county_crosswalk(year: int) -> pd.DataFrame:
    params = {"get": "NAME", "for": "county:*"}
    api_key = os.getenv("CENSUS_API_KEY")
    if api_key:
        params["key"] = api_key
    response = requests.get(COUNTY_CROSSWALK_URL.format(year=year), params=params, timeout=120)
    response.raise_for_status()
    rows = response.json()
    frame = pd.DataFrame(rows[1:], columns=rows[0])
    county_name = frame["NAME"].astype(str).str.split(",", n=1).str[0]
    frame["county_key"] = county_name.map(_normalize_county_name)
    frame["state_fips"] = frame["state"].astype(str).str.zfill(2)
    frame["county_fips"] = frame["state_fips"] + frame["county"].astype(str).str.zfill(3)
    return frame[["state_fips", "county_fips", "county_key"]].drop_duplicates()


def _normalize_head_start_frame(frame: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["state"] = data["state"].astype(str).str.upper().str.strip()
    data["state_fips"] = data["state"].map(STATE_ABBREV_TO_FIPS)
    data["county_key"] = data["county"].map(_normalize_county_name)
    data["funded_slots"] = pd.to_numeric(data["funded_slots"], errors="coerce")
    data["status"] = data["status"].astype(str).str.strip().str.lower()
    data = data.loc[
        data["state_fips"].notna()
        & data["county_key"].ne("")
        & data["funded_slots"].fillna(0).gt(0)
        & data["status"].eq("open")
    ].copy()

    merged = data.merge(crosswalk, on=["state_fips", "county_key"], how="left")
    merged = merged.loc[merged["county_fips"].notna()].copy()

    grouped = (
        merged.groupby(["county_fips", "state_fips"], as_index=False)
        .agg(
            head_start_slots=("funded_slots", "sum"),
            open_locations=("service_location_name", "nunique"),
        )
        .sort_values(["state_fips", "county_fips"])
        .reset_index(drop=True)
    )
    return grouped


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_head_start, refresh=refresh, dry_run=dry_run)

    raw_path = paths.raw / SPEC.name / Path(HEAD_START_URL).name
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"
    target_year = year or 2024
    if dry_run:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=HEAD_START_URL)
    if raw_path.exists() and normalized_path.exists() and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(_download_bytes(HEAD_START_URL))

    frame = pd.read_csv(io.BytesIO(raw_path.read_bytes()), dtype=str)
    crosswalk = _fetch_county_crosswalk(target_year)
    normalized = _normalize_head_start_frame(frame, crosswalk)
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
