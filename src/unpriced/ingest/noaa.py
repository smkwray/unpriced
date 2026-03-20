"""NOAA Storm Events Database ingestor.

Uses the NCEI bulk CSV download (no API key required) to produce
county-year storm exposure and precipitation-event measures for
the home-maintenance module.

Source: https://www.ncdc.noaa.gov/stormevents/
Bulk files: https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/

Geography: county_fips (5-digit, built from STATE_FIPS + CZ_FIPS where
CZ_TYPE == 'C').  Rows with CZ_TYPE != 'C' (NWS forecast zones) are
dropped because zone-to-county mapping is many-to-many and lossy.

Output columns
--------------
county_fips           5-digit FIPS built from state + county zone code
state_fips            2-digit state FIPS
year                  integer year
storm_event_count     total storm events in the county-year
storm_property_damage total property damage in dollars
storm_exposure        storm_event_count / max(storm_event_count) within the
                      year (0-1 scale, comparable to the previous placeholder)
precip_event_days     distinct calendar days with precipitation-type events
source_url            URL of the downloaded bulk CSV
"""
from __future__ import annotations

import gzip
import io
import json
import re

import pandas as pd
import requests

from unpriced.config import ProjectPaths
from unpriced.errors import SourceAccessError
from unpriced.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpriced.registry import append_registry, build_record
from unpriced.storage import write_parquet

SPEC = SourceSpec(
    name="noaa",
    citation="https://www.ncdc.noaa.gov/stormevents/",
    license_name="Public data",
    retrieval_method="download",
    landing_page="https://www.ncdc.noaa.gov/stormevents/",
)

BULK_CSV_DIR = "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}

PRECIP_EVENT_TYPES = frozenset({
    "Flash Flood",
    "Flood",
    "Heavy Rain",
    "Tropical Storm",
    "Hurricane",
    "Hurricane (Typhoon)",
    "Tropical Depression",
    "Ice Storm",
    "Sleet",
    "Winter Storm",
    "Winter Weather",
    "Heavy Snow",
    "Blizzard",
    "Lake-Effect Snow",
    "Coastal Flood",
    "Storm Surge/Tide",
    "Hail",
})


def _sample_noaa() -> pd.DataFrame:
    from unpriced.sample_data import noaa
    return noaa()


def _parse_damage(value: object) -> float:
    """Parse NOAA damage strings like '10.00K', '1.50M', '0.00K' into dollars."""
    if pd.isna(value):
        return 0.0
    text = str(value).strip().upper()
    if not text or text == "0" or text == "0.00":
        return 0.0
    multiplier = 1.0
    if text.endswith("K"):
        multiplier = 1_000.0
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text.endswith("B"):
        multiplier = 1_000_000_000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return 0.0


def _find_details_url(year: int) -> str:
    """Fetch the NCEI bulk CSV directory listing and find the details file for *year*."""
    try:
        response = requests.get(BULK_CSV_DIR, headers=BROWSER_HEADERS, timeout=60)
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to fetch NOAA bulk CSV listing: {exc}") from exc
    if response.status_code >= 400:
        raise SourceAccessError(
            f"NOAA bulk CSV listing returned HTTP {response.status_code}"
        )
    pattern = re.compile(
        rf"(StormEvents_details-ftp_v1\.0_d{year}_c\d{{8}}\.csv\.gz)"
    )
    matches = pattern.findall(response.text)
    if not matches:
        raise SourceAccessError(
            f"no Storm Events details file found for year {year} at {BULK_CSV_DIR}"
        )
    filename = sorted(matches)[-1]
    return BULK_CSV_DIR + filename


def _download_gzipped_csv(url: str) -> pd.DataFrame:
    """Download a gzipped CSV and return as DataFrame."""
    try:
        response = requests.get(url, headers=BROWSER_HEADERS, timeout=300)
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to download {url}: {exc}") from exc
    if response.status_code >= 400:
        raise SourceAccessError(
            f"NOAA download returned HTTP {response.status_code} for {url}"
        )
    raw_bytes = gzip.decompress(response.content)
    return pd.read_csv(io.BytesIO(raw_bytes), dtype=str, low_memory=False)


def normalize_storm_events(frame: pd.DataFrame, year: int, url: str) -> pd.DataFrame:
    """Normalize raw Storm Events details into county-year aggregates."""
    data = frame.copy()

    data["CZ_TYPE"] = data["CZ_TYPE"].astype(str).str.strip().str.upper()
    data = data.loc[data["CZ_TYPE"] == "C"].copy()
    if data.empty:
        return _empty_frame()

    data["state_fips"] = (
        pd.to_numeric(data["STATE_FIPS"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(2)
    )
    data["cz_fips"] = (
        pd.to_numeric(data["CZ_FIPS"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(3)
    )
    data["county_fips"] = data["state_fips"] + data["cz_fips"]

    data = data.loc[data["county_fips"].str.len().eq(5)].copy()
    if data.empty:
        return _empty_frame()

    data["property_damage"] = data["DAMAGE_PROPERTY"].apply(_parse_damage)

    data["EVENT_TYPE"] = data["EVENT_TYPE"].astype(str).str.strip()
    data["is_precip"] = data["EVENT_TYPE"].isin(PRECIP_EVENT_TYPES)

    data["BEGIN_YEARMONTH"] = data["BEGIN_YEARMONTH"].astype(str).str.strip()
    data["BEGIN_DAY"] = data["BEGIN_DAY"].astype(str).str.strip().str.zfill(2)
    data["event_date"] = data["BEGIN_YEARMONTH"] + data["BEGIN_DAY"]

    agg_rows: list[dict[str, object]] = []
    for (cfips, sfips), grp in data.groupby(["county_fips", "state_fips"], dropna=False):
        precip_sub = grp.loc[grp["is_precip"]]
        agg_rows.append({
            "county_fips": cfips,
            "state_fips": sfips,
            "storm_event_count": len(grp),
            "storm_property_damage": grp["property_damage"].sum(),
            "precip_event_days": precip_sub["event_date"].nunique(),
        })

    result = pd.DataFrame(agg_rows)
    result["year"] = int(year)

    max_events = result["storm_event_count"].max()
    result["storm_exposure"] = (
        result["storm_event_count"] / max_events if max_events > 0 else 0.0
    )

    result["source_url"] = url

    keep = [
        "county_fips",
        "state_fips",
        "year",
        "storm_event_count",
        "storm_property_damage",
        "storm_exposure",
        "precip_event_days",
        "source_url",
    ]
    return result[keep].sort_values(
        ["state_fips", "county_fips"], kind="stable"
    ).reset_index(drop=True)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "county_fips",
            "state_fips",
            "year",
            "storm_event_count",
            "storm_property_damage",
            "storm_exposure",
            "precip_event_days",
            "source_url",
        ]
    )


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, _sample_noaa, refresh=refresh, dry_run=dry_run)

    target_year = int(year or 2022)
    raw_path = paths.raw / SPEC.name / f"noaa_storm_{target_year}.json"
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"

    if dry_run:
        return IngestResult(
            SPEC.name, raw_path, normalized_path, False,
            dry_run=True, detail=str(target_year),
        )
    if raw_path.exists() and normalized_path.exists() and not refresh:
        return IngestResult(
            SPEC.name, raw_path, normalized_path, False,
            skipped=True, detail="cached",
        )

    url = _find_details_url(target_year)
    raw_frame = _download_gzipped_csv(url)
    normalized = normalize_storm_events(raw_frame, target_year, url)

    raw_meta = {
        "source": SPEC.name,
        "url": url,
        "year": target_year,
        "raw_rows": len(raw_frame),
        "county_rows": len(normalized),
        "columns": list(raw_frame.columns),
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw_meta, indent=2), encoding="utf-8")

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
