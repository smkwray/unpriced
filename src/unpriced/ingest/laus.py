from __future__ import annotations

import io
import json

import pandas as pd
import requests

from unpriced.config import ProjectPaths
from unpriced.errors import SourceAccessError
from unpriced.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpriced.logging import get_logger
from unpriced.registry import append_registry, build_record
from unpriced.storage import read_parquet, write_parquet

LOGGER = get_logger()

SPEC = SourceSpec(
    name="laus",
    citation="https://www.bls.gov/lau/",
    license_name="Public data",
    retrieval_method="download",
    landing_page="https://www.bls.gov/lau/",
)

BLS_TIMESERIES_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
COUNTY_DATA_URL = "https://download.bls.gov/pub/time.series/la/la.data.64.County"
STATE_DATA_URL = "https://download.bls.gov/pub/time.series/la/la.data.2.AllStatesU"
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}
MEASURE_MAP = {
    "03": "laus_unemployment_rate",
    "04": "laus_unemployed",
    "05": "laus_employment",
    "06": "laus_labor_force",
}

# LAUS flat files cover data from 1990 onward.
LAUS_FIRST_AVAILABLE_YEAR = 1990


def _sample_laus() -> pd.DataFrame:
    rows = [
        {"geography": "county", "state_fips": "06", "county_fips": "06037", "year": 2021, "laus_unemployment_rate": 0.074, "laus_unemployed": 362000.0, "laus_employment": 4520000.0, "laus_labor_force": 4882000.0},
        {"geography": "county", "state_fips": "06", "county_fips": "06073", "year": 2021, "laus_unemployment_rate": 0.059, "laus_unemployed": 109000.0, "laus_employment": 1744000.0, "laus_labor_force": 1853000.0},
        {"geography": "county", "state_fips": "48", "county_fips": "48113", "year": 2021, "laus_unemployment_rate": 0.068, "laus_unemployed": 89000.0, "laus_employment": 1220000.0, "laus_labor_force": 1309000.0},
        {"geography": "county", "state_fips": "48", "county_fips": "48201", "year": 2021, "laus_unemployment_rate": 0.072, "laus_unemployed": 184000.0, "laus_employment": 2375000.0, "laus_labor_force": 2559000.0},
        {"geography": "county", "state_fips": "36", "county_fips": "36061", "year": 2021, "laus_unemployment_rate": 0.087, "laus_unemployed": 71000.0, "laus_employment": 747000.0, "laus_labor_force": 818000.0},
        {"geography": "county", "state_fips": "36", "county_fips": "36029", "year": 2021, "laus_unemployment_rate": 0.051, "laus_unemployed": 5200.0, "laus_employment": 96800.0, "laus_labor_force": 102000.0},
        {"geography": "state", "state_fips": "06", "county_fips": pd.NA, "year": 2021, "laus_unemployment_rate": 0.076, "laus_unemployed": 1471000.0, "laus_employment": 17967000.0, "laus_labor_force": 19438000.0},
        {"geography": "state", "state_fips": "48", "county_fips": pd.NA, "year": 2021, "laus_unemployment_rate": 0.061, "laus_unemployed": 861000.0, "laus_employment": 13256000.0, "laus_labor_force": 14117000.0},
        {"geography": "state", "state_fips": "36", "county_fips": pd.NA, "year": 2021, "laus_unemployment_rate": 0.069, "laus_unemployed": 654000.0, "laus_employment": 8804000.0, "laus_labor_force": 9458000.0},
    ]
    return pd.DataFrame(rows)


def _county_series_id(county_fips: str, measure: str) -> str:
    state_fips = county_fips[:2]
    county_code = county_fips[2:]
    return f"LAUCN{state_fips}{county_code}00000000{measure}"


def _state_series_id(state_fips: str, measure: str) -> str:
    return f"LAUST{state_fips}00000000000{measure}"


def _fetch_laus_series(series_ids: list[str], year: int) -> dict[str, object]:
    try:
        response = requests.post(
            BLS_TIMESERIES_URL,
            json={
                "seriesid": series_ids,
                "startyear": str(year),
                "endyear": str(year),
                "annualaverage": True,
            },
            timeout=180,
        )
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to fetch LAUS API for year {year}: {exc}") from exc
    if response.status_code >= 400:
        raise SourceAccessError(f"LAUS API request failed for year {year} (HTTP {response.status_code})")
    payload = response.json()
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise SourceAccessError(f"LAUS API request did not succeed for year {year}")
    return payload


def _download_laus_text(url: str) -> str:
    try:
        response = requests.get(url, headers=BROWSER_HEADERS, timeout=240)
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to fetch LAUS flat file {url}: {exc}") from exc
    if response.status_code >= 400:
        raise SourceAccessError(f"LAUS flat file request failed for {url} (HTTP {response.status_code})")
    return response.text


def _normalize_laus_payload(payload: dict[str, object], year: int) -> pd.DataFrame:
    series = payload.get("Results", {}).get("series", [])
    rows: list[dict[str, object]] = []
    for item in series:
        series_id = str(item.get("seriesID", ""))
        annual = next((point for point in item.get("data", []) if point.get("period") == "M13"), None)
        if not series_id or not annual:
            continue
        measure = series_id[-2:]
        value = pd.to_numeric(annual.get("value"), errors="coerce")
        if measure == "03":
            value = value / 100.0
        area_code = series_id[3:-2]
        row: dict[str, object]
        if area_code.startswith("CN"):
            state_fips = area_code[2:4]
            county_fips = state_fips + area_code[4:7]
            row = {
                "geography": "county",
                "state_fips": state_fips,
                "county_fips": county_fips,
                "year": int(year),
            }
        elif area_code.startswith("ST"):
            state_fips = area_code[2:4]
            row = {
                "geography": "state",
                "state_fips": state_fips,
                "county_fips": pd.NA,
                "year": int(year),
            }
        else:
            continue
        row[MEASURE_MAP[measure]] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=["geography", "state_fips", "county_fips", "year", *MEASURE_MAP.values()]
        )

    group_keys = ["geography", "state_fips", "county_fips", "year"]
    normalized = frame.groupby(group_keys, dropna=False, as_index=False).first()
    return normalized.sort_values(["geography", "state_fips", "county_fips"], kind="stable").reset_index(drop=True)


def _normalize_laus_downloads(
    county_text: str,
    state_text: str,
    county_fips: list[str],
    state_fips: list[str],
    year: int,
) -> pd.DataFrame:
    county_series = {_county_series_id(code, measure) for code in county_fips for measure in MEASURE_MAP}
    state_series = {_state_series_id(code, measure) for code in state_fips for measure in MEASURE_MAP}

    county = pd.read_csv(io.StringIO(county_text), sep="\t", dtype=str)
    county.columns = county.columns.astype(str).str.strip()
    county["series_id"] = county["series_id"].astype(str).str.strip()
    county["year"] = pd.to_numeric(county["year"], errors="coerce").astype("Int64")
    county["period"] = county["period"].astype(str).str.strip()
    county = county.loc[
        county["year"].eq(year) & county["period"].eq("M13") & county["series_id"].isin(county_series)
    ].copy()

    state = pd.read_csv(io.StringIO(state_text), sep="\t", dtype=str)
    state.columns = state.columns.astype(str).str.strip()
    state["series_id"] = state["series_id"].astype(str).str.strip()
    state["year"] = pd.to_numeric(state["year"], errors="coerce").astype("Int64")
    state["period"] = state["period"].astype(str).str.strip()
    state = state.loc[
        state["year"].eq(year) & state["period"].eq("M13") & state["series_id"].isin(state_series)
    ].copy()

    flat_payload = {
        "Results": {
            "series": [
                {"seriesID": row["series_id"], "data": [{"period": "M13", "value": row["value"]}]}
                for row in pd.concat([county, state], ignore_index=True).to_dict(orient="records")
            ]
        }
    }
    return _normalize_laus_payload(flat_payload, year)


def _normalize_laus_downloads_multiyear(
    county_text: str,
    state_text: str,
    county_fips: list[str],
    state_fips: list[str],
    years: list[int],
) -> pd.DataFrame:
    """Extract multiple years from LAUS flat files in a single pass."""
    county_series = {_county_series_id(code, measure) for code in county_fips for measure in MEASURE_MAP}
    state_series = {_state_series_id(code, measure) for code in state_fips for measure in MEASURE_MAP}
    year_set = set(years)

    county = pd.read_csv(io.StringIO(county_text), sep="\t", dtype=str)
    county.columns = county.columns.astype(str).str.strip()
    county["series_id"] = county["series_id"].astype(str).str.strip()
    county["year"] = pd.to_numeric(county["year"], errors="coerce").astype("Int64")
    county["period"] = county["period"].astype(str).str.strip()
    county = county.loc[
        county["year"].isin(year_set) & county["period"].eq("M13") & county["series_id"].isin(county_series)
    ].copy()

    state = pd.read_csv(io.StringIO(state_text), sep="\t", dtype=str)
    state.columns = state.columns.astype(str).str.strip()
    state["series_id"] = state["series_id"].astype(str).str.strip()
    state["year"] = pd.to_numeric(state["year"], errors="coerce").astype("Int64")
    state["period"] = state["period"].astype(str).str.strip()
    state = state.loc[
        state["year"].isin(year_set) & state["period"].eq("M13") & state["series_id"].isin(state_series)
    ].copy()

    frames = []
    for yr in years:
        yr_county = county.loc[county["year"].eq(yr)]
        yr_state = state.loc[state["year"].eq(yr)]
        flat_payload = {
            "Results": {
                "series": [
                    {"seriesID": row["series_id"], "data": [{"period": "M13", "value": row["value"]}]}
                    for row in pd.concat([yr_county, yr_state], ignore_index=True).to_dict(orient="records")
                ]
            }
        }
        frames.append(_normalize_laus_payload(flat_payload, yr))
    if not frames:
        return pd.DataFrame(
            columns=["geography", "state_fips", "county_fips", "year", *MEASURE_MAP.values()]
        )
    return pd.concat(frames, ignore_index=True)


def _collect_laus_geographies(paths: ProjectPaths, year: int) -> tuple[list[str], list[str]]:
    acs_path = paths.interim / "acs" / "acs.parquet"
    if not acs_path.exists():
        raise SourceAccessError("LAUS ingest requires ACS county geography; run ACS ingest first")

    acs = read_parquet(acs_path)
    if year in set(pd.to_numeric(acs["year"], errors="coerce").dropna().astype(int)):
        acs = acs.loc[pd.to_numeric(acs["year"], errors="coerce").eq(year)].copy()
    county_fips = sorted(acs["county_fips"].astype(str).str.zfill(5).dropna().unique().tolist())
    state_fips = sorted(acs["state_fips"].astype(str).str.zfill(2).dropna().unique().tolist())
    return county_fips, state_fips


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


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
    combined = combined.sort_values(["geography", "state_fips", "county_fips", "year"], kind="stable").reset_index(drop=True)
    return combined


def existing_years(paths: ProjectPaths) -> set[int]:
    """Return the set of years already in the normalized LAUS parquet."""
    normalized_path = paths.interim / "laus" / "laus.parquet"
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
    """Fetch LAUS for every year in [start_year, end_year] not already present.

    Uses flat-file download for efficiency (single download covers all years).
    """
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"
    have = existing_years(paths) if not refresh else set()
    need = [y for y in range(max(start_year, LAUS_FIRST_AVAILABLE_YEAR), end_year + 1) if y not in have]
    if not need:
        return IngestResult(SPEC.name, normalized_path, normalized_path, False, skipped=True, detail="all years cached")

    # Use a representative year for geography (latest available ACS year)
    county_fips, state_fips = _collect_laus_geographies(paths, need[-1])
    LOGGER.info("fetching LAUS flat files for %s years (%s-%s)", len(need), need[0], need[-1])
    county_text = _download_laus_text(COUNTY_DATA_URL)
    state_text = _download_laus_text(STATE_DATA_URL)
    frame = _normalize_laus_downloads_multiyear(county_text, state_text, county_fips, state_fips, need)

    if frame.empty:
        return IngestResult(SPEC.name, normalized_path, normalized_path, False, skipped=True, detail="no data in flat files")

    combined = _merge_existing_years(normalized_path, frame)
    write_parquet(combined, normalized_path)
    LOGGER.info("LAUS parquet now covers years %s", sorted(combined["year"].unique().tolist()))
    return IngestResult(SPEC.name, normalized_path, normalized_path, False, detail=f"fetched {len(need)} years")


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, _sample_laus, refresh=refresh, dry_run=dry_run)

    target_year = int(year or 2022)
    raw_path = paths.raw / SPEC.name / f"laus_{target_year}.json"
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"
    if dry_run:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=str(target_year))
    have = existing_years(paths)
    if target_year in have and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")

    county_fips, state_fips = _collect_laus_geographies(paths, target_year)
    series_ids = []
    for county in county_fips:
        series_ids.extend(_county_series_id(county, measure) for measure in MEASURE_MAP)
    for state in state_fips:
        series_ids.extend(_state_series_id(state, measure) for measure in MEASURE_MAP)

    payloads = []
    frames = []
    for chunk in _chunked(series_ids, 50):
        try:
            payload = _fetch_laus_series(chunk, target_year)
        except SourceAccessError:
            payload = None
        if payload is not None:
            payloads.append(payload)
            frames.append(_normalize_laus_payload(payload, target_year))
            continue
        county_text = _download_laus_text(COUNTY_DATA_URL)
        state_text = _download_laus_text(STATE_DATA_URL)
        frame = _normalize_laus_downloads(county_text, state_text, county_fips, state_fips, target_year)
        payloads = [
            {
                "county_url": COUNTY_DATA_URL,
                "state_url": STATE_DATA_URL,
                "retrieval_method": "download-flat-file",
                "year": target_year,
            }
        ]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payloads, indent=2), encoding="utf-8")
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

    frame = pd.concat(frames, ignore_index=True) if frames else _normalize_laus_payload({}, target_year)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payloads, indent=2), encoding="utf-8")
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
