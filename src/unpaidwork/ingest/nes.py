from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests

from unpaidwork.config import ProjectPaths
from unpaidwork.errors import SourceAccessError
from unpaidwork.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpaidwork.registry import append_registry, build_record
from unpaidwork.sample_data import nes as sample_nes
from unpaidwork.storage import write_parquet

SPEC = SourceSpec(
    name="nes",
    citation="https://www.census.gov/data/developers/data-sets/nonemp-api.html",
    license_name="Public data",
    retrieval_method="api",
    landing_page="https://www.census.gov/data/developers/data-sets/nonemp-api.html",
)

NAICS_PRIMARY = "624410"
NAICS_FALLBACK = "62441"


def _naics_field(year: int) -> str:
    return "NAICS2017" if year <= 2021 else "NAICS2022"


def _normalize_nes_rows(rows: list[list[str]], year: int, source_url: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "county_fips",
                "state_fips",
                "year",
                "nonemployer_firms",
                "receipts",
                "source_url",
            ]
        )

    header = rows[0]
    lookup = {name: idx for idx, name in enumerate(header)}
    required = {"STATE", "COUNTY", "NESTAB", "NRCPTOT"}
    missing = sorted(required - set(lookup))
    if missing:
        raise ValueError(f"NES API response missing required columns: {', '.join(missing)}")

    records: list[dict[str, object]] = []
    for row in rows[1:]:
        state = str(row[lookup["STATE"]]).strip().zfill(2)
        county = str(row[lookup["COUNTY"]]).strip().zfill(3)
        if not state or not county:
            continue

        firms = pd.to_numeric(row[lookup["NESTAB"]], errors="coerce")
        receipts_thousands = pd.to_numeric(row[lookup["NRCPTOT"]], errors="coerce")
        records.append(
            {
                "county_fips": f"{state}{county}",
                "state_fips": state,
                "year": int(year),
                "nonemployer_firms": firms,
                "receipts": receipts_thousands * 1000.0,
                "source_url": source_url,
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "county_fips",
                "state_fips",
                "year",
                "nonemployer_firms",
                "receipts",
                "source_url",
            ]
        )

    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["nonemployer_firms"] = pd.to_numeric(frame["nonemployer_firms"], errors="coerce")
    frame["receipts"] = pd.to_numeric(frame["receipts"], errors="coerce")
    return frame.sort_values(["county_fips", "year"], kind="stable").reset_index(drop=True)


def _fetch_nes_rows(year: int) -> tuple[list[list[str]], str]:
    base_url = f"https://api.census.gov/data/{year}/nonemp"
    naics_field = _naics_field(year)
    common_params = {
        "get": f"STATE,COUNTY,{naics_field},LFO,RCPSZES,NESTAB,NRCPTOT",
        "for": "county:*",
        "in": "state:*",
        "LFO": "001",
        "RCPSZES": "001",
    }
    api_key = os.getenv("CENSUS_API_KEY")
    if api_key:
        common_params["key"] = api_key

    for naics in (NAICS_PRIMARY, NAICS_FALLBACK):
        params = dict(common_params)
        params[naics_field] = naics
        try:
            response = requests.get(base_url, params=params, timeout=180)
        except requests.RequestException as exc:
            raise SourceAccessError(f"failed to fetch NES API for year {year}: {exc}") from exc

        if response.status_code == 204 or not (response.text or "").strip():
            continue
        if response.status_code >= 400:
            raise SourceAccessError(
                f"NES API request failed for year {year} (HTTP {response.status_code})"
            )

        rows = response.json()
        if len(rows) > 1:
            return rows, response.url

    raise SourceAccessError(
        f"NES API returned no rows for {naics_field}={NAICS_PRIMARY} or fallback {NAICS_FALLBACK} in year {year}"
    )


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_nes, refresh=refresh, dry_run=dry_run)

    target_year = int(year or 2022)
    raw_path = paths.raw / SPEC.name / f"nes_{target_year}.json"
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"

    if dry_run:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=str(target_year))
    if raw_path.exists() and normalized_path.exists() and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")

    rows, source_url = _fetch_nes_rows(target_year)
    frame = _normalize_nes_rows(rows, target_year, source_url)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_payload = {
        "source": SPEC.name,
        "year": target_year,
        "source_url": source_url,
        "rows": rows,
    }
    raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    write_parquet(frame, normalized_path)
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
