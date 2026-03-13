from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from unpaidwork.config import ProjectPaths
from unpaidwork.errors import SourceAccessError
from unpaidwork.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpaidwork.registry import append_registry, build_record
from unpaidwork.storage import write_parquet

SPEC = SourceSpec(
    name="cbp",
    citation="https://www.census.gov/data/developers/data-sets/cbp-zbp/cbp-api.html",
    license_name="Public data",
    retrieval_method="api",
    landing_page="https://www.census.gov/data/developers/data-sets/cbp-zbp/cbp-api.html",
)


def _sample_cbp() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"county_fips": "06037", "state_fips": "06", "year": 2021, "employer_establishments": 430, "employer_employment": 53000.0, "employer_annual_payroll": 1522000000.0, "legal_form_code": "001", "source_url": "https://api.census.gov/data/2021/cbp"},
            {"county_fips": "06073", "state_fips": "06", "year": 2021, "employer_establishments": 164, "employer_employment": 17500.0, "employer_annual_payroll": 482000000.0, "legal_form_code": "001", "source_url": "https://api.census.gov/data/2021/cbp"},
            {"county_fips": "48113", "state_fips": "48", "year": 2021, "employer_establishments": 188, "employer_employment": 21500.0, "employer_annual_payroll": 451000000.0, "legal_form_code": "001", "source_url": "https://api.census.gov/data/2021/cbp"},
            {"county_fips": "48201", "state_fips": "48", "year": 2021, "employer_establishments": 245, "employer_employment": 25500.0, "employer_annual_payroll": 491000000.0, "legal_form_code": "001", "source_url": "https://api.census.gov/data/2021/cbp"},
            {"county_fips": "36061", "state_fips": "36", "year": 2021, "employer_establishments": 212, "employer_employment": 29100.0, "employer_annual_payroll": 689000000.0, "legal_form_code": "001", "source_url": "https://api.census.gov/data/2021/cbp"},
            {"county_fips": "36029", "state_fips": "36", "year": 2021, "employer_establishments": 42, "employer_employment": 9100.0, "employer_annual_payroll": 171000000.0, "legal_form_code": "001", "source_url": "https://api.census.gov/data/2021/cbp"},
        ]
    )


def _normalize_cbp_rows(rows: list[list[str]], year: int, source_url: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "county_fips",
                "state_fips",
                "year",
                "employer_establishments",
                "employer_employment",
                "employer_annual_payroll",
                "legal_form_code",
                "source_url",
            ]
        )

    header = rows[0]
    lookup = {name: idx for idx, name in enumerate(header)}
    required = {"ESTAB", "EMP", "PAYANN", "LFO", "state", "county"}
    missing = sorted(required - set(lookup))
    if missing:
        raise ValueError(f"CBP API response missing required columns: {', '.join(missing)}")

    records: list[dict[str, object]] = []
    for row in rows[1:]:
        state = str(row[lookup["state"]]).strip().zfill(2)
        county = str(row[lookup["county"]]).strip().zfill(3)
        if not state or not county:
            continue
        records.append(
            {
                "county_fips": f"{state}{county}",
                "state_fips": state,
                "year": int(year),
                "employer_establishments": pd.to_numeric(row[lookup["ESTAB"]], errors="coerce"),
                "employer_employment": pd.to_numeric(row[lookup["EMP"]], errors="coerce"),
                "employer_annual_payroll": pd.to_numeric(row[lookup["PAYANN"]], errors="coerce")
                * 1000.0,
                "legal_form_code": str(row[lookup["LFO"]]).strip(),
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
                "employer_establishments",
                "employer_employment",
                "employer_annual_payroll",
                "legal_form_code",
                "source_url",
            ]
        )

    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    return frame.sort_values(["county_fips", "year"], kind="stable").reset_index(drop=True)


def _fetch_cbp_rows(year: int) -> tuple[list[list[str]], str]:
    url = f"https://api.census.gov/data/{year}/cbp"
    params = {
        "get": "ESTAB,EMP,PAYANN,LFO,NAICS2017_LABEL",
        "NAICS2017": "624410",
        "LFO": "001",
        "for": "county:*",
    }
    try:
        response = requests.get(url, params=params, timeout=180)
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to fetch CBP API for year {year}: {exc}") from exc
    if response.status_code >= 400:
        raise SourceAccessError(f"CBP API request failed for year {year} (HTTP {response.status_code})")
    rows = response.json()
    if len(rows) <= 1:
        raise SourceAccessError(f"CBP API returned no county rows for NAICS 624410 in year {year}")
    return rows, response.url


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, _sample_cbp, refresh=refresh, dry_run=dry_run)

    target_year = int(year or 2022)
    raw_path = paths.raw / SPEC.name / f"cbp_{target_year}.json"
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"

    if dry_run:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=str(target_year))
    if raw_path.exists() and normalized_path.exists() and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")

    rows, source_url = _fetch_cbp_rows(target_year)
    frame = _normalize_cbp_rows(rows, target_year, source_url)

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
