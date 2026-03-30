from __future__ import annotations

import io
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import requests

from unpriced.config import ProjectPaths
from unpriced.errors import SourceAccessError
from unpriced.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpriced.registry import append_registry, build_record
from unpriced.sample_data import oews as sample_oews
from unpriced.storage import write_parquet

SPEC = SourceSpec(
    name="oews",
    citation="https://www.bls.gov/oes/oes_emp.htm",
    license_name="Public data",
    retrieval_method="download",
    landing_page="https://www.bls.gov/oes/oes_emp.htm",
)

CHILDCARE_OCC_CODE = "39-9011"
PRESCHOOL_TEACHER_OCC_CODE = "25-2011"
OUTSIDE_OPTION_OCC_CODE = "35-0000"
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
REQUIRED_NORMALIZED_COLUMNS = {
    "state_fips",
    "year",
    "oews_childcare_worker_wage",
    "oews_preschool_teacher_wage",
    "oews_outside_option_wage",
}


def _zip_url(year: int) -> str:
    yy = str(year)[-2:]
    return f"https://www.bls.gov/oes/special-requests/oesm{yy}st.zip"


def _download_oews_zip(url: str) -> bytes:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "unpriced/0.1 (+research repo)"},
            timeout=240,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to fetch {url}: {exc}") from exc
    return response.content


def _xlsx_name(names: list[str]) -> str:
    candidates = [
        name for name in names if name.lower().endswith(".xlsx") and not Path(name).name.startswith("~$")
    ]
    if not candidates:
        raise ValueError("OEWS zip did not contain the expected Excel workbook")
    return candidates[0]


def _normalize_oews_zip(path: Path, url: str, year: int) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        workbook_name = _xlsx_name(archive.namelist())
        with archive.open(workbook_name) as handle:
            frame = pd.read_excel(io.BytesIO(handle.read()))

    data = frame.copy()
    data["AREA_TYPE"] = pd.to_numeric(data["AREA_TYPE"], errors="coerce")
    data["state_fips"] = data["PRIM_STATE"].astype(str).str.upper().map(STATE_ABBREV_TO_FIPS)
    data["hourly_mean"] = pd.to_numeric(data["H_MEAN"], errors="coerce")
    data = data.loc[data["AREA_TYPE"].eq(2) & data["state_fips"].notna()].copy()

    childcare = data.loc[data["OCC_CODE"].astype(str).eq(CHILDCARE_OCC_CODE), ["state_fips", "hourly_mean"]]
    childcare = childcare.rename(columns={"hourly_mean": "oews_childcare_worker_wage"})

    preschool = data.loc[
        data["OCC_CODE"].astype(str).eq(PRESCHOOL_TEACHER_OCC_CODE),
        ["state_fips", "hourly_mean"],
    ]
    preschool = preschool.rename(columns={"hourly_mean": "oews_preschool_teacher_wage"})

    outside = data.loc[data["OCC_CODE"].astype(str).eq(OUTSIDE_OPTION_OCC_CODE), ["state_fips", "hourly_mean"]]
    outside = outside.rename(columns={"hourly_mean": "oews_outside_option_wage"})

    merged = childcare.merge(preschool, on="state_fips", how="outer")
    merged = merged.merge(outside, on="state_fips", how="outer")
    merged["year"] = int(year)
    merged["source_url"] = url
    return merged.sort_values("state_fips").reset_index(drop=True)


def _quoted(path: Path) -> str:
    return str(path).replace("'", "''")


def _has_required_schema(normalized_path: Path) -> bool:
    if not normalized_path.exists():
        return False
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_quoted(normalized_path)}')"
        ).fetchall()
    except Exception:
        return False
    finally:
        con.close()
    columns = {row[0] for row in rows}
    return REQUIRED_NORMALIZED_COLUMNS.issubset(columns)


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_oews, refresh=refresh, dry_run=dry_run)

    target_year = year or 2024
    url = _zip_url(target_year)
    raw_path = paths.raw / SPEC.name / Path(url).name
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"
    if dry_run:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=url)
    if raw_path.exists() and normalized_path.exists() and _has_required_schema(normalized_path) and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(_download_oews_zip(url))
    normalized = _normalize_oews_zip(raw_path, url, target_year)
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
