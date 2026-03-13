from __future__ import annotations

import io

import pandas as pd
import requests

from unpaidwork.config import ProjectPaths
from unpaidwork.errors import SourceAccessError
from unpaidwork.storage import read_parquet, write_parquet


CBSA_DELINEATION_URL = (
    "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/"
    "2023/delineation-files/list1_2023.xlsx"
)


def ensure_fips(value: str | int, width: int) -> str:
    return str(value).zfill(width)


def state_from_county(county_fips: str | int) -> str:
    return ensure_fips(county_fips, 5)[:2]


def harmonize_cbsa(value: str | int) -> str:
    return ensure_fips(value, 5)


def _download_cbsa_crosswalk() -> bytes:
    try:
        response = requests.get(CBSA_DELINEATION_URL, timeout=180)
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to fetch CBSA delineation workbook: {exc}") from exc
    if response.status_code >= 400:
        raise SourceAccessError(
            f"CBSA delineation workbook request failed (HTTP {response.status_code})"
        )
    return response.content


def _normalize_cbsa_crosswalk(raw_bytes: bytes) -> pd.DataFrame:
    frame = pd.read_excel(io.BytesIO(raw_bytes), header=2, dtype=str)
    frame = frame.rename(
        columns={
            "CBSA Code": "cbsa_code",
            "CBSA Title": "cbsa_title",
            "Metropolitan/Micropolitan Statistical Area": "cbsa_type",
            "County/County Equivalent": "county_name",
            "State Name": "state_name",
            "FIPS State Code": "state_fips",
            "FIPS County Code": "county_code",
            "Central/Outlying County": "county_role",
        }
    )
    keep = [
        "cbsa_code",
        "cbsa_title",
        "cbsa_type",
        "county_name",
        "state_name",
        "state_fips",
        "county_code",
        "county_role",
    ]
    normalized = frame.loc[:, keep].copy()
    normalized = normalized.loc[normalized["cbsa_code"].notna()].copy()
    normalized["cbsa_code"] = normalized["cbsa_code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)
    normalized["state_fips"] = normalized["state_fips"].astype(str).str.extract(r"(\d+)")[0].str.zfill(2)
    normalized["county_code"] = normalized["county_code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(3)
    normalized["county_fips"] = normalized["state_fips"] + normalized["county_code"]
    normalized = normalized.dropna(subset=["cbsa_code", "county_fips"])
    normalized = normalized.drop_duplicates(subset=["cbsa_code", "county_fips"]).reset_index(drop=True)
    return normalized


def load_cbsa_county_crosswalk(paths: ProjectPaths, refresh: bool = False) -> pd.DataFrame:
    raw_path = paths.raw / "geo" / "cbsa_delineation_2023.xlsx"
    normalized_path = paths.interim / "geo" / "cbsa_county_crosswalk.parquet"
    if normalized_path.exists() and raw_path.exists() and not refresh:
        return read_parquet(normalized_path)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not raw_path.exists():
        raw_path.write_bytes(_download_cbsa_crosswalk())

    normalized = _normalize_cbsa_crosswalk(raw_path.read_bytes())
    write_parquet(normalized, normalized_path)
    return normalized
