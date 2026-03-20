from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from unpriced.config import ProjectPaths
from unpriced.errors import SourceAccessError
from unpriced.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpriced.registry import append_registry, build_record
from unpriced.sample_data import ce as sample_ce
from unpriced.storage import write_parquet

SPEC = SourceSpec(
    name="ce",
    citation="https://www.bls.gov/cex/pumd.htm",
    license_name="Public data",
    retrieval_method="download",
    landing_page="https://www.bls.gov/cex/pumd.htm",
)

CE_URL = "https://www.bls.gov/cex/pumd/data/csv/intrvw23.zip"
FMLI_FILES = [
    "intrvw23/fmli232.csv",
    "intrvw23/fmli233.csv",
    "intrvw23/fmli234.csv",
    "intrvw23/fmli241.csv",
]
USECOLS = [
    "NEWID",
    "FINLWT21",
    "PERSLT18",
    "CHILDAGE",
    "QINTRVYR",
    "BBYDAYPQ",
    "TOTEXPPQ",
]


def _download_ce_zip(url: str) -> bytes:
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


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    valid = series.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(series.loc[valid], weights=weights.loc[valid]))


def _summarize(frame: pd.DataFrame, subgroup: str, mask: pd.Series) -> pd.DataFrame:
    subset = frame.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame()
    rows = []
    for year, group in subset.groupby("year", sort=True):
        weights = pd.to_numeric(group["weight"], errors="coerce").fillna(0.0)
        spend = pd.to_numeric(group["BBYDAYPQ"], errors="coerce").fillna(0.0)
        total = pd.to_numeric(group["TOTEXPPQ"], errors="coerce").fillna(0.0)
        payer = spend.gt(0)
        payer_weights = weights.where(payer, 0.0)
        valid_share = total.gt(0)
        rows.append(
            {
                "year": int(year),
                "subgroup": subgroup,
                "records": int(len(group)),
                "weight_sum": float(weights.sum()),
                "childcare_spender_rate": _weighted_mean(payer.astype(float), weights),
                "avg_childcare_spend_pq_all": _weighted_mean(spend, weights),
                "avg_childcare_spend_pq_payers": _weighted_mean(spend, payer_weights),
                "childcare_spend_share_pq": _weighted_mean(
                    spend.where(valid_share, np.nan) / total.where(valid_share, np.nan),
                    weights.where(valid_share, 0.0),
                ),
                "geography": "national",
            }
        )
    return pd.DataFrame(rows)


def _parse_ce_zip(path: Path, url: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        for name in FMLI_FILES:
            with archive.open(name) as handle:
                frame = pd.read_csv(handle, usecols=USECOLS, dtype=str)
            frame["year"] = pd.to_numeric(frame["QINTRVYR"], errors="coerce").astype("Int64")
            frame["weight"] = pd.to_numeric(frame["FINLWT21"], errors="coerce")
            frame["PERSLT18"] = pd.to_numeric(frame["PERSLT18"], errors="coerce")
            frame["CHILDAGE"] = pd.to_numeric(frame["CHILDAGE"], errors="coerce")
            frame["BBYDAYPQ"] = pd.to_numeric(frame["BBYDAYPQ"], errors="coerce")
            frame["TOTEXPPQ"] = pd.to_numeric(frame["TOTEXPPQ"], errors="coerce")
            frames.append(frame)

    all_frame = pd.concat(frames, ignore_index=True)
    normalized = pd.concat(
        [
            _summarize(all_frame, "with_children_u18", all_frame["PERSLT18"].fillna(0).gt(0)),
            _summarize(
                all_frame,
                "with_child_age_1_5",
                all_frame["CHILDAGE"].fillna(0).between(1, 5),
            ),
        ],
        ignore_index=True,
    )
    normalized["source_url"] = url
    return normalized.sort_values(["year", "subgroup"]).reset_index(drop=True)


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_ce, refresh=refresh, dry_run=dry_run)

    raw_path = paths.raw / SPEC.name / Path(CE_URL).name
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"
    if dry_run:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=CE_URL)
    if raw_path.exists() and normalized_path.exists() and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(_download_ce_zip(CE_URL))
    normalized = _parse_ce_zip(raw_path, CE_URL)
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
