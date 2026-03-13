from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from unpaidwork.config import ProjectPaths
from unpaidwork.errors import SourceAccessError
from unpaidwork.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpaidwork.registry import append_registry, build_record
from unpaidwork.sample_data import sipp as sample_sipp
from unpaidwork.storage import write_parquet

SPEC = SourceSpec(
    name="sipp",
    citation="https://www.census.gov/programs-surveys/sipp/data/datasets/2023-data/2023.html",
    license_name="Public data",
    retrieval_method="download",
    landing_page="https://www.census.gov/programs-surveys/sipp/data/datasets/2023-data/2023.html",
)

SIPP_URL = "https://www2.census.gov/programs-surveys/sipp/data/datasets/2023/pu2023_csv.zip"
USECOLS = [
    "SSUID",
    "PNUM",
    "SPANEL",
    "SWAVE",
    "ERP",
    "RANY5",
    "WPFINWGT",
    "EPAY",
    "TPAYWK",
    "EDAYCARE",
    "EFAM",
    "ENREL",
    "EHEADST",
    "ENUR",
]


def _download_sipp_zip(url: str) -> bytes:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "unpaidwork/0.1 (+research repo)"},
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


def _yes_indicator(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.eq(1).astype(float)


def _summarize_subgroup(frame: pd.DataFrame, subgroup: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            [
                {
                    "year": pd.NA,
                    "subgroup": subgroup,
                    "records": 0,
                    "weight_sum": 0.0,
                    "any_paid_childcare_rate": float("nan"),
                    "avg_weekly_paid_childcare_all": float("nan"),
                    "avg_weekly_paid_childcare_payers": float("nan"),
                    "center_care_rate": float("nan"),
                    "family_daycare_rate": float("nan"),
                    "nonrelative_care_rate": float("nan"),
                    "head_start_rate": float("nan"),
                    "nursery_preschool_rate": float("nan"),
                    "geography": "national",
                }
            ]
        )

    rows = []
    for year, group in frame.groupby("year", sort=True):
        weights = pd.to_numeric(group["weight"], errors="coerce").fillna(0.0)
        weekly_spend = pd.to_numeric(group["TPAYWK"], errors="coerce").fillna(0.0)
        payer_mask = _yes_indicator(group["EPAY"]).eq(1.0) | weekly_spend.gt(0)
        payer_weights = weights.where(payer_mask, 0.0)
        rows.append(
            {
                "year": int(year),
                "subgroup": subgroup,
                "records": int(len(group)),
                "weight_sum": float(weights.sum()),
                "any_paid_childcare_rate": _weighted_mean(_yes_indicator(group["EPAY"]), weights),
                "avg_weekly_paid_childcare_all": _weighted_mean(weekly_spend, weights),
                "avg_weekly_paid_childcare_payers": _weighted_mean(weekly_spend, payer_weights),
                "center_care_rate": _weighted_mean(_yes_indicator(group["EDAYCARE"]), weights),
                "family_daycare_rate": _weighted_mean(_yes_indicator(group["EFAM"]), weights),
                "nonrelative_care_rate": _weighted_mean(_yes_indicator(group["ENREL"]), weights),
                "head_start_rate": _weighted_mean(_yes_indicator(group["EHEADST"]), weights),
                "nursery_preschool_rate": _weighted_mean(_yes_indicator(group["ENUR"]), weights),
                "geography": "national",
            }
        )
    return pd.DataFrame(rows)


def _parse_sipp_zip(path: Path, url: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("SIPP zip did not contain a CSV payload")
        with archive.open(csv_names[0]) as handle:
            chunk_iter = pd.read_csv(
                handle,
                sep="|",
                dtype=str,
                chunksize=200_000,
                usecols=lambda name: name in USECOLS,
            )
            parent_chunks: list[pd.DataFrame] = []
            under5_chunks: list[pd.DataFrame] = []
            for chunk in chunk_iter:
                chunk["ERP"] = pd.to_numeric(chunk["ERP"], errors="coerce")
                chunk["RANY5"] = pd.to_numeric(chunk["RANY5"], errors="coerce")
                chunk["SPANEL"] = pd.to_numeric(chunk["SPANEL"], errors="coerce")
                chunk["WPFINWGT"] = pd.to_numeric(chunk["WPFINWGT"], errors="coerce")
                parents = chunk.loc[chunk["ERP"].eq(1) & chunk["SPANEL"].notna()].copy()
                if parents.empty:
                    continue
                parents["year"] = parents["SPANEL"].astype(int)
                parents["weight"] = parents["WPFINWGT"].fillna(0.0)
                parent_chunks.append(parents)
                under5 = parents.loc[parents["RANY5"].eq(1)].copy()
                if not under5.empty:
                    under5_chunks.append(under5)

    all_parents = pd.concat(parent_chunks, ignore_index=True) if parent_chunks else pd.DataFrame()
    under5_parents = pd.concat(under5_chunks, ignore_index=True) if under5_chunks else pd.DataFrame()

    normalized = pd.concat(
        [
            _summarize_subgroup(all_parents, "all_ref_parents"),
            _summarize_subgroup(under5_parents, "under5_ref_parents"),
        ],
        ignore_index=True,
    )
    normalized["source_url"] = url
    return normalized.dropna(subset=["year"]).sort_values(["year", "subgroup"]).reset_index(drop=True)


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_sipp, refresh=refresh, dry_run=dry_run)

    raw_path = paths.raw / SPEC.name / Path(SIPP_URL).name
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"
    if dry_run:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=SIPP_URL)
    if raw_path.exists() and normalized_path.exists() and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(_download_sipp_zip(SIPP_URL))
    normalized = _parse_sipp_zip(raw_path, SIPP_URL)
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
