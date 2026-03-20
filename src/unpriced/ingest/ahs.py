from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from unpriced.config import ProjectPaths
from unpriced.ingest.common import IngestResult, SourceSpec, ingest_remote_manifest, ingest_sample
from unpriced.sample_data import ahs as sample_ahs

SPEC = SourceSpec(
    name="ahs",
    citation="https://www.census.gov/programs-surveys/ahs/data/2023/ahs-2023-public-use-file--puf-.html",
    license_name="Public data",
    retrieval_method="download",
    landing_page="https://www.census.gov/programs-surveys/ahs/data/2023/ahs-2023-public-use-file--puf-.html",
)

JOB_TYPE_LABELS = {
    "01": "Earthquake damage required extensive repairs to home",
    "02": "Tornado or hurricane damage required extensive repairs to home",
    "03": "Landslide damage required extensive repairs to home",
    "04": "Fire damage required extensive repairs to home",
    "05": "Flood damage required extensive repairs to home",
    "06": "Other natural disaster damage required extensive repairs to home",
    "07": "Added bedroom or converted existing space to bedroom",
    "08": "Added bathroom or converted existing space to bathroom",
    "09": "Added recreation room or converted existing space to recreation room",
    "10": "Added kitchen or converted existing space to kitchen",
    "11": "Added other room or converted existing space to other room",
    "12": "Remodeled bathroom",
    "13": "Remodeled kitchen",
    "14": "Added attached garage or carport",
    "15": "Added porch, deck, patio, or terrace",
    "16": "Added or replaced roofing",
    "17": "Added or replaced siding",
    "18": "Added or replaced doors or windows",
    "19": "Added or replaced chimney, stairs or other part of exterior",
    "20": "Added or replaced insulation",
    "21": "Added or replaced interior water pipes",
    "22": "Added or replaced plumbing fixtures",
    "23": "Added or replaced electrical wiring, fuse boxes, breaker panels, or built-in lighting",
    "24": "Added or replaced security system",
    "25": "Installed carpeting, flooring, paneling, ceiling tiles or drywall",
    "26": "Added or replaced central air conditioning",
    "27": "Added or replaced built-in heating equipment",
    "28": "Added or replaced septic tank",
    "29": "Added or replaced water heater",
    "30": "Added or replaced built-in dishwasher or garbage disposal",
    "31": "Other major improvements or repairs inside home",
    "32": "Added or replaced driveways or walkways",
    "33": "Added or replaced fencing or walls",
    "34": "Added or replaced swimming pool, tennis court, or other recreational structure",
    "35": "Added or replaced shed, detached garage, or other building",
    "36": "Added or replaced landscaping or sprinkler system",
    "37": "Other major improvements or repairs to lot or yard",
}

JOB_TYPE_GROUPS = {
    "01": "disaster_repair",
    "02": "disaster_repair",
    "03": "disaster_repair",
    "04": "disaster_repair",
    "05": "disaster_repair",
    "06": "disaster_repair",
    "07": "room_addition",
    "08": "room_addition",
    "09": "room_addition",
    "10": "room_addition",
    "11": "room_addition",
    "12": "kitchen_bath_remodel",
    "13": "kitchen_bath_remodel",
    "14": "outdoor_structure_addition",
    "15": "outdoor_structure_addition",
    "16": "exterior_envelope",
    "17": "exterior_envelope",
    "18": "exterior_envelope",
    "19": "exterior_envelope",
    "20": "exterior_envelope",
    "21": "systems_and_fixtures",
    "22": "systems_and_fixtures",
    "23": "systems_and_fixtures",
    "24": "systems_and_fixtures",
    "25": "interior_finish_and_appliances",
    "26": "interior_finish_and_appliances",
    "27": "interior_finish_and_appliances",
    "28": "systems_and_fixtures",
    "29": "systems_and_fixtures",
    "30": "interior_finish_and_appliances",
    "31": "other_interior_improvement",
    "32": "lot_yard_and_outbuildings",
    "33": "lot_yard_and_outbuildings",
    "34": "lot_yard_and_outbuildings",
    "35": "lot_yard_and_outbuildings",
    "36": "lot_yard_and_outbuildings",
    "37": "lot_yard_and_outbuildings",
}


def _clean_string(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.strip("'").replace({"nan": pd.NA, "None": pd.NA})


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(_clean_string(series), errors="coerce")


def _job_type_code(series: pd.Series) -> pd.Series:
    numeric = _to_numeric(series).fillna(-1).astype(int)
    return numeric.map(lambda value: f"{value:02d}" if value >= 0 else "00")


def _job_type_label(series: pd.Series) -> pd.Series:
    codes = _job_type_code(series)
    return codes.map(JOB_TYPE_LABELS).fillna("Unknown major improvement or repair")


def _job_type_group(series: pd.Series) -> pd.Series:
    codes = _job_type_code(series)
    return codes.map(JOB_TYPE_GROUPS).fillna("other_major_improvement")


def _parse_ahs_archive(path: Path, _: str = "") -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        with archive.open("project.csv") as handle:
            project = pd.read_csv(handle, dtype=str)
        with archive.open("household.csv") as handle:
            household = pd.read_csv(handle, dtype=str)

    project["CONTROL"] = _clean_string(project["CONTROL"])
    household["CONTROL"] = _clean_string(household["CONTROL"])

    merged = project.merge(
        household[
            [
                "CONTROL",
                "OMB13CBSA",
                "WEIGHT",
                "JMARKETVAL",
                "HINCP",
                "FINCP",
                "JTENURE",
                "YRBUILT",
            ]
        ],
        on="CONTROL",
        how="left",
    )

    merged["job_cost"] = _to_numeric(merged["JOBCOST"])
    merged = merged.loc[merged["job_cost"].gt(0)].copy()
    merged["job_diy"] = _to_numeric(merged["JOBDIY"]).eq(1).astype(int)
    merged["year"] = 2023
    merged["job_type_code"] = _job_type_code(merged["JOBTYPE"])
    merged["job_type"] = _job_type_label(merged["JOBTYPE"])
    merged["job_group"] = _job_type_group(merged["JOBTYPE"])
    merged["cbsa_code"] = _clean_string(merged["OMB13CBSA"]).str.replace(r"\.0$", "", regex=True)
    merged["cbsa_code"] = merged["cbsa_code"].where(
        merged["cbsa_code"].notna() & merged["cbsa_code"].ne(""), "99999"
    )
    merged["cbsa_code"] = merged["cbsa_code"].astype(str).str.zfill(5)
    merged["weight"] = _to_numeric(merged["WEIGHT"]).fillna(1.0)
    merged["housing_vintage"] = _to_numeric(merged["YRBUILT"])
    merged["home_value"] = _to_numeric(merged["JMARKETVAL"])
    merged["household_income"] = _to_numeric(merged["HINCP"]).fillna(_to_numeric(merged["FINCP"]))
    merged["tenure_owner"] = _to_numeric(merged["JTENURE"]).eq(1).astype(int)
    merged["storm_exposure"] = 0.0
    merged["job_id"] = (
        merged["CONTROL"].fillna("missing")
        + "-"
        + _clean_string(merged["JOBTYPE"]).fillna("unk")
        + "-"
        + (merged.groupby("CONTROL").cumcount() + 1).astype(str)
    )

    normalized = merged[
        [
            "job_id",
            "cbsa_code",
            "year",
            "job_type_code",
            "job_type",
            "job_group",
            "job_cost",
            "job_diy",
            "weight",
            "housing_vintage",
            "home_value",
            "household_income",
            "tenure_owner",
            "storm_exposure",
        ]
    ].reset_index(drop=True)
    return normalized


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_ahs, refresh=refresh, dry_run=dry_run)
    return ingest_remote_manifest(
        paths,
        SPEC,
        "https://www2.census.gov/programs-surveys/ahs/2023/AHS%202023%20National%20PUF%20v1.1%20CSV.zip",
        refresh=refresh,
        dry_run=dry_run,
        parser=_parse_ahs_archive,
    )
