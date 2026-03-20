from __future__ import annotations

from pathlib import Path

import pandas as pd

from unpriced.config import ProjectPaths
from unpriced.errors import SourceAccessError
from unpriced.ingest.common import (
    IngestResult,
    SourceSpec,
    ingest_sample,
    require_manual_source_path,
)
from unpriced.registry import append_registry, build_record
from unpriced.sample_data import licensing_supply_shocks as sample_licensing_supply_shocks
from unpriced.storage import write_parquet

SPEC = SourceSpec(
    name="licensing",
    citation="https://licensingregulations.acf.hhs.gov/",
    license_name="Public data / manual curation required",
    retrieval_method="manual-curation",
    landing_page="https://licensingregulations.acf.hhs.gov/",
)

RAW_FILENAME = "licensing_supply_shocks.csv"
RAW_RULES_FILENAME = "licensing_rules_long.csv"
REQUIRED_COLUMNS = {"state_fips", "year"}
RULE_LEVEL_REQUIRED_COLUMNS = {"state_fips", "year", "provider_type", "age_group", "rule_family"}
ICPSR_2017_TSV = Path("icpsr_2017/ICPSR_37700/DS0001/37700-0001-Data.tsv")
ICPSR_2020_TSV = Path("icpsr_2020/ICPSR_38539/DS0001/38539-0001-Data.tsv")
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
ICPSR_2017_RULE_SPECS = (
    {
        "column": "C_CHR_ALLSTAFF",
        "provider_type": "center",
        "age_group": "all",
        "rule_family": "criminal_history_check_all_staff",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Criminal History Records Check Required for All Staff",
    },
    {
        "column": "F_CHR_ALLSTAFF",
        "provider_type": "family_home",
        "age_group": "all",
        "rule_family": "criminal_history_check_all_staff",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Criminal History Records Check Required for All Staff",
    },
    {
        "column": "LF_CHR_ALLSTAFF",
        "provider_type": "large_group_home",
        "age_group": "all",
        "rule_family": "criminal_history_check_all_staff",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Criminal History Records Check Required for All Staff",
    },
    {
        "column": "C_FINGERPRT_ALLSTAFF",
        "provider_type": "center",
        "age_group": "all",
        "rule_family": "fingerprint_check_all_staff",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Fingerprint Check Required for All Staff",
    },
    {
        "column": "F_FINGERPRT_ALLSTAFF",
        "provider_type": "family_home",
        "age_group": "all",
        "rule_family": "fingerprint_check_all_staff",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Fingerprint Check Required for All Staff",
    },
    {
        "column": "LF_FINGERPRT_ALLSTAFF",
        "provider_type": "large_group_home",
        "age_group": "all",
        "rule_family": "fingerprint_check_all_staff",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Fingerprint Check Required for All Staff",
    },
    {
        "column": "C_CAN_ALLSTAFF",
        "provider_type": "center",
        "age_group": "all",
        "rule_family": "child_abuse_registry_check_all_staff",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Child Abuse and Neglect Registry Check Required for All Staff",
    },
    {
        "column": "F_CAN_ALLSTAFF",
        "provider_type": "family_home",
        "age_group": "all",
        "rule_family": "child_abuse_registry_check_all_staff",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Child Abuse and Neglect Registry Check Required for All Staff",
    },
    {
        "column": "LF_CAN_ALLSTAFF",
        "provider_type": "large_group_home",
        "age_group": "all",
        "rule_family": "child_abuse_registry_check_all_staff",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Child Abuse and Neglect Registry Check Required for All Staff",
    },
    {
        "column": "C_SX_ALLSTAFF",
        "provider_type": "center",
        "age_group": "all",
        "rule_family": "sex_offender_registry_check_all_staff",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Sex Offender Registry Check Required for All Staff",
    },
    {
        "column": "F_SX_ALLSTAFF",
        "provider_type": "family_home",
        "age_group": "all",
        "rule_family": "sex_offender_registry_check_all_staff",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Sex Offender Registry Check Required for All Staff",
    },
    {
        "column": "LF_SX_ALLSTAFF",
        "provider_type": "large_group_home",
        "age_group": "all",
        "rule_family": "sex_offender_registry_check_all_staff",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Sex Offender Registry Check Required for All Staff",
    },
)
ICPSR_2020_RULE_SPECS = (
    {
        "column": "PQ_T_MINAGE",
        "provider_type": "center",
        "age_group": "all",
        "rule_family": "teacher_min_age",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Teacher Minimum Age",
    },
    {
        "column": "D_R_INFTOD",
        "provider_type": "center",
        "age_group": "infant_toddler",
        "rule_family": "separate_infant_toddler_regulations",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Separate Regulations for Infant and Toddler Care Programs",
    },
    {
        "column": "PS_MX_RATIO_YN",
        "provider_type": "center",
        "age_group": "mixed_age",
        "rule_family": "mixed_age_ratio_requirement",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Child-staff Ratio Requirements for Mixed-age Groups",
    },
    {
        "column": "PS_CNTRSIZE",
        "provider_type": "center",
        "age_group": "all",
        "rule_family": "supervision_based_on_center_size",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Supervision Requirements Based on Size of Center",
    },
    {
        "column": "PS_SIZERATIO",
        "provider_type": "center",
        "age_group": "all",
        "rule_family": "ratio_based_on_center_size",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Child-staff Ratios Based on Size of Center",
    },
    {
        "column": "PS_SIZEGRPSZ",
        "provider_type": "center",
        "age_group": "all",
        "rule_family": "group_size_based_on_center_size",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Group Sizes Based on Size of Center",
    },
    {
        "column": "PS_LGGROUP",
        "provider_type": "center",
        "age_group": "all",
        "rule_family": "large_group_activity_supervision",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Supervision Requirements for Large Group Activities - General",
    },
    {
        "column": "PS_GSEXCEED",
        "provider_type": "center",
        "age_group": "all",
        "rule_family": "allowed_to_exceed_group_sizes",
        "strictness_direction": "lower_is_stricter",
        "source_note": "Centers Allowed to Legally Exceed Group Sizes",
    },
    {
        "column": "SPND_CAPACITY",
        "provider_type": "center",
        "age_group": "special_needs",
        "rule_family": "special_needs_capacity_adjustment",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Adjust Center Capacity",
    },
    {
        "column": "SPND_RATIO",
        "provider_type": "center",
        "age_group": "special_needs",
        "rule_family": "special_needs_ratio_requirement",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Ratio Requirement - Special Needs",
    },
    {
        "column": "SPND_GROUPSIZE",
        "provider_type": "center",
        "age_group": "special_needs",
        "rule_family": "special_needs_group_size_requirement",
        "strictness_direction": "higher_is_stricter",
        "source_note": "Group Size Requirement - Special Needs",
    },
)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "state_fips" in data.columns:
        data["state_fips"] = data["state_fips"].astype(str).str.zfill(2)
    if "year" in data.columns:
        data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    for column in (
        "center_labor_intensity_index",
        "center_infant_ratio",
        "center_toddler_ratio",
        "center_infant_group_size",
        "center_toddler_group_size",
    ):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    keep = [
        column
        for column in (
            "state_fips",
            "year",
            "center_labor_intensity_index",
            "center_infant_ratio",
            "center_toddler_ratio",
            "center_infant_group_size",
            "center_toddler_group_size",
            "shock_label",
            "effective_date",
            "source_url",
            "source_note",
        )
        if column in data.columns
    ]
    normalized = data[keep].dropna(subset=["state_fips", "year"]).drop_duplicates(["state_fips", "year"])
    return normalized.sort_values(["state_fips", "year"]).reset_index(drop=True)


def _normalize_rule_level(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(RULE_LEVEL_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "Licensing rule-level CSV missing required columns: " + ", ".join(missing)
        )

    data = frame.copy()
    data["state_fips"] = data["state_fips"].astype(str).str.zfill(2)
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    for column in (
        "provider_type",
        "age_group",
        "rule_family",
        "rule_id",
        "rule_name",
        "rule_column_source",
        "strictness_direction",
        "shock_label",
        "effective_date",
        "source_url",
        "source_note",
    ):
        if column in data.columns:
            data[column] = data[column].astype(str)
    if "rule_value" in data.columns:
        data["rule_value"] = pd.to_numeric(data["rule_value"], errors="coerce")
    elif "rule_value_observed" in data.columns:
        data["rule_value"] = pd.to_numeric(data["rule_value_observed"], errors="coerce")
    elif "value_numeric" in data.columns:
        data["rule_value"] = pd.to_numeric(data["value_numeric"], errors="coerce")
    elif "value_text" in data.columns:
        data["rule_value"] = pd.to_numeric(data["value_text"], errors="coerce")
    else:
        data["rule_value"] = pd.NA
    if "rule_value_observed" in data.columns:
        data["rule_value_observed"] = pd.to_numeric(data["rule_value_observed"], errors="coerce")
    else:
        data["rule_value_observed"] = pd.to_numeric(data["rule_value"], errors="coerce")
    if "rule_column_source" not in data.columns:
        data["rule_column_source"] = data.get("rule_name", pd.Series("rule_value", index=data.index)).astype(str)
    keep = [
        column
        for column in (
            "state_fips",
            "year",
            "provider_type",
            "age_group",
            "rule_family",
            "rule_id",
            "rule_name",
            "rule_column_source",
            "strictness_direction",
            "rule_value",
            "rule_value_observed",
            "shock_label",
            "effective_date",
            "source_url",
            "source_note",
        )
        if column in data.columns
    ]
    normalized = (
        data[keep]
        .dropna(subset=["state_fips", "year", "provider_type", "age_group", "rule_family"])
        .drop_duplicates(
            ["state_fips", "year", "provider_type", "age_group", "rule_family", "rule_column_source"],
            keep="last",
        )
    )
    return normalized.sort_values(
        ["state_fips", "year", "provider_type", "age_group", "rule_family", "rule_column_source"],
        kind="stable",
    ).reset_index(drop=True)


def _coerce_state_fips(value: object) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if text in STATE_ABBREV_TO_FIPS:
        return STATE_ABBREV_TO_FIPS[text]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(2)


def _extract_icpsr_rule_rows(frame: pd.DataFrame, *, year: int, source_url: str, specs: tuple[dict[str, str], ...]) -> pd.DataFrame:
    data = frame.copy()
    data["state_fips"] = data["STATE"].map(_coerce_state_fips)
    data = data.dropna(subset=["state_fips"]).copy()
    records: list[dict[str, object]] = []
    for _, row in data.iterrows():
        for spec in specs:
            raw_value = row.get(spec["column"])
            value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            records.append(
                {
                    "state_fips": str(row["state_fips"]).zfill(2),
                    "year": year,
                    "provider_type": spec["provider_type"],
                    "age_group": spec["age_group"],
                    "rule_family": spec["rule_family"],
                    "rule_name": spec["column"].lower(),
                    "rule_column_source": spec["column"].lower(),
                    "strictness_direction": spec["strictness_direction"],
                    "rule_value": float(value),
                    "shock_label": f"icpsr_{year}",
                    "effective_date": str(year),
                    "source_url": source_url,
                    "source_note": spec["source_note"],
                }
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


def _build_rule_level_from_icpsr(paths: ProjectPaths) -> tuple[pd.DataFrame, list[Path]]:
    licensing_root = paths.raw / SPEC.name
    source_files: list[Path] = []
    pieces: list[pd.DataFrame] = []

    raw_2017_path = licensing_root / ICPSR_2017_TSV
    if raw_2017_path.exists():
        source_files.append(raw_2017_path)
        frame_2017 = pd.read_csv(raw_2017_path, sep="\t")
        pieces.append(
            _extract_icpsr_rule_rows(
                frame_2017,
                year=2017,
                source_url="https://www.childandfamilydataarchive.org/cfda/archives/cfda/studies/37700",
                specs=ICPSR_2017_RULE_SPECS,
            )
        )

    raw_2020_path = licensing_root / ICPSR_2020_TSV
    if raw_2020_path.exists():
        source_files.append(raw_2020_path)
        frame_2020 = pd.read_csv(raw_2020_path, sep="\t")
        pieces.append(
            _extract_icpsr_rule_rows(
                frame_2020,
                year=2020,
                source_url="https://www.childandfamilydataarchive.org/cfda/archives/cfda/studies/38539",
                specs=ICPSR_2020_RULE_SPECS,
            )
        )

    if not pieces:
        return pd.DataFrame(), []
    combined = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame(), source_files
    return _normalize_rule_level(combined), source_files


def _append_registry_record(
    paths: ProjectPaths,
    *,
    source_name: str,
    raw_path: Path,
    normalized_path: Path,
) -> None:
    append_registry(
        paths,
        build_record(
            source_name=source_name,
            raw_path=raw_path,
            normalized_path=normalized_path,
            license_name=SPEC.license_name,
            retrieval_method=SPEC.retrieval_method,
            citation=SPEC.citation,
            sample_mode=False,
        ),
    )


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_licensing_supply_shocks, refresh=refresh, dry_run=dry_run)

    raw_path = paths.raw / SPEC.name / RAW_FILENAME
    raw_rules_path = paths.raw / SPEC.name / RAW_RULES_FILENAME
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}_supply_shocks.parquet"
    normalized_rules_path = paths.interim / SPEC.name / "licensing_rules_long.parquet"
    icpsr_2017_path = paths.raw / SPEC.name / ICPSR_2017_TSV
    icpsr_2020_path = paths.raw / SPEC.name / ICPSR_2020_TSV
    if dry_run:
        return IngestResult(
            SPEC.name,
            raw_path if raw_path.exists() else raw_rules_path if raw_rules_path.exists() else icpsr_2020_path,
            normalized_path if raw_path.exists() else normalized_rules_path,
            False,
            dry_run=True,
            detail=f"shock={raw_path} rules={raw_rules_path} icpsr_2017={icpsr_2017_path} icpsr_2020={icpsr_2020_path}",
        )

    if not raw_path.exists() and not raw_rules_path.exists() and not icpsr_2017_path.exists() and not icpsr_2020_path.exists():
        raise SourceAccessError(
            f"missing manual source input for {SPEC.name}: expected local path at {raw_path} "
            f"or optional richer rule-level path at {raw_rules_path} "
            f"or ICPSR study files at {icpsr_2017_path} / {icpsr_2020_path}; download or curate the files "
            f"from {SPEC.landing_page} and place them there before running the real ingest"
        )

    wrote_any = False
    if raw_path.exists():
        if refresh or not normalized_path.exists() or normalized_path.stat().st_mtime < raw_path.stat().st_mtime:
            frame = pd.read_csv(raw_path)
            missing = REQUIRED_COLUMNS - set(frame.columns)
            if missing:
                raise ValueError(f"Licensing shock CSV missing required columns: {', '.join(sorted(missing))}")
            normalized = _normalize(frame)
            write_parquet(normalized, normalized_path)
            _append_registry_record(
                paths,
                source_name=SPEC.name,
                raw_path=raw_path,
                normalized_path=normalized_path,
            )
            wrote_any = True
    if raw_rules_path.exists():
        if (
            refresh
            or not normalized_rules_path.exists()
            or normalized_rules_path.stat().st_mtime < raw_rules_path.stat().st_mtime
        ):
            rules = pd.read_csv(raw_rules_path)
            normalized_rules = _normalize_rule_level(rules)
            write_parquet(normalized_rules, normalized_rules_path)
            _append_registry_record(
                paths,
                source_name=f"{SPEC.name}_rules_long",
                raw_path=raw_rules_path,
                normalized_path=normalized_rules_path,
            )
            wrote_any = True
    elif icpsr_2017_path.exists() or icpsr_2020_path.exists():
        candidate_sources = [path for path in (icpsr_2017_path, icpsr_2020_path) if path.exists()]
        latest_mtime = max(path.stat().st_mtime for path in candidate_sources)
        if refresh or not normalized_rules_path.exists() or normalized_rules_path.stat().st_mtime < latest_mtime:
            normalized_rules = pd.DataFrame()
            normalized_rules, source_files = _build_rule_level_from_icpsr(paths)
            if normalized_rules.empty:
                raise ValueError(
                    "ICPSR licensing study files were present but no rule-level rows could be extracted"
                )
            write_parquet(normalized_rules, normalized_rules_path)
            for source_file in source_files:
                _append_registry_record(
                    paths,
                    source_name=f"{SPEC.name}_rules_icpsr",
                    raw_path=source_file,
                    normalized_path=normalized_rules_path,
                )
            wrote_any = True

    if raw_path.exists():
        require_manual_source_path(raw_path, SPEC.landing_page, SPEC.name)
        if not wrote_any and normalized_path.exists():
            return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")
        return IngestResult(SPEC.name, raw_path, normalized_path, False)

    if raw_rules_path.exists():
        require_manual_source_path(raw_rules_path, SPEC.landing_page, f"{SPEC.name}_rules_long")
    elif icpsr_2020_path.exists():
        require_manual_source_path(icpsr_2020_path, SPEC.landing_page, f"{SPEC.name}_rules_icpsr_2020")
    elif icpsr_2017_path.exists():
        require_manual_source_path(icpsr_2017_path, SPEC.landing_page, f"{SPEC.name}_rules_icpsr_2017")
    if not wrote_any and normalized_rules_path.exists():
        return IngestResult(SPEC.name, raw_rules_path, normalized_rules_path, False, skipped=True, detail="cached")
    detail = "rules_long_only" if raw_rules_path.exists() else "icpsr_rule_level"
    source_path = raw_rules_path if raw_rules_path.exists() else icpsr_2020_path if icpsr_2020_path.exists() else icpsr_2017_path
    return IngestResult(SPEC.name, source_path, normalized_rules_path, False, detail=detail)
