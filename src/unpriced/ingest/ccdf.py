from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from unpriced.config import ProjectPaths
from unpriced.errors import SourceAccessError
from unpriced.ingest.common import IngestResult, SourceSpec, require_manual_source_path
from unpriced.registry import append_registry, build_record
from unpriced.sample_data import (
    ccdf as sample_ccdf,
    ccdf_admin_long as sample_ccdf_admin_long,
    ccdf_policy_long as sample_ccdf_policy_long,
)
from unpriced.storage import read_parquet, write_parquet

SPEC = SourceSpec(
    name="ccdf",
    citation="https://acf.gov/opre/project/child-care-and-development-fund-ccdf-policies-database ; https://acf.gov/occ/data/child-care-and-development-fund-statistics",
    license_name="Public data / manual download required",
    retrieval_method="manual-download",
    landing_page="https://acf.gov/opre/project/child-care-and-development-fund-ccdf-policies-database",
)

POLICIES_PAGE = "https://ccdf.urban.org/search-database"
ADMIN_PAGE = "https://acf.gov/occ/data/child-care-and-development-fund-statistics"
DEFAULT_SHEET_NAME = "__default__"
YEAR_PATTERN = re.compile(r"(20\d{2})")

MANIFEST_COLUMNS = [
    "source_component",
    "raw_relpath",
    "filename",
    "extension",
    "manual_download_required",
    "landing_page",
    "file_size",
]

PARSE_INVENTORY_COLUMNS = [
    "source_component",
    "raw_relpath",
    "filename",
    "source_sheet",
    "file_format",
    "parse_status",
    "output_table",
    "row_count",
    "parsed_row_count",
    "table_year",
    "parse_detail",
]

LONG_COLUMNS = [
    "source_component",
    "raw_relpath",
    "filename",
    "file_format",
    "landing_page",
    "source_sheet",
    "row_number",
    "column_name",
    "value_text",
    "value_numeric",
    "table_year",
    "parse_status",
]


STATE_HEADER_TOKENS = ("state/territory", "state")
CHILDREN_SERVED_SHEET_PREFIX = "1."
PAYMENT_TYPES_SHEET_PREFIX = "2."
CARE_BY_TYPE_SHEET_PREFIX = "3."
REGULATED_SHEET_PREFIX = "4."
RELATIVE_CARE_SHEET_PREFIX = "5."
SETTING_DETAIL_SHEET_PREFIX = "6."
EXPENDITURE_SHEET_NAME = "Table 3a"


def _ccdf_output_paths(paths: ProjectPaths) -> dict[str, Path]:
    base_dir = paths.interim / SPEC.name
    return {
        "manifest": base_dir / f"{SPEC.name}.parquet",
        "admin": base_dir / "ccdf_admin_long.parquet",
        "policy": base_dir / "ccdf_policy_long.parquet",
        "parse_inventory": base_dir / "ccdf_parse_inventory.parquet",
    }


def _empty_long_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=LONG_COLUMNS)


def _sample_manifest() -> pd.DataFrame:
    manifest = sample_ccdf().copy()
    manifest["file_size"] = 0
    return manifest[MANIFEST_COLUMNS]


def _sample_parse_inventory(admin_long: pd.DataFrame, policy_long: pd.DataFrame) -> pd.DataFrame:
    inventory = sample_ccdf().copy()
    inventory["source_sheet"] = inventory["source_component"].map(
        {"admin": "Table 1", "policies": DEFAULT_SHEET_NAME}
    )
    inventory["file_format"] = inventory["extension"].str.lstrip(".")
    inventory["parse_status"] = "parsed"
    inventory["output_table"] = inventory["source_component"].map(
        {
            "admin": "ccdf_admin_long",
            "policies": "ccdf_policy_long",
        }
    )
    inventory["row_count"] = inventory["source_component"].map(
        {
            "admin": int(len(admin_long)),
            "policies": int(len(policy_long)),
        }
    )
    inventory["parsed_row_count"] = inventory["row_count"]
    inventory["table_year"] = inventory["filename"].map(_extract_table_year)
    inventory["parse_detail"] = ""
    return inventory[PARSE_INVENTORY_COLUMNS]


def _normalize_column_names(columns: pd.Index) -> list[str]:
    normalized: list[str] = []
    seen: dict[str, int] = {}
    for idx, value in enumerate(columns):
        base = str(value).strip()
        if not base or base.lower() == "nan":
            base = f"unnamed_{idx + 1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        normalized.append(base if count == 0 else f"{base}_{count + 1}")
    return normalized


def _clean_tabular_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = _normalize_column_names(cleaned.columns)
    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return cleaned.reset_index(drop=True)


def _extract_table_year(*values: object) -> int | None:
    for value in values:
        if value is None:
            continue
        match = YEAR_PATTERN.search(str(value))
        if match:
            return int(match.group(1))
    return None


def _format_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _coerce_numeric(value_text: pd.Series) -> pd.Series:
    normalized = (
        value_text.fillna("")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
    )
    return pd.to_numeric(normalized, errors="coerce")


def _normalize_text_token(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ").replace("\n", " ").strip().lower()
    return re.sub(r"\s+", " ", text)


def _coerce_scalar_numeric(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace("\xa0", " ").strip()
    if not text or text in {"-", "—", "–"}:
        return None
    normalized = (
        text.replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .replace("(", "-")
        .replace(")", "")
    )
    numeric = pd.to_numeric(pd.Series([normalized]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def _find_state_header_row(frame: pd.DataFrame) -> int | None:
    for idx in range(len(frame)):
        row_values = [_normalize_text_token(value) for value in frame.iloc[idx].tolist()]
        non_empty = [value for value in row_values if value]
        if not non_empty:
            continue
        first = non_empty[0]
        if first in STATE_HEADER_TOKENS and len(non_empty) >= 2:
            return idx
    return None


def _structured_rows_to_long_records(
    rows: pd.DataFrame,
    *,
    source_component: str,
    raw_path: Path,
    landing_page: str,
    file_format: str,
    source_sheet: str,
) -> pd.DataFrame:
    if rows.empty:
        return _empty_long_frame()
    records: list[dict[str, object]] = []
    for row_number, row in rows.reset_index(drop=True).iterrows():
        for column_name, value in row.to_dict().items():
            value_text = _format_value(value)
            if value_text is None:
                continue
            numeric_value = _coerce_scalar_numeric(value)
            records.append(
                {
                    "source_component": source_component,
                    "raw_relpath": str(raw_path),
                    "filename": raw_path.name,
                    "file_format": file_format,
                    "landing_page": landing_page,
                    "source_sheet": source_sheet,
                    "row_number": int(row_number + 1),
                    "column_name": str(column_name),
                    "value_text": value_text,
                    "value_numeric": numeric_value,
                    "table_year": _extract_table_year(raw_path.name, source_sheet),
                    "parse_status": "parsed",
                }
            )
    if not records:
        return _empty_long_frame()
    return pd.DataFrame(records, columns=LONG_COLUMNS)


def _extract_structured_state_rows(frame: pd.DataFrame) -> tuple[int | None, pd.DataFrame]:
    header_row = _find_state_header_row(frame)
    if header_row is None:
        return None, pd.DataFrame()
    columns = [
        _normalize_text_token(value) or f"unnamed_{idx + 1}"
        for idx, value in enumerate(frame.iloc[header_row].tolist())
    ]
    data = frame.iloc[header_row + 1 :].copy().reset_index(drop=True)
    data.columns = columns
    data = data.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if data.empty:
        return header_row, pd.DataFrame()
    state_column = data.columns[0]
    state_text = data[state_column].map(_normalize_text_token)
    data = data.loc[
        state_text.ne("")
        & ~state_text.isin(
            {
                "total",
                "subtotal",
                "state/territory",
                "state",
                "mandatory",
                "matching federal",
                "matching state",
                "discretionary",
            }
        )
    ].copy()
    return header_row, data.reset_index(drop=True)


def _parse_children_served_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    _, data = _extract_structured_state_rows(frame)
    if data.empty:
        return pd.DataFrame(columns=["state", "average_number_of_families", "children_served_average_monthly"])
    state_column = data.columns[0]
    children_column = next((column for column in data.columns if "average number of children" in column), None)
    families_column = next((column for column in data.columns if "average number of families" in column), None)
    if children_column is None:
        return pd.DataFrame(columns=["state", "average_number_of_families", "children_served_average_monthly"])
    result = pd.DataFrame(
        {
            "state": data[state_column],
            "children_served_average_monthly": data[children_column],
        }
    )
    if families_column is not None:
        result["average_number_of_families"] = data[families_column]
    return result


def _parse_payment_types_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    _, data = _extract_structured_state_rows(frame)
    if data.empty:
        return pd.DataFrame(
            columns=[
                "state",
                "public_admin_share",
                "subsidized_private_share",
                "grants_contracts_percent",
                "certificates_percent",
                "cash_percent",
                "payment_method_total_children",
                "split_proxy_source",
            ]
        )
    state_column = data.columns[0]
    grants_column = next((column for column in data.columns if "grants/contracts" in column), None)
    certificates_column = next((column for column in data.columns if "certificates" in column), None)
    cash_column = next((column for column in data.columns if "cash" in column), None)
    total_column = next(
        (
            column
            for column in data.columns
            if column == "total" or ("total" in column and "child" in column)
        ),
        None,
    )
    if grants_column is None or certificates_column is None:
        return pd.DataFrame(
            columns=[
                "state",
                "public_admin_share",
                "subsidized_private_share",
                "grants_contracts_percent",
                "certificates_percent",
                "cash_percent",
                "payment_method_total_children",
                "split_proxy_source",
            ]
        )
    grants = data[grants_column].map(_coerce_scalar_numeric)
    certificates = data[certificates_column].map(_coerce_scalar_numeric)
    cash = data[cash_column].map(_coerce_scalar_numeric) if cash_column is not None else pd.Series([0.0] * len(data))
    public_admin_share = grants
    subsidized_private_share = certificates.fillna(0.0) + cash.fillna(0.0)
    result = pd.DataFrame(
        {
            "state": data[state_column],
            "public_admin_share": public_admin_share,
            "subsidized_private_share": subsidized_private_share,
            "grants_contracts_percent": grants,
            "certificates_percent": certificates,
            "cash_percent": cash,
            "split_proxy_source": "payment_method_shares",
        }
    )
    if total_column is not None:
        result["payment_method_total_children"] = data[total_column]
    return result


def _parse_care_by_type_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    _, data = _extract_structured_state_rows(frame)
    if data.empty:
        return pd.DataFrame(
            columns=[
                "state",
                "ccdf_care_type_child_home_share",
                "ccdf_care_type_family_home_share",
                "ccdf_care_type_group_home_share",
                "ccdf_care_type_center_share",
                "ccdf_care_type_invalid_share",
            ]
        )
    state_column = data.columns[0]
    child_home_column = next((column for column in data.columns if "child's home" in column), None)
    family_home_column = next((column for column in data.columns if "family home" in column), None)
    group_home_column = next((column for column in data.columns if "group home" in column), None)
    center_column = next((column for column in data.columns if column == "center"), None)
    invalid_column = next((column for column in data.columns if "invalid/not reported" in column), None)
    return pd.DataFrame(
        {
            "state": data[state_column],
            "ccdf_care_type_child_home_share": data[child_home_column] if child_home_column is not None else pd.NA,
            "ccdf_care_type_family_home_share": data[family_home_column] if family_home_column is not None else pd.NA,
            "ccdf_care_type_group_home_share": data[group_home_column] if group_home_column is not None else pd.NA,
            "ccdf_care_type_center_share": data[center_column] if center_column is not None else pd.NA,
            "ccdf_care_type_invalid_share": data[invalid_column] if invalid_column is not None else pd.NA,
        }
    )


def _parse_regulated_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    _, data = _extract_structured_state_rows(frame)
    if data.empty:
        return pd.DataFrame(
            columns=[
                "state",
                "ccdf_regulated_share",
                "ccdf_unregulated_share",
                "ccdf_regulation_invalid_share",
            ]
        )
    state_column = data.columns[0]
    regulated_column = next(
        (column for column in data.columns if "licensed/" in column or "licensed regulated" in column),
        None,
    )
    unregulated_column = next((column for column in data.columns if "legally operating" in column), None)
    invalid_column = next((column for column in data.columns if "invalid/" in column), None)
    return pd.DataFrame(
        {
            "state": data[state_column],
            "ccdf_regulated_share": data[regulated_column] if regulated_column is not None else pd.NA,
            "ccdf_unregulated_share": data[unregulated_column] if unregulated_column is not None else pd.NA,
            "ccdf_regulation_invalid_share": data[invalid_column] if invalid_column is not None else pd.NA,
        }
    )


def _parse_setting_detail_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    header_row = 4
    if len(frame) <= header_row + 3:
        return pd.DataFrame(
            columns=[
                "state",
                "ccdf_setting_detail_regulated_total_share",
                "ccdf_setting_detail_unregulated_total_share",
                "ccdf_setting_detail_unregulated_relative_total_share",
                "ccdf_setting_detail_unregulated_nonrelative_total_share",
                "ccdf_setting_detail_invalid_share",
            ]
        )
    data = frame.iloc[header_row + 3 :].copy().reset_index(drop=True)
    data = data.dropna(axis=0, how="all")
    if data.empty or len(data.columns) < 14:
        return pd.DataFrame(
            columns=[
                "state",
                "ccdf_setting_detail_regulated_total_share",
                "ccdf_setting_detail_unregulated_total_share",
                "ccdf_setting_detail_unregulated_relative_total_share",
                "ccdf_setting_detail_unregulated_nonrelative_total_share",
                "ccdf_setting_detail_invalid_share",
            ]
        )

    state_text = data.iloc[:, 0].map(_normalize_text_token)
    data = data.loc[
        state_text.ne("")
        & ~state_text.isin(
            {
                "total",
                "subtotal",
                "state/territory",
                "state",
            }
        )
    ].copy()
    if data.empty:
        return pd.DataFrame(
            columns=[
                "state",
                "ccdf_setting_detail_regulated_total_share",
                "ccdf_setting_detail_unregulated_total_share",
                "ccdf_setting_detail_unregulated_relative_total_share",
                "ccdf_setting_detail_unregulated_nonrelative_total_share",
                "ccdf_setting_detail_invalid_share",
            ]
        )

    regulated_child_home = data.iloc[:, 2].map(_coerce_scalar_numeric)
    regulated_family_home = data.iloc[:, 3].map(_coerce_scalar_numeric)
    regulated_group_home = data.iloc[:, 4].map(_coerce_scalar_numeric)
    regulated_center = data.iloc[:, 5].map(_coerce_scalar_numeric)
    unregulated_relative_child_home = data.iloc[:, 6].map(_coerce_scalar_numeric)
    unregulated_nonrelative_child_home = data.iloc[:, 7].map(_coerce_scalar_numeric)
    unregulated_relative_family_home = data.iloc[:, 8].map(_coerce_scalar_numeric)
    unregulated_nonrelative_family_home = data.iloc[:, 9].map(_coerce_scalar_numeric)
    unregulated_relative_group_home = data.iloc[:, 10].map(_coerce_scalar_numeric)
    unregulated_nonrelative_group_home = data.iloc[:, 11].map(_coerce_scalar_numeric)
    unregulated_center = data.iloc[:, 12].map(_coerce_scalar_numeric)
    invalid_share = data.iloc[:, 13].map(_coerce_scalar_numeric)

    return pd.DataFrame(
        {
            "state": data.iloc[:, 0],
            "ccdf_setting_detail_regulated_child_home_share": regulated_child_home,
            "ccdf_setting_detail_regulated_family_home_share": regulated_family_home,
            "ccdf_setting_detail_regulated_group_home_share": regulated_group_home,
            "ccdf_setting_detail_regulated_center_share": regulated_center,
            "ccdf_setting_detail_regulated_total_share": regulated_child_home.fillna(0.0)
            + regulated_family_home.fillna(0.0)
            + regulated_group_home.fillna(0.0)
            + regulated_center.fillna(0.0),
            "ccdf_setting_detail_unregulated_relative_child_home_share": unregulated_relative_child_home,
            "ccdf_setting_detail_unregulated_nonrelative_child_home_share": unregulated_nonrelative_child_home,
            "ccdf_setting_detail_unregulated_relative_family_home_share": unregulated_relative_family_home,
            "ccdf_setting_detail_unregulated_nonrelative_family_home_share": unregulated_nonrelative_family_home,
            "ccdf_setting_detail_unregulated_relative_group_home_share": unregulated_relative_group_home,
            "ccdf_setting_detail_unregulated_nonrelative_group_home_share": unregulated_nonrelative_group_home,
            "ccdf_setting_detail_unregulated_center_share": unregulated_center,
            "ccdf_setting_detail_unregulated_relative_total_share": unregulated_relative_child_home.fillna(0.0)
            + unregulated_relative_family_home.fillna(0.0)
            + unregulated_relative_group_home.fillna(0.0),
            "ccdf_setting_detail_unregulated_nonrelative_total_share": unregulated_nonrelative_child_home.fillna(0.0)
            + unregulated_nonrelative_family_home.fillna(0.0)
            + unregulated_nonrelative_group_home.fillna(0.0),
            "ccdf_setting_detail_unregulated_total_share": unregulated_relative_child_home.fillna(0.0)
            + unregulated_nonrelative_child_home.fillna(0.0)
            + unregulated_relative_family_home.fillna(0.0)
            + unregulated_nonrelative_family_home.fillna(0.0)
            + unregulated_relative_group_home.fillna(0.0)
            + unregulated_nonrelative_group_home.fillna(0.0)
            + unregulated_center.fillna(0.0),
            "ccdf_setting_detail_invalid_share": invalid_share,
        }
    )


def _parse_relative_care_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    _, data = _extract_structured_state_rows(frame)
    if data.empty:
        return pd.DataFrame(
            columns=[
                "state",
                "ccdf_relative_care_relative_share_within_unregulated",
                "ccdf_relative_care_nonrelative_share_within_unregulated",
                "ccdf_relative_care_total_children",
                "ccdf_relative_care_relative_children",
                "ccdf_relative_care_nonrelative_children",
            ]
        )
    state_column = data.columns[0]
    relative_column = next((column for column in data.columns if column == "relative"), None)
    nonrelative_column = next((column for column in data.columns if "non-relative" in column), None)
    total_count_column = next((column for column in data.columns if "total count" in column), None)
    if relative_column is None or nonrelative_column is None or total_count_column is None:
        return pd.DataFrame(
            columns=[
                "state",
                "ccdf_relative_care_relative_share_within_unregulated",
                "ccdf_relative_care_nonrelative_share_within_unregulated",
                "ccdf_relative_care_total_children",
                "ccdf_relative_care_relative_children",
                "ccdf_relative_care_nonrelative_children",
            ]
        )

    relative_share = data[relative_column].map(_coerce_scalar_numeric)
    nonrelative_share = data[nonrelative_column].map(_coerce_scalar_numeric)
    total_children = data[total_count_column].map(_coerce_scalar_numeric)
    return pd.DataFrame(
        {
            "state": data[state_column],
            "ccdf_relative_care_relative_share_within_unregulated": relative_share,
            "ccdf_relative_care_nonrelative_share_within_unregulated": nonrelative_share,
            "ccdf_relative_care_total_children": total_children,
            "ccdf_relative_care_relative_children": relative_share.fillna(0.0) * total_children.fillna(0.0),
            "ccdf_relative_care_nonrelative_children": nonrelative_share.fillna(0.0) * total_children.fillna(0.0),
        }
    )


def _parse_expenditure_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    _, data = _extract_structured_state_rows(frame)
    if data.empty:
        return pd.DataFrame(
            columns=[
                "state",
                "ccdf_admin_expenditures",
                "ccdf_quality_activities_expenditures",
                "ccdf_infant_toddler_quality_expenditures",
                "ccdf_direct_services_expenditures",
                "ccdf_nondirect_systems_expenditures",
                "ccdf_nondirect_cert_prog_eligibility_expenditures",
                "ccdf_nondirect_other_expenditures",
                "ccdf_total_expenditures",
            ]
        )
    state_column = data.columns[0]
    admin_column = next((column for column in data.columns if column == "admin"), None)
    quality_column = next((column for column in data.columns if "quality activities" in column), None)
    infant_toddler_quality_column = next((column for column in data.columns if "infant/toddler quality funds" in column), None)
    direct_services_column = next((column for column in data.columns if "direct services" in column), None)
    nondirect_systems_column = next((column for column in data.columns if "systems" in column), None)
    nondirect_eligibility_column = next(
        (column for column in data.columns if "cert prog elig/det" in column or "eligibility" in column),
        None,
    )
    nondirect_other_column = next((column for column in data.columns if "all other" in column), None)
    total_column = next((column for column in data.columns if "total expenditures" in column or column == "total"), None)
    if total_column is None and all(
        column is None
        for column in (
            admin_column,
            quality_column,
            infant_toddler_quality_column,
            direct_services_column,
            nondirect_systems_column,
            nondirect_eligibility_column,
            nondirect_other_column,
        )
    ):
        return pd.DataFrame(
            columns=[
                "state",
                "ccdf_admin_expenditures",
                "ccdf_quality_activities_expenditures",
                "ccdf_infant_toddler_quality_expenditures",
                "ccdf_direct_services_expenditures",
                "ccdf_nondirect_systems_expenditures",
                "ccdf_nondirect_cert_prog_eligibility_expenditures",
                "ccdf_nondirect_other_expenditures",
                "ccdf_total_expenditures",
            ]
        )
    return pd.DataFrame(
        {
            "state": data[state_column],
            "ccdf_admin_expenditures": data[admin_column] if admin_column is not None else pd.NA,
            "ccdf_quality_activities_expenditures": data[quality_column] if quality_column is not None else pd.NA,
            "ccdf_infant_toddler_quality_expenditures": data[infant_toddler_quality_column]
            if infant_toddler_quality_column is not None
            else pd.NA,
            "ccdf_direct_services_expenditures": data[direct_services_column]
            if direct_services_column is not None
            else pd.NA,
            "ccdf_nondirect_systems_expenditures": data[nondirect_systems_column]
            if nondirect_systems_column is not None
            else pd.NA,
            "ccdf_nondirect_cert_prog_eligibility_expenditures": data[nondirect_eligibility_column]
            if nondirect_eligibility_column is not None
            else pd.NA,
            "ccdf_nondirect_other_expenditures": data[nondirect_other_column]
            if nondirect_other_column is not None
            else pd.NA,
            "ccdf_total_expenditures": data[total_column] if total_column is not None else pd.NA,
        }
    )


def _admin_workbook_group(path: Path) -> str:
    name = path.name.lower()
    if "ccdf-data-tables" in name:
        return "data_tables"
    if "ccdf_data_tables" in name:
        return "data_tables"
    if "expenditures" in name:
        return "expenditures"
    return "other"


def _admin_workbook_priority(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if "final" in name:
        return (0, name)
    if "preliminary" in name:
        return (1, name)
    return (2, name)


def _preferred_admin_data_table_paths(raw_root: Path) -> set[Path]:
    candidates = [
        path
        for source_component, _, path in _manual_files(raw_root)
        if source_component == "admin" and path.suffix.lower() == ".xlsx" and _admin_workbook_group(path) == "data_tables"
    ]
    preferred: set[Path] = set()
    by_year: dict[int | None, list[Path]] = {}
    for path in candidates:
        by_year.setdefault(_extract_table_year(path.name), []).append(path)
    for paths in by_year.values():
        preferred.add(sorted(paths, key=_admin_workbook_priority)[0])
    return preferred


def _parse_special_admin_workbook(
    path: Path,
    landing_page: str,
) -> tuple[list[dict[str, object]], pd.DataFrame] | None:
    group = _admin_workbook_group(path)
    if path.suffix.lower() != ".xlsx" or group not in {"data_tables", "expenditures"}:
        return None
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    file_format = "xlsx"
    inventory_rows: list[dict[str, object]] = []
    records: list[pd.DataFrame] = []
    if group == "data_tables":
        target_parsers = {
            "1. Children Served": _parse_children_served_sheet,
            "2. Types of Payments": _parse_payment_types_sheet,
            "3. Care by Type": _parse_care_by_type_sheet,
            "4. Regulated vs Non-Regulated": _parse_regulated_sheet,
            "5. Relative Care": _parse_relative_care_sheet,
            "6. Setting Detail": _parse_setting_detail_sheet,
        }
    else:
        target_parsers = {
            EXPENDITURE_SHEET_NAME: _parse_expenditure_sheet,
        }
    for source_sheet, frame in sheets.items():
        parser = target_parsers.get(str(source_sheet))
        if parser is None:
            inventory_rows.append(
                {
                    "source_component": "admin",
                    "raw_relpath": str(path),
                    "filename": path.name,
                    "source_sheet": str(source_sheet or DEFAULT_SHEET_NAME),
                    "file_format": file_format,
                    "parse_status": "sheet_not_modeled",
                    "output_table": "ccdf_admin_long",
                    "row_count": int(len(_clean_tabular_frame(frame))),
                    "parsed_row_count": 0,
                    "table_year": _extract_table_year(path.name, source_sheet),
                    "parse_detail": "real workbook present but sheet-specific parser not yet implemented",
                }
            )
            continue
        structured_rows = parser(frame)
        long_records = _structured_rows_to_long_records(
            structured_rows,
            source_component="admin",
            raw_path=path,
            landing_page=landing_page,
            file_format=file_format,
            source_sheet=str(source_sheet),
        )
        inventory_rows.append(
            {
                "source_component": "admin",
                "raw_relpath": str(path),
                "filename": path.name,
                "source_sheet": str(source_sheet),
                "file_format": file_format,
                "parse_status": "parsed" if not long_records.empty else "empty_tabular",
                "output_table": "ccdf_admin_long",
                "row_count": int(len(structured_rows)),
                "parsed_row_count": int(len(long_records)),
                "table_year": _extract_table_year(path.name, source_sheet),
                "parse_detail": "",
            }
        )
        if not long_records.empty:
            records.append(long_records)
    combined = pd.concat(records, ignore_index=True) if records else _empty_long_frame()
    return inventory_rows, combined


def _read_json_frames(path: Path) -> dict[str, pd.DataFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {DEFAULT_SHEET_NAME: pd.DataFrame(payload)}
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return {DEFAULT_SHEET_NAME: pd.DataFrame(payload["rows"])}
        frames: dict[str, pd.DataFrame] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                frames[key] = pd.DataFrame(value)
            elif isinstance(value, dict):
                frames[key] = pd.DataFrame([value])
        if frames:
            return frames
        return {DEFAULT_SHEET_NAME: pd.DataFrame([payload])}
    raise ValueError(f"unsupported JSON payload structure in {path}")


def _read_tabular_frames(path: Path) -> tuple[str, dict[str, pd.DataFrame]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv", {DEFAULT_SHEET_NAME: pd.read_csv(path, dtype=object)}
    if suffix == ".tsv":
        return "tsv", {DEFAULT_SHEET_NAME: pd.read_csv(path, sep="\t", dtype=object)}
    if suffix == ".xlsx":
        return "xlsx", pd.read_excel(path, sheet_name=None, dtype=object)
    if suffix == ".json":
        return "json", _read_json_frames(path)
    if suffix == ".parquet":
        return "parquet", {DEFAULT_SHEET_NAME: read_parquet(path)}
    raise ValueError(f"unsupported manual file format: {suffix or '<no extension>'}")


def _frame_to_long_records(
    frame: pd.DataFrame,
    *,
    source_component: str,
    raw_path: Path,
    landing_page: str,
    file_format: str,
    source_sheet: str,
) -> pd.DataFrame:
    cleaned = _clean_tabular_frame(frame)
    if cleaned.empty or not len(cleaned.columns):
        return _empty_long_frame()

    long = (
        cleaned.reset_index(drop=True)
        .assign(row_number=lambda df: df.index + 1)
        .melt(id_vars=["row_number"], var_name="column_name", value_name="raw_value")
    )
    long["value_text"] = long["raw_value"].map(_format_value)
    long = long.loc[long["value_text"].notna()].copy()
    if long.empty:
        return _empty_long_frame()

    long["value_numeric"] = _coerce_numeric(long["value_text"])
    long["source_component"] = source_component
    long["raw_relpath"] = str(raw_path)
    long["filename"] = raw_path.name
    long["file_format"] = file_format
    long["landing_page"] = landing_page
    long["source_sheet"] = source_sheet
    long["table_year"] = _extract_table_year(raw_path.name, source_sheet)
    long["parse_status"] = "parsed"
    return long[LONG_COLUMNS].reset_index(drop=True)


def _manual_files(raw_root: Path) -> list[tuple[str, str, Path]]:
    files: list[tuple[str, str, Path]] = []
    for source_component, landing_page in (
        ("admin", ADMIN_PAGE),
        ("policies", POLICIES_PAGE),
    ):
        component_dir = raw_root / source_component
        if not component_dir.exists():
            continue
        files.extend(
            (source_component, landing_page, path)
            for path in sorted(path for path in component_dir.rglob("*") if path.is_file())
        )
    return files


def _first_manual_file(raw_root: Path) -> Path | None:
    manual_files = _manual_files(raw_root)
    if not manual_files:
        return None
    return manual_files[0][2]


def _build_manifest(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_component, landing_page, path in _manual_files(raw_root):
        rows.append(
            {
                "source_component": source_component,
                "raw_relpath": str(path),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "manual_download_required": True,
                "landing_page": landing_page,
                "file_size": int(path.stat().st_size),
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _build_parse_inventory_rows(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory_rows: list[dict[str, object]] = []
    admin_frames: list[pd.DataFrame] = []
    policy_frames: list[pd.DataFrame] = []
    preferred_data_tables = _preferred_admin_data_table_paths(raw_root)

    for source_component, landing_page, path in _manual_files(raw_root):
        base_row = {
            "source_component": source_component,
            "raw_relpath": str(path),
            "filename": path.name,
        }
        if (
            source_component == "admin"
            and path.suffix.lower() == ".xlsx"
            and _admin_workbook_group(path) == "data_tables"
            and path not in preferred_data_tables
        ):
            inventory_rows.append(
                {
                    **base_row,
                    "file_format": "xlsx",
                    "source_sheet": DEFAULT_SHEET_NAME,
                    "parse_status": "duplicate_nonpreferred",
                    "output_table": "ccdf_admin_long",
                    "row_count": 0,
                    "parsed_row_count": 0,
                    "table_year": _extract_table_year(path.name),
                    "parse_detail": "skipped because a preferred final workbook exists for the same year",
                }
            )
            continue
        special_admin_parse = (
            _parse_special_admin_workbook(path, landing_page)
            if source_component == "admin"
            else None
        )
        if special_admin_parse is not None:
            special_inventory_rows, special_admin_long = special_admin_parse
            inventory_rows.extend(special_inventory_rows)
            if not special_admin_long.empty:
                admin_frames.append(special_admin_long)
            continue
        try:
            file_format, sheets = _read_tabular_frames(path)
        except Exception as exc:
            parse_status = "unsupported_format" if isinstance(exc, ValueError) else "read_error"
            inventory_rows.append(
                {
                    **base_row,
                    "file_format": path.suffix.lower().lstrip("."),
                    "source_sheet": DEFAULT_SHEET_NAME,
                    "parse_status": parse_status,
                    "output_table": "ccdf_admin_long" if source_component == "admin" else "ccdf_policy_long",
                    "row_count": 0,
                    "parsed_row_count": 0,
                    "table_year": _extract_table_year(path.name),
                    "parse_detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for source_sheet, frame in sheets.items():
            records = _frame_to_long_records(
                frame,
                source_component=source_component,
                raw_path=path,
                landing_page=landing_page,
                file_format=file_format,
                source_sheet=str(source_sheet or DEFAULT_SHEET_NAME),
            )
            inventory_rows.append(
                {
                    **base_row,
                    "file_format": file_format,
                    "source_sheet": str(source_sheet or DEFAULT_SHEET_NAME),
                    "parse_status": "parsed" if not records.empty else "empty_tabular",
                    "output_table": "ccdf_admin_long" if source_component == "admin" else "ccdf_policy_long",
                    "row_count": int(len(_clean_tabular_frame(frame))),
                    "parsed_row_count": int(len(records)),
                    "table_year": _extract_table_year(path.name, source_sheet),
                    "parse_detail": "",
                }
            )
            if records.empty:
                continue
            if source_component == "admin":
                admin_frames.append(records)
            else:
                policy_frames.append(records)

    inventory = pd.DataFrame(inventory_rows, columns=PARSE_INVENTORY_COLUMNS)
    admin_long = pd.concat(admin_frames, ignore_index=True) if admin_frames else _empty_long_frame()
    policy_long = pd.concat(policy_frames, ignore_index=True) if policy_frames else _empty_long_frame()
    return inventory, admin_long, policy_long


def _latest_manual_mtime(raw_root: Path) -> float:
    timestamps = [path.stat().st_mtime for _, _, path in _manual_files(raw_root)]
    return max(timestamps) if timestamps else 0.0


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    del year
    output_paths = _ccdf_output_paths(paths)
    normalized_path = output_paths["manifest"]

    if sample:
        raw_path = paths.raw / SPEC.name / f"{SPEC.name}_sample.json"
        if dry_run:
            return IngestResult(SPEC.name, raw_path, normalized_path, True, dry_run=True, detail="dry-run")
        if (
            raw_path.exists()
            and all(path.exists() for path in output_paths.values())
            and not refresh
            and min(path.stat().st_mtime for path in output_paths.values()) >= raw_path.stat().st_mtime
        ):
            return IngestResult(SPEC.name, raw_path, normalized_path, True, skipped=True, detail="cached")

        admin_long = sample_ccdf_admin_long()
        policy_long = sample_ccdf_policy_long()
        manifest = _sample_manifest()
        parse_inventory = _sample_parse_inventory(admin_long, policy_long)

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_payload = {
            "source": SPEC.name,
            "inventory_rows": sample_ccdf().to_dict(orient="records"),
            "admin_rows": admin_long.to_dict(orient="records"),
            "policy_rows": policy_long.to_dict(orient="records"),
        }
        raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
        write_parquet(manifest, normalized_path)
        write_parquet(admin_long, output_paths["admin"])
        write_parquet(policy_long, output_paths["policy"])
        write_parquet(parse_inventory, output_paths["parse_inventory"])
        append_registry(
            paths,
            build_record(
                source_name=SPEC.name,
                raw_path=raw_path,
                normalized_path=normalized_path,
                license_name=SPEC.license_name,
                retrieval_method=f"{SPEC.retrieval_method}:sample",
                citation=SPEC.citation,
                sample_mode=True,
            ),
        )
        return IngestResult(SPEC.name, raw_path, normalized_path, True)

    raw_root = paths.raw / SPEC.name
    if dry_run:
        return IngestResult(SPEC.name, raw_root, normalized_path, False, dry_run=True, detail=str(raw_root))

    require_manual_source_path(raw_root, SPEC.landing_page, SPEC.name)
    manual_files = _manual_files(raw_root)
    if not manual_files:
        admin_dir = raw_root / "admin"
        policies_dir = raw_root / "policies"
        raise SourceAccessError(
            "missing CCDF manual files: expected one or more files under "
            f"{admin_dir} or {policies_dir}; download them from {ADMIN_PAGE} and/or {POLICIES_PAGE} first"
        )
    if (
        all(path.exists() for path in output_paths.values())
        and not refresh
        and min(path.stat().st_mtime for path in output_paths.values()) >= _latest_manual_mtime(raw_root)
    ):
        return IngestResult(SPEC.name, raw_root, normalized_path, False, skipped=True, detail="cached")

    manifest = _build_manifest(raw_root)
    parse_inventory, admin_long, policy_long = _build_parse_inventory_rows(raw_root)
    raw_record_path = _first_manual_file(raw_root)
    if raw_record_path is None:
        raise SourceAccessError(f"unable to locate a CCDF raw file under {raw_root}")

    write_parquet(manifest, normalized_path)
    write_parquet(admin_long, output_paths["admin"])
    write_parquet(policy_long, output_paths["policy"])
    write_parquet(parse_inventory, output_paths["parse_inventory"])
    append_registry(
        paths,
        build_record(
            source_name=SPEC.name,
            raw_path=raw_record_path,
            normalized_path=normalized_path,
            license_name=SPEC.license_name,
            retrieval_method=SPEC.retrieval_method,
            citation=SPEC.citation,
            sample_mode=False,
        ),
    )
    return IngestResult(SPEC.name, raw_root, normalized_path, False)
