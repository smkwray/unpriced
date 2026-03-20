from __future__ import annotations

import pandas as pd
import pytest

from unpriced.errors import SourceAccessError
from unpriced.ingest.ccdf import ingest as ingest_ccdf
from unpriced.storage import read_parquet


MANIFEST_REQUIRED_COLUMNS = {
    "source_component",
    "raw_relpath",
    "filename",
    "extension",
    "manual_download_required",
    "landing_page",
}
PARSE_METADATA_COLUMNS = {"source_component", "raw_relpath", "parse_status"}
INVENTORY_REQUIRED_COLUMNS = {"source_component", "raw_relpath", "parse_status", "output_table"}
INVENTORY_ROW_COUNT_COLUMNS = {"row_count", "parsed_row_count"}


def _ccdf_output_frames(project_paths):
    output_dir = project_paths.interim / "ccdf"
    parquet_paths = sorted(output_dir.glob("*.parquet"))
    return {path: read_parquet(path) for path in parquet_paths}


def _assert_extended_ccdf_outputs(project_paths, main_output_path):
    frames = _ccdf_output_frames(project_paths)
    assert main_output_path in frames

    extras = {path: frame for path, frame in frames.items() if path != main_output_path}
    assert len(extras) >= 3, "expected parsed admin/policy outputs plus inventory/status table"

    admin_tables = []
    policy_tables = []
    inventory_tables = []
    for path, frame in extras.items():
        columns = set(frame.columns)
        assert PARSE_METADATA_COLUMNS <= columns, f"missing parse metadata in {path.name}"
        assert frame["parse_status"].dropna().astype(str).str.len().gt(0).all()
        components = set(frame["source_component"].dropna().astype(str))
        if components and components <= {"admin"}:
            admin_tables.append(path)
        if components and components <= {"policies"}:
            policy_tables.append(path)
        if INVENTORY_REQUIRED_COLUMNS <= columns and columns.intersection(INVENTORY_ROW_COUNT_COLUMNS):
            inventory_tables.append(path)

    assert admin_tables, "expected at least one parsed admin table"
    assert policy_tables, "expected at least one parsed policy table"
    assert inventory_tables, "expected one inventory/status table with row counts"
    return frames, inventory_tables[0]


def test_ccdf_sample_ingest_writes_expected_columns(project_paths):
    result = ingest_ccdf(project_paths, sample=True, refresh=True)
    manifest = read_parquet(result.normalized_path)

    assert result.normalized_path.exists()
    assert MANIFEST_REQUIRED_COLUMNS <= set(manifest.columns)
    _assert_extended_ccdf_outputs(project_paths, result.normalized_path)


def test_ccdf_real_ingest_requires_manual_root(project_paths):
    raw_root = project_paths.raw / "ccdf"

    with pytest.raises(SourceAccessError) as exc_info:
        ingest_ccdf(project_paths, sample=False, refresh=True)

    message = str(exc_info.value)
    assert str(raw_root) in message
    assert "acf.gov/opre/project/child-care-and-development-fund-ccdf-policies-database" in message


def test_ccdf_real_ingest_builds_manifest_from_manual_files(project_paths):
    admin_dir = project_paths.raw / "ccdf" / "admin"
    policies_dir = project_paths.raw / "ccdf" / "policies"
    admin_dir.mkdir(parents=True, exist_ok=True)
    policies_dir.mkdir(parents=True, exist_ok=True)
    admin_file = admin_dir / "fy2023_table_1.csv"
    policy_file = policies_dir / "ccdf_policies_2023.csv"
    admin_file.write_text(
        "state_fips,year,children_served\n06,2023,1000\n48,2023,800\n",
        encoding="utf-8",
    )
    policy_file.write_text(
        "state_fips,year,policy_key,policy_value\n06,2023,copay_required,1\n48,2023,copay_required,0\n",
        encoding="utf-8",
    )

    result = ingest_ccdf(project_paths, sample=False, refresh=True)

    frame = read_parquet(result.normalized_path)
    assert result.normalized_path.name == "ccdf.parquet"
    assert set(frame["source_component"]) == {"admin", "policies"}
    assert set(frame["filename"]) == {"fy2023_table_1.csv", "ccdf_policies_2023.csv"}
    assert frame["file_size"].gt(0).all()
    _, inventory_path = _assert_extended_ccdf_outputs(project_paths, result.normalized_path)
    inventory = read_parquet(inventory_path)
    assert {
        str(admin_file),
        str(policy_file),
    }.issubset(set(inventory["raw_relpath"].astype(str)))


def test_ccdf_real_ingest_parses_real_style_admin_xlsx_into_canonical_long_columns(project_paths):
    admin_dir = project_paths.raw / "ccdf" / "admin"
    policies_dir = project_paths.raw / "ccdf" / "policies"
    admin_dir.mkdir(parents=True, exist_ok=True)
    policies_dir.mkdir(parents=True, exist_ok=True)

    stats_file = admin_dir / "fy-2023-ccdf-data-tables-preliminary.xlsx"
    expenditure_file = admin_dir / "ccdf-expenditures-for-fy-2023-all-appropriation-years.xlsx"
    policy_file = policies_dir / "ccdf_policies_2023.csv"

    children_sheet = pd.DataFrame(
        [
            ["Table 1", None, None],
            ["Child Care and Development Fund", None, None],
            ["Preliminary Estimates", None, None],
            ["Average Monthly Adjusted Number of Families and Children Served (FY 2023)", None, None],
            ["State/Territory", "Average Number of Families", "Average Number of Children"],
            ["California", "143800", "232500"],
            ["Texas", "65000", "102000"],
        ]
    )
    payments_sheet = pd.DataFrame(
        [
            ["Table 2", None, None, None, None],
            ["Child Care and Development Fund", None, None, None, None],
            ["Preliminary Estimates", None, None, None, None],
            ["Percent of Children Served by Payment Method (FY 2023)", None, None, None, None],
            ["State/Territory", "Grants/Contracts %", "Certificates %", "Cash %", "Total Children"],
            ["California", "0.1522", "0.8478", "0", "204837"],
            ["Texas", "0.1", "0.9", "0", "91000"],
        ]
    )
    care_by_type_sheet = pd.DataFrame(
        [
            ["Table 3", None, None, None, None, None],
            ["Child Care and Development Fund", None, None, None, None, None],
            ["Preliminary Estimates", None, None, None, None, None],
            ["Average Monthly Percentages of Children Served by Types of Care (FY 2023)", None, None, None, None, None],
            ["State/Territory", "Child's Home", "Family Home", "Group Home", "Center", "Invalid/Not Reported"],
            ["California", "0.0012", "0.3863", "0.2297", "0.3619", "0.0209"],
            ["Texas", "0.002", "0.28", "0.12", "0.59", "0.008"],
        ]
    )
    regulated_sheet = pd.DataFrame(
        [
            ["Table 4", None, None, None],
            ["Child Care and Development Fund", None, None, None],
            ["Preliminary Estimates", None, None, None],
            ["Average Monthly Percentages of Children Served in Regulated Settings (FY 2023)", None, None, None],
            ["State/Territory", "Licensed/Regulated", "Legally Operating Without Regulation", "Invalid/Not Reported"],
            ["California", "0.7607", "0.2183", "0.0209"],
            ["Texas", "0.89", "0.10", "0.01"],
        ]
    )
    relative_care_sheet = pd.DataFrame(
        [
            ["Table 5", None, None, None, None],
            ["Child Care and Development Fund", None, None, None, None],
            ["Preliminary Estimates", None, None, None, None],
            ["Children in Settings Legally Operating Without Regulation, Average Monthly Percent Served by Relatives vs. Non-Relatives (FY 2023)", None, None, None, None],
            ["State/Territory", "Relative", "Non-Relative", "Total %", "Total Count"],
            ["California", "0.756327", "0.243673", "1", "60491"],
            ["Texas", "0.799145", "0.200855", "1", "10663"],
        ]
    )
    setting_detail_sheet = pd.DataFrame(
        [
            ["Table 6", None, None, None, None, None, None, None, None, None, None, None, None, None],
            ["Child Care and Development Fund", None, None, None, None, None, None, None, None, None, None, None, None, None],
            ["Preliminary Estimates", None, None, None, None, None, None, None, None, None, None, None, None, None],
            ["Average Monthly Percentages of Children Served in All Types of Care (FY 2023)", None, None, None, None, None, None, None, None, None, None, None, None, None],
            ["State/Territory", "Total % of Children", "Licensed or Regulated Providers", None, None, None, "Providers Legally Operating without Regulation", None, None, None, None, None, None, "Invalid/ Not Reported"],
            [None, None, "Child's Home", "Family Home", "Group Home", "Center", "Child's Home", None, "Family Home", None, "Group Home", None, "Center", None],
            [None, None, None, None, None, None, "Relative", "Non-Relative", "Relative", "Non-Relative", "Relative", "Non-Relative", None, None],
            ["California", "1.0", "0.0012", "0.3863", "0.0125", "0.3607", "0.0410", "0.0102", "0.1220", "0.0151", "0.0090", "0.0020", "0.0209", "0.0191"],
            ["Texas", "1.0", "0.0020", "0.2800", "0.1200", "0.4880", "0.0300", "0.0100", "0.0400", "0.0080", "0.0050", "0.0020", "0.0100", "0.0050"],
        ]
    )
    with pd.ExcelWriter(stats_file) as writer:
        children_sheet.to_excel(writer, sheet_name="1. Children Served", header=False, index=False)
        payments_sheet.to_excel(writer, sheet_name="2. Types of Payments", header=False, index=False)
        care_by_type_sheet.to_excel(writer, sheet_name="3. Care by Type", header=False, index=False)
        regulated_sheet.to_excel(writer, sheet_name="4. Regulated vs Non-Regulated", header=False, index=False)
        relative_care_sheet.to_excel(writer, sheet_name="5. Relative Care", header=False, index=False)
        setting_detail_sheet.to_excel(writer, sheet_name="6. Setting Detail", header=False, index=False)

    expenditure_sheet = pd.DataFrame(
        [
            ["CHILD CARE AND DEVELOPMENT FUND (CCDF)", None, None, None, None, None, None, None],
            ["Table 3a - ALL EXPENDITURES BY STATE - DETAILED SUMMARY", None, None, None, None, None, None, None],
            ["(STATE AND FEDERAL EXPENDITURES INCLUDING MOE)", None, None, None, None, None, None, None],
            ["FISCAL YEAR 2023", None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            [
                "State",
                "Admin",
                "Quality Activities",
                "Infant/Toddler Quality Funds",
                "Direct Services",
                "N-Dir Svcs Systems",
                "N-Dir Svcs Cert Prog Elig/Det",
                "Total Expenditures",
            ],
            ["California", "16270449", "78231056", "39823459", "1313067602", "0", "0", "1472877994"],
            ["Texas", "8100000", "55000000", "21000000", "780000000", "1200000", "500000", "902000000"],
        ]
    )
    with pd.ExcelWriter(expenditure_file) as writer:
        expenditure_sheet.to_excel(writer, sheet_name="Table 3a", header=False, index=False)

    policy_file.write_text(
        "state_fips,year,policy_key,policy_value\n06,2023,copay_required,1\n48,2023,copay_required,0\n",
        encoding="utf-8",
    )

    result = ingest_ccdf(project_paths, sample=False, refresh=True)

    manifest = read_parquet(result.normalized_path)
    admin_long = read_parquet(project_paths.interim / "ccdf" / "ccdf_admin_long.parquet")
    inventory = read_parquet(project_paths.interim / "ccdf" / "ccdf_parse_inventory.parquet")

    assert set(manifest["filename"]) == {
        stats_file.name,
        expenditure_file.name,
        policy_file.name,
    }
    assert {
        "state",
        "children_served_average_monthly",
        "public_admin_share",
        "subsidized_private_share",
        "ccdf_total_expenditures",
        "ccdf_direct_services_expenditures",
        "ccdf_admin_expenditures",
        "ccdf_care_type_center_share",
        "ccdf_regulated_share",
        "ccdf_relative_care_total_children",
        "ccdf_relative_care_relative_children",
        "ccdf_setting_detail_regulated_total_share",
        "ccdf_setting_detail_unregulated_relative_total_share",
    } <= set(
        admin_long["column_name"].astype(str)
    )
    assert "split_proxy_source" in set(admin_long["column_name"].astype(str))
    payment_total_rows = admin_long.loc[
        admin_long["source_sheet"].astype(str).eq("2. Types of Payments")
        & admin_long["column_name"].astype(str).eq("payment_method_total_children")
    ].copy()
    assert not payment_total_rows.empty
    california_payment_total = payment_total_rows.loc[
        payment_total_rows["row_number"].eq(1),
        "value_numeric",
    ].iloc[0]
    assert california_payment_total == 204837.0
    assert "sheet_not_modeled" not in set(
        inventory.loc[
            inventory["source_sheet"].astype(str).isin(
                {
                    "1. Children Served",
                    "2. Types of Payments",
                    "3. Care by Type",
                    "4. Regulated vs Non-Regulated",
                    "5. Relative Care",
                    "6. Setting Detail",
                    "Table 3a",
                }
            ),
            "parse_status",
        ]
    )
