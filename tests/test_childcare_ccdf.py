from __future__ import annotations

import pandas as pd
import pytest

from unpaidwork.childcare.ccdf import (
    build_ccdf_admin_state_year,
    build_ccdf_policy_feature_audit,
    build_ccdf_policy_controls_coverage,
    build_ccdf_policy_promoted_controls_state_year,
    build_ccdf_policy_controls_state_year,
    build_ccdf_policy_features_state_year,
)
from unpaidwork.sample_data import ccdf_admin_long, ccdf_policy_long


def _admin_long_frame(
    cells: dict[str, object],
    *,
    row_number: int = 1,
    source_path: str = "data/raw/ccdf/admin/test.csv",
    source_sheet: str = "__default__",
    table_year: int = 2023,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for column_name, value in cells.items():
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        records.append(
            {
                "source_component": "admin",
                "raw_relpath": source_path,
                "source_sheet": source_sheet,
                "row_number": row_number,
                "column_name": column_name,
                "value_text": None if value is None else str(value),
                "value_numeric": None if pd.isna(numeric) else float(numeric),
                "table_year": table_year,
                "parse_status": "parsed",
            }
        )
    return pd.DataFrame(records)


def _policy_long_frame(
    rows: list[dict[str, object]],
    *,
    source_path: str = "data/raw/ccdf/policies/test.csv",
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows:
        row_number = int(row["row_number"])
        state = str(row["state"])
        cells = dict(row.get("cells", {}))
        row_cells = {"state": state} | cells
        for column_name, value in row_cells.items():
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            records.append(
                {
                    "source_component": "policies",
                    "raw_relpath": source_path,
                    "source_sheet": str(row.get("source_sheet", "__default__")),
                    "row_number": row_number,
                    "column_name": column_name,
                    "value_text": None if value is None else str(value),
                    "value_numeric": None if pd.isna(numeric) else float(numeric),
                    "table_year": int(row.get("table_year", 2023)),
                    "parse_status": "parsed",
                }
            )
    return pd.DataFrame(records)


def test_build_ccdf_admin_state_year_maps_sample_rows_to_state_year_totals():
    mapped = build_ccdf_admin_state_year(ccdf_admin_long())

    california = mapped.loc[(mapped["state_fips"] == "06") & (mapped["year"] == 2023)].iloc[0]
    texas = mapped.loc[(mapped["state_fips"] == "48") & (mapped["year"] == 2023)].iloc[0]

    assert california["ccdf_children_served"] == 145200.0
    assert california["ccdf_subsidized_private_slots"] == 120000.0
    assert california["ccdf_public_admin_slots"] == 25200.0
    assert california["ccdf_support_flag"] == "ccdf_explicit_split_observed"
    assert texas["ccdf_children_served"] == 98700.0
    assert texas["ccdf_subsidized_private_slots"] == 80000.0
    assert texas["ccdf_public_admin_slots"] == 18700.0


def test_build_ccdf_policy_features_state_year_maps_sample_policy_extract():
    features = build_ccdf_policy_features_state_year(ccdf_policy_long())

    california = features.loc[(features["state_fips"] == "06") & (features["year"] == 2023)].iloc[0]
    texas = features.loc[(features["state_fips"] == "48") & (features["year"] == 2023)].iloc[0]

    assert california["ccdf_policy_copayment_required"] == "yes"
    assert texas["ccdf_policy_copayment_required"] == "no"
    assert california["ccdf_policy_support_status"] == "observed_policy_long"


def test_build_ccdf_policy_controls_state_year_maps_allowlisted_controls():
    controls = build_ccdf_policy_controls_state_year(ccdf_policy_long())

    california = controls.loc[(controls["state_fips"] == "06") & (controls["year"] == 2023)].iloc[0]
    texas = controls.loc[(controls["state_fips"] == "48") & (controls["year"] == 2023)].iloc[0]

    assert california["ccdf_control_copayment_required"] == "yes"
    assert texas["ccdf_control_copayment_required"] == "no"
    assert california["ccdf_policy_control_count"] >= 1
    assert california["ccdf_policy_control_support_status"] == "observed_policy_controls"


def test_build_ccdf_policy_controls_coverage_summarizes_allowlisted_control_support():
    coverage = build_ccdf_policy_controls_coverage(ccdf_policy_long())

    copayment = coverage.loc[coverage["control_name"] == "ccdf_control_copayment_required"].iloc[0]

    assert copayment["control_kind"] == "text"
    assert copayment["policy_state_year_rows"] == 2
    assert copayment["observed_state_year_rows"] == 2
    assert copayment["missing_state_year_rows"] == 0
    assert copayment["state_year_coverage_rate"] == 1.0
    assert copayment["coverage_support_status"] == "full_state_year_coverage"
    assert len(coverage) == 1


def test_build_ccdf_policy_promoted_controls_state_year_promotes_only_copayment_at_default_threshold():
    promoted = build_ccdf_policy_promoted_controls_state_year(ccdf_policy_long())

    assert {"state_fips", "year", "ccdf_policy_control_count", "ccdf_policy_control_support_status"} <= set(
        promoted.columns
    )
    assert "ccdf_control_copayment_required" in promoted.columns
    assert set(promoted["ccdf_policy_promoted_controls_selected"]) == {"ccdf_control_copayment_required"}
    assert set(promoted["ccdf_policy_promoted_min_state_year_coverage"]) == {0.75}
    assert set(promoted["ccdf_policy_control_count"]) == {1}


def test_build_ccdf_policy_promoted_controls_state_year_threshold_controls_selection():
    policy_long = _policy_long_frame(
        [
            {
                "row_number": 1,
                "state": "California",
                "cells": {
                    "copayment_required": "yes",
                },
            },
            {
                "row_number": 2,
                "state": "Texas",
                "cells": {
                    "notes": "observed_without_control",
                },
            },
        ]
    )
    promoted_default = build_ccdf_policy_promoted_controls_state_year(policy_long)
    promoted_lower_threshold = build_ccdf_policy_promoted_controls_state_year(
        policy_long,
        min_state_year_coverage=0.5,
    )

    assert "ccdf_control_copayment_required" not in promoted_default.columns
    assert "ccdf_control_copayment_required" in promoted_lower_threshold.columns
    assert set(promoted_default["ccdf_policy_promoted_min_state_year_coverage"]) == {0.75}
    assert set(promoted_lower_threshold["ccdf_policy_promoted_min_state_year_coverage"]) == {0.5}
    assert len(promoted_lower_threshold) == 1
    california = promoted_lower_threshold.loc[promoted_lower_threshold["state_fips"] == "06"].iloc[0]
    assert california["ccdf_policy_control_count"] == 1
    assert california["ccdf_policy_control_support_status"] == "observed_policy_promoted_controls"


def test_build_ccdf_policy_promoted_controls_state_year_preserves_state_year_rows_and_metadata():
    controls = build_ccdf_policy_controls_state_year(ccdf_policy_long())
    promoted = build_ccdf_policy_promoted_controls_state_year(ccdf_policy_long())

    assert len(promoted) == len(controls)
    assert promoted[["state_fips", "year"]].equals(controls[["state_fips", "year"]])
    assert set(promoted["ccdf_policy_control_support_status"]) == {"observed_policy_promoted_controls"}
    assert set(promoted["ccdf_policy_promoted_control_rule"]) == {"state_year_coverage_gte_threshold"}


def test_build_ccdf_policy_feature_audit_preserves_feature_rows():
    audit = build_ccdf_policy_feature_audit(ccdf_policy_long())

    assert not audit.empty
    assert {"state_fips", "year", "feature_name", "feature_value_text"} <= set(audit.columns)
    assert "copayment_required" in set(audit["feature_name"])


def test_build_ccdf_policy_state_year_maps_effective_dated_workbook_rows():
    policy_long = _policy_long_frame(
        [
            {
                "row_number": 1,
                "source_sheet": "CopayAdmin",
                "table_year": 2025,
                "state": "6",
                "cells": {
                    "BeginDat": "2021/10/01",
                    "EndDat": "9999/12/31",
                    "FamilyGroup": 98,
                    "ProviderType": 98,
                    "ProviderSubtype": 0,
                    "CopayCollect": 1,
                },
            },
            {
                "row_number": 2,
                "source_sheet": "Redetermination",
                "table_year": 2025,
                "state": "6",
                "cells": {
                    "BeginDat": "2021/10/01",
                    "EndDat": "2023/09/30",
                    "FamilyGroup": 98,
                    "ProviderType": 98,
                    "ProviderSubtype": 0,
                    "RedetermPeriod": 12,
                    "RedetermAppDocNew": 1,
                    "RedetermDocMethodOnline": 2,
                },
            },
            {
                "row_number": 3,
                "source_sheet": "Waitlist",
                "table_year": 2025,
                "state": "6",
                "cells": {
                    "BeginDat": "2024/10/01",
                    "EndDat": "9999/12/31",
                    "FamilyGroup": 98,
                    "ProviderType": 98,
                    "ProviderSubtype": 0,
                    "WaitList": 1,
                },
            },
            {
                "row_number": 4,
                "source_sheet": "CopayExempt",
                "table_year": 2025,
                "state": "6",
                "cells": {
                    "BeginDat": "2024/10/01",
                    "EndDat": "9999/12/31",
                    "FamilyGroup": 98,
                    "ProviderType": 98,
                    "ProviderSubtype": 0,
                    "CopayPovertyExempt": 1,
                    "CopayTANFExempt": 2,
                },
            },
        ],
        source_path="data/raw/ccdf/policies/urban_full_data.xlsx",
    )

    features = build_ccdf_policy_features_state_year(policy_long)
    controls = build_ccdf_policy_controls_state_year(policy_long)
    coverage = build_ccdf_policy_controls_coverage(policy_long)
    promoted = build_ccdf_policy_promoted_controls_state_year(policy_long)

    california_2025 = features.loc[(features["state_fips"] == "06") & (features["year"] == 2025)].iloc[0]
    california_2023 = features.loc[(features["state_fips"] == "06") & (features["year"] == 2023)].iloc[0]
    california_control = controls.loc[(controls["state_fips"] == "06") & (controls["year"] == 2025)].iloc[0]
    copayment_coverage = coverage.loc[coverage["control_name"] == "ccdf_control_copayment_required"].iloc[0]

    assert california_2025["ccdf_policy_copayment_required"] == "yes"
    assert california_2023["ccdf_policy_redetermination_period_months"] == "12"
    assert california_2023["ccdf_policy_redetermination_requires_new_documentation"] == "yes"
    assert california_2023["ccdf_policy_redetermination_online_submission_available"] == "no"
    assert pd.isna(california_2023["ccdf_policy_waitlist_active"])
    assert california_2025["ccdf_policy_waitlist_active"] == "yes"
    assert california_2025["ccdf_policy_copay_poverty_exempt"] == "yes"
    assert california_2025["ccdf_policy_copay_tanf_exempt"] == "no"
    assert california_control["ccdf_control_copayment_required"] == "yes"
    assert "ccdf_control_copay_poverty_exempt" not in controls.columns
    assert copayment_coverage["observed_state_year_rows"] == 5
    assert copayment_coverage["state_year_coverage_rate"] == 1.0
    assert set(promoted["ccdf_policy_promoted_controls_selected"]) == {"ccdf_control_copayment_required"}
    assert {
        "redetermination_requires_new_documentation",
        "redetermination_online_submission_available",
        "waitlist_active",
        "copay_poverty_exempt",
        "copay_tanf_exempt",
    } <= set(build_ccdf_policy_feature_audit(policy_long)["feature_name"])


def test_build_ccdf_policy_state_year_ignores_ambiguous_binary_policy_codes():
    policy_long = _policy_long_frame(
        [
            {
                "row_number": 1,
                "source_sheet": "Redetermination",
                "table_year": 2025,
                "state": "48",
                "cells": {
                    "BeginDat": "2022/10/01",
                    "EndDat": "2025/09/30",
                    "FamilyGroup": 98,
                    "ProviderType": 98,
                    "ProviderSubtype": 0,
                    "RedetermAppDocNew": 92,
                    "RedetermDocMethodOnline": 0,
                },
            },
            {
                "row_number": 2,
                "source_sheet": "Waitlist",
                "table_year": 2025,
                "state": "48",
                "cells": {
                    "BeginDat": "2022/10/01",
                    "EndDat": "2025/09/30",
                    "FamilyGroup": 98,
                    "ProviderType": 98,
                    "ProviderSubtype": 0,
                    "WaitList": 92,
                },
            },
            {
                "row_number": 3,
                "source_sheet": "CopayExempt",
                "table_year": 2025,
                "state": "48",
                "cells": {
                    "BeginDat": "2022/10/01",
                    "EndDat": "2025/09/30",
                    "FamilyGroup": 98,
                    "ProviderType": 98,
                    "ProviderSubtype": 0,
                    "CopayTANFExempt": 0,
                },
            },
        ],
        source_path="data/raw/ccdf/policies/urban_full_data.xlsx",
    )

    features = build_ccdf_policy_features_state_year(policy_long)

    assert features.empty

def test_build_ccdf_admin_state_year_prefers_explicit_subsidized_private_and_public_admin_components():
    explicit_split_admin_long = _admin_long_frame(
        {
            "state_fips": "06",
            "year": "2023",
            "ccdf_subsidized_private_slots": "320",
            "ccdf_public_admin_slots": "180",
            "children_served_average_monthly": "999",
        },
        source_path="data/raw/ccdf/admin/explicit_split.csv",
    )

    mapped = build_ccdf_admin_state_year(explicit_split_admin_long)
    california = mapped.loc[(mapped["state_fips"] == "06") & (mapped["year"] == 2023)].iloc[0]

    assert california["ccdf_subsidized_private_slots"] == 320.0
    assert california["ccdf_public_admin_slots"] == 180.0
    assert california["ccdf_children_served"] == 999.0
    assert california["ccdf_support_flag"] == "ccdf_explicit_split_observed"
    assert california["ccdf_admin_support_status"] == "observed_long_explicit_split"


def test_build_ccdf_admin_state_year_inferrs_public_admin_complement_when_subsidized_and_children_served_present():
    inferred_public_admin_long = _admin_long_frame(
        {
            "state_fips": "06",
            "year": "2023",
            "children_served_average_monthly": "500",
            "subsidized_private_slots": "320",
        },
        source_path="data/raw/ccdf/admin/infer_public_admin.csv",
    )

    mapped = build_ccdf_admin_state_year(inferred_public_admin_long)
    california = mapped.loc[(mapped["state_fips"] == "06") & (mapped["year"] == 2023)].iloc[0]

    assert california["ccdf_children_served"] == 500.0
    assert california["ccdf_subsidized_private_slots"] == 320.0
    assert california["ccdf_public_admin_slots"] == 180.0
    assert (
        california["ccdf_support_flag"]
        == "ccdf_inferred_public_admin_from_children_served_minus_subsidized_private"
    )
    assert california["ccdf_admin_support_status"] == "observed_long_inferred_public_admin_complement"


def test_build_ccdf_admin_state_year_inferrs_subsidized_private_complement_when_public_and_children_served_present():
    inferred_subsidized_private_long = _admin_long_frame(
        {
            "state_fips": "06",
            "year": "2023",
            "children_served_average_monthly": "500",
            "public_admin_slots": "180",
        },
        source_path="data/raw/ccdf/admin/infer_subsidized_private.csv",
    )

    mapped = build_ccdf_admin_state_year(inferred_subsidized_private_long)
    california = mapped.loc[(mapped["state_fips"] == "06") & (mapped["year"] == 2023)].iloc[0]

    assert california["ccdf_children_served"] == 500.0
    assert california["ccdf_subsidized_private_slots"] == 320.0
    assert california["ccdf_public_admin_slots"] == 180.0
    assert (
        california["ccdf_support_flag"]
        == "ccdf_inferred_subsidized_private_from_children_served_minus_public_admin"
    )
    assert california["ccdf_admin_support_status"] == "observed_long_inferred_subsidized_private_complement"


def test_build_ccdf_admin_state_year_maps_split_from_share_fields():
    share_based_admin_long = _admin_long_frame(
        {
            "state_fips": "06",
            "year": "2023",
            "children_served_average_monthly": "500",
            "subsidized_private_share": "0.64",
            "public_admin_share": "0.36",
        },
        source_path="data/raw/ccdf/admin/share_split.csv",
    )

    mapped = build_ccdf_admin_state_year(share_based_admin_long)
    california = mapped.loc[(mapped["state_fips"] == "06") & (mapped["year"] == 2023)].iloc[0]

    assert california["ccdf_children_served"] == 500.0
    assert california["ccdf_subsidized_private_slots"] == 320.0
    assert california["ccdf_public_admin_slots"] == 180.0
    assert california["ccdf_support_flag"] == "ccdf_split_observed_from_shares_or_mixed_components"
    assert california["ccdf_admin_support_status"] == "observed_long_split_from_shares_or_mixed_components"


def test_build_ccdf_admin_state_year_maps_split_from_percent_fields():
    percent_based_admin_long = _admin_long_frame(
        {
            "state_fips": "06",
            "year": "2023",
            "children_served_average_monthly": "500",
            "subsidized_private_percent": "64",
            "public_admin_percent": "36",
        },
        source_path="data/raw/ccdf/admin/percent_split.csv",
    )

    mapped = build_ccdf_admin_state_year(percent_based_admin_long)
    california = mapped.loc[(mapped["state_fips"] == "06") & (mapped["year"] == 2023)].iloc[0]

    assert california["ccdf_subsidized_private_slots"] == 320.0
    assert california["ccdf_public_admin_slots"] == 180.0
    assert california["ccdf_support_flag"] == "ccdf_split_observed_from_shares_or_mixed_components"


def test_build_ccdf_admin_state_year_keeps_children_served_proxy_when_split_components_absent():
    proxy_only_long = _admin_long_frame(
        {
            "state_fips": "06",
            "year": "2023",
            "children_served_average_monthly": "500",
        },
        source_path="data/raw/ccdf/admin/proxy_only.csv",
    )

    mapped = build_ccdf_admin_state_year(proxy_only_long)
    california = mapped.loc[(mapped["state_fips"] == "06") & (mapped["year"] == 2023)].iloc[0]

    assert california["ccdf_children_served"] == 500.0
    assert california["ccdf_subsidized_private_slots"] == 500.0
    assert california["ccdf_public_admin_slots"] == 0.0
    assert california["ccdf_support_flag"] == "ccdf_children_served_proxy_for_subsidized_private"
    assert california["ccdf_admin_support_status"] == "observed_long_proxy_subsidized_private"


def test_build_ccdf_admin_state_year_combines_real_style_children_payments_and_expenditures_rows():
    source_path = "data/raw/ccdf/admin/fy-2023-ccdf-data-tables-preliminary.xlsx"
    children_row = _admin_long_frame(
        {
            "state": "California",
            "children_served_average_monthly": "232500",
        },
        row_number=10,
        source_path=source_path,
        source_sheet="1. Children Served",
        table_year=2023,
    )
    payment_row = _admin_long_frame(
        {
            "state": "California",
            "public_admin_share": "0.1522",
            "subsidized_private_share": "0.8478",
            "grants_contracts_percent": "0.1522",
            "certificates_percent": "0.8234",
            "cash_percent": "0.0244",
            "payment_method_total_children": "262162",
            "split_proxy_source": "payment_method_shares",
        },
        row_number=10,
        source_path=source_path,
        source_sheet="2. Types of Payments",
        table_year=2023,
    )
    care_type_row = _admin_long_frame(
        {
            "state": "California",
            "ccdf_care_type_center_share": "0.3619",
            "ccdf_care_type_family_home_share": "0.3863",
        },
        row_number=10,
        source_path=source_path,
        source_sheet="3. Care by Type",
        table_year=2023,
    )
    regulation_row = _admin_long_frame(
        {
            "state": "California",
            "ccdf_regulated_share": "0.7607",
            "ccdf_unregulated_share": "0.2183",
        },
        row_number=10,
        source_path=source_path,
        source_sheet="4. Regulated vs Non-Regulated",
        table_year=2023,
    )
    relative_care_row = _admin_long_frame(
        {
            "state": "California",
            "ccdf_relative_care_total_children": "60491",
            "ccdf_relative_care_relative_children": str(60491 * 0.756327),
            "ccdf_relative_care_nonrelative_children": str(60491 * 0.243673),
        },
        row_number=10,
        source_path=source_path,
        source_sheet="5. Relative Care",
        table_year=2023,
    )
    setting_detail_row = _admin_long_frame(
        {
            "state": "California",
            "ccdf_setting_detail_regulated_total_share": "0.7607",
            "ccdf_setting_detail_unregulated_total_share": "0.2202",
            "ccdf_setting_detail_unregulated_relative_total_share": "0.1720",
            "ccdf_setting_detail_unregulated_nonrelative_total_share": "0.0273",
            "ccdf_setting_detail_invalid_share": "0.0191",
        },
        row_number=10,
        source_path=source_path,
        source_sheet="6. Setting Detail",
        table_year=2023,
    )
    expenditure_row = _admin_long_frame(
        {
            "state": "California",
            "ccdf_total_expenditures": "1472877994",
            "ccdf_admin_expenditures": "16270449",
            "ccdf_quality_activities_expenditures": "78231056",
            "ccdf_direct_services_expenditures": "1313067602",
        },
        row_number=10,
        source_path="data/raw/ccdf/admin/ccdf-expenditures-for-fy-2023-all-appropriation-years.xlsx",
        source_sheet="Table 3a",
        table_year=2023,
    )

    mapped = build_ccdf_admin_state_year(
        pd.concat(
            [
                children_row,
                payment_row,
                care_type_row,
                regulation_row,
                relative_care_row,
                setting_detail_row,
                expenditure_row,
            ],
            ignore_index=True,
        )
    )
    california = mapped.loc[(mapped["state_fips"] == "06") & (mapped["year"] == 2023)].iloc[0]

    assert california["ccdf_children_served"] == 232500.0
    assert california["ccdf_total_expenditures"] == 1472877994.0
    assert california["ccdf_admin_expenditures"] == 16270449.0
    assert california["ccdf_quality_activities_expenditures"] == 78231056.0
    assert california["ccdf_direct_services_expenditures"] == 1313067602.0
    assert california["ccdf_subsidized_private_slots"] == 197113.5
    assert california["ccdf_public_admin_slots"] == 35386.5
    assert california["ccdf_grants_contracts_share"] == 0.1522
    assert california["ccdf_certificates_share"] == 0.8234
    assert california["ccdf_cash_share"] == 0.0244
    assert california["ccdf_payment_method_total_children"] == 262162.0
    assert california["ccdf_payment_method_gap_vs_children_served"] == 29662.0
    assert california["ccdf_payment_method_ratio_vs_children_served"] == 262162.0 / 232500.0
    assert california["ccdf_care_type_center_share"] == 0.3619
    assert california["ccdf_care_type_family_home_share"] == 0.3863
    assert california["ccdf_regulated_share"] == 0.7607
    assert california["ccdf_unregulated_share"] == 0.2202
    assert california["ccdf_relative_care_total_children"] == 60491.0
    assert california["ccdf_relative_care_relative_children"] == pytest.approx(60491 * 0.756327)
    assert california["ccdf_relative_care_nonrelative_children"] == pytest.approx(60491 * 0.243673)
    assert california["ccdf_unregulated_relative_share"] == 0.172
    assert california["ccdf_unregulated_nonrelative_share"] == 0.0273
    assert california["ccdf_care_type_support_status"] == "observed_long_mapped"
    assert california["ccdf_regulation_support_status"] == "observed_long_mapped"
    assert california["ccdf_support_flag"] == "ccdf_split_proxy_from_payment_method_shares_moderate_gap"
    assert california["ccdf_admin_support_status"] == "observed_long_payment_method_share_proxy_moderate_gap"


def test_build_ccdf_admin_state_year_classifies_payment_method_proxy_reliability_tiers():
    source_path = "data/raw/ccdf/admin/fy-2023-ccdf-data-tables-preliminary.xlsx"
    close_proxy_rows = _admin_long_frame(
        {
            "state": "California",
            "children_served_average_monthly": "20000",
            "public_admin_share": "0.10",
            "subsidized_private_share": "0.90",
            "payment_method_total_children": "22000",
            "split_proxy_source": "payment_method_shares",
        },
        row_number=1,
        source_path=source_path,
        source_sheet="2. Types of Payments",
        table_year=2023,
    )
    moderate_proxy_rows = _admin_long_frame(
        {
            "state": "Texas",
            "children_served_average_monthly": "20000",
            "public_admin_share": "0.10",
            "subsidized_private_share": "0.90",
            "payment_method_total_children": "30000",
            "split_proxy_source": "payment_method_shares",
        },
        row_number=2,
        source_path=source_path,
        source_sheet="2. Types of Payments",
        table_year=2023,
    )
    large_proxy_rows = _admin_long_frame(
        {
            "state": "Florida",
            "children_served_average_monthly": "20000",
            "public_admin_share": "0.10",
            "subsidized_private_share": "0.90",
            "payment_method_total_children": "42000",
            "split_proxy_source": "payment_method_shares",
        },
        row_number=3,
        source_path=source_path,
        source_sheet="2. Types of Payments",
        table_year=2023,
    )

    mapped = build_ccdf_admin_state_year(
        pd.concat([close_proxy_rows, moderate_proxy_rows, large_proxy_rows], ignore_index=True)
    )

    california = mapped.loc[(mapped["state_fips"] == "06") & (mapped["year"] == 2023)].iloc[0]
    texas = mapped.loc[(mapped["state_fips"] == "48") & (mapped["year"] == 2023)].iloc[0]
    florida = mapped.loc[(mapped["state_fips"] == "12") & (mapped["year"] == 2023)].iloc[0]

    assert california["ccdf_support_flag"] == "ccdf_split_proxy_from_payment_method_shares_close_gap"
    assert california["ccdf_admin_support_status"] == "observed_long_payment_method_share_proxy_close_gap"
    assert texas["ccdf_support_flag"] == "ccdf_split_proxy_from_payment_method_shares_moderate_gap"
    assert texas["ccdf_admin_support_status"] == "observed_long_payment_method_share_proxy_moderate_gap"
    assert florida["ccdf_subsidized_private_slots"] == 20000.0
    assert florida["ccdf_public_admin_slots"] == 0.0
    assert (
        florida["ccdf_support_flag"]
        == "ccdf_split_proxy_from_payment_method_shares_large_gap_downgraded_to_children_served_proxy"
    )
    assert (
        florida["ccdf_admin_support_status"]
        == "observed_long_payment_method_share_proxy_large_gap_downgraded_to_children_served_proxy"
    )


def test_build_ccdf_admin_state_year_reclassifies_boundary_payment_mix_rows_as_inferred_nonproxy():
    source_path = "data/raw/ccdf/admin/fy-2023-ccdf-data-tables-preliminary.xlsx"
    certificate_only_row = _admin_long_frame(
        {
            "state": "Texas",
            "children_served_average_monthly": "20000",
            "public_admin_share": "0.00",
            "subsidized_private_share": "1.00",
            "grants_contracts_percent": "0.00",
            "certificates_percent": "1.00",
            "cash_percent": "0.00",
            "payment_method_total_children": "30000",
            "split_proxy_source": "payment_method_shares",
        },
        row_number=1,
        source_path=source_path,
        source_sheet="2. Types of Payments",
        table_year=2023,
    )
    grants_only_row = _admin_long_frame(
        {
            "state": "Florida",
            "children_served_average_monthly": "20000",
            "public_admin_share": "1.00",
            "subsidized_private_share": "0.00",
            "grants_contracts_percent": "1.00",
            "certificates_percent": "0.00",
            "cash_percent": "0.00",
            "payment_method_total_children": "26000",
            "split_proxy_source": "payment_method_shares",
        },
        row_number=2,
        source_path=source_path,
        source_sheet="2. Types of Payments",
        table_year=2023,
    )
    nonboundary_row = _admin_long_frame(
        {
            "state": "California",
            "children_served_average_monthly": "20000",
            "public_admin_share": "0.10",
            "subsidized_private_share": "0.90",
            "grants_contracts_percent": "0.10",
            "certificates_percent": "0.90",
            "cash_percent": "0.00",
            "payment_method_total_children": "30000",
            "split_proxy_source": "payment_method_shares",
        },
        row_number=3,
        source_path=source_path,
        source_sheet="2. Types of Payments",
        table_year=2023,
    )
    mapped = build_ccdf_admin_state_year(
        pd.concat([certificate_only_row, grants_only_row, nonboundary_row], ignore_index=True)
    )
    texas = mapped.loc[(mapped["state_fips"] == "48") & (mapped["year"] == 2023)].iloc[0]
    florida = mapped.loc[(mapped["state_fips"] == "12") & (mapped["year"] == 2023)].iloc[0]
    california = mapped.loc[(mapped["state_fips"] == "06") & (mapped["year"] == 2023)].iloc[0]

    assert texas["ccdf_subsidized_private_slots"] == 20000.0
    assert texas["ccdf_public_admin_slots"] == 0.0
    assert texas["ccdf_support_flag"] == "ccdf_inferred_zero_public_admin_from_payment_method_mix"
    assert texas["ccdf_admin_support_status"] == "observed_long_inferred_zero_public_admin_payment_method_mix"

    assert florida["ccdf_subsidized_private_slots"] == 0.0
    assert florida["ccdf_public_admin_slots"] == 20000.0
    assert florida["ccdf_support_flag"] == "ccdf_inferred_zero_subsidized_private_from_payment_method_mix"
    assert florida["ccdf_admin_support_status"] == "observed_long_inferred_zero_subsidized_private_payment_method_mix"

    assert california["ccdf_support_flag"] == "ccdf_split_proxy_from_payment_method_shares_moderate_gap"
    proxy_flag_count = int(
        mapped["ccdf_support_flag"]
        .astype(str)
        .str.startswith("ccdf_split_proxy_from_payment_method_shares", na=False)
        .sum()
    )
    assert proxy_flag_count == 1


def test_build_ccdf_admin_state_year_uses_relative_care_counts_to_fill_missing_relative_shares():
    source_path = "data/raw/ccdf/admin/fy-2023-ccdf-data-tables-preliminary.xlsx"
    children_row = _admin_long_frame(
        {
            "state": "California",
            "children_served_average_monthly": "232500",
        },
        row_number=10,
        source_path=source_path,
        source_sheet="1. Children Served",
        table_year=2023,
    )
    regulated_row = _admin_long_frame(
        {
            "state": "California",
            "ccdf_regulated_share": "0.7607",
            "ccdf_unregulated_share": "0.2183",
        },
        row_number=10,
        source_path=source_path,
        source_sheet="4. Regulated vs Non-Regulated",
        table_year=2023,
    )
    relative_care_row = _admin_long_frame(
        {
            "state": "California",
            "ccdf_relative_care_total_children": "60491",
            "ccdf_relative_care_relative_children": str(60491 * 0.756327),
            "ccdf_relative_care_nonrelative_children": str(60491 * 0.243673),
        },
        row_number=10,
        source_path=source_path,
        source_sheet="5. Relative Care",
        table_year=2023,
    )

    mapped = build_ccdf_admin_state_year(
        pd.concat([children_row, regulated_row, relative_care_row], ignore_index=True)
    )
    california = mapped.loc[(mapped["state_fips"] == "06") & (mapped["year"] == 2023)].iloc[0]

    assert california["ccdf_unregulated_share"] == 0.2183
    assert california["ccdf_unregulated_relative_share"] == pytest.approx((60491 * 0.756327) / 232500)
    assert california["ccdf_unregulated_nonrelative_share"] == pytest.approx((60491 * 0.243673) / 232500)
