from __future__ import annotations

import numpy as np
import pandas as pd

from unpaidwork.childcare.licensing_harmonization import (
    build_licensing_backend_outputs,
    build_licensing_rules_harmonized,
    build_licensing_rules_raw_audit,
    build_licensing_stringency_index,
)


def _wide_licensing_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2020,
                "center_infant_ratio": 4.0,
                "center_toddler_ratio": 7.0,
                "center_infant_group_size": 8.0,
                "center_toddler_group_size": 14.0,
                "center_labor_intensity_index": 0.1964,
                "shock_label": "baseline",
                "source_note": "Original wording from source table A",
            },
            {
                "state_fips": "06",
                "year": 2021,
                "center_infant_ratio": np.nan,
                "center_toddler_ratio": 6.0,
                "center_infant_group_size": 6.0,
                "center_toddler_group_size": 12.0,
                "center_labor_intensity_index": 0.2361,
                "shock_label": "tightening",
                "source_note": "Original wording from source table B",
            },
            {
                "state_fips": "48",
                "year": 2021,
                "center_infant_ratio": 3.0,
                "center_toddler_ratio": 6.0,
                "center_infant_group_size": 6.0,
                "center_toddler_group_size": 12.0,
                "center_labor_intensity_index": 0.2361,
                "shock_label": "tightening",
                "source_note": "Original wording from source table B",
            },
        ]
    )


def _rule_level_fixture_single_rule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2020,
                "provider_type": "center",
                "age_group": "infant",
                "rule_family": "max_children_per_staff",
                "rule_name": "center_infant_ratio",
                "rule_value": 4.0,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "provider_type": "center",
                "age_group": "infant",
                "rule_family": "max_children_per_staff",
                "rule_name": "center_infant_ratio",
                "rule_value": 3.0,
            },
        ]
    )


def test_build_licensing_rules_raw_audit_preserves_source_wording_and_structure():
    licensing = _wide_licensing_fixture()
    audit = build_licensing_rules_raw_audit(licensing)

    assert len(audit) == len(licensing) * len(licensing.columns)
    assert {"state_fips", "year", "source_row_number", "raw_column_name", "value_text", "value_kind"} <= set(
        audit.columns
    )
    source_notes = set(
        audit.loc[audit["raw_column_name"].eq("source_note"), "value_text"].dropna().astype(str)
    )
    assert "Original wording from source table A" in source_notes
    assert "Original wording from source table B" in source_notes


def test_build_licensing_rules_harmonized_from_wide_inputs_sets_carry_forward_and_missing_flags():
    harmonized = build_licensing_rules_harmonized(_wide_licensing_fixture())

    # 2 states x 2 years x 5 rule specs
    assert len(harmonized) == 20
    required = {
        "provider_type",
        "age_group",
        "rule_family",
        "rule_value_observed",
        "rule_value",
        "licensing_rule_missing_original",
        "licensing_rule_carry_forward_applied",
        "licensing_rule_reference_year",
        "licensing_rule_support_status",
    }
    assert required <= set(harmonized.columns)

    california_infant_ratio_2021 = harmonized.loc[
        harmonized["state_fips"].eq("06")
        & harmonized["year"].eq(2021)
        & harmonized["provider_type"].eq("center")
        & harmonized["age_group"].eq("infant")
        & harmonized["rule_family"].eq("max_children_per_staff")
    ].iloc[0]
    assert bool(california_infant_ratio_2021["licensing_rule_missing_original"]) is True
    assert bool(california_infant_ratio_2021["licensing_rule_carry_forward_applied"]) is True
    assert int(california_infant_ratio_2021["licensing_rule_reference_year"]) == 2020
    assert float(california_infant_ratio_2021["rule_value"]) == 4.0
    assert california_infant_ratio_2021["licensing_rule_support_status"] == "carry_forward_rule"

    texas_2020 = harmonized.loc[harmonized["state_fips"].eq("48") & harmonized["year"].eq(2020)]
    assert texas_2020["licensing_rule_support_status"].eq("missing_rule").all()


def test_build_licensing_rules_harmonized_accepts_rule_level_input_contract():
    harmonized = build_licensing_rules_harmonized(_rule_level_fixture_single_rule())

    assert len(harmonized) == 2
    assert set(harmonized["rule_family"]) == {"max_children_per_staff"}
    assert set(harmonized["provider_type"]) == {"center"}
    assert harmonized["rule_column_source"].eq("center_infant_ratio").all()


def test_build_licensing_stringency_index_uses_documented_sparse_fallback():
    harmonized = build_licensing_rules_harmonized(_rule_level_fixture_single_rule())
    index = build_licensing_stringency_index(harmonized)

    assert set(index["stringency_pca_method"]) == {"equal_weight_fallback_sparse"}
    assert np.allclose(
        pd.to_numeric(index["stringency_equal_weight_index"], errors="coerce"),
        pd.to_numeric(index["stringency_pca_like_index"], errors="coerce"),
        equal_nan=True,
    )


def test_build_licensing_backend_outputs_returns_report_ready_tables():
    outputs = build_licensing_backend_outputs(_wide_licensing_fixture())

    assert set(outputs) == {
        "licensing_rules_raw_audit",
        "licensing_rules_harmonized",
        "licensing_stringency_index",
        "licensing_harmonization_summary",
    }
    summary = outputs["licensing_harmonization_summary"]
    assert {"state_fips", "year", "harmonized_rule_count", "harmonization_support_status"} <= set(summary.columns)
    assert summary["harmonized_rule_count"].ge(0).all()
