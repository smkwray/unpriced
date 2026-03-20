from __future__ import annotations

import numpy as np
import pandas as pd
from unpriced.childcare.utilization import (
    build_childcare_utilization_outputs,
    build_public_program_slots_state_year,
    build_segmented_quantity_panel,
    build_survey_paid_use_targets,
)
from unpriced.childcare.ccdf import (
    build_ccdf_policy_promoted_controls_state_year,
    build_ccdf_policy_controls_state_year,
    build_ccdf_policy_features_state_year,
)
from unpriced.sample_data import ccdf_policy_long


def _acs_state_year_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"state_fips": "06", "year": 2021, "under5_population": 1000.0},
            {"state_fips": "06", "year": 2022, "under5_population": 1100.0},
            {"state_fips": "48", "year": 2021, "under5_population": 800.0},
        ]
    )


def _sipp_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": 2021,
                "subgroup": "under5_ref_parents",
                "any_paid_childcare_rate": 0.50,
                "center_care_rate": 0.30,
                "family_daycare_rate": 0.10,
                "nonrelative_care_rate": 0.10,
                "head_start_rate": 0.05,
                "nursery_preschool_rate": 0.20,
            },
            {
                "year": 2023,
                "subgroup": "under5_ref_parents",
                "any_paid_childcare_rate": 0.60,
                "center_care_rate": 0.35,
                "family_daycare_rate": 0.10,
                "nonrelative_care_rate": 0.15,
                "head_start_rate": 0.06,
                "nursery_preschool_rate": 0.22,
            },
        ]
    )


def _head_start_cross_section() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"state_fips": "06", "county_fips": "06037", "head_start_slots": 100.0},
            {"state_fips": "06", "county_fips": "06073", "head_start_slots": 50.0},
            {"state_fips": "48", "county_fips": "48201", "head_start_slots": 80.0},
        ]
    )


def _segment_price_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "segment_id": "infant_center_private",
                "segment_label": "infant / center / private",
                "segment_order": 0,
                "segment_child_age": "infant",
                "segment_provider_type": "center",
                "segment_channel": "private",
                "segment_weight_sum": 2.0,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "segment_id": "preschool_home_private",
                "segment_label": "preschool / home / private",
                "segment_order": 1,
                "segment_child_age": "preschool",
                "segment_provider_type": "home",
                "segment_channel": "private",
                "segment_weight_sum": 1.0,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "segment_id": "infant_center_private",
                "segment_label": "infant / center / private",
                "segment_order": 0,
                "segment_child_age": "infant",
                "segment_provider_type": "center",
                "segment_channel": "private",
                "segment_weight_sum": 4.0,
            },
        ]
    )


def _ccdf_state_year_mapped() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "ccdf_children_served": 200.0,
                "ccdf_public_admin_slots": 200.0,
                "ccdf_support_flag": "ccdf_admin_observed",
            },
            {
                "state_fips": "06",
                "year": 2022,
                "ccdf_children_served": 210.0,
                "ccdf_public_admin_slots": 210.0,
                "ccdf_support_flag": "ccdf_admin_observed",
            },
            {
                "state_fips": "48",
                "year": 2021,
                "ccdf_children_served": 120.0,
                "ccdf_public_admin_slots": 120.0,
                "ccdf_support_flag": "ccdf_admin_observed",
            },
        ]
    )


def _ccdf_state_year_explicit_split() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "ccdf_children_served": 260.0,
                "ccdf_public_admin_slots": 120.0,
                "ccdf_subsidized_private_slots": 140.0,
                "ccdf_grants_contracts_share": 0.46,
                "ccdf_certificates_share": 0.44,
                "ccdf_cash_share": 0.10,
                "ccdf_payment_method_total_children": 300.0,
                "ccdf_payment_method_gap_vs_children_served": 40.0,
                "ccdf_payment_method_ratio_vs_children_served": 300.0 / 260.0,
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_explicit_split",
            },
            {
                "state_fips": "06",
                "year": 2022,
                "ccdf_children_served": 240.0,
                "ccdf_public_admin_slots": 110.0,
                "ccdf_subsidized_private_slots": 130.0,
                "ccdf_grants_contracts_share": 0.45,
                "ccdf_certificates_share": 0.45,
                "ccdf_cash_share": 0.10,
                "ccdf_payment_method_total_children": 270.0,
                "ccdf_payment_method_gap_vs_children_served": 30.0,
                "ccdf_payment_method_ratio_vs_children_served": 270.0 / 240.0,
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_explicit_split",
            },
            {
                "state_fips": "48",
                "year": 2021,
                "ccdf_children_served": 150.0,
                "ccdf_public_admin_slots": 80.0,
                "ccdf_subsidized_private_slots": 70.0,
                "ccdf_grants_contracts_share": 0.53,
                "ccdf_certificates_share": 0.40,
                "ccdf_cash_share": 0.07,
                "ccdf_payment_method_total_children": 180.0,
                "ccdf_payment_method_gap_vs_children_served": 30.0,
                "ccdf_payment_method_ratio_vs_children_served": 180.0 / 150.0,
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_explicit_split",
            },
        ]
    )


def _ccdf_state_year_proxy_reliability_labeled() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "ccdf_children_served": 260.0,
                "ccdf_public_admin_slots": 120.0,
                "ccdf_subsidized_private_slots": 140.0,
                "ccdf_grants_contracts_share": 0.46,
                "ccdf_certificates_share": 0.44,
                "ccdf_cash_share": 0.10,
                "ccdf_payment_method_total_children": 300.0,
                "ccdf_payment_method_gap_vs_children_served": 40.0,
                "ccdf_payment_method_ratio_vs_children_served": 300.0 / 260.0,
                "ccdf_support_flag": "ccdf_split_proxy_from_payment_method_shares_close_gap",
                "ccdf_admin_support_status": "observed_long_payment_method_share_proxy_close_gap",
            },
            {
                "state_fips": "06",
                "year": 2022,
                "ccdf_children_served": 240.0,
                "ccdf_public_admin_slots": 110.0,
                "ccdf_subsidized_private_slots": 130.0,
                "ccdf_grants_contracts_share": 0.45,
                "ccdf_certificates_share": 0.45,
                "ccdf_cash_share": 0.10,
                "ccdf_payment_method_total_children": 330.0,
                "ccdf_payment_method_gap_vs_children_served": 90.0,
                "ccdf_payment_method_ratio_vs_children_served": 330.0 / 240.0,
                "ccdf_support_flag": "ccdf_split_proxy_from_payment_method_shares_moderate_gap",
                "ccdf_admin_support_status": "observed_long_payment_method_share_proxy_moderate_gap",
            },
            {
                "state_fips": "48",
                "year": 2021,
                "ccdf_children_served": 150.0,
                "ccdf_public_admin_slots": 80.0,
                "ccdf_subsidized_private_slots": 70.0,
                "ccdf_grants_contracts_share": 0.53,
                "ccdf_certificates_share": 0.40,
                "ccdf_cash_share": 0.07,
                "ccdf_payment_method_total_children": 360.0,
                "ccdf_payment_method_gap_vs_children_served": 210.0,
                "ccdf_payment_method_ratio_vs_children_served": 360.0 / 150.0,
                "ccdf_support_flag": "ccdf_split_proxy_from_payment_method_shares_large_gap",
                "ccdf_admin_support_status": "observed_long_payment_method_share_proxy_large_gap",
            },
        ]
    )


def _assert_ccdf_support_metadata(frame: pd.DataFrame) -> None:
    metadata_columns = [col for col in frame.columns if "support" in col or "flag" in col]
    assert metadata_columns, "expected support/status metadata columns"
    has_explicit_ccdf_metadata = False
    for column in metadata_columns:
        values = frame[column].dropna().astype(str).str.lower()
        if values.str.contains("ccdf").any():
            has_explicit_ccdf_metadata = True
            break
    assert has_explicit_ccdf_metadata, "expected explicit ccdf support metadata when ccdf slots are used"


def _ccdf_policy_features_state_year() -> pd.DataFrame:
    frame = build_ccdf_policy_features_state_year(ccdf_policy_long())
    frame["year"] = 2021
    return frame


def _ccdf_policy_controls_state_year() -> pd.DataFrame:
    frame = build_ccdf_policy_controls_state_year(ccdf_policy_long())
    frame["year"] = 2021
    return frame


def _ccdf_policy_promoted_controls_state_year() -> pd.DataFrame:
    frame = build_ccdf_policy_promoted_controls_state_year(ccdf_policy_long())
    frame["year"] = 2021
    return frame


def test_build_public_program_slots_state_year_adds_ccdf_on_top_of_head_start():
    public_programs = build_public_program_slots_state_year(
        _acs_state_year_frame(),
        _head_start_cross_section(),
        ccdf_state_year=_ccdf_state_year_mapped(),
    )

    california_2021 = public_programs.loc[
        (public_programs["state_fips"] == "06") & (public_programs["year"] == 2021)
    ].iloc[0]
    texas_2021 = public_programs.loc[
        (public_programs["state_fips"] == "48") & (public_programs["year"] == 2021)
    ].iloc[0]

    assert np.isclose(california_2021["head_start_slots"], 150.0)
    assert np.isclose(california_2021["ccdf_public_admin_slots"], 200.0)
    assert np.isclose(california_2021["public_admin_slots"], 350.0)
    assert np.isclose(texas_2021["public_admin_slots"], 200.0)
    _assert_ccdf_support_metadata(public_programs)


def test_build_segmented_quantity_panel_uses_ccdf_augmented_public_admin_slots():
    public_programs = build_public_program_slots_state_year(
        _acs_state_year_frame(),
        _head_start_cross_section(),
        ccdf_state_year=_ccdf_state_year_mapped(),
    )
    survey_targets = build_survey_paid_use_targets(_acs_state_year_frame(), _sipp_frame())
    quantity_panel = build_segmented_quantity_panel(public_programs, survey_targets, _segment_price_panel())

    california_2021_rows = quantity_panel.loc[
        (quantity_panel["state_fips"] == "06") & (quantity_panel["year"] == 2021)
    ]
    california_2021_public = california_2021_rows.loc[
        california_2021_rows["quantity_component"] == "public_admin"
    ].iloc[0]
    california_2021_private = california_2021_rows.loc[
        california_2021_rows["quantity_component"] == "private_unsubsidized"
    ]

    assert np.isclose(float(california_2021_public["public_admin_slots"]), 350.0)
    assert np.isclose(float(california_2021_public["quantity_slots"]), 350.0)
    assert np.isclose(float(california_2021_private["quantity_slots"].sum()), 150.0)
    assert np.isclose(float(california_2021_rows["quantity_slots"].sum()), 500.0)
    _assert_ccdf_support_metadata(california_2021_rows)


def test_build_segmented_quantity_panel_propagates_proxy_reliability_support_labels():
    public_programs = build_public_program_slots_state_year(
        _acs_state_year_frame(),
        _head_start_cross_section(),
        ccdf_state_year=_ccdf_state_year_proxy_reliability_labeled(),
    )
    survey_targets = build_survey_paid_use_targets(_acs_state_year_frame(), _sipp_frame())
    quantity_panel = build_segmented_quantity_panel(public_programs, survey_targets, _segment_price_panel())

    california_2021_public = public_programs.loc[
        (public_programs["state_fips"] == "06") & (public_programs["year"] == 2021)
    ].iloc[0]
    texas_2021_public = public_programs.loc[
        (public_programs["state_fips"] == "48") & (public_programs["year"] == 2021)
    ].iloc[0]
    california_2022_rows = quantity_panel.loc[
        (quantity_panel["state_fips"] == "06") & (quantity_panel["year"] == 2022)
    ]

    assert california_2021_public["ccdf_support_flag"] == "ccdf_split_proxy_from_payment_method_shares_close_gap"
    assert (
        california_2021_public["ccdf_admin_support_status"]
        == "observed_long_payment_method_share_proxy_close_gap"
    )
    assert texas_2021_public["ccdf_public_admin_slots"] == 0.0
    assert texas_2021_public["ccdf_subsidized_private_slots"] == 150.0
    assert (
        texas_2021_public["ccdf_support_flag"]
        == "ccdf_split_proxy_from_payment_method_shares_large_gap_downgraded_to_children_served_proxy"
    )
    assert (
        texas_2021_public["ccdf_admin_support_status"]
        == "observed_long_payment_method_share_proxy_large_gap_downgraded_to_children_served_proxy"
    )
    assert "ccdf_split_proxy_from_payment_method_shares_moderate_gap" in set(
        california_2022_rows["q0_support_flag"]
    )


def test_build_childcare_utilization_outputs_keeps_surface_and_adds_ccdf_state_year_inputs():
    outputs = build_childcare_utilization_outputs(
        acs_frame=_acs_state_year_frame(),
        sipp_frame=_sipp_frame(),
        head_start_frame=_head_start_cross_section(),
        ccdf_state_year=_ccdf_state_year_mapped(),
        ccdf_policy_features_state_year=_ccdf_policy_features_state_year(),
        ccdf_policy_controls_state_year=_ccdf_policy_promoted_controls_state_year(),
        segment_price_panel=_segment_price_panel(),
        config={"year_window": {"start": 2021, "end": 2022}},
    )

    # Enforce additive integration: no output-surface rewrite to support CCDF inputs.
    assert set(outputs) == {
        "public_program_slots",
        "survey_paid_use_targets",
        "quantity_by_segment",
        "reconciliation_diagnostics",
    }
    assert not outputs["public_program_slots"].empty
    assert not outputs["quantity_by_segment"].empty
    _assert_ccdf_support_metadata(outputs["public_program_slots"])
    _assert_ccdf_support_metadata(outputs["quantity_by_segment"])
    diagnostics = outputs["reconciliation_diagnostics"]
    california_2021 = diagnostics.loc[
        (diagnostics["state_fips"] == "06") & (diagnostics["year"] == 2021)
    ].iloc[0]
    assert california_2021["ccdf_policy_control_count"] >= 1
    assert california_2021["ccdf_policy_control_support_status"] == "observed_policy_promoted_controls"
    assert california_2021["ccdf_control_copayment_required"] == "yes"
    assert california_2021["ccdf_policy_promoted_control_rule"] == "state_year_coverage_gte_threshold"
    assert california_2021["ccdf_policy_feature_count"] >= 1
    assert california_2021["ccdf_policy_support_status"] == "observed_policy_long"
    assert california_2021["ccdf_policy_copayment_required"] == "yes"

def test_build_segmented_quantity_panel_separates_subsidized_private_from_unsubsidized_private():
    public_programs = build_public_program_slots_state_year(
        _acs_state_year_frame(),
        _head_start_cross_section(),
        ccdf_state_year=_ccdf_state_year_explicit_split(),
    )
    survey_targets = build_survey_paid_use_targets(_acs_state_year_frame(), _sipp_frame())
    quantity_panel = build_segmented_quantity_panel(public_programs, survey_targets, _segment_price_panel())

    california_2021_rows = quantity_panel.loc[
        (quantity_panel["state_fips"] == "06") & (quantity_panel["year"] == 2021)
    ]
    components = set(california_2021_rows["quantity_component"])
    assert {"private_subsidized", "private_unsubsidized", "public_admin"} <= components

    subsidized = california_2021_rows.loc[
        california_2021_rows["quantity_component"] == "private_subsidized"
    ]["quantity_slots"].sum()
    unsubsidized = california_2021_rows.loc[
        california_2021_rows["quantity_component"] == "private_unsubsidized"
    ]["quantity_slots"].sum()
    public_admin = california_2021_rows.loc[
        california_2021_rows["quantity_component"] == "public_admin"
    ]["quantity_slots"].sum()

    assert np.isclose(float(subsidized), 140.0)
    assert np.isclose(float(public_admin), 270.0)  # 150 head start + 120 CCDF public-admin
    assert np.isclose(float(unsubsidized), 90.0)  # total target 500 - public_admin 270 - subsidized 140
    assert np.isclose(float(california_2021_rows["quantity_slots"].sum()), 500.0)


def test_build_public_program_slots_state_year_marks_cross_section_carryforward():
    acs_state_year = _acs_state_year_frame()

    public_programs = build_public_program_slots_state_year(acs_state_year, _head_start_cross_section())

    california_2021 = public_programs.loc[
        (public_programs["state_fips"] == "06") & (public_programs["year"] == 2021)
    ].iloc[0]
    california_2022 = public_programs.loc[
        (public_programs["state_fips"] == "06") & (public_programs["year"] == 2022)
    ].iloc[0]

    assert california_2021["public_admin_slots"] == 150.0
    assert california_2022["public_admin_slots"] == 150.0
    assert california_2021["public_program_support_status"] == "cross_section_carry_forward"
    assert bool(california_2021["head_start_carry_forward"]) is True


def test_build_survey_paid_use_targets_uses_exact_and_carryforward_support_flags():
    targets = build_survey_paid_use_targets(_acs_state_year_frame(), _sipp_frame())

    california_2021 = targets.loc[(targets["state_fips"] == "06") & (targets["year"] == 2021)].iloc[0]
    california_2022 = targets.loc[(targets["state_fips"] == "06") & (targets["year"] == 2022)].iloc[0]

    assert california_2021["survey_target_support_status"] == "exact_year"
    assert california_2021["total_paid_slots_target"] == 500.0
    assert california_2022["survey_target_support_status"] == "carry_forward"
    assert california_2022["total_paid_slots_target"] == 550.0


def test_build_segmented_quantity_panel_reconciles_and_preserves_nonnegativity():
    public_programs = build_public_program_slots_state_year(_acs_state_year_frame(), _head_start_cross_section())
    survey_targets = build_survey_paid_use_targets(_acs_state_year_frame(), _sipp_frame())

    quantity_panel = build_segmented_quantity_panel(
        public_programs,
        survey_targets,
        _segment_price_panel(),
    )
    outputs = build_childcare_utilization_outputs(
        acs_frame=_acs_state_year_frame(),
        sipp_frame=_sipp_frame(),
        head_start_frame=_head_start_cross_section(),
        ccdf_policy_features_state_year=_ccdf_policy_features_state_year(),
        ccdf_policy_controls_state_year=_ccdf_policy_promoted_controls_state_year(),
        segment_price_panel=_segment_price_panel(),
        config={"year_window": {"start": 2021, "end": 2022}},
    )
    diagnostics = outputs["reconciliation_diagnostics"]

    california_2021 = diagnostics.loc[
        (diagnostics["state_fips"] == "06") & (diagnostics["year"] == 2021)
    ].iloc[0]
    texas_2021 = diagnostics.loc[
        (diagnostics["state_fips"] == "48") & (diagnostics["year"] == 2021)
    ].iloc[0]
    california_2022_rows = quantity_panel.loc[
        (quantity_panel["state_fips"] == "06") & (quantity_panel["year"] == 2022)
    ].copy()

    assert np.isclose(california_2021["private_paid_slots"], 350.0)
    assert np.isclose(california_2021["component_slots_sum"], 500.0)
    assert np.isclose(california_2021["component_sum_gap"], 0.0)
    assert bool(texas_2021["any_negative_quantity"]) is False
    assert "private_all" in set(california_2022_rows["segment_id"])
    assert np.isclose(california_2022_rows["quantity_slots"].sum(), 550.0)


def test_build_childcare_utilization_outputs_returns_all_artifacts():
    outputs = build_childcare_utilization_outputs(
        acs_frame=_acs_state_year_frame(),
        sipp_frame=_sipp_frame(),
        head_start_frame=_head_start_cross_section(),
        ccdf_state_year=_ccdf_state_year_explicit_split(),
        ccdf_policy_features_state_year=_ccdf_policy_features_state_year(),
        ccdf_policy_controls_state_year=_ccdf_policy_promoted_controls_state_year(),
        segment_price_panel=_segment_price_panel(),
        config={"year_window": {"start": 2021, "end": 2022}},
    )

    assert set(outputs) == {
        "public_program_slots",
        "survey_paid_use_targets",
        "quantity_by_segment",
        "reconciliation_diagnostics",
    }
    assert not outputs["public_program_slots"].empty
    assert not outputs["survey_paid_use_targets"].empty
    assert not outputs["quantity_by_segment"].empty
    assert not outputs["reconciliation_diagnostics"].empty
    assert "ccdf_grants_contracts_share" in outputs["public_program_slots"].columns
    assert "ccdf_payment_method_gap_vs_children_served" in outputs["public_program_slots"].columns
    assert "ccdf_policy_control_count" in outputs["reconciliation_diagnostics"].columns
    assert "ccdf_control_copayment_required" in outputs["reconciliation_diagnostics"].columns
    assert "ccdf_policy_promoted_control_rule" in outputs["reconciliation_diagnostics"].columns
    assert "ccdf_policy_feature_count" in outputs["reconciliation_diagnostics"].columns

    california_public = outputs["public_program_slots"].loc[
        (outputs["public_program_slots"]["state_fips"] == "06")
        & (outputs["public_program_slots"]["year"] == 2021)
    ].iloc[0]
    assert california_public["ccdf_grants_contracts_share"] == 0.46
    assert california_public["ccdf_payment_method_gap_vs_children_served"] == 40.0
