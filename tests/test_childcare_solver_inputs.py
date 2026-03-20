from __future__ import annotations

import numpy as np
import pandas as pd

from unpriced.childcare.solver_inputs import (
    build_childcare_solver_inputs,
    build_solver_baseline_state_year,
    build_solver_channel_quantities,
    build_solver_elasticity_mapping,
    build_solver_policy_controls_state_year,
)


def _q0_segmented() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "segment_id": "seg_a",
                "quantity_component": "private_unsubsidized",
                "quantity_slots": 60.0,
                "segment_allocation_fallback": False,
                "ndcp_segment_support": True,
                "public_program_support_status": "head_start_plus_ccdf_observed",
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_long_explicit_split",
                "q0_support_flag": "ccdf_plus_observed_segment_weights",
            },
            {
                "state_fips": "06",
                "year": 2021,
                "segment_id": "seg_b",
                "quantity_component": "private_unsubsidized",
                "quantity_slots": 40.0,
                "segment_allocation_fallback": True,
                "ndcp_segment_support": False,
                "public_program_support_status": "head_start_plus_ccdf_observed",
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_long_explicit_split",
                "q0_support_flag": "ccdf_plus_private_unallocated_fallback",
            },
            {
                "state_fips": "06",
                "year": 2021,
                "segment_id": "seg_a",
                "quantity_component": "private_subsidized",
                "quantity_slots": 20.0,
                "segment_allocation_fallback": False,
                "ndcp_segment_support": True,
                "public_program_support_status": "head_start_plus_ccdf_observed",
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_long_explicit_split",
                "q0_support_flag": "ccdf_subsidized_private_allocated_by_observed_segment_weights",
            },
            {
                "state_fips": "06",
                "year": 2021,
                "segment_id": "seg_b",
                "quantity_component": "private_subsidized",
                "quantity_slots": 10.0,
                "segment_allocation_fallback": False,
                "ndcp_segment_support": True,
                "public_program_support_status": "head_start_plus_ccdf_observed",
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_long_explicit_split",
                "q0_support_flag": "ccdf_subsidized_private_allocated_by_observed_segment_weights",
            },
            {
                "state_fips": "06",
                "year": 2021,
                "segment_id": "public_head_start",
                "quantity_component": "public_admin",
                "quantity_slots": 30.0,
                "segment_allocation_fallback": False,
                "ndcp_segment_support": False,
                "public_program_support_status": "head_start_plus_ccdf_observed",
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_long_explicit_split",
                "q0_support_flag": "ccdf_explicit_split_observed",
            },
            {
                "state_fips": "48",
                "year": 2021,
                "segment_id": "private_all",
                "quantity_component": "private_unsubsidized",
                "quantity_slots": 50.0,
                "segment_allocation_fallback": True,
                "ndcp_segment_support": False,
                "public_program_support_status": "ccdf_observed",
                "ccdf_support_flag": "ccdf_inferred_public_admin_from_children_served_minus_subsidized_private",
                "ccdf_admin_support_status": "observed_long_inferred_public_admin_complement",
                "q0_support_flag": "private_unallocated_fallback",
            },
            {
                "state_fips": "48",
                "year": 2021,
                "segment_id": "private_all",
                "quantity_component": "private_subsidized",
                "quantity_slots": 5.0,
                "segment_allocation_fallback": True,
                "ndcp_segment_support": False,
                "public_program_support_status": "ccdf_observed",
                "ccdf_support_flag": "ccdf_inferred_public_admin_from_children_served_minus_subsidized_private",
                "ccdf_admin_support_status": "observed_long_inferred_public_admin_complement",
                "q0_support_flag": "ccdf_subsidized_private_plus_private_unallocated_fallback",
            },
            {
                "state_fips": "48",
                "year": 2021,
                "segment_id": "public_head_start",
                "quantity_component": "public_admin",
                "quantity_slots": 20.0,
                "segment_allocation_fallback": False,
                "ndcp_segment_support": False,
                "public_program_support_status": "ccdf_observed",
                "ccdf_support_flag": "ccdf_inferred_public_admin_from_children_served_minus_subsidized_private",
                "ccdf_admin_support_status": "observed_long_inferred_public_admin_complement",
                "q0_support_flag": "ccdf_inferred_public_admin_from_children_served_minus_subsidized_private",
            },
        ]
    )


def _promoted_controls() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "ccdf_policy_control_count": 1,
                "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                "ccdf_policy_promoted_controls_selected": "ccdf_control_copayment_required",
                "ccdf_policy_promoted_control_rule": "state_year_coverage_gte_threshold",
                "ccdf_policy_promoted_min_state_year_coverage": 0.75,
                "ccdf_control_copayment_required": "yes",
                "ccdf_policy_copayment_required": "yes",
            }
        ]
    )


def _segment_price_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"state_fips": "06", "year": 2021, "segment_id": "seg_a", "segment_channel": "private"},
            {"state_fips": "06", "year": 2021, "segment_id": "seg_b", "segment_channel": "private"},
            {"state_fips": "48", "year": 2021, "segment_id": "private_all", "segment_channel": "private"},
        ]
    )


def test_build_solver_channel_quantities_maps_locked_channels_and_preserves_state_year_totals():
    q0_segmented = _q0_segmented()
    solver_quantities = build_solver_channel_quantities(q0_segmented)

    assert set(solver_quantities["solver_channel"]) == {
        "private_unsubsidized",
        "private_subsidized",
        "public_admin",
    }
    assert len(solver_quantities) == 6

    california_private_unsub = solver_quantities.loc[
        (solver_quantities["state_fips"] == "06")
        & (solver_quantities["year"] == 2021)
        & (solver_quantities["solver_channel"] == "private_unsubsidized")
    ].iloc[0]
    texas_public_admin = solver_quantities.loc[
        (solver_quantities["state_fips"] == "48")
        & (solver_quantities["year"] == 2021)
        & (solver_quantities["solver_channel"] == "public_admin")
    ].iloc[0]

    assert np.isclose(float(california_private_unsub["quantity_slots"]), 100.0)
    assert bool(california_private_unsub["price_responsive"]) is True
    assert bool(california_private_unsub["any_segment_allocation_fallback"]) is True
    assert california_private_unsub["source_quantity_component"] == "private_unsubsidized"
    assert bool(texas_public_admin["price_responsive"]) is False

    q0_totals = (
        q0_segmented.groupby(["state_fips", "year"], as_index=False)
        .agg(total_quantity_slots=("quantity_slots", "sum"))
    )
    solver_totals = (
        solver_quantities.groupby(["state_fips", "year"], as_index=False)
        .agg(total_quantity_slots=("quantity_slots", "sum"))
    )
    assert q0_totals.equals(solver_totals)


def test_build_solver_baseline_state_year_wide_fields_and_exogenous_public_quantity():
    solver_quantities = build_solver_channel_quantities(_q0_segmented())
    baseline = build_solver_baseline_state_year(solver_quantities)

    assert len(baseline) == 2
    california = baseline.loc[(baseline["state_fips"] == "06") & (baseline["year"] == 2021)].iloc[0]
    texas = baseline.loc[(baseline["state_fips"] == "48") & (baseline["year"] == 2021)].iloc[0]

    assert np.isclose(float(california["solver_private_unsubsidized_slots"]), 100.0)
    assert np.isclose(float(california["solver_private_subsidized_slots"]), 30.0)
    assert np.isclose(float(california["solver_public_admin_slots"]), 30.0)
    assert np.isclose(float(california["solver_total_paid_slots"]), 160.0)
    assert np.isclose(float(california["solver_exogenous_public_admin_slots"]), 30.0)
    assert np.isclose(float(texas["solver_total_private_slots"]), 55.0)


def test_build_solver_elasticity_mapping_marks_private_channels_active_and_public_exogenous():
    mapping = build_solver_elasticity_mapping()

    assert list(mapping["solver_channel"]) == ["private_unsubsidized", "private_subsidized", "public_admin"]
    private_rows = mapping.loc[mapping["solver_channel"].isin(["private_unsubsidized", "private_subsidized"])]
    public_row = mapping.loc[mapping["solver_channel"] == "public_admin"].iloc[0]

    assert set(private_rows["elasticity_family"]) == {"pooled_childcare_demand"}
    assert private_rows["active_in_price_solver"].all()
    assert bool(public_row["active_in_price_solver"]) is False
    assert public_row["elasticity_family"] == "exogenous_non_price"


def test_build_solver_policy_controls_state_year_keeps_promoted_controls_and_state_years():
    solver_quantities = build_solver_channel_quantities(_q0_segmented())
    controls = build_solver_policy_controls_state_year(
        promoted_controls=_promoted_controls(),
        solver_channel_quantities=solver_quantities,
    )

    assert len(controls) == 2
    assert "ccdf_control_copayment_required" in controls.columns
    assert "ccdf_policy_copayment_required" not in controls.columns

    california = controls.loc[(controls["state_fips"] == "06") & (controls["year"] == 2021)].iloc[0]
    texas = controls.loc[(controls["state_fips"] == "48") & (controls["year"] == 2021)].iloc[0]

    assert california["ccdf_control_copayment_required"] == "yes"
    assert texas["ccdf_policy_control_count"] == 0
    assert texas["ccdf_policy_control_support_status"] == "missing_policy_promoted_controls"


def test_build_childcare_solver_inputs_returns_all_artifacts_and_ndcp_support_summary():
    outputs = build_childcare_solver_inputs(
        q0_segmented=_q0_segmented(),
        promoted_controls=_promoted_controls(),
        ndcp_segment_prices=_segment_price_panel(),
    )

    assert set(outputs) == {
        "solver_channel_quantities",
        "solver_baseline_state_year",
        "solver_elasticity_mapping",
        "solver_policy_controls_state_year",
    }
    solver_quantities = outputs["solver_channel_quantities"]
    assert {"ndcp_private_segment_count", "ndcp_price_panel_support_status"} <= set(solver_quantities.columns)
    california_private_unsub = solver_quantities.loc[
        (solver_quantities["state_fips"] == "06")
        & (solver_quantities["year"] == 2021)
        & (solver_quantities["solver_channel"] == "private_unsubsidized")
    ].iloc[0]
    assert california_private_unsub["ndcp_private_segment_count"] == 2
    assert california_private_unsub["ndcp_price_panel_support_status"] == "observed_private_segments"
