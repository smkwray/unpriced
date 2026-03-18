from __future__ import annotations

import numpy as np
import pandas as pd

from unpaidwork.childcare.report_tables import (
    build_childcare_report_tables,
    build_state_year_channel_summary,
    build_state_year_policy_quantity_summary,
    build_state_year_support_summary,
)


def _q0_segmented_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "segment_id": "infant_center_private",
                "quantity_component": "private_unsubsidized",
                "quantity_slots": 60.0,
                "segment_allocation_fallback": False,
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_long_explicit_split",
                "q0_support_flag": "ccdf_plus_observed_segment_weights",
                "public_program_support_status": "head_start_plus_ccdf_observed",
                "total_paid_slots_target": 200.0,
                "reconciled_paid_slots": 200.0,
                "accounting_gap_from_target": 0.0,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "segment_id": "preschool_home_private",
                "quantity_component": "private_unsubsidized",
                "quantity_slots": 30.0,
                "segment_allocation_fallback": False,
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_long_explicit_split",
                "q0_support_flag": "ccdf_plus_observed_segment_weights",
                "public_program_support_status": "head_start_plus_ccdf_observed",
                "total_paid_slots_target": 200.0,
                "reconciled_paid_slots": 200.0,
                "accounting_gap_from_target": 0.0,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "segment_id": "infant_center_private",
                "quantity_component": "private_subsidized",
                "quantity_slots": 20.0,
                "segment_allocation_fallback": False,
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_long_explicit_split",
                "q0_support_flag": "ccdf_subsidized_private_allocated_by_observed_segment_weights",
                "public_program_support_status": "head_start_plus_ccdf_observed",
                "total_paid_slots_target": 200.0,
                "reconciled_paid_slots": 200.0,
                "accounting_gap_from_target": 0.0,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "segment_id": "preschool_home_private",
                "quantity_component": "private_subsidized",
                "quantity_slots": 10.0,
                "segment_allocation_fallback": False,
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_long_explicit_split",
                "q0_support_flag": "ccdf_subsidized_private_allocated_by_observed_segment_weights",
                "public_program_support_status": "head_start_plus_ccdf_observed",
                "total_paid_slots_target": 200.0,
                "reconciled_paid_slots": 200.0,
                "accounting_gap_from_target": 0.0,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "segment_id": "public_head_start",
                "quantity_component": "public_admin",
                "quantity_slots": 80.0,
                "segment_allocation_fallback": False,
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_long_explicit_split",
                "q0_support_flag": "ccdf_explicit_split_observed",
                "public_program_support_status": "head_start_plus_ccdf_observed",
                "total_paid_slots_target": 200.0,
                "reconciled_paid_slots": 200.0,
                "accounting_gap_from_target": 0.0,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "segment_id": "infant_center_private",
                "quantity_component": "private_unsubsidized",
                "quantity_slots": 40.0,
                "segment_allocation_fallback": True,
                "ccdf_support_flag": "ccdf_inferred_public_admin_from_children_served_minus_subsidized_private",
                "ccdf_admin_support_status": "observed_long_inferred_public_admin_complement",
                "q0_support_flag": "ccdf_plus_private_unallocated_fallback",
                "public_program_support_status": "ccdf_observed",
                "total_paid_slots_target": 120.0,
                "reconciled_paid_slots": 120.0,
                "accounting_gap_from_target": 0.0,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "segment_id": "infant_center_private",
                "quantity_component": "private_subsidized",
                "quantity_slots": 10.0,
                "segment_allocation_fallback": True,
                "ccdf_support_flag": "ccdf_inferred_public_admin_from_children_served_minus_subsidized_private",
                "ccdf_admin_support_status": "observed_long_inferred_public_admin_complement",
                "q0_support_flag": "ccdf_subsidized_private_plus_private_unallocated_fallback",
                "public_program_support_status": "ccdf_observed",
                "total_paid_slots_target": 120.0,
                "reconciled_paid_slots": 120.0,
                "accounting_gap_from_target": 0.0,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "segment_id": "public_head_start",
                "quantity_component": "public_admin",
                "quantity_slots": 70.0,
                "segment_allocation_fallback": False,
                "ccdf_support_flag": "ccdf_children_served_proxy",
                "ccdf_admin_support_status": "proxy_children_served_split",
                "q0_support_flag": "ccdf_children_served_proxy",
                "public_program_support_status": "ccdf_observed",
                "total_paid_slots_target": 120.0,
                "reconciled_paid_slots": 120.0,
                "accounting_gap_from_target": 0.0,
            },
        ]
    )


def _promoted_controls_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "ccdf_control_copayment_required": "yes",
                "ccdf_policy_control_count": 1,
                "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                "ccdf_policy_promoted_controls_selected": "ccdf_control_copayment_required",
                "ccdf_policy_promoted_control_rule": "state_year_coverage_gte_threshold",
                "ccdf_policy_promoted_min_state_year_coverage": 0.75,
            }
        ]
    )


def _utilization_diagnostics_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "component_sum_gap": 0.0,
                "any_private_allocation_fallback": False,
                "any_negative_quantity": False,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "component_sum_gap": 0.0,
                "any_private_allocation_fallback": True,
                "any_negative_quantity": False,
            },
        ]
    )


def test_build_state_year_channel_summary_computes_channel_shares_and_private_public_splits():
    summary = build_state_year_channel_summary(_q0_segmented_fixture())
    assert {"private_unsubsidized", "private_subsidized", "public_admin"} == set(summary["solver_channel"])

    california = summary.loc[(summary["state_fips"] == "06") & (summary["year"] == 2021)]
    assert np.isclose(float(california.loc[california["solver_channel"] == "private_unsubsidized", "quantity_slots"].iloc[0]), 90.0)
    assert np.isclose(float(california.loc[california["solver_channel"] == "private_subsidized", "quantity_slots"].iloc[0]), 30.0)
    assert np.isclose(float(california.loc[california["solver_channel"] == "public_admin", "quantity_slots"].iloc[0]), 80.0)
    assert np.isclose(float(california["channel_share_of_paid"].sum()), 1.0)
    assert np.isclose(float(california["private_share_of_paid"].iloc[0]), 0.60)
    assert np.isclose(float(california["public_share_of_paid"].iloc[0]), 0.40)


def test_build_state_year_policy_quantity_summary_joins_promoted_copayment_controls():
    channel_summary = build_state_year_channel_summary(_q0_segmented_fixture())
    policy_summary = build_state_year_policy_quantity_summary(channel_summary, _promoted_controls_fixture())

    assert len(policy_summary) == len(channel_summary)
    california = policy_summary.loc[(policy_summary["state_fips"] == "06") & (policy_summary["year"] == 2021)]
    texas = policy_summary.loc[(policy_summary["state_fips"] == "48") & (policy_summary["year"] == 2021)]

    assert set(california["ccdf_control_copayment_required"]) == {"yes"}
    assert set(california["ccdf_policy_control_support_status"]) == {"observed_policy_promoted_controls"}
    assert california["promoted_control_observed"].all()
    assert texas["ccdf_control_copayment_required"].isna().all()
    assert (texas["ccdf_policy_control_count"] == 0).all()


def test_build_state_year_support_summary_tracks_explicit_inferred_proxy_and_keeps_rows():
    support = build_state_year_support_summary(
        q0_segmented=_q0_segmented_fixture(),
        utilization_diagnostics=_utilization_diagnostics_fixture(),
        promoted_controls=_promoted_controls_fixture(),
    )

    assert len(support) == 2
    california = support.loc[(support["state_fips"] == "06") & (support["year"] == 2021)].iloc[0]
    texas = support.loc[(support["state_fips"] == "48") & (support["year"] == 2021)].iloc[0]

    assert california["explicit_ccdf_row_count"] == 5
    assert california["inferred_ccdf_row_count"] == 0
    assert california["proxy_ccdf_row_count"] == 0
    assert california["ccdf_policy_control_support_status"] == "observed_policy_promoted_controls"
    assert bool(california["promoted_control_observed"]) is True

    assert texas["explicit_ccdf_row_count"] == 0
    assert texas["inferred_ccdf_row_count"] == 2
    assert texas["proxy_ccdf_row_count"] == 1
    assert bool(texas["any_private_allocation_fallback"]) is True
    assert pd.isna(texas["ccdf_control_copayment_required"])
    assert texas["ccdf_policy_control_count"] == 0


def test_build_childcare_report_tables_returns_required_additive_outputs():
    outputs = build_childcare_report_tables(
        q0_segmented=_q0_segmented_fixture(),
        promoted_controls=_promoted_controls_fixture(),
        utilization_diagnostics=_utilization_diagnostics_fixture(),
    )

    assert set(outputs) == {
        "state_year_channel_summary",
        "state_year_policy_quantity_summary",
        "state_year_support_summary",
    }
    assert not outputs["state_year_channel_summary"].empty
    assert not outputs["state_year_policy_quantity_summary"].empty
    assert not outputs["state_year_support_summary"].empty
