from __future__ import annotations

import numpy as np
import pandas as pd

from unpriced.childcare.segmented_reports import (
    build_childcare_segmented_reports,
    build_segmented_channel_response_summary,
    build_segmented_headline_summary,
    build_segmented_state_fallback_summary,
)


def _segmented_channel_scenarios() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "solver_channel": "private_unsubsidized",
                "alpha": 0.0,
                "market_quantity_proxy": 20.0,
                "unpaid_quantity_proxy": 6.0,
                "price_responsive": True,
                "p_baseline": 100.0,
                "p_shadow_marginal": 120.0,
                "p_alpha": 100.0,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "solver_channel": "private_unsubsidized",
                "alpha": 0.5,
                "market_quantity_proxy": 20.0,
                "unpaid_quantity_proxy": 6.0,
                "price_responsive": True,
                "p_baseline": 100.0,
                "p_shadow_marginal": 120.0,
                "p_alpha": 110.0,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "solver_channel": "private_subsidized",
                "alpha": 0.0,
                "market_quantity_proxy": 10.0,
                "unpaid_quantity_proxy": 2.0,
                "price_responsive": True,
                "p_baseline": 90.0,
                "p_shadow_marginal": 95.0,
                "p_alpha": 90.0,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "solver_channel": "private_subsidized",
                "alpha": 0.5,
                "market_quantity_proxy": 10.0,
                "unpaid_quantity_proxy": 2.0,
                "price_responsive": True,
                "p_baseline": 90.0,
                "p_shadow_marginal": 95.0,
                "p_alpha": 93.0,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "solver_channel": "public_admin",
                "alpha": 0.0,
                "market_quantity_proxy": 15.0,
                "unpaid_quantity_proxy": 0.0,
                "price_responsive": False,
                "p_baseline": 80.0,
                "p_shadow_marginal": 80.0,
                "p_alpha": 80.0,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "solver_channel": "public_admin",
                "alpha": 0.5,
                "market_quantity_proxy": 15.0,
                "unpaid_quantity_proxy": 0.0,
                "price_responsive": False,
                "p_baseline": 80.0,
                "p_shadow_marginal": 80.0,
                "p_alpha": 80.0,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "solver_channel": "private_unsubsidized",
                "alpha": 0.5,
                "market_quantity_proxy": 25.0,
                "unpaid_quantity_proxy": 4.0,
                "price_responsive": True,
                "p_baseline": 95.0,
                "p_shadow_marginal": 105.0,
                "p_alpha": 100.0,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "solver_channel": "private_subsidized",
                "alpha": 0.5,
                "market_quantity_proxy": 5.0,
                "unpaid_quantity_proxy": 1.0,
                "price_responsive": True,
                "p_baseline": 90.0,
                "p_shadow_marginal": 96.0,
                "p_alpha": 93.0,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "solver_channel": "public_admin",
                "alpha": 0.5,
                "market_quantity_proxy": 20.0,
                "unpaid_quantity_proxy": 0.0,
                "price_responsive": False,
                "p_baseline": 70.0,
                "p_shadow_marginal": 70.0,
                "p_alpha": 70.0,
            },
        ]
    )


def _segmented_state_year_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "alpha": 0.0,
                "market_quantity_total": 45.0,
                "unpaid_quantity_total": 8.0,
                "private_market_quantity_total": 30.0,
                "public_market_quantity_total": 15.0,
                "private_unpaid_quantity_total": 8.0,
                "public_unpaid_quantity_total": 0.0,
                "quantity_weighted_p_baseline": 93.333333,
                "quantity_weighted_p_shadow_marginal": 104.444444,
                "quantity_weighted_p_alpha": 93.333333,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "alpha": 0.5,
                "market_quantity_total": 45.0,
                "unpaid_quantity_total": 8.0,
                "private_market_quantity_total": 30.0,
                "public_market_quantity_total": 15.0,
                "private_unpaid_quantity_total": 8.0,
                "public_unpaid_quantity_total": 0.0,
                "quantity_weighted_p_baseline": 93.333333,
                "quantity_weighted_p_shadow_marginal": 104.444444,
                "quantity_weighted_p_alpha": 98.222222,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "alpha": 0.5,
                "market_quantity_total": 50.0,
                "unpaid_quantity_total": 5.0,
                "private_market_quantity_total": 30.0,
                "public_market_quantity_total": 20.0,
                "private_unpaid_quantity_total": 5.0,
                "public_unpaid_quantity_total": 0.0,
                "quantity_weighted_p_baseline": 85.500000,
                "quantity_weighted_p_shadow_marginal": 92.100000,
                "quantity_weighted_p_alpha": 88.700000,
            },
        ]
    )


def _segmented_state_year_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_rows": 9,
                "scenario_states": 2,
                "scenario_year_min": 2021,
                "scenario_year_max": 2021,
                "alpha_count": 2,
                "price_responsive_rows": 5,
                "non_price_responsive_rows": 4,
                "public_admin_invariant_prices": True,
                "unpaid_public_quantity_total": 0.0,
                "demand_elasticity_at_mean": -0.4,
                "solver_demand_elasticity_magnitude": 0.4,
                "supply_elasticity": 0.6,
            }
        ]
    )


def _state_year_support_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "any_segment_allocation_fallback": False,
                "any_private_allocation_fallback": False,
                "ccdf_support_flag": "ccdf_explicit_split_observed",
                "ccdf_admin_support_status": "observed_long_explicit_split",
                "q0_support_flag": "ccdf_plus_observed_segment_weights",
                "public_program_support_status": "head_start_plus_ccdf_observed",
                "explicit_ccdf_row_count": 3,
                "inferred_ccdf_row_count": 0,
                "proxy_ccdf_row_count": 0,
                "missing_ccdf_row_count": 0,
                "promoted_control_observed": True,
                "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                "ccdf_policy_control_count": 1,
                "ccdf_control_copayment_required": "yes",
            },
            {
                "state_fips": "48",
                "year": 2021,
                "any_segment_allocation_fallback": True,
                "any_private_allocation_fallback": True,
                "ccdf_support_flag": "ccdf_inferred_public_admin_from_children_served_minus_subsidized_private",
                "ccdf_admin_support_status": "observed_long_inferred_public_admin_complement",
                "q0_support_flag": "ccdf_plus_private_unallocated_fallback",
                "public_program_support_status": "ccdf_observed",
                "explicit_ccdf_row_count": 0,
                "inferred_ccdf_row_count": 2,
                "proxy_ccdf_row_count": 1,
                "missing_ccdf_row_count": 0,
                "promoted_control_observed": False,
                "ccdf_policy_control_support_status": "missing_policy_promoted_controls",
                "ccdf_policy_control_count": 0,
                "ccdf_control_copayment_required": pd.NA,
            },
        ]
    )


def test_channel_response_summary_computes_baseline_vs_alpha_response_columns():
    summary = build_segmented_channel_response_summary(_segmented_channel_scenarios())

    private_half = summary.loc[
        (summary["state_fips"] == "06")
        & (summary["year"] == 2021)
        & (summary["solver_channel"] == "private_unsubsidized")
        & (summary["alpha"].eq(0.5))
    ].iloc[0]
    public_half = summary.loc[
        (summary["state_fips"] == "06")
        & (summary["year"] == 2021)
        & (summary["solver_channel"] == "public_admin")
        & (summary["alpha"].eq(0.5))
    ].iloc[0]

    assert np.isclose(float(private_half["p_alpha_delta_vs_baseline"]), 10.0)
    assert np.isclose(float(private_half["p_alpha_pct_change_vs_baseline"]), 0.1)
    assert np.isclose(float(private_half["p_shadow_delta_vs_baseline"]), 20.0)
    assert bool(private_half["public_admin_price_invariant"]) is False
    assert bool(public_half["public_admin_price_invariant"]) is True


def test_fallback_summary_preserves_support_metadata_after_join():
    fallback = build_segmented_state_fallback_summary(
        segmented_state_year_summary=_segmented_state_year_summary(),
        state_year_support_summary=_state_year_support_summary(),
    )

    assert len(fallback) == 2
    california = fallback.loc[(fallback["state_fips"] == "06") & (fallback["year"] == 2021)].iloc[0]
    texas = fallback.loc[(fallback["state_fips"] == "48") & (fallback["year"] == 2021)].iloc[0]

    assert np.isclose(float(california["headline_alpha"]), 0.5)
    assert california["ccdf_support_flag"] == "ccdf_explicit_split_observed"
    assert bool(california["promoted_control_observed"]) is True
    assert bool(california["has_support_row"]) is True

    assert np.isclose(float(texas["headline_alpha"]), 0.5)
    assert bool(texas["any_segment_allocation_fallback"]) is True
    assert bool(texas["any_private_allocation_fallback"]) is True
    assert texas["ccdf_admin_support_status"] == "observed_long_inferred_public_admin_complement"


def test_headline_summary_marks_public_invariance_and_counts_fallback_states():
    headline = build_segmented_headline_summary(
        segmented_channel_scenarios=_segmented_channel_scenarios(),
        segmented_state_year_summary=_segmented_state_year_summary(),
        segmented_state_year_diagnostics=_segmented_state_year_diagnostics(),
        state_year_support_summary=_state_year_support_summary(),
    )

    assert np.isclose(float(headline["headline_alpha"]), 0.5)
    assert bool(headline["public_admin_invariant_prices"]) is True
    assert int(headline["fallback_state_year_count"]) == 1
    assert int(headline["fallback_state_count"]) == 1
    assert int(headline["state_year_count"]) == 2


def test_top_level_output_contract_and_markdown_smoke():
    outputs = build_childcare_segmented_reports(
        segmented_channel_scenarios=_segmented_channel_scenarios(),
        segmented_state_year_summary=_segmented_state_year_summary(),
        segmented_state_year_diagnostics=_segmented_state_year_diagnostics(),
        state_year_support_summary=_state_year_support_summary(),
    )

    assert set(outputs) == {
        "segmented_channel_response_summary",
        "segmented_state_fallback_summary",
        "segmented_headline_summary",
        "segmented_report_markdown",
    }
    assert not outputs["segmented_channel_response_summary"].empty
    assert not outputs["segmented_state_fallback_summary"].empty
    assert isinstance(outputs["segmented_headline_summary"], dict)
    markdown = outputs["segmented_report_markdown"]
    assert isinstance(markdown, str)
    assert "# segmented_childcare_report" in markdown
    assert "headline_alpha: 0.500000" in markdown
    assert "public_admin_invariant_prices: true" in markdown
