from __future__ import annotations

import numpy as np
import pandas as pd

from unpaidwork.childcare.segmented_scenarios import (
    build_childcare_segmented_scenarios,
    build_segmented_scenario_diagnostics,
    build_segmented_scenario_inputs,
    build_segmented_state_year_summary,
    simulate_segmented_childcare_scenarios,
)


def _state_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "state_price_index": 120.0,
                "unpaid_quantity_proxy": 30.0,
                "benchmark_replacement_cost": 100.0,
                "state_price_observation_status": "observed_ndcp_support",
                "state_price_nowcast": False,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "state_price_index": 100.0,
                "unpaid_quantity_proxy": 10.0,
                "benchmark_replacement_cost": 85.0,
                "state_price_observation_status": "observed_ndcp_support",
                "state_price_nowcast": False,
            },
        ]
    )


def _solver_baseline_state_year() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "solver_private_unsubsidized_slots": 90.0,
                "solver_private_subsidized_slots": 30.0,
                "solver_public_admin_slots": 80.0,
                "solver_total_private_slots": 120.0,
                "solver_total_paid_slots": 200.0,
                "solver_exogenous_public_admin_slots": 80.0,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "solver_private_unsubsidized_slots": 40.0,
                "solver_private_subsidized_slots": 0.0,
                "solver_public_admin_slots": 20.0,
                "solver_total_private_slots": 40.0,
                "solver_total_paid_slots": 60.0,
                "solver_exogenous_public_admin_slots": 20.0,
            },
        ]
    )


def _solver_elasticity_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "solver_channel": "private_unsubsidized",
                "elasticity_family": "pooled_childcare_demand",
                "active_in_price_solver": True,
            },
            {
                "solver_channel": "private_subsidized",
                "elasticity_family": "pooled_childcare_demand",
                "active_in_price_solver": True,
            },
            {
                "solver_channel": "public_admin",
                "elasticity_family": "exogenous_non_price",
                "active_in_price_solver": False,
            },
        ]
    )


def _solver_policy_controls() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "ccdf_policy_control_count": 1,
                "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                "ccdf_control_copayment_required": "yes",
            }
        ]
    )


def _demand_summary() -> dict[str, float | int | bool]:
    return {
        "mode": "observed_core",
        "specification_profile": "full_controls",
        "n_obs": 200,
        "n_states": 2,
        "n_years": 1,
        "elasticity_at_mean": -0.4,
        "economically_admissible": True,
    }


def _supply_summary() -> dict[str, float | str]:
    return {
        "supply_elasticity": 0.6,
        "estimation_method": "sample_iv",
    }


def test_build_segmented_scenario_inputs_allocates_unpaid_only_across_private_channels():
    inputs = build_segmented_scenario_inputs(
        state_frame=_state_frame(),
        solver_baseline_state_year=_solver_baseline_state_year(),
        solver_elasticity_mapping=_solver_elasticity_mapping(),
        solver_policy_controls_state_year=_solver_policy_controls(),
    )

    california = inputs.loc[(inputs["state_fips"] == "06") & (inputs["year"] == 2021)]
    ca_unsubsidized = california.loc[california["solver_channel"] == "private_unsubsidized"].iloc[0]
    ca_subsidized = california.loc[california["solver_channel"] == "private_subsidized"].iloc[0]
    ca_public = california.loc[california["solver_channel"] == "public_admin"].iloc[0]

    assert np.isclose(float(ca_unsubsidized["market_quantity_proxy"]), 90.0)
    assert np.isclose(float(ca_subsidized["market_quantity_proxy"]), 30.0)
    assert np.isclose(float(ca_public["market_quantity_proxy"]), 80.0)
    assert np.isclose(float(ca_unsubsidized["unpaid_quantity_proxy"]), 22.5)
    assert np.isclose(float(ca_subsidized["unpaid_quantity_proxy"]), 7.5)
    assert np.isclose(float(ca_public["unpaid_quantity_proxy"]), 0.0)
    assert ca_unsubsidized["state_price_observation_status"] == "observed_ndcp_support"
    assert ca_unsubsidized["ccdf_control_copayment_required"] == "yes"


def test_simulate_segmented_childcare_scenarios_keeps_public_channel_price_invariant():
    inputs = build_segmented_scenario_inputs(
        state_frame=_state_frame(),
        solver_baseline_state_year=_solver_baseline_state_year(),
        solver_elasticity_mapping=_solver_elasticity_mapping(),
        solver_policy_controls_state_year=_solver_policy_controls(),
    )
    scenarios = simulate_segmented_childcare_scenarios(
        channel_inputs=inputs,
        alphas=[0.0, 0.5, 1.0],
        demand_summary=_demand_summary(),
        supply_summary=_supply_summary(),
    )

    public_rows = scenarios.loc[scenarios["solver_channel"] == "public_admin"].copy()
    private_rows = scenarios.loc[scenarios["solver_channel"] == "private_unsubsidized"].copy()

    assert not public_rows.empty
    assert np.isclose(
        pd.to_numeric(public_rows["p_alpha"], errors="coerce"),
        pd.to_numeric(public_rows["p_baseline"], errors="coerce"),
    ).all()
    assert np.isclose(
        pd.to_numeric(public_rows["p_shadow_marginal"], errors="coerce"),
        pd.to_numeric(public_rows["p_baseline"], errors="coerce"),
    ).all()
    assert (~public_rows["price_responsive"].astype(bool)).all()

    private_alpha0 = private_rows.loc[private_rows["alpha"].eq(0.0), "p_alpha"].iloc[0]
    private_alpha1 = private_rows.loc[private_rows["alpha"].eq(1.0), "p_alpha"].iloc[0]
    assert float(private_alpha1) >= float(private_alpha0)


def test_build_segmented_state_year_summary_preserves_quantity_accounting():
    inputs = build_segmented_scenario_inputs(
        state_frame=_state_frame(),
        solver_baseline_state_year=_solver_baseline_state_year(),
        solver_elasticity_mapping=_solver_elasticity_mapping(),
        solver_policy_controls_state_year=_solver_policy_controls(),
    )
    scenarios = simulate_segmented_childcare_scenarios(
        channel_inputs=inputs,
        alphas=[0.0, 0.5],
        demand_summary=_demand_summary(),
        supply_summary=_supply_summary(),
    )
    summary = build_segmented_state_year_summary(scenarios)

    expected_totals = (
        inputs.groupby(["state_fips", "year"], as_index=False)
        .agg(
            expected_market_total=("market_quantity_proxy", "sum"),
            expected_unpaid_total=("unpaid_quantity_proxy", "sum"),
        )
    )
    merged = summary.merge(expected_totals, on=["state_fips", "year"], how="left")
    assert np.isclose(
        pd.to_numeric(merged["market_quantity_total"], errors="coerce"),
        pd.to_numeric(merged["expected_market_total"], errors="coerce"),
    ).all()
    assert np.isclose(
        pd.to_numeric(merged["unpaid_quantity_total"], errors="coerce"),
        pd.to_numeric(merged["expected_unpaid_total"], errors="coerce"),
    ).all()

    california_half = summary.loc[
        (summary["state_fips"] == "06") & (summary["year"] == 2021) & (summary["alpha"].eq(0.5))
    ].iloc[0]
    alpha_slice = scenarios.loc[
        (scenarios["state_fips"] == "06") & (scenarios["year"] == 2021) & (scenarios["alpha"].eq(0.5))
    ].copy()
    manual_weighted = (
        pd.to_numeric(alpha_slice["p_alpha"], errors="coerce")
        * pd.to_numeric(alpha_slice["market_quantity_proxy"], errors="coerce")
    ).sum() / pd.to_numeric(alpha_slice["market_quantity_proxy"], errors="coerce").sum()
    assert np.isclose(float(california_half["quantity_weighted_p_alpha"]), float(manual_weighted))


def test_build_childcare_segmented_scenarios_returns_expected_output_contract():
    outputs = build_childcare_segmented_scenarios(
        state_frame=_state_frame(),
        solver_baseline_state_year=_solver_baseline_state_year(),
        solver_elasticity_mapping=_solver_elasticity_mapping(),
        solver_policy_controls_state_year=_solver_policy_controls(),
        alphas=[0.0, 0.5],
        demand_summary=_demand_summary(),
        supply_summary=_supply_summary(),
    )

    assert set(outputs) == {
        "segmented_channel_inputs",
        "segmented_channel_scenarios",
        "segmented_state_year_summary",
        "segmented_state_year_diagnostics",
    }
    assert not outputs["segmented_channel_inputs"].empty
    assert not outputs["segmented_channel_scenarios"].empty
    assert not outputs["segmented_state_year_summary"].empty
    diagnostics = outputs["segmented_state_year_diagnostics"]
    assert len(diagnostics) == 1
    diag = diagnostics.iloc[0]
    assert bool(diag["public_admin_invariant_prices"]) is True
    assert np.isclose(float(diag["unpaid_public_quantity_total"]), 0.0)

    required_input_columns = {
        "solver_channel",
        "price_responsive",
        "market_quantity_proxy",
        "unpaid_quantity_proxy",
        "state_price_observation_status",
    }
    assert required_input_columns <= set(outputs["segmented_channel_inputs"].columns)

    # Smoke the standalone diagnostics helper explicitly.
    diagnostic_frame = build_segmented_scenario_diagnostics(
        channel_scenarios=outputs["segmented_channel_scenarios"],
        demand_summary=_demand_summary(),
        supply_summary=_supply_summary(),
    )
    assert len(diagnostic_frame) == 1
    assert bool(diagnostic_frame.iloc[0]["public_admin_invariant_prices"]) is True

