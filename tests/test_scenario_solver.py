from __future__ import annotations

import pandas as pd
import pytest

from unpriced.models.scenario_solver import (
    SolverMetadata,
    bootstrap_childcare_intervals,
    dual_shift_zero_price_frontier,
    solve_price,
    solve_price_dual_shift,
    summarize_childcare_scenario_diagnostics,
)
from unpriced.models.supply_curve import summarize_supply_elasticity


def _make_state_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "01",
                "unpaid_childcare_hours": 100.0,
                "state_price_index": 100.0,
                "outside_option_wage": 10.0,
                "parent_employment_rate": 0.75,
                "single_parent_share": 0.28,
                "median_income": 60000.0,
                "unemployment_rate": 0.04,
            }
            for _ in range(10)
        ]
    )


def _make_county_frame() -> pd.DataFrame:
    rows = []
    for idx in range(10):
        rows.append(
            {
                "state_fips": "01",
                "under5_population": 1000.0,
                "annual_price": 100.0 + idx,
                "provider_density": 10.0 + idx,
            }
        )
    return pd.DataFrame(rows)


def _make_scenarios() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_fips": ["01", "01"],
            "year": [2020, 2021],
            "p_baseline": [100.0, 110.0],
            "market_quantity_proxy": [10.0, 11.0],
            "unpaid_quantity_proxy": [0.0, 0.0],
            "benchmark_replacement_cost": [4.0, 4.0],
            "alpha": [0.2, 0.5],
            "p_alpha": [100.0, 110.0],
            "p_shadow_marginal": [100.0, 110.0],
        }
    )


def test_solve_price_returns_solver_metadata_when_requested():
    result = solve_price(
        baseline_price=100.0,
        market_quantity=10.0,
        unpaid_quantity=0.0,
        demand_elasticity=-1.0,
        supply_elasticity=0.5,
        alpha=0.2,
        return_metadata=True,
    )
    assert isinstance(result, SolverMetadata)
    assert result.status == "converged"
    assert result.price > 0


def test_solve_price_raises_when_root_not_bracketed():
    with pytest.raises(RuntimeError, match="failed to bracket root"):
        solve_price(
            baseline_price=100.0,
            market_quantity=10.0,
            unpaid_quantity=1.0,
            demand_elasticity=0.0,
            supply_elasticity=0.0,
            alpha=0.5,
            return_metadata=True,
        )


def test_bootstrap_childcare_intervals_returns_metadata_and_counts(monkeypatch):
    call_count = {"demand": 0}

    def fake_estimate_childcare_demand_summary(*args, **kwargs):
        call_count["demand"] += 1
        if call_count["demand"] == 2:
            return {"elasticity_at_mean": 0.5, "economically_admissible": False}, None
        return {"elasticity_at_mean": -0.7, "economically_admissible": True}, None

    def fake_calibrate_supply_elasticity(*args, **kwargs):
        return 0.5

    monkeypatch.setattr("unpriced.models.scenario_solver.estimate_childcare_demand_summary", fake_estimate_childcare_demand_summary)
    monkeypatch.setattr("unpriced.models.scenario_solver.calibrate_supply_elasticity", fake_calibrate_supply_elasticity)

    enriched, meta = bootstrap_childcare_intervals(
        state_frame=_make_state_frame(),
        county_frame=_make_county_frame(),
        scenarios=_make_scenarios(),
        demand_mode="broad_complete",
        demand_specification_profile="full_controls",
        n_boot=3,
        seed=123,
    )

    assert meta["bootstrap_draws_requested"] == 3
    assert meta["bootstrap_draws_accepted"] == 2
    assert meta["bootstrap_draws_rejected"] == 1
    assert abs(meta["bootstrap_acceptance_rate"] - 2 / 3) < 1e-12
    assert meta["bootstrap_failed"] is False
    assert meta["bootstrap_rejection_reasons"]["demand_fit:ValueError"] == 1
    assert "p_shadow_marginal_lower" in enriched.columns
    assert "p_alpha_upper" in enriched.columns


def test_bootstrap_childcare_intervals_records_solver_failures(monkeypatch):
    monkeypatch.setattr("unpriced.models.scenario_solver.estimate_childcare_demand_summary", lambda *_, **__: ({"elasticity_at_mean": -0.5, "economically_admissible": True}, None))
    monkeypatch.setattr("unpriced.models.scenario_solver.calibrate_supply_elasticity", lambda *_, **__: 0.5)
    monkeypatch.setattr("unpriced.models.scenario_solver.solve_price", lambda *_, **__: (_ for _ in ()).throw(RuntimeError("unbracketed")))

    enriched, meta = bootstrap_childcare_intervals(
        state_frame=_make_state_frame(),
        county_frame=_make_county_frame(),
        scenarios=_make_scenarios(),
        n_boot=2,
        seed=123,
    )

    assert meta["bootstrap_draws_accepted"] == 0
    assert meta["bootstrap_draws_rejected"] == 2
    assert meta["bootstrap_rejection_reasons"]["solver:RuntimeError"] == 2


def test_summarize_supply_elasticity_exposes_positive_all_and_gap():
    frame = pd.DataFrame(
        {
            "state_fips": ["01"] * 12 + ["02"] * 12,
            "year": [2020] * 24,
            "under5_population": [1000.0] * 24,
            "annual_price": [1, 2, 4, 5, 6, 8, 1, 2, 4, 5, 6, 8] * 2,
            "provider_density": [10, 20, 40, 80, 160, 320, 320, 160, 80, 40, 20, 10] * 2,
        }
    )
    summary = summarize_supply_elasticity(frame)

    assert "supply_elasticity_positive_weighted_median" in summary
    assert "supply_elasticity_all_weighted_median" in summary
    assert "supply_elasticity_weighted_median_gap" in summary
    assert summary["supply_elasticity_all_weighted_median"] is not None


def test_solve_price_dual_shift_matches_fixed_supply_when_kappas_are_zero():
    fixed = solve_price(
        baseline_price=100.0,
        market_quantity=10.0,
        unpaid_quantity=4.0,
        demand_elasticity=0.3,
        supply_elasticity=0.7,
        alpha=0.5,
        return_metadata=True,
    )
    dual = solve_price_dual_shift(
        baseline_price=100.0,
        market_quantity=10.0,
        unpaid_quantity=4.0,
        demand_elasticity=0.3,
        supply_elasticity=0.7,
        alpha=0.5,
        kappa_q=0.0,
        kappa_c=0.0,
        return_metadata=True,
    )

    assert isinstance(fixed, SolverMetadata)
    assert isinstance(dual, SolverMetadata)
    assert abs(fixed.price - dual.price) < 1e-8


def test_solve_price_dual_shift_can_lower_price_with_large_entry_shift():
    dual = solve_price_dual_shift(
        baseline_price=100.0,
        market_quantity=10.0,
        unpaid_quantity=1.0,
        demand_elasticity=0.3,
        supply_elasticity=1.0,
        alpha=1.0,
        kappa_q=1.0,
        kappa_c=0.0,
        return_metadata=True,
    )

    assert isinstance(dual, SolverMetadata)
    assert dual.price < 100.0


def test_solve_price_dual_shift_cost_pressure_raises_price_relative_to_same_entry_shift():
    low_cost = solve_price_dual_shift(
        baseline_price=100.0,
        market_quantity=10.0,
        unpaid_quantity=1.0,
        demand_elasticity=0.3,
        supply_elasticity=1.0,
        alpha=1.0,
        kappa_q=0.4,
        kappa_c=0.0,
        return_metadata=True,
    )
    high_cost = solve_price_dual_shift(
        baseline_price=100.0,
        market_quantity=10.0,
        unpaid_quantity=1.0,
        demand_elasticity=0.3,
        supply_elasticity=1.0,
        alpha=1.0,
        kappa_q=0.4,
        kappa_c=0.2,
        return_metadata=True,
    )

    assert isinstance(low_cost, SolverMetadata)
    assert isinstance(high_cost, SolverMetadata)
    assert high_cost.price > low_cost.price


def test_dual_shift_zero_price_frontier_matches_formula():
    frontier = dual_shift_zero_price_frontier(
        market_quantity=10.0,
        unpaid_quantity=2.5,
        supply_elasticity=0.8,
        kappa_c=0.1,
    )

    assert abs(frontier - (2.5 / 10.0 + 0.8 * 0.1)) < 1e-12


def test_summarize_childcare_scenario_diagnostics_counts_nonconverged_rows_correctly():
    scenarios = pd.DataFrame(
        {
            "state_fips": ["01", "01", "02"],
            "year": [2020, 2020, 2021],
            "alpha": [0.0, 0.5, 0.5],
            "p_baseline": [100.0, 100.0, 120.0],
            "p_baseline_direct_care": [70.0, 70.0, 85.0],
            "p_baseline_non_direct_care": [30.0, 30.0, 35.0],
            "p_alpha": [100.0, 110.0, 130.0],
            "p_alpha_direct_care": [70.0, 78.0, 92.0],
            "p_alpha_non_direct_care": [30.0, 32.0, 38.0],
            "p_shadow_marginal": [100.0, 100.1, 120.2],
            "p_shadow_marginal_lower": [99.9, 100.0, 120.0],
            "p_shadow_marginal_upper": [100.1, 100.2, 120.4],
            "p_alpha_lower": [99.8, 109.0, 129.0],
            "p_alpha_upper": [100.2, 111.0, 131.0],
            "wage_baseline_implied": [10.0, 10.0, 11.0],
            "wage_alpha_implied": [10.0, 10.8, 12.0],
            "direct_care_labor_share": [0.7, 0.71, 0.72],
            "direct_care_price_clip_binding": [False, False, True],
            "unpaid_quantity_proxy": [0.0, 1.0, 2.0],
            "solver_status": ["converged", "root_at_low", "failed_to_bracket"],
            "solver_iterations": [12, 3, 0],
            "solver_expansion_steps": [0, 0, 5],
        }
    )
    diagnostics = summarize_childcare_scenario_diagnostics(scenarios, skipped_state_rows=0)

    assert diagnostics["solver_status_counts"] == {
        "converged": 1,
        "failed_to_bracket": 1,
        "root_at_low": 1,
    }
    assert diagnostics["solver_root_at_boundary_rows"] == 1
    assert diagnostics["solver_nonconverged_rows"] == 1
