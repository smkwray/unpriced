from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from unpriced.reports.export import (
    build_childcare_headline_summary,
    build_childcare_satellite_account,
    build_markdown_report,
)
from unpriced.storage import read_json, write_json


def test_build_childcare_headline_summary_surfaces_mode_bootstrap_and_supply_traceability(tmp_path: Path) -> None:
    demand_iv_path = tmp_path / "demand.json"
    scenario_diagnostics_path = tmp_path / "diagnostics.json"
    decomposition_path = tmp_path / "decomposition.json"
    sensitivity_path = tmp_path / "sensitivity.json"
    output_json_path = tmp_path / "headline.json"
    output_markdown_path = tmp_path / "headline.md"

    write_json({"instrument": "outside_option_wage"}, demand_iv_path)
    write_json(
        {
            "current_mode": "real",
            "demand_sample_name": "observed_core",
            "demand_instrument": "outside_option_wage",
            "demand_specification_profile": "household_parsimonious",
            "scenario_rows": 42,
            "scenario_states": 20,
            "demand_elasticity_at_mean": -0.2,
            "demand_first_stage_r2": 0.4,
            "demand_loo_state_fips_r2": 0.1,
            "demand_loo_year_r2": 0.2,
            "bootstrap_draws_requested": 100,
            "bootstrap_draws_accepted": 85,
            "bootstrap_draws_rejected": 15,
            "bootstrap_acceptance_rate": 0.85,
            "bootstrap_failed": False,
            "supply_elasticity": 0.6,
            "supply_estimation_method": "within_state_year_positive_weighted_median",
            "supply_within_state_year_weighted_median_positive_slope": 0.6,
            "supply_within_state_year_weighted_median_all_slope": 0.3,
            "supply_within_state_year_weighted_median_gap": 0.3,
        },
        scenario_diagnostics_path,
    )
    write_json(
        {
            "canonical": {
                "years": [2012, 2013],
                "baseline_price_p50": 100.0,
                "baseline_direct_care_price_p50": 70.0,
                "baseline_non_direct_care_price_p50": 30.0,
                "baseline_implied_wage_p50": 18.0,
                "baseline_direct_care_labor_share_p50": 0.7,
                "baseline_direct_care_clip_binding_row_share": 0.1,
                "alphas": {
                    "0.50": {
                        "price_p50": 110.0,
                        "direct_care_price_p50": 77.0,
                        "non_direct_care_price_p50": 33.0,
                        "implied_wage_p50": 20.0,
                    }
                },
            }
        },
        decomposition_path,
    )
    write_json(
        {
            "cases": [
                {
                    "baseline_direct_care_price_p50": 68.0,
                    "baseline_implied_wage_p50": 17.0,
                },
                {
                    "baseline_direct_care_price_p50": 72.0,
                    "baseline_implied_wage_p50": 19.0,
                },
            ]
        },
        sensitivity_path,
    )

    build_childcare_headline_summary(
        demand_iv_path=demand_iv_path,
        scenario_diagnostics_path=scenario_diagnostics_path,
        price_decomposition_path=decomposition_path,
        output_json_path=output_json_path,
        output_markdown_path=output_markdown_path,
        price_decomposition_sensitivity_path=sensitivity_path,
    )

    summary = read_json(output_json_path)
    markdown = output_markdown_path.read_text(encoding="utf-8")

    assert summary["current_mode"] == "real"
    assert summary["demand_instrument"] == "outside_option_wage"
    assert summary["bootstrap_draws_requested"] == 100
    assert summary["bootstrap_draws_accepted"] == 85
    assert summary["bootstrap_acceptance_rate"] == 0.85
    assert summary["supply_positive_only_weighted_median_slope"] == 0.6
    assert summary["supply_all_slopes_weighted_median_slope"] == 0.3
    assert summary["supply_positivity_selection_gap"] == 0.3
    assert "Current mode: `real`" in markdown
    assert "Demand instrument: `outside_option_wage`" in markdown
    assert "Bootstrap draws requested/accepted/rejected: `100` / `85` / `15`" in markdown
    assert "Supply positivity-selection gap: `0.3000`" in markdown


def test_build_markdown_report_surfaces_mode_bootstrap_and_supply_sensitivity(tmp_path: Path) -> None:
    price_surface_path = tmp_path / "price_surface.json"
    demand_iv_path = tmp_path / "demand_iv.json"
    scenario_diagnostics_path = tmp_path / "scenario_diagnostics.json"
    output_path = tmp_path / "report.md"

    write_json({"holdout_year": 2022, "rmse_test": 12.5}, price_surface_path)
    write_json(
        {
            "mode": "observed_core",
            "n_obs": 120,
            "n_states": 20,
            "year_min": 2012,
            "year_max": 2022,
            "instrument": "outside_option_wage",
            "specification_profile": "household_parsimonious",
            "price_coefficient": -0.02,
            "elasticity_at_mean": -0.03,
            "economically_admissible": True,
            "first_stage_r2": 0.4,
            "loo_state_fips_r2": 0.1,
            "loo_year_r2": 0.2,
        },
        demand_iv_path,
    )
    write_json(
        {
            "current_mode": "real",
            "demand_sample_name": "observed_core",
            "scenario_rows": 2,
            "scenario_price_observed_support_rows": 2,
            "scenario_price_nowcast_rows": 0,
            "baseline_price_p50": 100.0,
            "baseline_direct_care_price_p50": 70.0,
            "baseline_non_direct_care_price_p50": 30.0,
            "baseline_implied_wage_p50": 18.0,
            "baseline_direct_care_labor_share_p50": 0.7,
            "baseline_direct_care_clip_binding_row_share": 0.1,
            "alpha_50_price_p50": 110.0,
            "alpha_50_direct_care_price_p50": 77.0,
            "alpha_50_implied_wage_p50": 20.0,
            "skipped_state_rows": 0,
            "bootstrap_resampling_unit": "state_fips_cluster",
            "bootstrap_draws_requested": 100,
            "bootstrap_draws_accepted": 90,
            "bootstrap_draws_rejected": 10,
            "bootstrap_acceptance_rate": 0.9,
            "bootstrap_failed": False,
            "demand_sample_selection_reason": "observed_core_passes_minimum_support",
            "shadow_width_p10": 1.0,
            "shadow_width_p50": 2.0,
            "shadow_width_p90": 3.0,
            "alpha_width_p10": 1.5,
            "alpha_width_p50": 2.5,
            "alpha_width_p90": 3.5,
            "zero_width_shadow_count": 0,
            "zero_width_alpha_count": 0,
            "zero_unpaid_quantity_count": 0,
            "demand_first_stage_r2": 0.4,
            "demand_loo_state_fips_r2": 0.1,
            "demand_loo_year_r2": 0.2,
            "solver_demand_elasticity_magnitude": 0.03,
            "supply_elasticity": 0.6,
            "supply_estimation_method": "within_state_year_positive_weighted_median",
            "supply_within_state_year_weighted_median_positive_slope": 0.6,
            "supply_within_state_year_weighted_median_all_slope": 0.3,
            "supply_within_state_year_weighted_median_gap": 0.3,
            "demand_fit_quarantined": False,
        },
        scenario_diagnostics_path,
    )
    scenarios = pd.DataFrame(
        [
            {"state_fips": "06", "year": 2022, "alpha": 0.5, "p_alpha": 110.0},
            {"state_fips": "12", "year": 2022, "alpha": 1.0, "p_alpha": 120.0},
        ]
    )

    build_markdown_report(
        price_surface_path=price_surface_path,
        demand_iv_path=demand_iv_path,
        scenario_diagnostics_path=scenario_diagnostics_path,
        scenarios=scenarios,
        output_path=output_path,
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert "- current mode: real" in markdown
    assert "- bootstrap draws requested/accepted/rejected: 100 / 90 / 10" in markdown
    assert "- bootstrap acceptance rate: 90.0%" in markdown
    assert "- supply positive-only weighted median slope: 0.6" in markdown
    assert "- supply all-slopes weighted median slope: 0.3" in markdown
    assert "- supply positivity-selection gap: 0.3" in markdown


def test_build_childcare_satellite_account_nationalizes_from_person_equivalent_weights(tmp_path: Path) -> None:
    county = pd.DataFrame(
        [
            {
                "year": 2018,
                "under5_population": 1_000.0,
                "annual_price": 10_400.0,
                "direct_care_price_index": 5_200.0,
                "non_direct_care_price_index": 5_200.0,
                "benchmark_childcare_wage": 10.0,
                "specialist_childcare_wage": 12.0,
            },
            {
                "year": 2019,
                "under5_population": 1_000.0,
                "annual_price": 10_400.0,
                "direct_care_price_index": 5_200.0,
                "non_direct_care_price_index": 5_200.0,
                "benchmark_childcare_wage": 10.0,
                "specialist_childcare_wage": 12.0,
            },
            {
                "year": 2020,
                "under5_population": 1_000.0,
                "annual_price": 10_400.0,
                "direct_care_price_index": 5_200.0,
                "non_direct_care_price_index": 5_200.0,
                "benchmark_childcare_wage": 10.0,
                "specialist_childcare_wage": 12.0,
            },
            {
                "year": 2022,
                "under5_population": 1_000.0,
                "annual_price": 10_400.0,
                "direct_care_price_index": 5_200.0,
                "non_direct_care_price_index": 5_200.0,
                "benchmark_childcare_wage": 10.0,
                "specialist_childcare_wage": 12.0,
            }
        ]
    )
    state = pd.DataFrame(
        [
            {
                "year": 2018,
                "unpaid_active_childcare_hours": 210.0,
                "unpaid_active_household_childcare_hours": 160.0,
                "unpaid_active_nonhousehold_childcare_hours": 50.0,
                "unpaid_supervisory_childcare_hours": 90.0,
                "atus_weight_sum": 1_000.0,
                "is_sensitivity_year": False,
            },
            {
                "year": 2019,
                "unpaid_active_childcare_hours": 190.0,
                "unpaid_active_household_childcare_hours": 145.0,
                "unpaid_active_nonhousehold_childcare_hours": 45.0,
                "unpaid_supervisory_childcare_hours": 95.0,
                "atus_weight_sum": 1_000.0,
                "is_sensitivity_year": False,
            },
            {
                "year": 2020,
                "unpaid_active_childcare_hours": 400.0,
                "unpaid_active_household_childcare_hours": 320.0,
                "unpaid_active_nonhousehold_childcare_hours": 80.0,
                "unpaid_supervisory_childcare_hours": 200.0,
                "atus_weight_sum": 1_000.0,
                "is_sensitivity_year": True,
            },
            {
                "year": 2022,
                "unpaid_active_childcare_hours": 200.0,
                "unpaid_active_household_childcare_hours": 150.0,
                "unpaid_active_nonhousehold_childcare_hours": 50.0,
                "unpaid_supervisory_childcare_hours": 100.0,
                "atus_weight_sum": 1_000.0,
                "is_sensitivity_year": False,
            }
        ]
    )
    acs = pd.DataFrame(
        [{"year": year, "under5_population": 1_000.0} for year in (2018, 2019, 2020, 2022)]
    )
    output_json_path = tmp_path / "satellite.json"
    output_markdown_path = tmp_path / "satellite.md"
    output_table_path = tmp_path / "satellite.csv"

    build_childcare_satellite_account(
        county=county,
        state=state,
        acs=acs,
        childcare_assumptions={"market_hours_per_child_per_week": 20.0},
        output_json_path=output_json_path,
        output_markdown_path=output_markdown_path,
        output_table_path=output_table_path,
    )

    summary = read_json(output_json_path)
    latest = summary["latest_year"]
    methodologies = summary["benchmark_methodologies"]
    bridge = methodologies["active_care_bridge_benchmark"]
    annual_hours = methodologies["annual_hours_childcare_account"]

    assert latest["national_active_childcare_hours_total"] == 200_000.0
    assert latest["national_active_household_childcare_hours_total"] == 150_000.0
    assert latest["national_active_nonhousehold_childcare_hours_total"] == 50_000.0
    assert latest["national_supervisory_childcare_hours_total"] == 100_000.0
    assert latest["national_total_childcare_hours_total"] == 300_000.0
    assert latest["price_support_population_share"] == 1.0
    assert latest["national_total_childcare_hours_total"] < 24.0 * 366.0 * 1_000.0
    assert summary["headline_window"]["window_years"] == [2018, 2019]
    assert summary["headline_window"]["excludes_sensitivity_years"] is True
    assert summary["headline_window"]["preferred_value_mean"] == 1_462_500.0
    assert summary["default_methodology"] == "annual_hours_childcare_account"
    assert "Top-level preferred_series" in summary["compatibility_note"]
    assert bridge["label"] == "Active-care bridge benchmark (scaled to under-5 population)"
    assert bridge["latest_year"]["quantity_slots"] == 200_000.0 / (52.0 * 20.0)
    assert bridge["latest_year"]["preferred_value"] == pytest.approx(1_000_000.0)
    assert bridge["headline_window"]["preferred_value_mean"] == pytest.approx(1_000_000.0)
    assert bridge["annual"][-1]["active_care_bridge_quantity_slots"] == pytest.approx(200_000.0 / (52.0 * 20.0))
    assert "active_childcare_hours_per_respondent_year" in bridge["latest_year"]
    assert "average_unpaid_childcare_hours_per_child_year" not in bridge["latest_year"]
    assert annual_hours["annual"][-1]["national_active_nonhousehold_childcare_hours_total"] == 50_000.0
    assert annual_hours["latest_year"]["supervisory_hours_total"] == 100_000.0


def test_build_markdown_report_adds_dual_shift_section_only_when_artifact_exists(tmp_path: Path) -> None:
    price_surface_path = tmp_path / "price_surface.json"
    demand_iv_path = tmp_path / "demand_iv.json"
    scenario_diagnostics_path = tmp_path / "scenario_diagnostics.json"
    dual_shift_summary_path = tmp_path / "dual_shift.json"
    output_path = tmp_path / "report.md"

    write_json({"holdout_year": 2022, "rmse_test": 12.5}, price_surface_path)
    write_json(
        {
            "mode": "observed_core",
            "n_obs": 120,
            "n_states": 20,
            "year_min": 2012,
            "year_max": 2022,
            "instrument": "outside_option_wage",
            "specification_profile": "household_parsimonious",
            "price_coefficient": -0.02,
            "elasticity_at_mean": -0.03,
            "economically_admissible": True,
            "first_stage_r2": 0.4,
            "loo_state_fips_r2": 0.1,
            "loo_year_r2": 0.2,
        },
        demand_iv_path,
    )
    write_json(
        {
            "current_mode": "real",
            "demand_sample_name": "observed_core",
            "scenario_rows": 2,
            "scenario_price_observed_support_rows": 2,
            "scenario_price_nowcast_rows": 0,
            "baseline_price_p50": 100.0,
            "baseline_direct_care_price_p50": 70.0,
            "baseline_non_direct_care_price_p50": 30.0,
            "baseline_implied_wage_p50": 18.0,
            "baseline_direct_care_labor_share_p50": 0.7,
            "baseline_direct_care_clip_binding_row_share": 0.1,
            "alpha_50_price_p50": 110.0,
            "alpha_50_direct_care_price_p50": 77.0,
            "alpha_50_implied_wage_p50": 20.0,
            "skipped_state_rows": 0,
            "bootstrap_resampling_unit": "state_fips_cluster",
            "bootstrap_draws_requested": 100,
            "bootstrap_draws_accepted": 90,
            "bootstrap_draws_rejected": 10,
            "bootstrap_acceptance_rate": 0.9,
            "bootstrap_failed": False,
            "demand_sample_selection_reason": "observed_core_passes_minimum_support",
            "shadow_width_p10": 1.0,
            "shadow_width_p50": 2.0,
            "shadow_width_p90": 3.0,
            "alpha_width_p10": 1.5,
            "alpha_width_p50": 2.5,
            "alpha_width_p90": 3.5,
            "zero_width_shadow_count": 0,
            "zero_width_alpha_count": 0,
            "zero_unpaid_quantity_count": 0,
            "demand_first_stage_r2": 0.4,
            "demand_loo_state_fips_r2": 0.1,
            "demand_loo_year_r2": 0.2,
            "solver_demand_elasticity_magnitude": 0.03,
            "supply_elasticity": 0.6,
            "supply_estimation_method": "within_state_year_positive_weighted_median",
            "supply_within_state_year_weighted_median_positive_slope": 0.6,
            "supply_within_state_year_weighted_median_all_slope": 0.3,
            "supply_within_state_year_weighted_median_gap": 0.3,
            "demand_fit_quarantined": False,
        },
        scenario_diagnostics_path,
    )
    scenarios = pd.DataFrame(
        [
            {"state_fips": "06", "year": 2022, "alpha": 0.5, "p_alpha": 110.0},
            {"state_fips": "12", "year": 2022, "alpha": 1.0, "p_alpha": 120.0},
        ]
    )

    build_markdown_report(
        price_surface_path=price_surface_path,
        demand_iv_path=demand_iv_path,
        scenario_diagnostics_path=scenario_diagnostics_path,
        scenarios=scenarios,
        output_path=output_path,
    )
    markdown_without = output_path.read_text(encoding="utf-8")
    assert "## Dual-shift marketization sensitivity" not in markdown_without

    write_json(
        {
            "headline_alpha": 0.5,
            "short_run_fixed_supply_headline_alpha_price_p50": 110.0,
            "short_run_fixed_supply_headline_alpha_pct_change_p50": 0.10,
            "bootstrap_draws_requested": 100,
            "bootstrap_draws_accepted": 80,
            "bootstrap_draws_rejected": 20,
            "bootstrap_acceptance_rate": 0.8,
            "headline_alpha_table": [
                {
                    "kappa_q": 0.0,
                    "kappa_c": 0.0,
                    "median_p_alpha": 110.0,
                    "median_p_alpha_pct_change": 0.10,
                    "share_price_increase": 1.0,
                    "share_price_decrease": 0.0,
                },
                {
                    "kappa_q": 1.0,
                    "kappa_c": 0.0,
                    "median_p_alpha": 95.0,
                    "median_p_alpha_pct_change": -0.05,
                    "share_price_increase": 0.0,
                    "share_price_decrease": 1.0,
                },
            ],
            "frontier_summary": [
                {
                    "kappa_c": 0.0,
                    "kappa_q_zero_price_frontier_p10": 0.2,
                    "kappa_q_zero_price_frontier_p50": 0.3,
                    "kappa_q_zero_price_frontier_p90": 0.4,
                }
            ],
        },
        dual_shift_summary_path,
    )
    build_markdown_report(
        price_surface_path=price_surface_path,
        demand_iv_path=demand_iv_path,
        scenario_diagnostics_path=scenario_diagnostics_path,
        scenarios=scenarios,
        output_path=output_path,
        dual_shift_summary_path=dual_shift_summary_path,
    )
    markdown_with = output_path.read_text(encoding="utf-8")
    assert "## Dual-shift marketization sensitivity" in markdown_with
    assert "Short-run fixed-supply benchmark" in markdown_with
    assert "Medium-run dual-shift sensitivity" in markdown_with
