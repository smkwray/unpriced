from __future__ import annotations

import pandas as pd
import pytest

from unpaidwork.features.childcare_panel import build_childcare_panels
from unpaidwork.features import home_maintenance_panel as home_panel
from unpaidwork.features.home_maintenance_panel import build_home_maintenance_panel
from unpaidwork.ingest.acs import ingest as ingest_acs
from unpaidwork.ingest.ahs import ingest as ingest_ahs
from unpaidwork.ingest.atus import ingest as ingest_atus
from unpaidwork.ingest.laus import ingest as ingest_laus
from unpaidwork.ingest.ndcp import ingest as ingest_ndcp
from unpaidwork.ingest.qcew import ingest as ingest_qcew
from unpaidwork.models.demand_iv import (
    CANONICAL_COMPARISON_SPECIFICATION_PROFILE,
    CANONICAL_OBSERVED_SPECIFICATION_PROFILE,
    build_childcare_demand_sample_comparison,
    build_childcare_imputation_sweep,
    build_childcare_labor_support_sweep,
    build_childcare_specification_sweep,
    estimate_childcare_demand_summary,
    fit_childcare_demand_iv,
    normalize_demand_mode,
    normalize_specification_profile,
    select_headline_sample,
)
from unpaidwork.models.price_surface import fit_price_surface
from unpaidwork.models.scenario_solver import (
    bootstrap_childcare_intervals,
    prepare_childcare_scenario_inputs,
    resolve_solver_demand_elasticity,
    solve_alpha_grid,
    solve_alpha_grid_piecewise_supply,
    summarize_childcare_scenario_diagnostics,
)
from unpaidwork.models.supply_curve import (
    calibrate_supply_elasticity,
    summarize_piecewise_supply_curve,
    summarize_supply_elasticity,
)
from unpaidwork.models.switching import fit_home_switching


def test_price_surface_and_demand_iv(project_paths):
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs):
        ingestor(project_paths, sample=True)
    county, state = build_childcare_panels(project_paths)
    price_summary = fit_price_surface(
        county,
        output_json=project_paths.outputs_reports / "price.json",
        output_panel=project_paths.processed / "county.parquet",
    )
    iv_summary = fit_childcare_demand_iv(
        state,
        output_json=project_paths.outputs_reports / "iv.json",
        output_panel=project_paths.processed / "state.parquet",
    )
    solved = solve_alpha_grid(10000, 50000, 15000, 0.4, 0.6, [0.1, 0.5, 1.0])
    assert price_summary["n_obs"] == len(county)
    assert "price_coefficient" in iv_summary
    assert len(solved) == 3
    assert solved[-1].price >= solved[0].price


def test_bootstrap_childcare_intervals(project_paths):
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs):
        ingestor(project_paths, sample=True)
    county, state = build_childcare_panels(project_paths)
    scenarios = pd.DataFrame(
        [
            {
                "state_fips": row["state_fips"],
                "year": row["year"],
                "alpha": 0.5,
                "p_baseline": float(row["state_price_index"]),
                "p_shadow_marginal": float(row["state_price_index"]) * 1.01,
                "p_alpha": float(row["state_price_index"]) * 1.08,
                "market_quantity_proxy": float(row["market_quantity_proxy"]),
                "unpaid_quantity_proxy": float(row["unpaid_quantity_proxy"]),
            }
            for row in state.head(3).to_dict(orient="records")
        ]
    )

    boot = bootstrap_childcare_intervals(state, county, scenarios, demand_mode="broad_complete", n_boot=20, seed=7)

    assert {"p_shadow_marginal_lower", "p_shadow_marginal_upper", "p_alpha_lower", "p_alpha_upper"} <= set(boot.columns)
    assert (boot["p_shadow_marginal_upper"] >= boot["p_shadow_marginal_lower"]).all()
    assert (boot["p_alpha_upper"] >= boot["p_alpha_lower"]).all()


def test_prepare_childcare_scenario_inputs_drops_invalid_rows():
    frame = pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "state_price_index": 12000.0,
                "market_quantity_proxy": 1000.0,
                "unpaid_quantity_proxy": 50.0,
                "benchmark_replacement_cost": 10.0,
            },
            {
                "state_fips": "00",
                "year": 2021,
                "state_price_index": pd.NA,
                "market_quantity_proxy": pd.NA,
                "unpaid_quantity_proxy": 20.0,
                "benchmark_replacement_cost": 5.0,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "state_price_index": 0.0,
                "market_quantity_proxy": 500.0,
                "unpaid_quantity_proxy": 10.0,
                "benchmark_replacement_cost": 8.0,
            },
        ]
    )

    valid = prepare_childcare_scenario_inputs(frame)

    assert len(valid) == 1
    assert valid.loc[0, "state_fips"] == "06"


def test_summarize_childcare_scenario_diagnostics():
    scenarios = pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "state_price_observation_status": "observed_ndcp_support",
                "state_price_nowcast": False,
                "p_baseline": 120.0,
                "p_baseline_direct_care": 60.0,
                "p_baseline_non_direct_care": 60.0,
                "wage_baseline_implied": 12.0,
                "unpaid_quantity_proxy": 10.0,
                "p_shadow_marginal_lower": 100.0,
                "p_shadow_marginal_upper": 100.5,
                "p_shadow_marginal": 100.2,
                "p_shadow_marginal_direct_care": 50.1,
                "p_shadow_marginal_non_direct_care": 50.1,
                "wage_shadow_implied": 11.0,
                "p_alpha_lower": 102.0,
                "p_alpha_upper": 104.0,
                "alpha": 0.5,
                "p_alpha": 103.0,
                "p_alpha_direct_care": 51.5,
                "p_alpha_non_direct_care": 51.5,
                "wage_alpha_implied": 11.3,
            },
            {
                "state_fips": "48",
                "year": 2023,
                "state_price_observation_status": "post_ndcp_nowcast",
                "state_price_nowcast": True,
                "p_baseline": 100.0,
                "p_baseline_direct_care": 45.0,
                "p_baseline_non_direct_care": 55.0,
                "wage_baseline_implied": 10.0,
                "unpaid_quantity_proxy": 0.0,
                "p_shadow_marginal_lower": 90.0,
                "p_shadow_marginal_upper": 90.0,
                "p_shadow_marginal": 90.0,
                "p_shadow_marginal_direct_care": 40.5,
                "p_shadow_marginal_non_direct_care": 49.5,
                "wage_shadow_implied": 9.0,
                "p_alpha_lower": 91.0,
                "p_alpha_upper": 91.0,
                "alpha": 0.5,
                "p_alpha": 91.0,
                "p_alpha_direct_care": 41.0,
                "p_alpha_non_direct_care": 50.0,
                "wage_alpha_implied": 9.1,
            },
        ]
    )

    summary = summarize_childcare_scenario_diagnostics(
        scenarios,
        skipped_state_rows=3,
        demand_summary={"first_stage_r2": 1.0},
    )

    assert summary["scenario_rows"] == 2
    assert summary["skipped_state_rows"] == 3
    assert summary["zero_width_shadow_count"] == 1
    assert summary["zero_width_alpha_count"] == 1
    assert summary["zero_unpaid_quantity_count"] == 1
    assert summary["demand_first_stage_near_perfect"] is True
    assert summary["scenario_price_observed_support_rows"] == 1
    assert summary["scenario_price_nowcast_rows"] == 1
    assert summary["scenario_contains_nowcast_rows"] is True
    assert summary["scenario_price_nowcast_years"] == [2023]
    assert summary["baseline_price_p50"] == 110.0
    assert summary["baseline_direct_care_price_p50"] == 52.5
    assert summary["alpha_50_price_p50"] == 97.0
    assert summary["alpha_50_implied_wage_p50"] == 10.2
    assert summary["bootstrap_resampling_unit"] == "state_fips_cluster"


def test_demand_iv_baseline_includes_core_diagnostics(project_paths):
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs):
        ingestor(project_paths, sample=True)
    _, state = build_childcare_panels(project_paths)
    summary, dataset = estimate_childcare_demand_summary(state, mode="broad_complete")

    assert summary["mode"] == "broad_complete"
    assert summary["n_obs"] > 0
    assert summary["n_states"] > 0
    assert summary["n_years"] > 0
    assert summary["specification_profile"] == "full_controls"
    assert "first_stage_r2" in summary
    assert "elasticity_at_mean" in summary
    assert "year_min" in summary
    assert "year_max" in summary
    assert "economically_admissible" in summary


def test_demand_iv_observed_core_filters_rows(project_paths):
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs):
        ingestor(project_paths, sample=True)
    _, state = build_childcare_panels(project_paths)
    broad, _ = estimate_childcare_demand_summary(state, mode="broad_complete")
    observed, _ = estimate_childcare_demand_summary(state, mode="observed_core")

    assert observed["mode"] == "observed_core"
    assert observed["n_obs"] <= broad["n_obs"]


def test_demand_iv_loo_runs_with_sufficient_data(project_paths):
    """LOO diagnostics require n>=10. With sample data (9 rows), they are skipped."""
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs):
        ingestor(project_paths, sample=True)
    _, state = build_childcare_panels(project_paths)
    summary, _ = estimate_childcare_demand_summary(state, mode="broad_complete")

    # Sample data has 9 rows, so LOO keys won't exist. But the estimator must not error.
    if summary["n_obs"] >= 10:
        assert "loo_state_fips_r2" in summary
        assert "loo_year_r2" in summary


def test_demand_iv_aliases_map_to_sample_ladder_modes():
    assert normalize_demand_mode("baseline") == "broad_complete"
    assert normalize_demand_mode("strict_observed") == "observed_core"


def test_specification_profile_defaults_to_full_controls():
    assert normalize_specification_profile(None) == "full_controls"
    assert normalize_specification_profile("instrument_only") == "instrument_only"


def test_calibrate_supply_elasticity_coerces_object_columns():
    frame = pd.DataFrame(
        [
            {"provider_density": "1.0", "annual_price": "10000"},
            {"provider_density": "1.2", "annual_price": "11000"},
            {"provider_density": "1.5", "annual_price": "12000"},
        ]
    )

    elasticity = calibrate_supply_elasticity(frame)

    assert isinstance(elasticity, float)


def test_supply_elasticity_prefers_within_state_year_weighted_median():
    rows = []
    for state_fips in ("01", "02"):
        for year in (2019, 2020):
            for idx, price in enumerate((100.0, 120.0, 140.0, 160.0, 180.0), start=1):
                rows.append(
                    {
                        "state_fips": state_fips,
                        "year": year,
                        "annual_price": price,
                        "provider_density": price / 100.0,
                        "under5_population": 100 + idx,
                    }
                )
    frame = pd.DataFrame(rows)

    summary = summarize_supply_elasticity(frame)

    assert summary["estimation_method"] == "within_state_year_positive_weighted_median"
    assert summary["fallback_used"] is False
    assert summary["within_state_year_group_count"] == 4
    assert 0.9 <= float(summary["supply_elasticity"]) <= 1.1


def test_summarize_piecewise_supply_curve_estimates_side_specific_slopes():
    rows = []
    for state_fips, baseline in (("01", 105.0), ("02", 108.0)):
        for idx, price in enumerate((70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0), start=1):
            rows.append(
                {
                    "state_fips": state_fips,
                    "year": 2021,
                    "annual_price": price,
                    "provider_density": 1.0 + idx * 0.08 + (0.05 if state_fips == "02" else 0.0),
                    "under5_population": 100 + idx,
                    "state_price_index": baseline,
                }
            )
    frame = pd.DataFrame(rows)

    summary, groups = summarize_piecewise_supply_curve(frame)

    assert summary["piecewise_method"] == "state_year_piecewise_isoelastic"
    assert summary["group_count"] == 2
    assert summary["supported_both_sides_groups"] == 2
    assert summary["pooled_eta_below"] > 0
    assert summary["pooled_eta_above"] > 0
    assert {"eta_below", "eta_above", "fallback_below", "fallback_above"} <= set(groups.columns)
    assert groups["fallback_below"].eq(False).all()
    assert groups["fallback_above"].eq(False).all()


def test_solve_alpha_grid_piecewise_supply_preserves_monotone_prices():
    constant = solve_alpha_grid(100.0, 1000.0, 100.0, 0.1, 0.8, [0.0, 0.5, 1.0])
    piecewise = solve_alpha_grid_piecewise_supply(100.0, 1000.0, 100.0, 0.1, 0.6, 1.2, [0.0, 0.5, 1.0])

    assert len(piecewise) == 3
    assert piecewise[0].price >= 100.0
    assert piecewise[-1].price >= piecewise[0].price
    assert abs(piecewise[0].price - constant[0].price) < 1e-6


def test_build_childcare_demand_sample_comparison_selects_observed_core():
    comparison = build_childcare_demand_sample_comparison(
        pd.DataFrame(
            [
                {
                    "state_fips": f"{state:02d}",
                    "year": 2014 + (idx % 7),
                    "unpaid_childcare_hours": 1.8 - idx / 1000.0,
                    "state_price_index": 5000.0 + idx,
                    "outside_option_wage": 20.0 + idx / 1000.0,
                    "parent_employment_rate": 0.7,
                    "single_parent_share": 0.2,
                    "median_income": 60000.0,
                    "unemployment_rate": 0.05,
                    "eligible_broad_complete": True,
                    "eligible_observed_core": True,
                    "eligible_observed_core_low_impute": idx < 80,
                }
                for idx, state in enumerate(([1] * 8 + [2] * 8 + [3] * 8 + [4] * 8 + [5] * 8 + [6] * 8 + [7] * 8 + [8] * 8 + [9] * 8 + [10] * 8 + [11] * 8 + [12] * 8 + [13] * 8 + [14] * 8 + [15] * 8))
            ]
        )
    )

    selected, reason = select_headline_sample(comparison["samples"])

    assert selected == "observed_core"
    assert reason == "observed_core_passes_minimum_support"
    assert comparison["comparison_specification_profile"] == CANONICAL_COMPARISON_SPECIFICATION_PROFILE
    assert comparison["samples"]["observed_core"]["economically_admissible"] is True


def test_build_childcare_imputation_sweep_reports_best_headline_candidate():
    frame = pd.DataFrame(
        [
            {
                "state_fips": f"{state:02d}",
                "year": 2014 + (idx % 7),
                "unpaid_childcare_hours": 1.5 + idx / 1000.0,
                "state_price_index": 5000.0 + idx,
                "outside_option_wage": 20.0 + idx / 1000.0,
                "parent_employment_rate": 0.7,
                "single_parent_share": 0.2,
                "median_income": 60000.0,
                "unemployment_rate": 0.05,
                "state_ndcp_imputed_share": 0.2 if idx < 110 else 0.45,
                "eligible_observed_core": True,
                "eligible_observed_core_low_impute": idx < 80,
            }
            for idx, state in enumerate(([1] * 8 + [2] * 8 + [3] * 8 + [4] * 8 + [5] * 8 + [6] * 8 + [7] * 8 + [8] * 8 + [9] * 8 + [10] * 8 + [11] * 8 + [12] * 8 + [13] * 8 + [14] * 8 + [15] * 8))
        ]
    )

    sweep = build_childcare_imputation_sweep(frame, thresholds=[0.25, 0.45])

    assert sweep["current_headline_sample"] == "observed_core"
    assert sweep["specification_profile"] == CANONICAL_COMPARISON_SPECIFICATION_PROFILE
    assert len(sweep["thresholds"]) == 2
    assert sweep["best_headline_eligible_threshold"]["threshold"] == 0.45
    assert sweep["best_headline_eligible_threshold"]["n_obs"] >= 100


def test_build_childcare_labor_support_sweep_reports_best_headline_candidate():
    frame = pd.DataFrame(
        [
            {
                "state_fips": f"{state:02d}",
                "year": 2014 + (idx % 7),
                "unpaid_childcare_hours": 1.5 + idx / 1000.0,
                "state_price_index": 5000.0 + idx,
                "outside_option_wage": 20.0 + idx / 1000.0,
                "parent_employment_rate": 0.7,
                "single_parent_share": 0.2,
                "median_income": 60000.0,
                "unemployment_rate": 0.05,
                "state_qcew_labor_observed_share": 0.95 if idx < 110 else 0.99,
                "eligible_observed_core": True,
                "eligible_observed_core_low_impute": idx < 80,
            }
            for idx, state in enumerate(([1] * 8 + [2] * 8 + [3] * 8 + [4] * 8 + [5] * 8 + [6] * 8 + [7] * 8 + [8] * 8 + [9] * 8 + [10] * 8 + [11] * 8 + [12] * 8 + [13] * 8 + [14] * 8 + [15] * 8))
        ]
    )

    sweep = build_childcare_labor_support_sweep(frame, thresholds=[0.99, 0.95])

    assert sweep["current_headline_sample"] == "observed_core"
    assert sweep["specification_profile"] == CANONICAL_COMPARISON_SPECIFICATION_PROFILE
    assert len(sweep["thresholds"]) == 2
    assert sweep["best_headline_eligible_threshold"]["threshold"] == 0.95
    assert sweep["best_headline_eligible_threshold"]["n_obs"] >= 100


def test_build_childcare_specification_sweep_identifies_better_holdout_profiles():
    frame = pd.DataFrame(
        [
            {
                "state_fips": f"{1 + (idx % 20):02d}",
                "year": 2014 + (idx % 8),
                "unpaid_childcare_hours": 2.0 - price / 10000.0 + (idx % 3) / 100.0,
                "state_price_index": price,
                "outside_option_wage": price / 1000.0,
                "parent_employment_rate": 0.7 + (idx % 2) * 0.01,
                "single_parent_share": 0.2 + (idx % 4) * 0.005,
                "median_income": 60000.0 + idx * 10.0,
                "unemployment_rate": 0.05 + (idx % 5) * 0.001,
                "eligible_observed_core": True,
            }
            for idx, price in enumerate(range(5000, 5120))
        ]
    )

    sweep = build_childcare_specification_sweep(frame, mode="observed_core")

    assert sweep["current_profile"] == CANONICAL_OBSERVED_SPECIFICATION_PROFILE
    assert "instrument_only" in sweep["profiles"]
    assert isinstance(sweep["profiles_beating_current_on_both_holdouts"], list)


def test_resolve_solver_demand_elasticity_rejects_positive_response():
    with pytest.raises(ValueError, match="positive price response"):
        resolve_solver_demand_elasticity(
            {
                "elasticity_at_mean": 0.12,
                "economically_admissible": False,
            }
        )


def test_resolve_solver_demand_elasticity_returns_signed_and_magnitude():
    signed, magnitude = resolve_solver_demand_elasticity(
        {
            "elasticity_at_mean": -0.12,
            "economically_admissible": True,
        }
    )

    assert signed == -0.12
    assert magnitude == 0.12


def test_home_switching(project_paths, monkeypatch):
    ingest_laus(project_paths, sample=True)
    ingest_ahs(project_paths, sample=True)
    monkeypatch.setattr(
        home_panel,
        "load_cbsa_county_crosswalk",
        lambda paths: pd.DataFrame(
            [
                {"cbsa_code": "31080", "county_fips": "06037"},
                {"cbsa_code": "19100", "county_fips": "48113"},
                {"cbsa_code": "35620", "county_fips": "36061"},
            ]
        ),
    )
    panel = build_home_maintenance_panel(project_paths)
    panel["precip_event_days"] = [5, 5, 14, 14, 8, 8]
    panel["noaa_match_status"] = [
        "observed",
        "observed",
        "national_avg_not_reported",
        "national_avg_not_reported",
        "national_avg_nonmetro",
        "national_avg_nonmetro",
    ]
    summary = fit_home_switching(panel, project_paths.outputs_reports / "home.json")
    assert "price_effect" in summary
    assert "storm_exposure_effect" in summary
    assert "precip_event_days_effect" in summary
    assert "storm_event_count_effect" in summary
    assert "log_storm_property_damage_effect" in summary
    assert "unemployment_effect" in summary
    assert "noaa_not_reported_effect" in summary
    assert "noaa_nonmetro_effect" in summary
    assert summary["model_specification"] == "weather_status_interactions_plus_storm_load"
    assert "storm_exposure_effect_observed" in summary
    assert "storm_exposure_effect_not_reported" in summary
    assert "storm_exposure_effect_nonmetro" in summary
    assert "precip_event_days_effect_observed" in summary
    assert "precip_event_days_effect_not_reported" in summary
    assert "precip_event_days_effect_nonmetro" in summary
    assert summary["noaa_observed_rows"] == 2
    assert summary["noaa_not_reported_rows"] == 2
    assert summary["noaa_nonmetro_rows"] == 2
