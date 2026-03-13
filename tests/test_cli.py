from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from unpaidwork.assumptions import childcare_model_assumptions
from unpaidwork import cli
from unpaidwork.errors import UnpaidWorkError
from unpaidwork.storage import read_json, read_parquet, write_json, write_parquet


def test_real_pull_dry_run_is_ok_from_cli():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unpaidwork.cli",
            "pull-core",
            "--real",
            "--dry-run",
            "--year",
            "2024",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "planned qcew" in result.stdout or "planned qcew" in result.stderr


def _write_childcare_simulation_inputs(project_paths) -> None:
    state = pd.DataFrame(
        [
            {
                "state_fips": "01",
                "year": 2009,
                "subgroup": "all",
                "state_price_index": 10000.0,
                "unpaid_childcare_hours": 18.0,
                "atus_weight_sum": 100.0,
                "outside_option_wage": 20.0,
                "parent_employment_rate": 0.72,
                "single_parent_share": 0.24,
                "median_income": 55000.0,
                "unemployment_rate": 0.05,
                "market_quantity_proxy": 1000.0,
                "unpaid_quantity_proxy": 200.0,
                "benchmark_replacement_cost": 8000.0,
                "state_direct_care_price_index": 4200.0,
                "state_direct_care_price_index_raw": 4300.0,
                "state_non_direct_care_price_index": 5800.0,
                "state_direct_care_labor_share": 0.42,
                "state_direct_care_price_clip_binding_share": 0.0,
                "state_direct_care_price_clip_binding": False,
                "state_effective_children_per_worker": 5.0,
                "state_implied_direct_care_hourly_wage": 8.08,
                "eligible_broad_complete": True,
                "eligible_observed_core": False,
                "eligible_observed_core_low_impute": False,
                "state_controls_source": "acs_observed",
                "state_unemployment_source": "laus_observed",
                "state_price_support_window": "out_of_support",
                "state_price_observation_status": "pre_ndcp_support_gap",
                "state_price_nowcast": False,
                "state_ndcp_imputed_share": 0.35,
                "state_qcew_wage_observed_share": 0.98,
                "state_qcew_employment_observed_share": 0.98,
                "state_qcew_labor_observed_share": 0.98,
                "observed_core_exclusion_reason": "outside_year_window",
                "observed_core_low_impute_exclusion_reason": "outside_year_window",
                "is_sensitivity_year": False,
            },
            {
                "state_fips": "01",
                "year": 2014,
                "subgroup": "all",
                "state_price_index": 11000.0,
                "unpaid_childcare_hours": 17.0,
                "atus_weight_sum": 120.0,
                "outside_option_wage": 21.0,
                "parent_employment_rate": 0.74,
                "single_parent_share": 0.22,
                "median_income": 57000.0,
                "unemployment_rate": 0.045,
                "market_quantity_proxy": 1100.0,
                "unpaid_quantity_proxy": 180.0,
                "benchmark_replacement_cost": 8200.0,
                "state_direct_care_price_index": 4800.0,
                "state_direct_care_price_index_raw": 4900.0,
                "state_non_direct_care_price_index": 6200.0,
                "state_direct_care_labor_share": 4800.0 / 11000.0,
                "state_direct_care_price_clip_binding_share": 0.0,
                "state_direct_care_price_clip_binding": False,
                "state_effective_children_per_worker": 5.5,
                "state_implied_direct_care_hourly_wage": 8.73,
                "eligible_broad_complete": True,
                "eligible_observed_core": True,
                "eligible_observed_core_low_impute": True,
                "state_controls_source": "acs_observed",
                "state_unemployment_source": "laus_observed",
                "state_price_support_window": "in_support",
                "state_price_observation_status": "observed_ndcp_support",
                "state_price_nowcast": False,
                "state_ndcp_imputed_share": 0.12,
                "state_qcew_wage_observed_share": 0.99,
                "state_qcew_employment_observed_share": 0.99,
                "state_qcew_labor_observed_share": 0.99,
                "observed_core_exclusion_reason": "",
                "observed_core_low_impute_exclusion_reason": "",
                "is_sensitivity_year": False,
            },
            {
                "state_fips": "02",
                "year": 2015,
                "subgroup": "all",
                "state_price_index": 12000.0,
                "unpaid_childcare_hours": 16.0,
                "atus_weight_sum": 130.0,
                "outside_option_wage": 22.0,
                "parent_employment_rate": 0.75,
                "single_parent_share": 0.21,
                "median_income": 59000.0,
                "unemployment_rate": 0.04,
                "market_quantity_proxy": 1200.0,
                "unpaid_quantity_proxy": 150.0,
                "benchmark_replacement_cost": 8400.0,
                "state_direct_care_price_index": 5200.0,
                "state_direct_care_price_index_raw": 13000.0,
                "state_non_direct_care_price_index": 6800.0,
                "state_direct_care_labor_share": 5200.0 / 12000.0,
                "state_direct_care_price_clip_binding_share": 1.0,
                "state_direct_care_price_clip_binding": True,
                "state_effective_children_per_worker": 5.0,
                "state_implied_direct_care_hourly_wage": 10.42,
                "eligible_broad_complete": True,
                "eligible_observed_core": True,
                "eligible_observed_core_low_impute": False,
                "state_controls_source": "acs_observed",
                "state_unemployment_source": "laus_observed",
                "state_price_support_window": "in_support",
                "state_price_observation_status": "observed_ndcp_support",
                "state_price_nowcast": False,
                "state_ndcp_imputed_share": 0.31,
                "state_qcew_wage_observed_share": 0.97,
                "state_qcew_employment_observed_share": 0.97,
                "state_qcew_labor_observed_share": 0.97,
                "observed_core_exclusion_reason": "",
                "observed_core_low_impute_exclusion_reason": "imputation_share_above_threshold",
                "is_sensitivity_year": False,
            },
            {
                "state_fips": "02",
                "year": 2023,
                "subgroup": "all",
                "state_price_index": 12500.0,
                "unpaid_childcare_hours": 15.5,
                "atus_weight_sum": 140.0,
                "outside_option_wage": 22.5,
                "parent_employment_rate": 0.76,
                "single_parent_share": 0.2,
                "median_income": 60000.0,
                "unemployment_rate": 0.038,
                "market_quantity_proxy": 1210.0,
                "unpaid_quantity_proxy": 140.0,
                "benchmark_replacement_cost": 8500.0,
                "state_direct_care_price_index": 5300.0,
                "state_direct_care_price_index_raw": 5400.0,
                "state_non_direct_care_price_index": 7200.0,
                "state_direct_care_labor_share": 5300.0 / 12500.0,
                "state_direct_care_price_clip_binding_share": 0.0,
                "state_direct_care_price_clip_binding": False,
                "state_effective_children_per_worker": 5.0,
                "state_implied_direct_care_hourly_wage": 10.61,
                "eligible_broad_complete": True,
                "eligible_observed_core": False,
                "eligible_observed_core_low_impute": False,
                "state_controls_source": "acs_observed",
                "state_unemployment_source": "laus_observed",
                "state_price_support_window": "out_of_support",
                "state_price_observation_status": "post_ndcp_nowcast",
                "state_price_nowcast": True,
                "state_ndcp_imputed_share": 0.28,
                "state_qcew_wage_observed_share": 0.96,
                "state_qcew_employment_observed_share": 0.96,
                "state_qcew_labor_observed_share": 0.96,
                "observed_core_exclusion_reason": "outside_year_window",
                "observed_core_low_impute_exclusion_reason": "outside_year_window",
                "is_sensitivity_year": False,
            },
        ]
    )
    county = pd.DataFrame(
        [
            {
                "state_fips": "01",
                "year": 2014,
                "provider_density": 1.0,
                "annual_price": 10000.0,
                "under5_population": 1000.0,
                "direct_care_price_index": 4800.0,
                "non_direct_care_price_index": 5200.0,
                "benchmark_childcare_wage": 8.4,
            },
            {
                "state_fips": "02",
                "year": 2015,
                "provider_density": 1.1,
                "annual_price": 10200.0,
                "under5_population": 1200.0,
                "direct_care_price_index": 5200.0,
                "non_direct_care_price_index": 5000.0,
                "benchmark_childcare_wage": 8.8,
            },
        ]
    )
    acs = pd.DataFrame(
        [
            {"county_fips": "01001", "state_fips": "01", "year": 2014, "under5_population": 1300.0},
            {"county_fips": "02001", "state_fips": "02", "year": 2015, "under5_population": 1500.0},
        ]
    )
    write_parquet(state, project_paths.processed / "childcare_state_year_panel.parquet")
    write_parquet(county, project_paths.processed / "childcare_county_year_price_panel.parquet")
    write_parquet(acs, project_paths.interim / "acs" / "acs.parquet")
    write_json(
        {
            "comparison_specification_profile": "household_parsimonious",
            "samples": {
                "broad_complete": {
                    "n_obs": 228,
                    "n_states": 22,
                    "n_years": 15,
                    "economically_admissible": True,
                    "headline_eligible": False,
                },
                "observed_core": {
                    "n_obs": 139,
                    "n_states": 21,
                    "n_years": 8,
                    "economically_admissible": True,
                    "headline_eligible": True,
                },
                "observed_core_low_impute": {
                    "n_obs": 55,
                    "n_states": 13,
                    "n_years": 8,
                    "economically_admissible": True,
                    "headline_eligible": False,
                },
            }
        },
        project_paths.outputs_reports / "childcare_demand_sample_comparison.json",
    )
    for sample_name, elasticity in (
        ("broad_complete", -0.15),
        ("observed_core", 0.30),
        ("observed_core_low_impute", 0.80),
    ):
        write_json(
            {
                "mode": sample_name,
                "specification_profile": "full_controls",
                "price_coefficient": -0.001,
                "elasticity_at_mean": elasticity,
                "first_stage_r2": 0.8,
                "n_obs": 10,
                "n_states": 2,
                "n_years": 2,
                "year_min": 2014,
                "year_max": 2015,
                "economically_admissible": elasticity <= 0,
            },
            project_paths.outputs_reports / f"childcare_demand_iv_{sample_name}.json",
        )
    write_json(
        {
            "mode": "broad_complete",
            "specification_profile": "household_parsimonious",
            "price_coefficient": -0.001,
            "elasticity_at_mean": -0.14,
            "first_stage_r2": 0.79,
            "n_obs": 12,
            "n_states": 2,
            "n_years": 3,
            "year_min": 2009,
            "year_max": 2023,
            "economically_admissible": True,
        },
        project_paths.outputs_reports / "childcare_demand_iv_broad_complete_household_parsimonious.json",
    )
    write_json(
        {
            "mode": "observed_core",
            "specification_profile": "household_parsimonious",
            "price_coefficient": -0.001,
            "elasticity_at_mean": -0.11,
            "first_stage_r2": 0.78,
            "n_obs": 10,
            "n_states": 2,
            "n_years": 2,
            "year_min": 2014,
            "year_max": 2015,
            "economically_admissible": True,
        },
        project_paths.outputs_reports / "childcare_demand_iv_observed_core_household_parsimonious.json",
    )
    write_json(
        {
            "mode": "observed_core_low_impute",
            "specification_profile": "household_parsimonious",
            "price_coefficient": -0.001,
            "elasticity_at_mean": -0.10,
            "first_stage_r2": 0.77,
            "n_obs": 8,
            "n_states": 2,
            "n_years": 2,
            "year_min": 2014,
            "year_max": 2015,
            "economically_admissible": True,
        },
        project_paths.outputs_reports / "childcare_demand_iv_observed_core_low_impute_household_parsimonious.json",
    )
    write_json(
        {
            "mode": "observed_core",
            "specification_profile": "instrument_only",
            "price_coefficient": -0.001,
            "elasticity_at_mean": -0.09,
            "first_stage_r2": 0.75,
            "n_obs": 10,
            "n_states": 2,
            "n_years": 2,
            "year_min": 2014,
            "year_max": 2015,
            "economically_admissible": True,
        },
        project_paths.outputs_reports / "childcare_demand_iv_observed_core_instrument_only.json",
    )
    write_json(
        {
            "profiles": {
                "household_parsimonious": {},
                "instrument_only": {},
            }
        },
        project_paths.outputs_reports / "childcare_demand_specification_sweep.json",
    )


def test_simulate_childcare_writes_all_sample_outputs(project_paths):
    _write_childcare_simulation_inputs(project_paths)

    cli.simulate_childcare(project_paths)

    canonical = read_parquet(project_paths.processed / "childcare_marketization_scenarios.parquet")
    combined = read_parquet(project_paths.processed / "childcare_marketization_scenarios_all_samples.parquet")
    diagnostics = read_json(project_paths.outputs_reports / "childcare_scenario_diagnostics.json")
    comparison = read_json(project_paths.outputs_reports / "childcare_scenario_sample_comparison.json")
    spec_comparison = read_json(project_paths.outputs_reports / "childcare_scenario_specification_comparison.json")
    decomposition = read_json(project_paths.outputs_reports / "childcare_price_decomposition.json")
    supply = read_json(project_paths.outputs_reports / "childcare_supply_elasticity.json")
    piecewise = read_json(project_paths.outputs_reports / "childcare_piecewise_supply_demo.json")

    assert canonical["demand_sample_name"].eq("observed_core").all()
    assert set(combined["demand_sample_name"].unique()) == {
        "broad_complete",
        "observed_core",
        "observed_core_low_impute",
    }
    assert diagnostics["demand_sample_name"] == "observed_core"
    assert diagnostics["demand_specification_profile"] == "household_parsimonious"
    assert diagnostics["scenario_price_nowcast_rows"] == 0
    assert diagnostics["demand_fit_quarantined"] is False
    assert diagnostics["supply_estimation_method"] == supply["estimation_method"]
    assert comparison["selected_headline_sample"] == "observed_core"
    assert comparison["comparison_specification_profile"] == "household_parsimonious"
    assert comparison["samples"]["broad_complete"]["scenario_price_nowcast_rows"] == 4
    assert comparison["samples"]["broad_complete"]["demand_sample_selection_reason"] == "comparison_only"
    assert comparison["samples"]["observed_core"]["demand_sample_selection_reason"] == "observed_core_passes_minimum_support"
    assert comparison["samples"]["broad_complete"]["demand_specification_profile"] == "household_parsimonious"
    assert set(spec_comparison["profiles"]) == {"household_parsimonious", "instrument_only"}
    assert spec_comparison["comparison_specification_profile"] == "household_parsimonious"
    assert "state_price_observation_status" in canonical.columns
    assert "state_price_nowcast" in canonical.columns
    assert "p_alpha_direct_care" in canonical.columns
    assert "wage_alpha_implied" in canonical.columns
    assert "demand_elasticity_signed" in canonical.columns
    assert "solver_demand_elasticity_magnitude" in canonical.columns
    assert canonical["p_alpha_direct_care"].gt(0).all()
    assert decomposition["selected_headline_sample"] == "observed_core"
    assert decomposition["canonical"]["baseline_direct_care_clip_binding_row_share"] > 0
    assert supply["supply_elasticity"] > 0
    assert "0.50" in decomposition["canonical"]["alphas"]
    assert piecewise["demo_sample_name"] == "observed_core"
    assert piecewise["piecewise_method"] == "state_year_piecewise_isoelastic"
    assert (project_paths.processed / "childcare_piecewise_supply_demo.parquet").exists()


def test_simulate_childcare_writes_price_decomposition_sensitivity(project_paths):
    _write_childcare_simulation_inputs(project_paths)

    cli.simulate_childcare(project_paths)

    sensitivity = read_json(
        project_paths.outputs_reports / "childcare_price_decomposition_sensitivity.json"
    )
    assert sensitivity["n_cases"] == 9
    assert len(sensitivity["cases"]) == 9

    # Check that the base/base case matches the canonical decomposition.
    base_case = [
        c for c in sensitivity["cases"]
        if c["staffing_case"] == "base" and c["fringe_case"] == "base"
    ]
    assert len(base_case) == 1
    canonical = read_json(
        project_paths.outputs_reports / "childcare_price_decomposition.json"
    )
    # Base/base baseline price should match canonical baseline price.
    assert abs(
        base_case[0]["baseline_price_p50"] - canonical["canonical"]["baseline_price_p50"]
    ) < 0.01

    # Low staffing (more workers per child) should yield higher direct-care prices
    # than high staffing (fewer workers per child), holding fringe constant.
    low_staff_base_fringe = [
        c for c in sensitivity["cases"]
        if c["staffing_case"] == "low" and c["fringe_case"] == "base"
    ][0]
    high_staff_base_fringe = [
        c for c in sensitivity["cases"]
        if c["staffing_case"] == "high" and c["fringe_case"] == "base"
    ][0]
    assert (
        low_staff_base_fringe["baseline_direct_care_price_p50"]
        >= high_staff_base_fringe["baseline_direct_care_price_p50"]
    )

    # Higher fringe should yield higher direct-care prices, holding staffing constant.
    base_staff_low_fringe = [
        c for c in sensitivity["cases"]
        if c["staffing_case"] == "base" and c["fringe_case"] == "low"
    ][0]
    base_staff_high_fringe = [
        c for c in sensitivity["cases"]
        if c["staffing_case"] == "base" and c["fringe_case"] == "high"
    ][0]
    assert (
        base_staff_high_fringe["baseline_direct_care_price_p50"]
        >= base_staff_low_fringe["baseline_direct_care_price_p50"]
    )


def test_recompute_decomposition_preserves_gross_prices(project_paths):
    """Sensitivity sweep must not change gross prices."""
    frame = pd.DataFrame(
        [
            {
                "state_fips": "01",
                "year": 2015,
                "alpha": 0.50,
                "p_baseline": 11000.0,
                "p_alpha": 11500.0,
                "effective_children_per_worker": 5.5,
                "direct_care_fringe_multiplier": 1.20,
                "direct_care_labor_share": 0.44,
            },
        ]
    )

    result = cli._recompute_decomposition_under_assumptions(
        frame,
        0.80,
        1.35,
        childcare_model_assumptions(project_paths),
    )

    # Gross baseline and alpha prices are untouched.
    assert result["baseline_price_p50"] == 11000.0
    assert result["alphas"]["0.50"]["price_p50"] == 11500.0
    # Direct-care price must not exceed gross.
    assert result["baseline_direct_care_price_p50"] <= 11000.0
    assert result["alphas"]["0.50"]["direct_care_price_p50"] <= 11500.0


def test_simulate_childcare_fails_without_defensible_observed_core(project_paths):
    _write_childcare_simulation_inputs(project_paths)
    write_json(
        {
            "samples": {
                "broad_complete": {
                    "n_obs": 228,
                    "n_states": 22,
                    "n_years": 15,
                },
                "observed_core": {
                    "n_obs": 20,
                    "n_states": 5,
                    "n_years": 3,
                },
                "observed_core_low_impute": {
                    "n_obs": 10,
                    "n_states": 4,
                    "n_years": 2,
                },
            }
        },
        project_paths.outputs_reports / "childcare_demand_sample_comparison.json",
    )

    with pytest.raises(UnpaidWorkError, match="only exploratory samples are available"):
        cli.simulate_childcare(project_paths)


def test_simulate_childcare_quarantines_inadmissible_comparison_fit(project_paths):
    _write_childcare_simulation_inputs(project_paths)
    write_json(
        {
            "mode": "broad_complete",
            "specification_profile": "household_parsimonious",
            "price_coefficient": 0.001,
            "elasticity_at_mean": 0.14,
            "first_stage_r2": 0.79,
            "n_obs": 12,
            "n_states": 2,
            "n_years": 3,
            "year_min": 2009,
            "year_max": 2023,
            "economically_admissible": False,
        },
        project_paths.outputs_reports / "childcare_demand_iv_broad_complete_household_parsimonious.json",
    )

    cli.simulate_childcare(project_paths)

    comparison = read_json(project_paths.outputs_reports / "childcare_scenario_sample_comparison.json")
    combined = read_parquet(project_paths.processed / "childcare_marketization_scenarios_all_samples.parquet")

    assert comparison["samples"]["broad_complete"]["demand_fit_quarantined"] is True
    assert "positive price response" in comparison["samples"]["broad_complete"]["demand_fit_quarantine_reason"]
    assert set(combined["demand_sample_name"].unique()) == {"observed_core", "observed_core_low_impute"}


def test_report_writes_figure_assets(project_paths):
    _write_childcare_simulation_inputs(project_paths)
    cli.simulate_childcare(project_paths)
    write_json(
        {"holdout_year": 2022, "rmse_test": 123.4},
        project_paths.outputs_reports / "childcare_price_surface.json",
    )
    write_json(
        {
            "state_year_rows": 3,
            "county_year_rows": 2,
            "births_cdc_wonder_observed": 2,
            "state_controls_acs_observed": 2,
            "county_controls_acs_direct": 1,
            "county_wage_observed": 1,
            "county_employment_observed": 1,
            "county_laus_observed": 2,
        },
        project_paths.outputs_reports / "childcare_pipeline_diagnostics.json",
    )
    shock_panel_path = project_paths.interim / "licensing" / "licensing_supply_shocks.parquet"
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "year": 2017,
                    "center_labor_intensity_index": 0.20,
                    "center_labor_intensity_shock": 0.0,
                },
                {
                    "state_fips": "01",
                    "year": 2019,
                    "center_labor_intensity_index": 0.24,
                    "center_labor_intensity_shock": 0.04,
                },
            ]
        ),
        shock_panel_path,
    )
    write_json(
        {
            "status": "ok",
            "pilot_scope": "single_state_pilot",
            "sample_mode": False,
            "n_obs": 364,
            "n_counties": 91,
            "n_states": 1,
            "year_min": 2017,
            "year_max": 2022,
            "shock_state_count": 1,
            "treated_state_fips": ["01"],
            "first_stage_strength_flag": "weak_or_unknown",
            "shock_panel_path": str(shock_panel_path),
            "first_stage_price": {"beta": 1.868},
            "reduced_form_provider_density": {"beta": 29.796},
            "reduced_form_employer_establishment_density": {"beta": 0.0},
            "iv_supply_elasticity_provider_density": 15.951,
            "local_iv_supply_elasticity_provider_density": 15.951,
            "secondary_supply_estimate": {
                "name": "local_iv_supply_elasticity_provider_density",
                "value": 15.951,
                "scope": "treated_state_local_wald",
            },
        },
        project_paths.outputs_reports / "childcare_supply_iv.json",
    )

    cli.report(project_paths)

    assert (project_paths.outputs_reports / "childcare_mvp_report.md").exists()
    assert (project_paths.outputs_reports / "childcare_headline_summary.json").exists()
    assert (project_paths.outputs_reports / "childcare_headline_readout.md").exists()
    assert (project_paths.outputs_reports / "childcare_satellite_account.json").exists()
    assert (project_paths.outputs_reports / "childcare_satellite_account.md").exists()
    assert (project_paths.outputs_tables / "childcare_satellite_account_annual.csv").exists()
    assert (project_paths.outputs_reports / "model_assumption_audit.json").exists()
    assert (project_paths.outputs_figures / "childcare_marketization_diagram.svg").exists()
    assert (project_paths.outputs_figures / "childcare_sample_ladder.svg").exists()
    assert (project_paths.outputs_figures / "childcare_alpha_intervals.svg").exists()
    assert (project_paths.outputs_figures / "childcare_price_decomposition_by_alpha.svg").exists()
    assert (project_paths.outputs_figures / "childcare_alpha_examples.svg").exists()
    assert (project_paths.outputs_figures / "childcare_solver_implied_curves.svg").exists()
    assert (project_paths.outputs_figures / "childcare_support_boundary.svg").exists()
    assert (project_paths.outputs_figures / "childcare_scenario_specification_comparison.svg").exists()
    assert (project_paths.outputs_figures / "childcare_piecewise_supply_demo.svg").exists()
    assert (project_paths.outputs_figures / "childcare_supply_iv_pilot.svg").exists()
    assert (project_paths.outputs_figures / "childcare_local_iv_marketization_demo.svg").exists()
    manifest = (project_paths.outputs_figures / "figure_manifest.md").read_text(encoding="utf-8")
    report = (project_paths.outputs_reports / "childcare_mvp_report.md").read_text(encoding="utf-8")
    readout = (project_paths.outputs_reports / "childcare_headline_readout.md").read_text(encoding="utf-8")
    satellite = read_json(project_paths.outputs_reports / "childcare_satellite_account.json")
    satellite_md = (project_paths.outputs_reports / "childcare_satellite_account.md").read_text(encoding="utf-8")
    assert "Stylized marketization diagram" in manifest
    assert "Price decomposition by alpha" in manifest
    assert "Alpha examples with implied wages" in manifest
    assert "Solver-implied supply and demand curves" in manifest
    assert "Piecewise supply demo" in manifest
    assert "Supply IV pilot" in manifest
    assert "Local IV-informed marketization demo" in manifest
    assert "## Piecewise supply demo" in report
    assert "## National benchmark satellite account" in report
    assert "secondary local IV supply elasticity (provider density)" in report
    assert "Childcare Headline Readout" in readout
    assert satellite["preferred_series"] == "direct_care_nationalized_value"
    assert satellite["latest_year"]["price_support_population_share"] < 1.0
    assert "marginal replacement price x unpaid child-equivalent quantity" in satellite_md
