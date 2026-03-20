from __future__ import annotations

import pandas as pd
import unpriced.childcare.licensing_iv_backend as licensing_iv_backend

from unpriced.childcare.licensing_iv_backend import (
    build_licensing_event_study_results,
    build_licensing_first_stage_diagnostics,
    build_licensing_iv_backend_outputs,
    build_licensing_iv_results,
    build_licensing_leave_one_state_out,
    build_licensing_treatment_timing,
)
from unpriced.features.childcare_panel import build_childcare_panels
from unpriced.ingest.acs import ingest as ingest_acs
from unpriced.ingest.atus import ingest as ingest_atus
from unpriced.ingest.head_start import ingest as ingest_head_start
from unpriced.ingest.nces_ccd import ingest as ingest_nces_ccd
from unpriced.ingest.ndcp import ingest as ingest_ndcp
from unpriced.ingest.oews import ingest as ingest_oews
from unpriced.ingest.qcew import ingest as ingest_qcew


def _build_sample_childcare_panels(project_paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    for ingestor in (
        ingest_atus,
        ingest_ndcp,
        ingest_qcew,
        ingest_acs,
        ingest_head_start,
        ingest_nces_ccd,
        ingest_oews,
    ):
        ingestor(project_paths, sample=True)
    return build_childcare_panels(project_paths)


def _build_synthetic_licensing_inputs(state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_year = (
        state[["state_fips", "year"]]
        .drop_duplicates(["state_fips", "year"], keep="first")
        .sort_values(["state_fips", "year"], kind="stable")
        .reset_index(drop=True)
    )
    index_values: list[float] = []
    for row in state_year.itertuples(index=False):
        if row.state_fips == "06":
            if int(row.year) <= 2019:
                index_values.append(1.0)
            elif int(row.year) == 2020:
                index_values.append(1.2)
            else:
                index_values.append(1.4)
        elif row.state_fips == "36":
            if int(row.year) <= 2020:
                index_values.append(1.0)
            else:
                index_values.append(1.3)
        else:
            index_values.append(1.0)
    stringency_index = state_year.assign(licensing_stringency_equal_weight=index_values)

    harmonized_rows: list[dict[str, object]] = []
    for row in state_year.itertuples(index=False):
        for rule_idx, rule_family in enumerate(("ratio", "group_size", "staffing")):
            harmonized_rows.append(
                {
                    "state_fips": row.state_fips,
                    "year": int(row.year),
                    "provider_type": "center",
                    "age_band": "infant",
                    "rule_family": rule_family,
                    "rule_missing": bool(rule_idx == 2 and row.state_fips == "36" and int(row.year) == 2020),
                    "source_text": f"{rule_family} rule text",
                }
            )
    harmonized_rules = pd.DataFrame(harmonized_rows)
    return harmonized_rules, stringency_index


def test_build_licensing_treatment_timing_detects_start_years_and_rule_counts(project_paths):
    county, state = _build_sample_childcare_panels(project_paths)
    del county
    harmonized_rules, stringency_index = _build_synthetic_licensing_inputs(state)

    timing = build_licensing_treatment_timing(harmonized_rules, stringency_index)

    assert {
        "state_fips",
        "treatment_start_year",
        "ever_treated",
        "peak_abs_stringency_shock",
        "rule_row_count",
        "rule_missing_count",
        "rule_observed_count",
    } <= set(timing.columns)
    california = timing.loc[timing["state_fips"] == "06"].iloc[0]
    texas = timing.loc[timing["state_fips"] == "48"].iloc[0]
    assert int(california["treatment_start_year"]) == 2020
    assert bool(california["ever_treated"]) is True
    assert bool(texas["ever_treated"]) is False
    assert int(california["rule_row_count"]) > 0


def test_build_licensing_event_study_results_returns_report_ready_rows(project_paths):
    county, state = _build_sample_childcare_panels(project_paths)
    harmonized_rules, stringency_index = _build_synthetic_licensing_inputs(state)

    event_study = build_licensing_event_study_results(
        harmonized_rules,
        stringency_index,
        county,
        state,
    )

    assert {
        "outcome",
        "event_time",
        "estimate",
        "status",
        "n_treated_state_year",
        "n_control_state_year",
    } <= set(event_study.columns)
    assert set(event_study["outcome"]) >= {"log_annual_price", "log_provider_density"}
    assert -1 in set(event_study["event_time"])
    assert 1 in set(event_study["event_time"])
    assert set(event_study["status"]).issubset({"ok", "insufficient_overlap"})


def test_build_licensing_iv_results_first_stage_and_leave_one_out(project_paths):
    county, state = _build_sample_childcare_panels(project_paths)
    harmonized_rules, stringency_index = _build_synthetic_licensing_inputs(state)

    iv_results = build_licensing_iv_results(harmonized_rules, stringency_index, county, state)
    first_stage = build_licensing_first_stage_diagnostics(harmonized_rules, stringency_index, county, state)
    leave_one_out = build_licensing_leave_one_state_out(harmonized_rules, stringency_index, county, state)

    assert len(iv_results) == 2
    assert set(iv_results["outcome"]) == {"provider_density", "employer_establishment_density"}
    assert {
        "first_stage_beta",
        "first_stage_f_stat",
        "first_stage_strength_tier",
        "usable_for_headline",
        "treated_state_count_sufficiency_flag",
        "cluster_count_warning_flag",
        "recommended_use_tier",
        "shock_state_count",
        "treated_state_count",
    } <= set(iv_results.columns)
    assert len(first_stage) == 1
    assert {
        "first_stage_strength_flag",
        "first_stage_strength_tier",
        "usable_for_headline",
        "treated_state_count_sufficiency_flag",
        "cluster_count_warning_flag",
        "recommended_use_tier",
        "first_stage_f_stat",
        "reduced_form_provider_beta",
        "reduced_form_employer_beta",
    } <= set(first_stage.columns)
    assert "baseline" in set(leave_one_out["omitted_state_fips"])
    assert {"delta_provider_vs_baseline", "delta_employer_vs_baseline"} <= set(leave_one_out.columns)


def test_build_licensing_iv_backend_outputs_returns_tables_and_summary(project_paths):
    county, state = _build_sample_childcare_panels(project_paths)
    harmonized_rules, stringency_index = _build_synthetic_licensing_inputs(state)

    outputs = build_licensing_iv_backend_outputs(
        harmonized_rules,
        stringency_index,
        county,
        state,
    )

    assert {
        "event_study_results",
        "iv_results",
        "first_stage_diagnostics",
        "treatment_timing",
        "leave_one_state_out",
        "iv_usability_summary",
        "summary",
    } <= set(outputs.keys())
    assert isinstance(outputs["summary"], dict)
    assert len(outputs["iv_usability_summary"]) == len(outputs["iv_results"])
    assert outputs["summary"]["iv_result_row_count"] == len(outputs["iv_results"])
    assert outputs["summary"]["treatment_timing_row_count"] == len(outputs["treatment_timing"])
    assert "iv_statuses" in outputs["summary"]
    assert "iv_usability_counts" in outputs["summary"]
    assert "iv_usability_summary_rows" in outputs["summary"]


def test_build_licensing_iv_results_handles_missing_optional_county_columns(project_paths):
    county, state = _build_sample_childcare_panels(project_paths)
    harmonized_rules, stringency_index = _build_synthetic_licensing_inputs(state)
    county_missing = county.drop(
        columns=["employer_establishments", "nonemployer_firms", "under5_population"],
        errors="ignore",
    )

    iv_results = build_licensing_iv_results(harmonized_rules, stringency_index, county_missing, state)
    first_stage = build_licensing_first_stage_diagnostics(
        harmonized_rules, stringency_index, county_missing, state
    )

    assert len(iv_results) == 2
    assert len(first_stage) == 1
    assert iv_results["status"].notna().all()


def test_licensing_iv_quality_flags_mark_weak_sparse_design_as_diagnostics_only(monkeypatch):
    def _stub_fit_iv_summary(county_panel, shock_panel):
        del county_panel, shock_panel
        return (
            {
                "status": "ok",
                "design": "county_fe_state_year_fe_exposure_shock",
                "pilot_scope": "multi_state_demo",
                "first_stage_strength_flag": "weak_or_unknown",
                "first_stage_price": {
                    "beta": 0.05,
                    "f_stat": 2.4,
                    "n_clusters": 4,
                    "n_obs": 120,
                    "se_cluster_state": 0.04,
                    "t_cluster_state": 1.25,
                    "within_r2": 0.01,
                },
                "reduced_form_provider_density": {"beta": 0.1, "t_cluster_state": 0.7, "within_r2": 0.01},
                "reduced_form_employer_establishment_density": {
                    "beta": 0.05,
                    "t_cluster_state": 0.4,
                    "within_r2": 0.01,
                },
                "iv_supply_elasticity_provider_density": 0.3,
                "iv_supply_elasticity_employer_establishment_density": 0.2,
                "shock_state_count": 2,
                "treated_state_fips": ["01", "02"],
                "n_obs": 120,
                "n_states": 2,
                "year_min": 2019,
                "year_max": 2021,
            },
            pd.DataFrame(),
        )

    monkeypatch.setattr(licensing_iv_backend, "_fit_iv_summary", _stub_fit_iv_summary)

    stringency_index = pd.DataFrame(
        {
            "state_fips": ["01", "01", "02", "02"],
            "year": [2019, 2020, 2019, 2020],
            "licensing_stringency_equal_weight": [1.0, 1.1, 1.0, 1.05],
        }
    )
    county_panel = pd.DataFrame(
        {
            "state_fips": ["01", "01", "02", "02"],
            "county_fips": ["01001", "01001", "02001", "02001"],
            "year": [2019, 2020, 2019, 2020],
            "annual_price": [1000.0, 1050.0, 900.0, 920.0],
            "provider_density": [2.0, 2.1, 1.8, 1.85],
            "under5_population": [100.0, 100.0, 80.0, 80.0],
            "employer_establishments": [20.0, 20.0, 10.0, 10.0],
            "nonemployer_firms": [1.0, 1.0, 1.0, 1.0],
        }
    )

    iv_results = build_licensing_iv_results(None, stringency_index, county_panel, None)
    first_stage = build_licensing_first_stage_diagnostics(None, stringency_index, county_panel, None)
    outputs = build_licensing_iv_backend_outputs(None, stringency_index, county_panel, None)

    assert set(iv_results["first_stage_strength_tier"]) == {"weak"}
    assert set(iv_results["treated_state_count_sufficiency_flag"]) == {"sparse"}
    assert set(iv_results["cluster_count_warning_flag"]) == {True}
    assert set(iv_results["usable_for_headline"]) == {False}
    assert set(iv_results["recommended_use_tier"]) == {"diagnostics_only"}
    assert first_stage.iloc[0]["first_stage_strength_tier"] == "weak"
    assert bool(first_stage.iloc[0]["usable_for_headline"]) is False
    assert first_stage.iloc[0]["treated_state_count_sufficiency_flag"] == "sparse"
    assert bool(first_stage.iloc[0]["cluster_count_warning_flag"]) is True
    assert first_stage.iloc[0]["recommended_use_tier"] == "diagnostics_only"
    assert outputs["summary"]["iv_usability_counts"]["diagnostics_only"] == len(iv_results)
