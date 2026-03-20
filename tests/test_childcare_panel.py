from __future__ import annotations

import numpy as np
import pandas as pd

from unpriced.features.childcare_panel import build_childcare_panels, diagnose_childcare_pipeline
from unpriced.features.demand import aggregate_counties_to_state
from unpriced.ingest.acs import ingest as ingest_acs
from unpriced.ingest.atus import ingest as ingest_atus
from unpriced.ingest.head_start import ingest as ingest_head_start
from unpriced.ingest.nces_ccd import ingest as ingest_nces_ccd
from unpriced.ingest.ndcp import ingest as ingest_ndcp
from unpriced.ingest.oews import ingest as ingest_oews
from unpriced.ingest.qcew import ingest as ingest_qcew
from unpriced.storage import read_parquet, write_parquet


def test_build_childcare_panels(project_paths):
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs, ingest_head_start, ingest_nces_ccd, ingest_oews):
        ingestor(project_paths, sample=True)
    county, state = build_childcare_panels(project_paths)
    assert {
        "provider_density",
        "benchmark_childcare_wage",
        "direct_care_price_index",
        "non_direct_care_price_index",
        "direct_care_labor_share",
        "effective_children_per_worker",
        "head_start_slots",
        "public_school_option_index",
    } <= set(county.columns)
    assert {
        "state_price_index",
        "state_direct_care_price_index",
        "state_non_direct_care_price_index",
        "state_direct_care_labor_share",
        "state_effective_children_per_worker",
        "state_implied_direct_care_hourly_wage",
        "unpaid_quantity_proxy",
        "state_controls_source",
        "state_unemployment_source",
        "state_price_support_window",
        "state_price_observation_status",
        "state_price_nowcast",
        "state_ndcp_imputed_share",
        "state_qcew_wage_observed_share",
        "state_qcew_employment_observed_share",
        "state_qcew_labor_observed_share",
        "eligible_broad_complete",
        "eligible_observed_core",
        "eligible_observed_core_low_impute",
        "observed_core_exclusion_reason",
        "observed_core_low_impute_exclusion_reason",
    } <= set(state.columns)
    assert len(state) >= 3
    assert county["direct_care_price_index"].le(county["annual_price"]).all()
    assert state["state_direct_care_price_index"].le(state["state_price_index"]).fillna(True).all()


def test_build_childcare_panels_backfills_county_covariates(project_paths):
    ingest_atus(project_paths, sample=True)
    ingest_ndcp(project_paths, sample=True)
    ingest_qcew(project_paths, sample=True)
    ingest_acs(project_paths, sample=True)
    ingest_head_start(project_paths, sample=True)
    ingest_nces_ccd(project_paths, sample=True)
    ingest_oews(project_paths, sample=True)

    acs_path = project_paths.interim / "acs" / "acs.parquet"
    acs = read_parquet(acs_path)
    for column in ["single_parent_share", "median_income", "unemployment_rate"]:
        acs[column] = pd.to_numeric(acs[column], errors="coerce").astype(float)
    acs.loc[:, ["single_parent_share", "median_income", "unemployment_rate"]] = np.nan
    write_parquet(acs, acs_path)

    county, _ = build_childcare_panels(project_paths)

    assert county["single_parent_share"].notna().all()
    assert county["median_income"].notna().all()
    assert county["unemployment_rate"].notna().all()


def test_build_childcare_panels_uses_head_start_slots(project_paths):
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs, ingest_head_start, ingest_nces_ccd, ingest_oews):
        ingestor(project_paths, sample=True)

    county, _ = build_childcare_panels(project_paths)
    los_angeles = county.loc[county["county_fips"] == "06037"].iloc[0]

    assert los_angeles["head_start_slots"] == 3120.0
    assert los_angeles["head_start_capacity"] == 3120.0
    assert los_angeles["head_start_slot_share"] > 0


def test_build_childcare_panels_uses_ccd_public_school_index(project_paths):
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs, ingest_head_start, ingest_nces_ccd, ingest_oews):
        ingestor(project_paths, sample=True)

    county, _ = build_childcare_panels(project_paths)
    california = county.loc[county["state_fips"] == "06"].iloc[0]

    assert california["public_school_option_index"] == 0.609


def test_build_childcare_panels_fills_missing_qcew_wages_from_oews(project_paths):
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs, ingest_head_start, ingest_nces_ccd, ingest_oews):
        ingestor(project_paths, sample=True)

    qcew_path = project_paths.interim / "qcew" / "qcew.parquet"
    qcew = read_parquet(qcew_path)
    mask = (qcew["county_fips"] == "06037") & (qcew["year"] == 2021)
    qcew.loc[mask, ["childcare_worker_wage", "outside_option_wage"]] = np.nan
    write_parquet(qcew, qcew_path)

    county, _ = build_childcare_panels(project_paths)
    california = county.loc[(county["county_fips"] == "06037") & (county["year"] == 2021)].iloc[0]

    assert california["childcare_worker_wage"] == 18.4
    assert california["outside_option_wage"] == 21.2
    assert california["outside_option_wage_source"] == "oews_state_observed"


def test_diagnose_childcare_pipeline_produces_artifact(project_paths):
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs, ingest_head_start, ingest_nces_ccd, ingest_oews):
        ingestor(project_paths, sample=True)
    county, state = build_childcare_panels(project_paths)
    diag = diagnose_childcare_pipeline(county, state, project_paths)

    assert diag["state_year_rows"] > 0
    assert diag["county_year_rows"] > 0
    assert "births_cdc_wonder_observed" in diag
    assert "births_atus_reported" in diag
    assert "ndcp_mean_imputed_share" in diag
    assert "county_controls_acs_direct" in diag
    assert "county_wage_price_derived" in diag
    assert "county_employment_synthetic" in diag
    assert "atus_sensitivity_year_rows" in diag
    assert "eligible_observed_core" in diag
    assert "observed_core_exclusion_counts" in diag
    assert "state_qcew_labor_observed_share_mean" in diag
    assert "state_price_observation_status_counts" in diag
    assert "state_price_post_support_nowcast_rows" in diag
    assert "state_direct_care_labor_share_mean" in diag
    assert "state_implied_direct_care_hourly_wage_p50" in diag

    artifact_path = project_paths.outputs_reports / "childcare_pipeline_diagnostics.json"
    assert artifact_path.exists()

    from unpriced.storage import read_json
    loaded = read_json(artifact_path)
    assert loaded["state_year_rows"] == diag["state_year_rows"]


def test_diagnose_childcare_pipeline_counts_are_consistent(project_paths):
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs, ingest_head_start, ingest_nces_ccd, ingest_oews):
        ingestor(project_paths, sample=True)
    county, state = build_childcare_panels(project_paths)
    diag = diagnose_childcare_pipeline(county, state, project_paths)

    # Births sources must sum to total state rows.
    births_total = (
        diag["births_cdc_wonder_observed"]
        + diag["births_atus_reported"]
        + diag["births_cdc_suppressed_fallback"]
        + diag["births_atus_unmatched_fallback"]
    )
    assert births_total == diag["state_year_rows"]

    # ACS observed + ATUS synthetic = total
    assert (
        diag["state_controls_acs_observed"] + diag["state_controls_atus_synthetic"]
        == diag["state_year_rows"]
    )

    # County ACS direct + state backfill = total
    assert (
        diag["county_controls_acs_direct"] + diag["county_controls_state_backfill"]
        == diag["county_year_rows"]
    )


def test_diagnose_childcare_pipeline_reports_acs_year_coverage(project_paths):
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs, ingest_head_start, ingest_nces_ccd, ingest_oews):
        ingestor(project_paths, sample=True)
    county, state = build_childcare_panels(project_paths)
    diag = diagnose_childcare_pipeline(county, state, project_paths)

    assert "acs_available_years" in diag
    assert "county_panel_years" in diag
    assert "acs_missing_county_years" in diag
    # Sample data has matching years, so no missing years
    assert len(diag["acs_missing_county_years"]) == 0
    assert diag["county_controls_acs_direct"] == diag["county_year_rows"]


def test_build_childcare_panels_assigns_state_sources(project_paths):
    for ingestor in (ingest_atus, ingest_ndcp, ingest_qcew, ingest_acs, ingest_head_start, ingest_nces_ccd, ingest_oews):
        ingestor(project_paths, sample=True)

    _, state = build_childcare_panels(project_paths)

    assert set(state["state_controls_source"].unique()) <= {"acs_observed", "atus_fallback"}
    assert set(state["state_unemployment_source"].unique()) <= {"laus_observed", "atus_fallback"}
    assert set(state["state_price_observation_status"].unique()) <= {
        "observed_ndcp_support",
        "pre_ndcp_support_gap",
        "post_ndcp_nowcast",
    }


def test_aggregate_counties_to_state_handles_missing_weights():
    frame = pd.DataFrame(
        {
            "state_fips": ["01", "01", "01"],
            "year": [2022, 2022, 2022],
            "under5_population": [pd.NA, 0.0, 10.0],
            "annual_price": [100.0, 120.0, 140.0],
        }
    )

    result = aggregate_counties_to_state(frame, ["annual_price"])

    assert len(result) == 1
    assert result.loc[0, "annual_price"] == 140.0
    assert result.loc[0, "under5_population"] == 10.0
