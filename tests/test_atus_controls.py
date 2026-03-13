from __future__ import annotations

import pandas as pd

from unpaidwork.clean.atus import build_state_year_panel
from unpaidwork.ingest.acs import ingest as ingest_acs
from unpaidwork.ingest.atus import ingest as ingest_atus
from unpaidwork.ingest.cdc_wonder import ingest as ingest_cdc_wonder
from unpaidwork.ingest.laus import ingest as ingest_laus
from unpaidwork.storage import read_parquet, write_parquet


def test_state_year_panel_prefers_observed_acs_and_laus_controls(project_paths):
    ingest_atus(project_paths, sample=True)
    ingest_acs(project_paths, sample=True)
    ingest_laus(project_paths, sample=True)
    ingest_cdc_wonder(project_paths, sample=True)

    acs_path = project_paths.interim / "acs" / "acs.parquet"
    acs = read_parquet(acs_path)
    mask = (acs["state_fips"] == "06") & (acs["year"] == 2021)
    acs.loc[mask, "under6_population"] = [100.0, 200.0]
    acs.loc[mask, "single_parent_share"] = [0.20, 0.40]
    acs.loc[mask, "parent_employment_rate"] = [0.70, 0.85]
    acs.loc[mask, "median_income"] = [90000, 120000]
    write_parquet(acs, acs_path)

    panel = build_state_year_panel(project_paths)
    california = panel.loc[(panel["state_fips"] == "06") & (panel["year"] == 2021)].iloc[0]

    assert california["single_parent_share"] == 1 / 3
    assert california["parent_employment_rate"] == 0.80
    assert california["median_income"] == 110000.0
    assert california["unemployment_rate"] == 0.076
    assert california["births"] == 420786.0


def test_state_year_panel_tracks_births_source_and_suppression_status(project_paths):
    ingest_atus(project_paths, sample=True)
    ingest_acs(project_paths, sample=True)
    ingest_laus(project_paths, sample=True)
    ingest_cdc_wonder(project_paths, sample=True)

    cdc_path = project_paths.interim / "cdc_wonder" / "cdc_wonder.parquet"
    cdc = read_parquet(cdc_path)
    cdc["suppression_note"] = cdc["suppression_note"].astype("object")
    cdc.loc[cdc["state_fips"] == "48", "births"] = pd.NA
    cdc.loc[cdc["state_fips"] == "48", "births_suppressed"] = True
    cdc.loc[cdc["state_fips"] == "48", "suppression_note"] = "Suppressed for confidentiality"
    cdc = cdc.loc[cdc["state_fips"] != "36"].copy()
    write_parquet(cdc, cdc_path)

    panel = build_state_year_panel(project_paths)
    california = panel.loc[(panel["state_fips"] == "06") & (panel["year"] == 2021)].iloc[0]
    texas = panel.loc[(panel["state_fips"] == "48") & (panel["year"] == 2021)].iloc[0]
    new_york = panel.loc[(panel["state_fips"] == "36") & (panel["year"] == 2021)].iloc[0]

    assert california["births_value_source"] == "cdc_wonder_observed"
    assert bool(california["births_observed_available"]) is True
    assert bool(california["births_suppressed_observed"]) is False

    assert texas["births_value_source"] == "cdc_wonder_suppressed_fallback"
    assert bool(texas["births_observed_available"]) is False
    assert bool(texas["births_suppressed_observed"]) is True
    assert "Suppressed" in str(texas["births_suppression_note"])

    assert new_york["births_value_source"] == "atus_reported"
    assert bool(new_york["births_observed_available"]) is False
    assert bool(new_york["births_suppressed_observed"]) is False
