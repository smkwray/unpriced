from __future__ import annotations

import pandas as pd

from unpriced.features import home_maintenance_panel as home_panel
from unpriced.features.home_maintenance_panel import (
    AHS_NONGEOGRAPHIC_CBSA,
    build_home_maintenance_panel,
)
from unpriced.ingest.ahs import ingest as ingest_ahs
from unpriced.ingest.laus import ingest as ingest_laus
from unpriced.ingest.noaa import ingest as ingest_noaa


def _mock_crosswalk(paths):
    return pd.DataFrame(
        [
            {"cbsa_code": "31080", "county_fips": "06037"},
            {"cbsa_code": "19100", "county_fips": "48113"},
            {"cbsa_code": "35620", "county_fips": "36061"},
        ]
    )


def test_build_home_maintenance_panel(project_paths):
    ingest_ahs(project_paths, sample=True)
    panel = build_home_maintenance_panel(project_paths)
    assert "predicted_job_cost" in panel.columns
    assert panel["predicted_job_cost"].notna().all()
    assert "job_group" in panel.columns
    assert "precip_event_days" in panel.columns
    assert "noaa_match_status" in panel.columns


def test_build_home_maintenance_panel_with_laus_controls(project_paths, monkeypatch):
    ingest_ahs(project_paths, sample=True)
    ingest_laus(project_paths, sample=True)

    monkeypatch.setattr(home_panel, "load_cbsa_county_crosswalk", _mock_crosswalk)

    panel = build_home_maintenance_panel(project_paths)
    observed = panel.groupby("cbsa_code", as_index=False)["cbsa_unemployment_rate"].first()
    rates = dict(zip(observed["cbsa_code"], observed["cbsa_unemployment_rate"]))

    assert abs(rates["31080"] - (362000.0 / 4882000.0)) < 1e-9
    assert abs(rates["19100"] - (89000.0 / 1309000.0)) < 1e-9
    assert abs(rates["35620"] - (71000.0 / 818000.0)) < 1e-9
    assert "cbsa_unemployment_year" in panel.columns


def test_noaa_merge_labels_observed_rows(project_paths, monkeypatch):
    ingest_ahs(project_paths, sample=True)
    ingest_noaa(project_paths, sample=True)

    monkeypatch.setattr(home_panel, "load_cbsa_county_crosswalk", _mock_crosswalk)

    panel = build_home_maintenance_panel(project_paths)
    assert "noaa_match_status" in panel.columns
    assert (panel["noaa_match_status"] == "observed").all()
    assert panel["storm_exposure"].gt(0).all()
    assert panel["precip_event_days"].notna().all()


def test_noaa_merge_fills_nongeographic_cbsa_with_national_avg(project_paths, monkeypatch):
    """Rows with AHS non-geographic CBSAs get national-average NOAA values."""
    ingest_noaa(project_paths, sample=True)

    # Build a fake AHS panel that includes a 99999 non-metro CBSA row.
    ahs_rows = pd.DataFrame(
        [
            {"job_id": "a", "cbsa_code": "31080", "year": 2021, "job_type_code": "36",
             "job_type": "Landscaping", "job_group": "lot_yard_and_outbuildings",
             "job_cost": 2000, "job_diy": 0, "weight": 1.0, "housing_vintage": 1990,
             "home_value": 500000, "household_income": 80000, "tenure_owner": 1,
             "storm_exposure": 0.0},
            {"job_id": "b", "cbsa_code": "99999", "year": 2021, "job_type_code": "16",
             "job_type": "Roofing", "job_group": "exterior_envelope",
             "job_cost": 5000, "job_diy": 1, "weight": 1.0, "housing_vintage": 1980,
             "home_value": 200000, "household_income": 50000, "tenure_owner": 1,
             "storm_exposure": 0.0},
            {"job_id": "c", "cbsa_code": "99998", "year": 2021, "job_type_code": "29",
             "job_type": "Water heater", "job_group": "systems_and_fixtures",
             "job_cost": 800, "job_diy": 1, "weight": 1.0, "housing_vintage": 1975,
             "home_value": 150000, "household_income": 40000, "tenure_owner": 1,
             "storm_exposure": 0.0},
        ]
    )
    from unpriced.storage import write_parquet
    write_parquet(ahs_rows, project_paths.interim / "ahs" / "ahs.parquet")

    monkeypatch.setattr(home_panel, "load_cbsa_county_crosswalk", _mock_crosswalk)

    panel = build_home_maintenance_panel(project_paths)

    metro = panel.loc[panel["cbsa_code"] == "31080"].iloc[0]
    nonmetro = panel.loc[panel["cbsa_code"] == "99999"].iloc[0]
    not_reported = panel.loc[panel["cbsa_code"] == "99998"].iloc[0]

    assert metro["noaa_match_status"] == "observed"
    assert nonmetro["noaa_match_status"] == "national_avg_nonmetro"
    assert not_reported["noaa_match_status"] == "national_avg_not_reported"

    # Non-geographic rows should have non-zero national-average storm_exposure.
    assert nonmetro["storm_exposure"] > 0
    assert not_reported["storm_exposure"] > 0
    assert nonmetro["precip_event_days"] > 0
    assert not_reported["precip_event_days"] > 0


def test_ahs_nongeographic_cbsa_codes():
    assert "99998" in AHS_NONGEOGRAPHIC_CBSA
    assert "99999" in AHS_NONGEOGRAPHIC_CBSA
