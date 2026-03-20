from __future__ import annotations

from unpriced.features.childcare_panel import build_childcare_panels
from unpriced.ingest.acs import ingest as ingest_acs
from unpriced.ingest.atus import ingest as ingest_atus
from unpriced.ingest.head_start import ingest as ingest_head_start
from unpriced.ingest.laus import _normalize_laus_payload, ingest as ingest_laus
from unpriced.ingest.nces_ccd import ingest as ingest_nces_ccd
from unpriced.ingest.ndcp import ingest as ingest_ndcp
from unpriced.ingest.oews import ingest as ingest_oews
from unpriced.ingest.qcew import ingest as ingest_qcew
from unpriced.storage import read_parquet


def test_normalize_laus_payload_builds_county_and_state_rows():
    payload = {
        "Results": {
            "series": [
                {"seriesID": "LAUCN060370000000003", "data": [{"period": "M13", "value": "5.0"}]},
                {"seriesID": "LAUCN060370000000004", "data": [{"period": "M13", "value": "250056"}]},
                {"seriesID": "LAUST060000000000003", "data": [{"period": "M13", "value": "4.3"}]},
            ]
        }
    }

    frame = _normalize_laus_payload(payload, 2022)

    assert {"geography", "state_fips", "county_fips", "laus_unemployment_rate", "laus_unemployed"} <= set(frame.columns)
    county = frame.loc[frame["geography"] == "county"].iloc[0]
    state = frame.loc[frame["geography"] == "state"].iloc[0]
    assert county["county_fips"] == "06037"
    assert county["laus_unemployment_rate"] == 0.05
    assert county["laus_unemployed"] == 250056
    assert state["state_fips"] == "06"


def test_laus_sample_ingest_writes_normalized_panel(project_paths):
    ingest_laus(project_paths, sample=True)
    frame = read_parquet(project_paths.interim / "laus" / "laus.parquet")

    assert {"geography", "state_fips", "year", "laus_unemployment_rate"} <= set(frame.columns)
    assert set(frame["geography"]) == {"county", "state"}


def test_childcare_panel_prefers_laus_unemployment_rate(project_paths):
    for ingestor in (
        ingest_atus,
        ingest_ndcp,
        ingest_qcew,
        ingest_acs,
        ingest_head_start,
        ingest_nces_ccd,
        ingest_oews,
        ingest_laus,
    ):
        ingestor(project_paths, sample=True)

    county, state = build_childcare_panels(project_paths)
    los_angeles = county.loc[(county["county_fips"] == "06037") & (county["year"] == 2021)].iloc[0]
    california = state.loc[(state["state_fips"] == "06") & (state["year"] == 2021)].iloc[0]

    assert los_angeles["laus_unemployment_rate"] == 0.074
    assert los_angeles["unemployment_rate"] == 0.074
    assert california["laus_unemployment_rate"] == 0.076
    assert california["unemployment_rate"] == 0.076
