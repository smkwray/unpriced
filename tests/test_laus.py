from __future__ import annotations

from unpriced.features.childcare_panel import build_childcare_panels
from unpriced.errors import SourceAccessError
from unpriced.ingest.acs import ingest as ingest_acs
from unpriced.ingest.atus import ingest as ingest_atus
from unpriced.ingest.head_start import ingest as ingest_head_start
from unpriced.ingest.laus import _normalize_laus_payload, ingest as ingest_laus, ingest_year_range
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


def test_laus_ingest_year_range_falls_back_to_api_when_flat_files_blocked(project_paths, monkeypatch):
    ingest_acs(project_paths, sample=True)

    def _fail_download(url: str) -> str:
        raise SourceAccessError(f"blocked {url}")

    def _fake_fetch_range(series_ids: list[str], start_year: int, end_year: int) -> dict[str, object]:
        assert start_year == 2021
        assert end_year == 2022
        return {
            "Results": {
                "series": [
                    {
                        "seriesID": "LAUCN060370000000003",
                        "data": [
                            {"year": "2021", "period": "M13", "value": "7.4"},
                            {"year": "2022", "period": "M13", "value": "5.0"},
                        ],
                    },
                    {
                        "seriesID": "LAUST060000000000003",
                        "data": [
                            {"year": "2021", "period": "M13", "value": "7.6"},
                            {"year": "2022", "period": "M13", "value": "4.3"},
                        ],
                    },
                ]
            }
        }

    monkeypatch.setattr("unpriced.ingest.laus._download_laus_text", _fail_download)
    monkeypatch.setattr("unpriced.ingest.laus._fetch_laus_series_range", _fake_fetch_range)

    result = ingest_year_range(project_paths, 2021, 2022, refresh=True)

    assert result.skipped is False
    frame = read_parquet(project_paths.interim / "laus" / "laus.parquet")
    assert set(frame["year"]) == {2021, 2022}
    assert set(frame["geography"]) == {"county", "state"}
    county_2022 = frame.loc[(frame["geography"] == "county") & (frame["year"] == 2022)].iloc[0]
    assert county_2022["laus_unemployment_rate"] == 0.05
