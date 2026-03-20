from __future__ import annotations

import pandas as pd

from unpriced.ingest.noaa import (
    _parse_damage,
    normalize_storm_events,
    ingest as ingest_noaa,
)
from unpriced.storage import read_parquet

EXPECTED_COLUMNS = {
    "county_fips",
    "state_fips",
    "year",
    "storm_event_count",
    "storm_property_damage",
    "storm_exposure",
    "precip_event_days",
    "source_url",
}


def test_parse_damage_thousands():
    assert _parse_damage("10.00K") == 10_000.0


def test_parse_damage_millions():
    assert _parse_damage("1.50M") == 1_500_000.0


def test_parse_damage_billions():
    assert _parse_damage("2.00B") == 2_000_000_000.0


def test_parse_damage_zero():
    assert _parse_damage("0.00K") == 0.0
    assert _parse_damage(None) == 0.0
    assert _parse_damage("") == 0.0


def test_normalize_storm_events_filters_county_only():
    raw = pd.DataFrame(
        [
            {
                "STATE_FIPS": "6",
                "CZ_TYPE": "C",
                "CZ_FIPS": "37",
                "EVENT_TYPE": "Thunderstorm Wind",
                "DAMAGE_PROPERTY": "5.00K",
                "BEGIN_YEARMONTH": "202201",
                "BEGIN_DAY": "15",
            },
            {
                "STATE_FIPS": "6",
                "CZ_TYPE": "Z",
                "CZ_FIPS": "100",
                "EVENT_TYPE": "Flood",
                "DAMAGE_PROPERTY": "50.00K",
                "BEGIN_YEARMONTH": "202203",
                "BEGIN_DAY": "10",
            },
            {
                "STATE_FIPS": "6",
                "CZ_TYPE": "C",
                "CZ_FIPS": "37",
                "EVENT_TYPE": "Flash Flood",
                "DAMAGE_PROPERTY": "25.00K",
                "BEGIN_YEARMONTH": "202206",
                "BEGIN_DAY": "01",
            },
        ]
    )
    result = normalize_storm_events(raw, 2022, "https://example.com/storm.csv.gz")

    assert EXPECTED_COLUMNS <= set(result.columns)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["county_fips"] == "06037"
    assert row["state_fips"] == "06"
    assert row["year"] == 2022
    assert row["storm_event_count"] == 2
    assert row["storm_property_damage"] == 30_000.0
    assert row["storm_exposure"] == 1.0
    assert row["precip_event_days"] == 1


def test_normalize_multiple_counties():
    raw = pd.DataFrame(
        [
            {"STATE_FIPS": "48", "CZ_TYPE": "C", "CZ_FIPS": "113", "EVENT_TYPE": "Hail", "DAMAGE_PROPERTY": "10.00K", "BEGIN_YEARMONTH": "202204", "BEGIN_DAY": "05"},
            {"STATE_FIPS": "48", "CZ_TYPE": "C", "CZ_FIPS": "113", "EVENT_TYPE": "Hail", "DAMAGE_PROPERTY": "5.00K", "BEGIN_YEARMONTH": "202205", "BEGIN_DAY": "12"},
            {"STATE_FIPS": "48", "CZ_TYPE": "C", "CZ_FIPS": "201", "EVENT_TYPE": "Flash Flood", "DAMAGE_PROPERTY": "100.00K", "BEGIN_YEARMONTH": "202206", "BEGIN_DAY": "01"},
            {"STATE_FIPS": "48", "CZ_TYPE": "C", "CZ_FIPS": "201", "EVENT_TYPE": "Thunderstorm Wind", "DAMAGE_PROPERTY": "0.00K", "BEGIN_YEARMONTH": "202206", "BEGIN_DAY": "01"},
            {"STATE_FIPS": "48", "CZ_TYPE": "C", "CZ_FIPS": "201", "EVENT_TYPE": "Flash Flood", "DAMAGE_PROPERTY": "50.00K", "BEGIN_YEARMONTH": "202207", "BEGIN_DAY": "20"},
        ]
    )
    result = normalize_storm_events(raw, 2022, "https://example.com/storm.csv.gz")

    assert len(result) == 2
    dallas = result.loc[result["county_fips"] == "48113"].iloc[0]
    houston = result.loc[result["county_fips"] == "48201"].iloc[0]
    assert dallas["storm_event_count"] == 2
    assert houston["storm_event_count"] == 3
    assert houston["storm_exposure"] == 1.0
    assert dallas["storm_exposure"] < 1.0
    assert dallas["precip_event_days"] == 2
    assert houston["precip_event_days"] == 2


def test_normalize_empty_after_zone_filter():
    raw = pd.DataFrame(
        [
            {"STATE_FIPS": "6", "CZ_TYPE": "Z", "CZ_FIPS": "100", "EVENT_TYPE": "Flood", "DAMAGE_PROPERTY": "0.00K", "BEGIN_YEARMONTH": "202201", "BEGIN_DAY": "01"},
        ]
    )
    result = normalize_storm_events(raw, 2022, "https://example.com/storm.csv.gz")
    assert result.empty
    assert EXPECTED_COLUMNS <= set(result.columns)


def test_noaa_sample_ingest_writes_normalized_panel(project_paths):
    ingest_noaa(project_paths, sample=True)
    frame = read_parquet(project_paths.interim / "noaa" / "noaa.parquet")

    assert EXPECTED_COLUMNS <= set(frame.columns)
    assert len(frame) == 6
    assert set(frame["state_fips"]) == {"06", "48", "36"}
    houston = frame.loc[frame["county_fips"] == "48201"].iloc[0]
    assert houston["storm_exposure"] == 1.0
