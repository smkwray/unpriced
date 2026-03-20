from __future__ import annotations

from unpriced.features.childcare_panel import build_childcare_panels
from unpriced.ingest.acs import ingest as ingest_acs
from unpriced.ingest.atus import ingest as ingest_atus
from unpriced.ingest.cbp import _normalize_cbp_rows, ingest as ingest_cbp
from unpriced.ingest.head_start import ingest as ingest_head_start
from unpriced.ingest.nces_ccd import ingest as ingest_nces_ccd
from unpriced.ingest.ndcp import ingest as ingest_ndcp
from unpriced.ingest.nes import ingest as ingest_nes
from unpriced.ingest.oews import ingest as ingest_oews
from unpriced.ingest.qcew import ingest as ingest_qcew
from unpriced.storage import read_parquet


def test_normalize_cbp_rows_builds_county_panel():
    rows = [
        ["ESTAB", "EMP", "PAYANN", "LFO", "NAICS2017_LABEL", "state", "county"],
        ["43", "491", "13362", "001", "Child day care services", "01", "003"],
        ["10", "172", "3130", "001", "Child day care services", "01", "001"],
    ]

    frame = _normalize_cbp_rows(rows, 2022, "https://api.census.gov/data/2022/cbp")

    assert list(frame.columns) == [
        "county_fips",
        "state_fips",
        "year",
        "employer_establishments",
        "employer_employment",
        "employer_annual_payroll",
        "legal_form_code",
        "source_url",
    ]
    assert frame.loc[0, "county_fips"] == "01001"
    assert frame.loc[1, "employer_annual_payroll"] == 13362000.0


def test_cbp_sample_ingest_writes_establishment_panel(project_paths):
    ingest_cbp(project_paths, sample=True)
    frame = read_parquet(project_paths.interim / "cbp" / "cbp.parquet")

    assert {"county_fips", "employer_establishments", "employer_employment"} <= set(frame.columns)
    assert frame["employer_establishments"].gt(0).all()


def test_childcare_panel_uses_cbp_and_nes_provider_counts(project_paths):
    for ingestor in (
        ingest_atus,
        ingest_ndcp,
        ingest_qcew,
        ingest_acs,
        ingest_head_start,
        ingest_nces_ccd,
        ingest_oews,
        ingest_cbp,
        ingest_nes,
    ):
        ingestor(project_paths, sample=True)

    county, _ = build_childcare_panels(project_paths)
    los_angeles = county.loc[(county["county_fips"] == "06037") & (county["year"] == 2021)].iloc[0]
    expected_density = (430.0 + 1186.0) / 620000.0 * 1000.0

    assert los_angeles["employer_establishments"] == 430.0
    assert los_angeles["nonemployer_firms"] == 1186.0
    assert los_angeles["total_provider_firms"] == 1616.0
    assert los_angeles["provider_density"] == expected_density
