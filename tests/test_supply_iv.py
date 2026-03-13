from __future__ import annotations

import pandas as pd

from unpaidwork import cli
from unpaidwork.models.supply_iv import build_supply_iv_panel, fit_supply_iv_exposure_design
from unpaidwork.storage import read_json, read_parquet, write_parquet


def _county_panel() -> pd.DataFrame:
    rows = []
    for county_fips, state_fips, exposure_base in (
        ("01001", "01", 0.8),
        ("01003", "01", 0.7),
        ("02001", "02", 0.2),
        ("02003", "02", 0.3),
    ):
        for year, shock in ((2020, 0.0), (2021, 0.2)):
            price = 10000.0 + 1500.0 * exposure_base * shock
            provider_density = 1.0 + 0.30 * exposure_base * shock
            employer_establishments = 80.0 * exposure_base + 5.0
            nonemployer_firms = 80.0 * (1.0 - exposure_base) + 5.0
            rows.append(
                {
                    "county_fips": county_fips,
                    "state_fips": state_fips,
                    "year": year,
                    "annual_price": price,
                    "provider_density": provider_density,
                    "under5_population": 1000.0,
                    "employer_establishments": employer_establishments,
                    "nonemployer_firms": nonemployer_firms,
                }
            )
    return pd.DataFrame(rows)


def _shock_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"state_fips": "01", "year": 2020, "center_infant_ratio": 4.0, "center_toddler_ratio": 7.0, "center_infant_group_size": 8.0, "center_toddler_group_size": 14.0},
            {"state_fips": "01", "year": 2021, "center_infant_ratio": 3.0, "center_toddler_ratio": 6.0, "center_infant_group_size": 6.0, "center_toddler_group_size": 12.0},
            {"state_fips": "02", "year": 2020, "center_infant_ratio": 4.0, "center_toddler_ratio": 7.0, "center_infant_group_size": 8.0, "center_toddler_group_size": 14.0},
            {"state_fips": "02", "year": 2021, "center_infant_ratio": 3.0, "center_toddler_ratio": 6.0, "center_infant_group_size": 6.0, "center_toddler_group_size": 12.0},
        ]
    )


def test_build_supply_iv_panel_constructs_exposure_shock():
    panel = build_supply_iv_panel(_county_panel(), _shock_panel())

    assert not panel.empty
    assert {"reg_shock_ct", "center_exposure_share_pre", "center_labor_intensity_shock"} <= set(panel.columns)
    assert panel["reg_shock_ct"].abs().max() > 0


def test_fit_supply_iv_exposure_design_returns_iv_summary():
    summary, panel = fit_supply_iv_exposure_design(_county_panel(), _shock_panel())

    assert not panel.empty
    assert summary["status"] == "ok"
    assert summary["pilot_scope"] == "multi_state_demo"
    assert summary["first_stage_price"]["beta"] > 0
    assert summary["iv_supply_elasticity_provider_density"] > 0
    assert summary["local_iv_supply_elasticity_provider_density"] > 0
    assert summary["secondary_supply_estimate"]["name"] == "local_iv_supply_elasticity_provider_density"


def test_fit_supply_iv_cli_writes_missing_artifact_without_shock_panel(project_paths):
    write_parquet(_county_panel(), project_paths.processed / "childcare_county_year_price_panel.parquet")

    cli.fit_supply_iv(project_paths)

    artifact = read_json(project_paths.outputs_reports / "childcare_supply_iv.json")
    panel = read_parquet(project_paths.processed / "childcare_supply_iv_panel.parquet")
    assert artifact["status"] == "missing_licensing_shock_panel"
    assert panel.empty


def test_fit_supply_iv_exposure_design_flags_missing_identification():
    shock_panel = _shock_panel().copy()
    shock_panel.loc[shock_panel["year"] == 2021, ["center_infant_ratio", "center_toddler_ratio", "center_infant_group_size", "center_toddler_group_size"]] = [4.0, 7.0, 8.0, 14.0]

    summary, panel = fit_supply_iv_exposure_design(_county_panel(), shock_panel)

    assert not panel.empty
    assert summary["status"] == "insufficient_identification"
    assert summary["shock_state_count"] == 0
