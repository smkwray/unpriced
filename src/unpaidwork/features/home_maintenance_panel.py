from __future__ import annotations

import pandas as pd

from unpaidwork.assumptions import home_maintenance_assumptions
from unpaidwork.clean.ahs import build_job_panel
from unpaidwork.config import ProjectPaths
from unpaidwork.errors import SourceAccessError
from unpaidwork.geo.crosswalks import load_cbsa_county_crosswalk
from unpaidwork.logging import get_logger
from unpaidwork.storage import read_parquet, write_parquet

LOGGER = get_logger()


def _ensure_storm_columns(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    for column in ("precip_event_days", "storm_event_count", "storm_property_damage"):
        if column not in result.columns:
            result[column] = pd.NA
    return result


def _fallback_unemployment(panel: pd.DataFrame, assumptions: dict[str, float]) -> pd.DataFrame:
    panel["cbsa_unemployment_rate"] = (
        float(assumptions["cbsa_unemployment_base"])
        + panel["storm_exposure"] * float(assumptions["cbsa_unemployment_storm_slope"])
    )
    panel["cbsa_unemployment_year"] = pd.NA
    return panel


def _choose_laus_year(target_year: int, available_years: list[int]) -> int:
    eligible = [year for year in available_years if year <= target_year]
    if eligible:
        return max(eligible)
    return max(available_years)


def _attach_laus_controls(panel: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    assumptions = home_maintenance_assumptions(paths)
    laus_path = paths.interim / "laus" / "laus.parquet"
    if not laus_path.exists():
        return _fallback_unemployment(panel, assumptions)

    laus = read_parquet(laus_path)
    laus = laus.loc[laus["geography"].eq("county")].copy()
    if laus.empty:
        return _fallback_unemployment(panel, assumptions)

    try:
        crosswalk = load_cbsa_county_crosswalk(paths)
    except SourceAccessError:
        return _fallback_unemployment(panel, assumptions)

    laus["county_fips"] = laus["county_fips"].astype(str).str.zfill(5)
    laus["year"] = pd.to_numeric(laus["year"], errors="coerce").astype("Int64")
    laus["laus_unemployment_rate"] = pd.to_numeric(laus["laus_unemployment_rate"], errors="coerce")
    laus["laus_unemployed"] = pd.to_numeric(laus["laus_unemployed"], errors="coerce")
    laus["laus_labor_force"] = pd.to_numeric(laus["laus_labor_force"], errors="coerce")
    laus = laus.dropna(subset=["county_fips", "year"]).copy()
    available_years = sorted(laus["year"].dropna().astype(int).unique().tolist())
    if not available_years:
        return _fallback_unemployment(panel, assumptions)

    year_map = pd.DataFrame(
        {
            "year": sorted(pd.to_numeric(panel["year"], errors="coerce").dropna().astype(int).unique().tolist())
        }
    )
    year_map["cbsa_unemployment_year"] = year_map["year"].map(
        lambda value: _choose_laus_year(int(value), available_years)
    )

    cbsa_counties = crosswalk.loc[:, ["cbsa_code", "county_fips"]].drop_duplicates()
    cbsa_laus = cbsa_counties.merge(laus, on="county_fips", how="inner")
    cbsa_laus = cbsa_laus.merge(
        year_map.rename(columns={"year": "panel_year"}),
        left_on="year",
        right_on="cbsa_unemployment_year",
        how="inner",
    )

    aggregated = (
        cbsa_laus.groupby(["cbsa_code", "panel_year", "cbsa_unemployment_year"], as_index=False)
        .agg(
            laus_unemployed=("laus_unemployed", "sum"),
            laus_labor_force=("laus_labor_force", "sum"),
            county_mean_unemployment=("laus_unemployment_rate", "mean"),
        )
        .rename(columns={"panel_year": "year"})
    )
    aggregated["cbsa_unemployment_rate"] = aggregated["laus_unemployed"] / aggregated["laus_labor_force"]
    aggregated["cbsa_unemployment_rate"] = aggregated["cbsa_unemployment_rate"].where(
        aggregated["laus_labor_force"].gt(0), aggregated["county_mean_unemployment"]
    )

    national = (
        laus.groupby("year", as_index=False)
        .agg(
            laus_unemployed=("laus_unemployed", "sum"),
            laus_labor_force=("laus_labor_force", "sum"),
            county_mean_unemployment=("laus_unemployment_rate", "mean"),
        )
        .rename(columns={"year": "cbsa_unemployment_year"})
    )
    national["fallback_unemployment_rate"] = national["laus_unemployed"] / national["laus_labor_force"]
    national["fallback_unemployment_rate"] = national["fallback_unemployment_rate"].where(
        national["laus_labor_force"].gt(0), national["county_mean_unemployment"]
    )

    merged = panel.merge(year_map, on="year", how="left")
    merged = merged.merge(
        aggregated.loc[:, ["cbsa_code", "year", "cbsa_unemployment_rate"]],
        on=["cbsa_code", "year"],
        how="left",
    )
    merged = merged.merge(
        national.loc[:, ["cbsa_unemployment_year", "fallback_unemployment_rate"]],
        on="cbsa_unemployment_year",
        how="left",
    )
    merged["cbsa_unemployment_rate"] = merged["cbsa_unemployment_rate"].fillna(
        merged["fallback_unemployment_rate"]
    )
    merged = merged.drop(columns=["fallback_unemployment_rate"])
    return merged


def _national_noaa_averages(noaa: pd.DataFrame) -> dict[str, float]:
    """Return national county-mean storm measures for use as a fallback."""
    return {
        "storm_event_count": float(noaa["storm_event_count"].mean()),
        "storm_property_damage": float(noaa["storm_property_damage"].mean()),
        "precip_event_days": float(noaa["precip_event_days"].mean()),
        "storm_exposure": float(noaa["storm_event_count"].mean() / noaa["storm_event_count"].max())
        if noaa["storm_event_count"].max() > 0
        else 0.0,
    }


# AHS uses these synthetic CBSA codes for observations whose metro area is
# not identified.  They have no county geography, so county→CBSA storm joins
# cannot reach them.
AHS_NONGEOGRAPHIC_CBSA = frozenset({
    "99998",  # CBSA not reported / withheld for confidentiality
    "99999",  # Non-metropolitan area
})


def _attach_noaa_storm(panel: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    """Replace the AHS-embedded storm_exposure placeholder with observed NOAA county→CBSA aggregates.

    Rows whose ``cbsa_code`` is in ``AHS_NONGEOGRAPHIC_CBSA`` cannot be joined
    to county-level NOAA data.  For these rows the function fills weather
    columns with national county-mean values and labels them explicitly via
    ``noaa_match_status``.
    """
    panel = _ensure_storm_columns(panel)
    panel["noaa_match_status"] = "pending"
    noaa_path = paths.interim / "noaa" / "noaa.parquet"
    if not noaa_path.exists():
        LOGGER.info("NOAA parquet not found; keeping AHS storm_exposure column as-is")
        panel["noaa_match_status"] = "noaa_unavailable"
        return panel

    noaa = read_parquet(noaa_path)
    if noaa.empty:
        panel["noaa_match_status"] = "noaa_empty"
        return panel

    try:
        crosswalk = load_cbsa_county_crosswalk(paths)
    except SourceAccessError:
        LOGGER.info("CBSA crosswalk unavailable; keeping AHS storm_exposure column as-is")
        panel["noaa_match_status"] = "crosswalk_unavailable"
        return panel

    noaa["county_fips"] = noaa["county_fips"].astype(str).str.zfill(5)
    noaa["year"] = pd.to_numeric(noaa["year"], errors="coerce").astype("Int64")
    noaa["storm_event_count"] = pd.to_numeric(noaa["storm_event_count"], errors="coerce")
    noaa["storm_property_damage"] = pd.to_numeric(noaa["storm_property_damage"], errors="coerce")
    noaa["precip_event_days"] = pd.to_numeric(noaa["precip_event_days"], errors="coerce")

    national_avg = _national_noaa_averages(noaa)

    cbsa_counties = crosswalk.loc[:, ["cbsa_code", "county_fips"]].drop_duplicates()
    cbsa_noaa = cbsa_counties.merge(noaa, on="county_fips", how="inner")

    available_years = sorted(cbsa_noaa["year"].dropna().astype(int).unique().tolist())
    if not available_years:
        panel["noaa_match_status"] = "no_noaa_years"
        return panel
    panel_years = sorted(pd.to_numeric(panel["year"], errors="coerce").dropna().astype(int).unique().tolist())
    year_map = pd.DataFrame({"year": panel_years})
    year_map["noaa_year"] = year_map["year"].map(
        lambda value: _choose_laus_year(int(value), available_years)
    )

    cbsa_noaa = cbsa_noaa.merge(
        year_map.rename(columns={"year": "panel_year"}),
        left_on="year",
        right_on="noaa_year",
        how="inner",
    )

    aggregated = (
        cbsa_noaa.groupby(["cbsa_code", "panel_year"], as_index=False)
        .agg(
            noaa_storm_event_count=("storm_event_count", "sum"),
            noaa_storm_property_damage=("storm_property_damage", "sum"),
            noaa_precip_event_days=("precip_event_days", "sum"),
        )
        .rename(columns={"panel_year": "year"})
    )

    max_events = aggregated["noaa_storm_event_count"].max()
    aggregated["noaa_storm_exposure"] = (
        aggregated["noaa_storm_event_count"] / max_events if max_events > 0 else 0.0
    )

    merged = panel.merge(aggregated, on=["cbsa_code", "year"], how="left")

    # --- Classify and fill each row ---
    has_noaa = merged["noaa_storm_exposure"].notna()
    is_nongeographic = merged["cbsa_code"].isin(AHS_NONGEOGRAPHIC_CBSA)

    # 1. Rows with a direct CBSA→county NOAA match.
    merged.loc[has_noaa, "storm_exposure"] = merged.loc[has_noaa, "noaa_storm_exposure"]
    merged.loc[has_noaa, "precip_event_days"] = merged.loc[has_noaa, "noaa_precip_event_days"]
    merged.loc[has_noaa, "storm_event_count"] = merged.loc[has_noaa, "noaa_storm_event_count"]
    merged.loc[has_noaa, "storm_property_damage"] = merged.loc[has_noaa, "noaa_storm_property_damage"]
    merged.loc[has_noaa, "noaa_match_status"] = "observed"

    # 2. Non-geographic AHS rows (99998/99999): fill with national county-mean.
    national_fill = is_nongeographic & ~has_noaa
    merged.loc[national_fill, "storm_exposure"] = national_avg["storm_exposure"]
    merged.loc[national_fill, "precip_event_days"] = national_avg["precip_event_days"]
    merged.loc[national_fill, "storm_event_count"] = national_avg["storm_event_count"]
    merged.loc[national_fill, "storm_property_damage"] = national_avg["storm_property_damage"]
    for code in AHS_NONGEOGRAPHIC_CBSA:
        mask = merged["cbsa_code"].eq(code) & national_fill
        if code == "99999":
            merged.loc[mask, "noaa_match_status"] = "national_avg_nonmetro"
        else:
            merged.loc[mask, "noaa_match_status"] = "national_avg_not_reported"

    # 3. Any remaining unmatched rows (CBSA in crosswalk but no county NOAA data).
    still_pending = merged["noaa_match_status"].eq("pending")
    merged.loc[still_pending, "noaa_match_status"] = "cbsa_no_county_match"

    # --- Diagnostics ---
    status_counts = merged["noaa_match_status"].value_counts().to_dict()
    LOGGER.info(
        "NOAA storm merge: %s",
        ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items())),
    )

    merged = merged.drop(
        columns=[
            "noaa_storm_exposure",
            "noaa_storm_event_count",
            "noaa_storm_property_damage",
            "noaa_precip_event_days",
        ],
        errors="ignore",
    )
    return merged


def build_home_maintenance_panel(paths: ProjectPaths) -> pd.DataFrame:
    jobs = build_job_panel(paths)
    price_bucket = "job_group" if "job_group" in jobs.columns else "job_type"
    outsourced_reference = (
        jobs.loc[jobs["job_diy"].eq(0)]
        .groupby(price_bucket, as_index=False)["job_cost"]
        .mean()
        .rename(columns={"job_cost": "predicted_job_cost"})
    )
    panel = jobs.merge(outsourced_reference, on=price_bucket, how="left")
    panel["predicted_job_cost"] = panel["predicted_job_cost"].fillna(panel["job_cost"])
    panel = _attach_noaa_storm(panel, paths)
    panel = _attach_laus_controls(panel, paths)
    write_parquet(panel, paths.processed / "home_maintenance_panel.parquet")
    return panel
