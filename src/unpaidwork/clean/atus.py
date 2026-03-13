from __future__ import annotations

import numpy as np
import pandas as pd

from unpaidwork.config import ProjectPaths
from unpaidwork.geo.harmonize import add_sensitivity_flag, annualize
from unpaidwork.storage import read_parquet, write_parquet
from unpaidwork.validate import require_columns

WEEKS_PER_YEAR = 52.0


def _weighted_mean(frame: pd.DataFrame, value: str, weight: str) -> float:
    return float(np.average(frame[value], weights=frame[weight]))


def _merge_observed_state_controls(grouped: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    result = grouped.copy()
    result["state_controls_source"] = "atus_fallback"
    result["state_unemployment_source"] = "atus_fallback"

    acs_path = paths.interim / "acs" / "acs.parquet"
    if acs_path.exists():
        acs = read_parquet(acs_path)
        if {"state_fips", "year", "under5_population", "median_income"} <= set(acs.columns):
            weights = pd.to_numeric(
                acs["under6_population"] if "under6_population" in acs.columns else acs["under5_population"],
                errors="coerce",
            ).fillna(0.0)
            acs = acs.assign(_weight=weights)
            state_rows = []
            for (state_fips, year), frame in acs.groupby(["state_fips", "year"], as_index=False):
                weight = frame["_weight"]
                positive = weight.gt(0)
                if not positive.any():
                    continue
                row = {"state_fips": state_fips, "year": year}
                for column in ("median_income", "single_parent_share", "parent_employment_rate"):
                    if column in frame.columns:
                        values = pd.to_numeric(frame[column], errors="coerce")
                        mask = positive & values.notna()
                        if mask.any():
                            row[column] = float(np.average(values[mask], weights=weight[mask]))
                state_rows.append(row)
            if state_rows:
                observed_acs = pd.DataFrame(state_rows).rename(
                    columns={
                        "median_income": "median_income_observed",
                        "single_parent_share": "single_parent_share_observed",
                        "parent_employment_rate": "parent_employment_rate_observed",
                    }
                )
                result = result.merge(observed_acs, on=["state_fips", "year"], how="left")
                observed_control_columns = [
                    column
                    for column in (
                        "median_income_observed",
                        "single_parent_share_observed",
                        "parent_employment_rate_observed",
                    )
                    if column in result.columns
                ]
                for column in ("median_income", "single_parent_share", "parent_employment_rate"):
                    observed = f"{column}_observed"
                    if observed in result.columns:
                        result[column] = pd.to_numeric(result[observed], errors="coerce").fillna(result[column])
                if observed_control_columns:
                    observed_controls_mask = result[observed_control_columns].notna().all(axis=1)
                    result.loc[observed_controls_mask, "state_controls_source"] = "acs_observed"

    laus_path = paths.interim / "laus" / "laus.parquet"
    if laus_path.exists():
        laus = read_parquet(laus_path)
        if {"geography", "state_fips", "year", "laus_unemployment_rate"} <= set(laus.columns):
            state_laus = laus.loc[laus["geography"].eq("state"), ["state_fips", "year", "laus_unemployment_rate"]]
            result = result.merge(state_laus, on=["state_fips", "year"], how="left")
            observed_unemployment_mask = pd.to_numeric(result["laus_unemployment_rate"], errors="coerce").notna()
            result["unemployment_rate"] = pd.to_numeric(
                result["laus_unemployment_rate"], errors="coerce"
            ).fillna(result["unemployment_rate"])
            result.loc[observed_unemployment_mask, "state_unemployment_source"] = "laus_observed"

    cdc_path = paths.interim / "cdc_wonder" / "cdc_wonder.parquet"
    if cdc_path.exists():
        cdc = read_parquet(cdc_path)
        if {"state_fips", "year", "births", "births_suppressed"} <= set(cdc.columns):
            cdc = cdc.loc[:, [column for column in ("state_fips", "year", "births", "births_suppressed", "suppression_note") if column in cdc.columns]].copy()
            cdc = cdc.rename(
                columns={
                    "births": "births_observed",
                    "births_suppressed": "births_suppressed_observed",
                    "suppression_note": "births_suppression_note",
                }
            )
            result = result.merge(cdc, on=["state_fips", "year"], how="left")
            observed_mask = pd.to_numeric(result["births_observed"], errors="coerce").notna()
            result["births"] = pd.to_numeric(result["births_observed"], errors="coerce").fillna(result["births"])
            result["births_observed_available"] = observed_mask
            if "births_suppressed_observed" in result.columns:
                result["births_suppressed_observed"] = result["births_suppressed_observed"].fillna(False).astype(bool)
            result.loc[observed_mask, "births_value_source"] = "cdc_wonder_observed"
            suppressed_mask = (~observed_mask) & result["births_suppressed_observed"].astype(bool)
            result.loc[suppressed_mask, "births_value_source"] = "cdc_wonder_suppressed_fallback"
            unmatched_mask = (
                (~observed_mask)
                & (~result["births_suppressed_observed"].astype(bool))
                & result["state_fips"].astype(str).eq("00")
            )
            result.loc[unmatched_mask, "births_value_source"] = "atus_unmatched_state_fallback"

    if "births_observed_available" not in result.columns:
        result["births_observed_available"] = False
    if "births_suppressed_observed" not in result.columns:
        result["births_suppressed_observed"] = False
    if "births_suppression_note" not in result.columns:
        result["births_suppression_note"] = pd.NA
    if "births_value_source" not in result.columns:
        result["births_value_source"] = "atus_reported"

    result["births_observed_available"] = result["births_observed_available"].fillna(False).astype(bool)
    result["births_suppressed_observed"] = result["births_suppressed_observed"].fillna(False).astype(bool)
    result["births_value_source"] = result["births_value_source"].fillna("atus_reported")
    result["state_controls_source"] = result["state_controls_source"].fillna("atus_fallback")
    result["state_unemployment_source"] = result["state_unemployment_source"].fillna("atus_fallback")

    drop_columns = [
        column
        for column in (
            "median_income_observed",
            "single_parent_share_observed",
            "parent_employment_rate_observed",
            "laus_unemployment_rate",
            "births_observed",
        )
        if column in result.columns
    ]
    return result.drop(columns=drop_columns)


def build_state_year_panel(paths: ProjectPaths) -> pd.DataFrame:
    frame = read_parquet(paths.interim / "atus" / "atus.parquet")
    require_columns(
        frame,
        [
            "state_fips",
            "year",
            "subgroup",
            "childcare_hours",
            "weight",
            "parent_employment_rate",
            "single_parent_share",
            "median_income",
            "unemployment_rate",
            "births",
        ],
        "ATUS",
    )
    frame = annualize(frame)
    grouped = (
        frame.groupby(["state_fips", "year", "subgroup"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    # The normalized ATUS respondent file stores weekly-equivalent childcare hours.
                    # Annualize here so replacement-cost outputs are on the same annual scale as NDCP prices.
                    "unpaid_childcare_hours": _weighted_mean(g, "childcare_hours", "weight") * WEEKS_PER_YEAR,
                    "parent_employment_rate": _weighted_mean(
                        g, "parent_employment_rate", "weight"
                    ),
                    "single_parent_share": _weighted_mean(g, "single_parent_share", "weight"),
                    "median_income": _weighted_mean(g, "median_income", "weight"),
                    "unemployment_rate": _weighted_mean(g, "unemployment_rate", "weight"),
                    "births": _weighted_mean(g, "births", "weight"),
                    "atus_weight_sum": float(g["weight"].sum()),
                }
            )
        )
        .reset_index(drop=True)
    )
    grouped = _merge_observed_state_controls(grouped, paths)
    grouped = add_sensitivity_flag(grouped)
    write_parquet(grouped, paths.processed / "atus_state_year_childcare.parquet")
    return grouped
