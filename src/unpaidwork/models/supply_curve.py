from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SUPPLY_ELASTICITY = 0.5
MIN_GROUP_ROWS = 5
MIN_SIDE_ROWS = 4
MIN_SIDE_UNIQUE_PRICES = 3


def _coerce_supply_frame(frame: pd.DataFrame) -> pd.DataFrame:
    dataset = frame.copy()
    for column in ("provider_density", "annual_price", "under5_population", "year"):
        if column in dataset.columns:
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    dataset = dataset.dropna(subset=["provider_density", "annual_price"])
    dataset = dataset.loc[
        dataset["provider_density"].gt(0)
        & dataset["annual_price"].gt(0)
    ].copy()
    return dataset


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    ordered = pd.DataFrame({"value": values, "weight": weights}).dropna().sort_values("value").reset_index(drop=True)
    if ordered.empty:
        return float("nan")
    total_weight = float(ordered["weight"].sum())
    if total_weight <= 0:
        return float(ordered["value"].median())
    cumulative = ordered["weight"].cumsum() / total_weight
    return float(ordered.loc[cumulative.ge(0.5).idxmax(), "value"])


def _fit_loglog_slope(frame: pd.DataFrame) -> float:
    if len(frame) < MIN_SIDE_ROWS:
        return float("nan")
    if frame["annual_price"].nunique() < MIN_SIDE_UNIQUE_PRICES or frame["provider_density"].nunique() < MIN_SIDE_UNIQUE_PRICES:
        return float("nan")
    return float(np.polyfit(np.log(frame["annual_price"]), np.log(frame["provider_density"]), deg=1)[0])


def summarize_supply_elasticity(frame: pd.DataFrame) -> dict[str, float | int | bool | str]:
    dataset = _coerce_supply_frame(frame)
    if len(dataset) < 2:
        return {
            "supply_elasticity": DEFAULT_SUPPLY_ELASTICITY,
            "estimation_method": "fallback_default",
            "fallback_used": True,
            "n_obs": int(len(dataset)),
            "pooled_loglog_slope": float("nan"),
            "year_demeaned_loglog_slope": float("nan"),
            "within_state_year_group_count": 0,
            "within_state_year_positive_group_count": 0,
            "within_state_year_positive_group_share": 0.0,
            "within_state_year_weighted_median_positive_slope": float("nan"),
            "within_state_year_weighted_median_all_slope": float("nan"),
        }

    dataset["log_density"] = np.log(dataset["provider_density"])
    dataset["log_price"] = np.log(dataset["annual_price"])
    pooled_slope = float(np.polyfit(dataset["log_price"], dataset["log_density"], deg=1)[0])

    if "year" in dataset.columns and dataset["year"].notna().any():
        year_demeaned = dataset.copy()
        year_demeaned["log_density_dm"] = (
            year_demeaned["log_density"] - year_demeaned.groupby("year")["log_density"].transform("mean")
        )
        year_demeaned["log_price_dm"] = (
            year_demeaned["log_price"] - year_demeaned.groupby("year")["log_price"].transform("mean")
        )
        year_demeaned = year_demeaned.loc[
            year_demeaned["log_density_dm"].abs().add(year_demeaned["log_price_dm"].abs()).gt(0)
        ]
        if len(year_demeaned) >= 2:
            year_demeaned_slope = float(
                np.polyfit(year_demeaned["log_price_dm"], year_demeaned["log_density_dm"], deg=1)[0]
            )
        else:
            year_demeaned_slope = float("nan")
    else:
        year_demeaned_slope = float("nan")

    group_rows: list[dict[str, float | int | str]] = []
    if {"state_fips", "year"}.issubset(dataset.columns):
        for (state_fips, year), group in dataset.groupby(["state_fips", "year"], dropna=True):
            if len(group) < MIN_GROUP_ROWS:
                continue
            if group["log_price"].nunique() < 3 or group["log_density"].nunique() < 3:
                continue
            group_rows.append(
                {
                    "state_fips": str(state_fips),
                    "year": int(year),
                    "group_rows": int(len(group)),
                    "weight": float(
                        pd.to_numeric(
                            group["under5_population"] if "under5_population" in group.columns else pd.Series(1.0, index=group.index),
                            errors="coerce",
                        )
                        .fillna(0.0)
                        .sum()
                    ),
                    "slope": float(np.polyfit(group["log_price"], group["log_density"], deg=1)[0]),
                }
            )
    groups = pd.DataFrame(group_rows)
    if groups.empty:
        positive_groups = groups
        weighted_positive = float("nan")
        weighted_all = float("nan")
    else:
        groups["weight"] = groups["weight"].clip(lower=1.0)
        positive_groups = groups.loc[groups["slope"].gt(0)].copy()
        weighted_positive = (
            _weighted_median(positive_groups["slope"], positive_groups["weight"]) if not positive_groups.empty else float("nan")
        )
        weighted_all = _weighted_median(groups["slope"], groups["weight"])

    if np.isfinite(weighted_positive):
        supply_elasticity = float(weighted_positive)
        estimation_method = "within_state_year_positive_weighted_median"
        fallback_used = False
    elif np.isfinite(year_demeaned_slope) and year_demeaned_slope > 0:
        supply_elasticity = float(year_demeaned_slope)
        estimation_method = "year_demeaned_loglog_slope"
        fallback_used = False
    elif pooled_slope > 0:
        supply_elasticity = float(pooled_slope)
        estimation_method = "pooled_loglog_slope"
        fallback_used = False
    else:
        supply_elasticity = DEFAULT_SUPPLY_ELASTICITY
        estimation_method = "fallback_default"
        fallback_used = True

    return {
        "supply_elasticity": float(supply_elasticity),
        "estimation_method": estimation_method,
        "fallback_used": fallback_used,
        "n_obs": int(len(dataset)),
        "pooled_loglog_slope": pooled_slope,
        "year_demeaned_loglog_slope": year_demeaned_slope,
        "within_state_year_group_count": int(len(groups)),
        "within_state_year_positive_group_count": int(len(positive_groups)),
        "within_state_year_positive_group_share": float(len(positive_groups) / max(len(groups), 1)),
        "within_state_year_weighted_median_positive_slope": weighted_positive,
        "within_state_year_weighted_median_all_slope": weighted_all,
    }


def summarize_piecewise_supply_curve(
    frame: pd.DataFrame,
    baseline_column: str = "state_price_index",
) -> tuple[dict[str, float | int | bool | str], pd.DataFrame]:
    dataset = _coerce_supply_frame(frame)
    if baseline_column in dataset.columns:
        dataset[baseline_column] = pd.to_numeric(dataset[baseline_column], errors="coerce")
    else:
        dataset[baseline_column] = np.nan
    dataset = dataset.dropna(subset=[baseline_column]).copy()
    overall = summarize_supply_elasticity(dataset)
    if dataset.empty or not {"state_fips", "year"}.issubset(dataset.columns):
        summary = {
            "piecewise_method": "state_year_piecewise_isoelastic",
            "n_obs": int(len(dataset)),
            "group_count": 0,
            "supported_below_groups": 0,
            "supported_above_groups": 0,
            "supported_both_sides_groups": 0,
            "pooled_eta_below": float(overall["supply_elasticity"]),
            "pooled_eta_above": float(overall["supply_elasticity"]),
            "fallback_share_any_side": 1.0,
            "fallback_share_below": 1.0,
            "fallback_share_above": 1.0,
            "constant_supply_elasticity": float(overall["supply_elasticity"]),
            "constant_supply_method": overall["estimation_method"],
        }
        return summary, pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "state_price_index",
                "rows_total",
                "rows_below",
                "rows_above",
                "eta_below_raw",
                "eta_above_raw",
                "eta_below",
                "eta_above",
                "fallback_below",
                "fallback_above",
            ]
        )

    group_rows: list[dict[str, float | int | bool | str]] = []
    for (state_fips, year), group in dataset.groupby(["state_fips", "year"], dropna=True):
        baseline_price = float(pd.to_numeric(group[baseline_column], errors="coerce").dropna().iloc[0])
        below = group.loc[group["annual_price"].le(baseline_price)].copy()
        above = group.loc[group["annual_price"].gt(baseline_price)].copy()
        eta_below_raw = _fit_loglog_slope(below)
        eta_above_raw = _fit_loglog_slope(above)
        group_rows.append(
            {
                "state_fips": str(state_fips),
                "year": int(year),
                "state_price_index": baseline_price,
                "rows_total": int(len(group)),
                "rows_below": int(len(below)),
                "rows_above": int(len(above)),
                "weight": max(
                    float(
                        pd.to_numeric(
                            group["under5_population"] if "under5_population" in group.columns else pd.Series(1.0, index=group.index),
                            errors="coerce",
                        )
                        .fillna(0.0)
                        .sum()
                    ),
                    1.0,
                ),
                "eta_below_raw": eta_below_raw,
                "eta_above_raw": eta_above_raw,
            }
        )
    groups = pd.DataFrame(group_rows)
    supported_below = groups["eta_below_raw"].replace([np.inf, -np.inf], np.nan).dropna()
    supported_above = groups["eta_above_raw"].replace([np.inf, -np.inf], np.nan).dropna()
    supported_below_positive = groups.loc[pd.to_numeric(groups["eta_below_raw"], errors="coerce").gt(0)].copy()
    supported_above_positive = groups.loc[pd.to_numeric(groups["eta_above_raw"], errors="coerce").gt(0)].copy()
    pooled_eta_below = (
        _weighted_median(supported_below_positive["eta_below_raw"], supported_below_positive["weight"])
        if not supported_below_positive.empty
        else float(overall["supply_elasticity"])
    )
    pooled_eta_above = (
        _weighted_median(supported_above_positive["eta_above_raw"], supported_above_positive["weight"])
        if not supported_above_positive.empty
        else float(overall["supply_elasticity"])
    )
    groups["eta_below"] = pd.to_numeric(groups["eta_below_raw"], errors="coerce")
    groups["eta_above"] = pd.to_numeric(groups["eta_above_raw"], errors="coerce")
    groups["fallback_below"] = groups["eta_below"].isna() | groups["eta_below"].le(0)
    groups["fallback_above"] = groups["eta_above"].isna() | groups["eta_above"].le(0)
    groups.loc[groups["fallback_below"], "eta_below"] = pooled_eta_below
    groups.loc[groups["fallback_above"], "eta_above"] = pooled_eta_above
    summary = {
        "piecewise_method": "state_year_piecewise_isoelastic",
        "n_obs": int(len(dataset)),
        "group_count": int(len(groups)),
        "supported_below_groups": int(groups["fallback_below"].eq(False).sum()),
        "supported_above_groups": int(groups["fallback_above"].eq(False).sum()),
        "supported_both_sides_groups": int((groups["fallback_below"].eq(False) & groups["fallback_above"].eq(False)).sum()),
        "pooled_eta_below": float(pooled_eta_below),
        "pooled_eta_above": float(pooled_eta_above),
        "fallback_share_any_side": float((groups["fallback_below"] | groups["fallback_above"]).mean()) if len(groups) else 1.0,
        "fallback_share_below": float(groups["fallback_below"].mean()) if len(groups) else 1.0,
        "fallback_share_above": float(groups["fallback_above"].mean()) if len(groups) else 1.0,
        "constant_supply_elasticity": float(overall["supply_elasticity"]),
        "constant_supply_method": overall["estimation_method"],
    }
    return summary, groups


def calibrate_supply_elasticity(frame: pd.DataFrame) -> float:
    return float(summarize_supply_elasticity(frame)["supply_elasticity"])
