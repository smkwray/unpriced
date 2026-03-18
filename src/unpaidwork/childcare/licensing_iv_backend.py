from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from unpaidwork.models.supply_iv import fit_supply_iv_exposure_design

TREATMENT_TIMING_COLUMNS = [
    "state_fips",
    "treatment_start_year",
    "ever_treated",
    "peak_abs_stringency_shock",
    "treated_state_year_count",
    "state_year_count",
    "stringency_index_column",
    "rule_row_count",
    "rule_missing_count",
    "rule_observed_count",
]

EVENT_STUDY_COLUMNS = [
    "outcome",
    "event_time",
    "estimate",
    "treated_mean",
    "control_mean",
    "treated_reference_mean",
    "control_reference_mean",
    "n_treated_state_year",
    "n_control_state_year",
    "n_treated_states",
    "included_year_count",
    "reference_period",
    "status",
]

IV_RESULTS_COLUMNS = [
    "outcome",
    "estimate",
    "first_stage_beta",
    "first_stage_f_stat",
    "status",
    "first_stage_strength_tier",
    "usable_for_headline",
    "treated_state_count_sufficiency_flag",
    "cluster_count_warning_flag",
    "recommended_use_tier",
    "design",
    "pilot_scope",
    "shock_state_count",
    "treated_state_count",
    "n_obs",
    "n_states",
    "year_min",
    "year_max",
]

FIRST_STAGE_COLUMNS = [
    "status",
    "design",
    "pilot_scope",
    "first_stage_strength_flag",
    "first_stage_strength_tier",
    "usable_for_headline",
    "treated_state_count_sufficiency_flag",
    "cluster_count_warning_flag",
    "recommended_use_tier",
    "first_stage_beta",
    "first_stage_se_cluster_state",
    "first_stage_t_cluster_state",
    "first_stage_f_stat",
    "first_stage_within_r2",
    "first_stage_n_obs",
    "first_stage_n_clusters",
    "reduced_form_provider_beta",
    "reduced_form_provider_t_cluster_state",
    "reduced_form_provider_within_r2",
    "reduced_form_employer_beta",
    "reduced_form_employer_t_cluster_state",
    "reduced_form_employer_within_r2",
    "shock_state_count",
    "treated_state_count",
    "n_obs",
    "n_states",
]

LEAVE_ONE_OUT_COLUMNS = [
    "omitted_state_fips",
    "status",
    "n_obs",
    "n_states",
    "shock_state_count",
    "first_stage_f_stat",
    "iv_supply_elasticity_provider_density",
    "iv_supply_elasticity_employer_establishment_density",
    "delta_provider_vs_baseline",
    "delta_employer_vs_baseline",
]

IV_USABILITY_COLUMNS = [
    "outcome",
    "status",
    "estimate_finite",
    "first_stage_f_stat",
    "first_stage_strength_tier",
    "treated_state_count",
    "treated_state_count_sufficiency_flag",
    "first_stage_n_clusters",
    "cluster_count_warning_flag",
    "usable_for_headline",
    "recommended_use_tier",
]


@dataclass(frozen=True)
class _LicensingCoreInputs:
    stringency_panel: pd.DataFrame
    harmonized_summary: pd.DataFrame
    shock_panel: pd.DataFrame
    stringency_column: str


def _first_stage_strength_tier(f_stat: float | None) -> str:
    if f_stat is None or not np.isfinite(f_stat):
        return "unknown"
    if float(f_stat) >= 10.0:
        return "strong"
    if float(f_stat) >= 5.0:
        return "moderate"
    return "weak"


def _treated_state_count_sufficiency_flag(treated_state_count: int) -> str:
    if treated_state_count >= 5:
        return "sufficient"
    if treated_state_count >= 3:
        return "limited"
    return "sparse"


def _cluster_count_warning_flag(cluster_count: int) -> bool:
    return int(cluster_count) < 30


def _recommended_use_tier(
    *,
    status: str,
    estimate_finite: bool,
    first_stage_strength_tier: str,
    treated_state_count_sufficiency_flag: str,
    cluster_count_warning_flag: bool,
) -> str:
    if status != "ok" or not estimate_finite:
        return "diagnostics_only"
    if (
        first_stage_strength_tier == "strong"
        and treated_state_count_sufficiency_flag == "sufficient"
        and not cluster_count_warning_flag
    ):
        return "headline"
    if (
        first_stage_strength_tier in {"strong", "moderate"}
        and treated_state_count_sufficiency_flag in {"sufficient", "limited"}
    ):
        return "appendix"
    return "diagnostics_only"


def _normalize_state_year(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["state_fips"] = working["state_fips"].astype(str).str.zfill(2)
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype("Int64")
    working = working.dropna(subset=["year"]).copy()
    working["year"] = working["year"].astype(int)
    return working


def _first_existing(columns: Iterable[str], frame_columns: set[str]) -> str | None:
    for column in columns:
        if column in frame_columns:
            return column
    return None


def _resolve_stringency_column(
    stringency_index: pd.DataFrame,
    index_column: str | None,
) -> str:
    if index_column is not None:
        if index_column not in stringency_index.columns:
            raise KeyError(f"Requested stringency index column not found: {index_column}")
        return index_column
    selected = _first_existing(
        (
            "stringency_equal_weight_index",
            "licensing_stringency_equal_weight",
            "stringency_equal_weight",
            "stringency_pca_like_index",
            "equal_weight_index",
            "licensing_stringency_index",
            "stringency_index",
        ),
        set(stringency_index.columns),
    )
    if selected is None:
        raise KeyError(
            "Stringency frame is missing a recognized index column; "
            "expected one of licensing_stringency_equal_weight, stringency_equal_weight, "
            "equal_weight_index, licensing_stringency_index, stringency_index"
        )
    return selected


def _build_harmonized_summary(harmonized_rules: pd.DataFrame | None) -> pd.DataFrame:
    if harmonized_rules is None or harmonized_rules.empty:
        return pd.DataFrame(columns=["state_fips", "year", "rule_row_count", "rule_missing_count", "rule_observed_count"])
    required = {"state_fips", "year"}
    missing = sorted(required - set(harmonized_rules.columns))
    if missing:
        raise KeyError(f"Harmonized rules frame missing required columns: {', '.join(missing)}")
    working = _normalize_state_year(harmonized_rules)
    missing_flag_column = _first_existing(
        (
            "licensing_rule_missing_original",
            "rule_missing",
            "is_missing",
            "missing_flag",
            "carry_forward_missing",
        ),
        set(working.columns),
    )
    if missing_flag_column is None:
        working["_rule_missing"] = False
    else:
        working["_rule_missing"] = pd.to_numeric(working[missing_flag_column], errors="coerce").fillna(0.0).gt(0)
    grouped = (
        working.groupby(["state_fips", "year"], as_index=False)
        .agg(
            rule_row_count=("state_fips", "size"),
            rule_missing_count=("_rule_missing", "sum"),
        )
    )
    grouped["rule_missing_count"] = pd.to_numeric(grouped["rule_missing_count"], errors="coerce").fillna(0).astype(int)
    grouped["rule_row_count"] = grouped["rule_row_count"].astype(int)
    grouped["rule_observed_count"] = (grouped["rule_row_count"] - grouped["rule_missing_count"]).clip(lower=0)
    return grouped.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def _build_core_inputs(
    harmonized_rules: pd.DataFrame | None,
    stringency_index: pd.DataFrame,
    index_column: str | None,
) -> _LicensingCoreInputs:
    required = {"state_fips", "year"}
    missing = sorted(required - set(stringency_index.columns))
    if missing:
        raise KeyError(f"Stringency frame missing required columns: {', '.join(missing)}")
    stringency_column = _resolve_stringency_column(stringency_index, index_column=index_column)
    working = _normalize_state_year(stringency_index)
    working[stringency_column] = pd.to_numeric(working[stringency_column], errors="coerce")
    working = working.dropna(subset=[stringency_column]).copy()
    if working.empty:
        empty = pd.DataFrame(columns=["state_fips", "year", "stringency_index", "stringency_shock"])
        harmonized_summary = _build_harmonized_summary(harmonized_rules)
        return _LicensingCoreInputs(
            stringency_panel=empty,
            harmonized_summary=harmonized_summary,
            shock_panel=empty,
            stringency_column=stringency_column,
        )
    working = working.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)
    baseline = working.groupby("state_fips")[stringency_column].transform("first")
    working["stringency_index"] = working[stringency_column]
    working["stringency_shock"] = working["stringency_index"] - baseline

    harmonized_summary = _build_harmonized_summary(harmonized_rules)
    if not harmonized_summary.empty:
        working = working.merge(
            harmonized_summary,
            on=["state_fips", "year"],
            how="left",
        )
    for column in ("rule_row_count", "rule_missing_count", "rule_observed_count"):
        if column not in working.columns:
            working[column] = 0
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0).astype(int)
    shock_panel = working[
        [
            "state_fips",
            "year",
            "stringency_index",
            "stringency_shock",
        ]
    ].rename(
        columns={
            "stringency_index": "center_labor_intensity_index",
            "stringency_shock": "center_labor_intensity_shock",
        }
    )
    return _LicensingCoreInputs(
        stringency_panel=working.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True),
        harmonized_summary=harmonized_summary,
        shock_panel=shock_panel.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True),
        stringency_column=stringency_column,
    )


def _safe_group_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _numeric_series(frame: pd.DataFrame, column: str, default: float = float("nan")) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _prepare_state_year_outcomes(county_panel: pd.DataFrame) -> pd.DataFrame:
    required = {"state_fips", "year", "annual_price", "provider_density"}
    missing = sorted(required - set(county_panel.columns))
    if missing:
        raise KeyError(f"County panel missing required columns: {', '.join(missing)}")
    working = _normalize_state_year(county_panel)
    working["annual_price"] = pd.to_numeric(working["annual_price"], errors="coerce")
    working["provider_density"] = pd.to_numeric(working["provider_density"], errors="coerce")
    working["employer_establishments"] = _numeric_series(working, "employer_establishments")
    weights = _numeric_series(working, "under5_population", default=1.0).fillna(1.0).clip(lower=0.0)
    working["_weight"] = weights
    working["log_annual_price"] = np.log(working["annual_price"].where(working["annual_price"].gt(0)))
    working["log_provider_density"] = np.log(working["provider_density"].where(working["provider_density"].gt(0)))
    under5_population = pd.to_numeric(_numeric_series(working, "under5_population"), errors="coerce").replace(
        {0: np.nan}
    )
    employer_density = pd.to_numeric(working["employer_establishments"], errors="coerce").div(
        under5_population
    ).mul(1000.0)
    working["log_employer_establishment_density"] = np.log(employer_density.where(employer_density.gt(0)))

    def weighted_mean(group: pd.DataFrame, column: str) -> float:
        values = pd.to_numeric(group[column], errors="coerce")
        mask = values.notna() & group["_weight"].gt(0)
        if not mask.any():
            return float("nan")
        return float(np.average(values.loc[mask], weights=group.loc[mask, "_weight"]))

    grouped_rows: list[dict[str, Any]] = []
    for (state_fips, year), group in working.groupby(["state_fips", "year"], sort=True):
        grouped_rows.append(
            {
                "state_fips": str(state_fips),
                "year": int(year),
                "log_annual_price": weighted_mean(group, "log_annual_price"),
                "log_provider_density": weighted_mean(group, "log_provider_density"),
                "log_employer_establishment_density": weighted_mean(
                    group, "log_employer_establishment_density"
                ),
            }
        )
    return pd.DataFrame(grouped_rows)


def _normalize_county_panel_for_supply_iv(county_panel: pd.DataFrame) -> pd.DataFrame:
    required = {"state_fips", "county_fips", "year", "annual_price", "provider_density"}
    missing = sorted(required - set(county_panel.columns))
    if missing:
        raise KeyError(f"County panel missing required columns for IV backend: {', '.join(missing)}")
    working = _normalize_state_year(county_panel)
    working["county_fips"] = working["county_fips"].astype(str).str.zfill(5)
    working["annual_price"] = _numeric_series(working, "annual_price")
    working["provider_density"] = _numeric_series(working, "provider_density")
    working["under5_population"] = _numeric_series(working, "under5_population", default=1.0).fillna(1.0)
    working.loc[working["under5_population"].le(0), "under5_population"] = 1.0
    working["employer_establishments"] = _numeric_series(
        working, "employer_establishments", default=0.0
    ).fillna(0.0)
    working["nonemployer_firms"] = _numeric_series(working, "nonemployer_firms", default=0.0).fillna(0.0)
    return working


def _fit_iv_summary(
    county_panel: pd.DataFrame,
    shock_panel: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    if shock_panel.empty:
        return (
            {
                "status": "missing_stringency_support",
                "design": "county_fe_state_year_fe_exposure_shock",
                "note": "No non-missing stringency panel rows were available.",
            },
            pd.DataFrame(),
        )
    summary, panel = fit_supply_iv_exposure_design(
        _normalize_county_panel_for_supply_iv(county_panel),
        shock_panel,
    )
    return summary, panel


def build_licensing_treatment_timing(
    harmonized_rules: pd.DataFrame | None,
    stringency_index: pd.DataFrame,
    *,
    index_column: str | None = None,
    shock_threshold: float = 1e-9,
) -> pd.DataFrame:
    core = _build_core_inputs(harmonized_rules, stringency_index, index_column=index_column)
    if core.stringency_panel.empty:
        return pd.DataFrame(columns=TREATMENT_TIMING_COLUMNS)
    panel = core.stringency_panel.copy()
    panel["abs_stringency_shock"] = pd.to_numeric(panel["stringency_shock"], errors="coerce").abs()
    panel["treated_state_year"] = panel["abs_stringency_shock"].gt(shock_threshold)
    start_year = (
        panel.loc[panel["treated_state_year"]]
        .groupby("state_fips", as_index=False)["year"]
        .min()
        .rename(columns={"year": "treatment_start_year"})
    )
    grouped = (
        panel.groupby("state_fips", as_index=False)
        .agg(
            peak_abs_stringency_shock=("abs_stringency_shock", "max"),
            treated_state_year_count=("treated_state_year", "sum"),
            state_year_count=("year", "size"),
            rule_row_count=("rule_row_count", "sum"),
            rule_missing_count=("rule_missing_count", "sum"),
            rule_observed_count=("rule_observed_count", "sum"),
        )
    )
    grouped = grouped.merge(start_year, on="state_fips", how="left")
    grouped["ever_treated"] = grouped["treated_state_year_count"].gt(0)
    grouped["stringency_index_column"] = core.stringency_column
    grouped["treatment_start_year"] = pd.to_numeric(grouped["treatment_start_year"], errors="coerce").astype("Int64")
    result = grouped[TREATMENT_TIMING_COLUMNS].sort_values(["state_fips"], kind="stable").reset_index(drop=True)
    return result


def build_licensing_event_study_results(
    harmonized_rules: pd.DataFrame | None,
    stringency_index: pd.DataFrame,
    county_panel: pd.DataFrame,
    state_panel: pd.DataFrame | None = None,
    *,
    index_column: str | None = None,
    event_window: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3),
    reference_period: int = -1,
) -> pd.DataFrame:
    del state_panel  # State panel is optional context; core event-study uses county outcomes + state-year treatment.
    timing = build_licensing_treatment_timing(
        harmonized_rules,
        stringency_index,
        index_column=index_column,
    )
    if timing.empty:
        return pd.DataFrame(columns=EVENT_STUDY_COLUMNS)
    outcomes = _prepare_state_year_outcomes(county_panel)
    merged = outcomes.merge(
        timing[["state_fips", "treatment_start_year", "ever_treated"]],
        on="state_fips",
        how="left",
    )
    merged["ever_treated"] = merged["ever_treated"].fillna(False)
    merged["event_time"] = pd.to_numeric(merged["year"], errors="coerce") - pd.to_numeric(
        merged["treatment_start_year"], errors="coerce"
    )
    event_grid = sorted(set(int(k) for k in event_window))
    rows: list[dict[str, Any]] = []
    for outcome in (
        "log_annual_price",
        "log_provider_density",
        "log_employer_establishment_density",
    ):
        if outcome not in merged.columns:
            continue
        treated_all = merged.loc[merged["ever_treated"]].copy()
        treated_reference = treated_all.loc[treated_all["event_time"].eq(reference_period)].copy()
        for event_time in event_grid:
            treated_event = treated_all.loc[treated_all["event_time"].eq(event_time)].copy()
            years = sorted(pd.to_numeric(treated_event["year"], errors="coerce").dropna().astype(int).unique().tolist())
            control_event = merged.loc[
                (~merged["ever_treated"]) & (merged["year"].isin(years))
            ].copy()
            reference_years = sorted(
                pd.to_numeric(treated_reference["year"], errors="coerce").dropna().astype(int).unique().tolist()
            )
            control_reference = merged.loc[
                (~merged["ever_treated"]) & (merged["year"].isin(reference_years))
            ].copy()
            treated_mean = _safe_group_mean(treated_event, outcome)
            control_mean = _safe_group_mean(control_event, outcome)
            treated_ref_mean = _safe_group_mean(treated_reference, outcome)
            control_ref_mean = _safe_group_mean(control_reference, outcome)
            if (
                np.isfinite(treated_mean)
                and np.isfinite(control_mean)
                and np.isfinite(treated_ref_mean)
                and np.isfinite(control_ref_mean)
            ):
                estimate = (treated_mean - control_mean) - (treated_ref_mean - control_ref_mean)
                status = "ok"
            else:
                estimate = float("nan")
                status = "insufficient_overlap"
            rows.append(
                {
                    "outcome": outcome,
                    "event_time": int(event_time),
                    "estimate": float(estimate) if np.isfinite(estimate) else float("nan"),
                    "treated_mean": treated_mean,
                    "control_mean": control_mean,
                    "treated_reference_mean": treated_ref_mean,
                    "control_reference_mean": control_ref_mean,
                    "n_treated_state_year": int(len(treated_event)),
                    "n_control_state_year": int(len(control_event)),
                    "n_treated_states": int(treated_event["state_fips"].nunique()),
                    "included_year_count": int(len(years)),
                    "reference_period": int(reference_period),
                    "status": status,
                }
            )
    if not rows:
        return pd.DataFrame(columns=EVENT_STUDY_COLUMNS)
    result = pd.DataFrame(rows, columns=EVENT_STUDY_COLUMNS)
    return result.sort_values(["outcome", "event_time"], kind="stable").reset_index(drop=True)


def build_licensing_iv_results(
    harmonized_rules: pd.DataFrame | None,
    stringency_index: pd.DataFrame,
    county_panel: pd.DataFrame,
    state_panel: pd.DataFrame | None = None,
    *,
    index_column: str | None = None,
) -> pd.DataFrame:
    del harmonized_rules, state_panel
    core = _build_core_inputs(None, stringency_index, index_column=index_column)
    summary, _ = _fit_iv_summary(county_panel, core.shock_panel)
    first_stage = summary.get("first_stage_price", {}) or {}
    first_stage_f_stat = float(first_stage.get("f_stat", float("nan")))
    first_stage_strength_tier = _first_stage_strength_tier(first_stage_f_stat)
    first_stage_n_clusters = int(first_stage.get("n_clusters", 0))
    cluster_warning = _cluster_count_warning_flag(first_stage_n_clusters)
    treated_states = summary.get("treated_state_fips", [])
    treated_state_count = len(treated_states) if isinstance(treated_states, list) else 0
    treated_state_count_flag = _treated_state_count_sufficiency_flag(int(treated_state_count))
    status = str(summary.get("status", "unknown"))

    provider_estimate = float(summary.get("iv_supply_elasticity_provider_density", float("nan")))
    provider_estimate_finite = bool(np.isfinite(provider_estimate))
    provider_recommended_use = _recommended_use_tier(
        status=status,
        estimate_finite=provider_estimate_finite,
        first_stage_strength_tier=first_stage_strength_tier,
        treated_state_count_sufficiency_flag=treated_state_count_flag,
        cluster_count_warning_flag=cluster_warning,
    )

    employer_estimate = float(
        summary.get("iv_supply_elasticity_employer_establishment_density", float("nan"))
    )
    employer_estimate_finite = bool(np.isfinite(employer_estimate))
    employer_recommended_use = _recommended_use_tier(
        status=status,
        estimate_finite=employer_estimate_finite,
        first_stage_strength_tier=first_stage_strength_tier,
        treated_state_count_sufficiency_flag=treated_state_count_flag,
        cluster_count_warning_flag=cluster_warning,
    )

    rows = [
        {
            "outcome": "provider_density",
            "estimate": provider_estimate,
            "first_stage_beta": float(first_stage.get("beta", float("nan"))),
            "first_stage_f_stat": first_stage_f_stat,
            "status": status,
            "first_stage_strength_tier": first_stage_strength_tier,
            "usable_for_headline": bool(provider_recommended_use == "headline"),
            "treated_state_count_sufficiency_flag": treated_state_count_flag,
            "cluster_count_warning_flag": bool(cluster_warning),
            "recommended_use_tier": provider_recommended_use,
            "design": str(summary.get("design", "county_fe_state_year_fe_exposure_shock")),
            "pilot_scope": str(summary.get("pilot_scope", "unknown")),
            "shock_state_count": int(summary.get("shock_state_count", 0)),
            "treated_state_count": int(treated_state_count),
            "n_obs": int(summary.get("n_obs", 0)),
            "n_states": int(summary.get("n_states", 0)),
            "year_min": int(summary.get("year_min", 0)) if summary.get("year_min") is not None else 0,
            "year_max": int(summary.get("year_max", 0)) if summary.get("year_max") is not None else 0,
        },
        {
            "outcome": "employer_establishment_density",
            "estimate": employer_estimate,
            "first_stage_beta": float(first_stage.get("beta", float("nan"))),
            "first_stage_f_stat": first_stage_f_stat,
            "status": status,
            "first_stage_strength_tier": first_stage_strength_tier,
            "usable_for_headline": bool(employer_recommended_use == "headline"),
            "treated_state_count_sufficiency_flag": treated_state_count_flag,
            "cluster_count_warning_flag": bool(cluster_warning),
            "recommended_use_tier": employer_recommended_use,
            "design": str(summary.get("design", "county_fe_state_year_fe_exposure_shock")),
            "pilot_scope": str(summary.get("pilot_scope", "unknown")),
            "shock_state_count": int(summary.get("shock_state_count", 0)),
            "treated_state_count": int(treated_state_count),
            "n_obs": int(summary.get("n_obs", 0)),
            "n_states": int(summary.get("n_states", 0)),
            "year_min": int(summary.get("year_min", 0)) if summary.get("year_min") is not None else 0,
            "year_max": int(summary.get("year_max", 0)) if summary.get("year_max") is not None else 0,
        },
    ]
    return pd.DataFrame(rows, columns=IV_RESULTS_COLUMNS)


def build_licensing_first_stage_diagnostics(
    harmonized_rules: pd.DataFrame | None,
    stringency_index: pd.DataFrame,
    county_panel: pd.DataFrame,
    state_panel: pd.DataFrame | None = None,
    *,
    index_column: str | None = None,
) -> pd.DataFrame:
    del harmonized_rules, state_panel
    core = _build_core_inputs(None, stringency_index, index_column=index_column)
    summary, _ = _fit_iv_summary(county_panel, core.shock_panel)
    first_stage = summary.get("first_stage_price", {}) or {}
    reduced_provider = summary.get("reduced_form_provider_density", {}) or {}
    reduced_employer = summary.get("reduced_form_employer_establishment_density", {}) or {}
    treated_states = summary.get("treated_state_fips", [])
    treated_state_count = len(treated_states) if isinstance(treated_states, list) else 0
    first_stage_f_stat = float(first_stage.get("f_stat", float("nan")))
    first_stage_tier = _first_stage_strength_tier(first_stage_f_stat)
    treated_count_flag = _treated_state_count_sufficiency_flag(int(treated_state_count))
    first_stage_n_clusters = int(first_stage.get("n_clusters", 0))
    cluster_warning = _cluster_count_warning_flag(first_stage_n_clusters)
    status = str(summary.get("status", "unknown"))
    recommended_use = _recommended_use_tier(
        status=status,
        estimate_finite=True,
        first_stage_strength_tier=first_stage_tier,
        treated_state_count_sufficiency_flag=treated_count_flag,
        cluster_count_warning_flag=cluster_warning,
    )
    row = {
        "status": status,
        "design": str(summary.get("design", "county_fe_state_year_fe_exposure_shock")),
        "pilot_scope": str(summary.get("pilot_scope", "unknown")),
        "first_stage_strength_flag": str(summary.get("first_stage_strength_flag", "weak_or_unknown")),
        "first_stage_strength_tier": first_stage_tier,
        "usable_for_headline": bool(recommended_use == "headline"),
        "treated_state_count_sufficiency_flag": treated_count_flag,
        "cluster_count_warning_flag": bool(cluster_warning),
        "recommended_use_tier": recommended_use,
        "first_stage_beta": float(first_stage.get("beta", float("nan"))),
        "first_stage_se_cluster_state": float(first_stage.get("se_cluster_state", float("nan"))),
        "first_stage_t_cluster_state": float(first_stage.get("t_cluster_state", float("nan"))),
        "first_stage_f_stat": first_stage_f_stat,
        "first_stage_within_r2": float(first_stage.get("within_r2", float("nan"))),
        "first_stage_n_obs": int(first_stage.get("n_obs", 0)),
        "first_stage_n_clusters": first_stage_n_clusters,
        "reduced_form_provider_beta": float(reduced_provider.get("beta", float("nan"))),
        "reduced_form_provider_t_cluster_state": float(
            reduced_provider.get("t_cluster_state", float("nan"))
        ),
        "reduced_form_provider_within_r2": float(reduced_provider.get("within_r2", float("nan"))),
        "reduced_form_employer_beta": float(reduced_employer.get("beta", float("nan"))),
        "reduced_form_employer_t_cluster_state": float(
            reduced_employer.get("t_cluster_state", float("nan"))
        ),
        "reduced_form_employer_within_r2": float(reduced_employer.get("within_r2", float("nan"))),
        "shock_state_count": int(summary.get("shock_state_count", 0)),
        "treated_state_count": int(treated_state_count),
        "n_obs": int(summary.get("n_obs", 0)),
        "n_states": int(summary.get("n_states", 0)),
    }
    return pd.DataFrame([row], columns=FIRST_STAGE_COLUMNS)


def build_licensing_leave_one_state_out(
    harmonized_rules: pd.DataFrame | None,
    stringency_index: pd.DataFrame,
    county_panel: pd.DataFrame,
    state_panel: pd.DataFrame | None = None,
    *,
    index_column: str | None = None,
) -> pd.DataFrame:
    del harmonized_rules, state_panel
    core = _build_core_inputs(None, stringency_index, index_column=index_column)
    baseline_summary, _ = _fit_iv_summary(county_panel, core.shock_panel)
    baseline_provider = float(
        baseline_summary.get("iv_supply_elasticity_provider_density", float("nan"))
    )
    baseline_employer = float(
        baseline_summary.get("iv_supply_elasticity_employer_establishment_density", float("nan"))
    )
    treated_states = baseline_summary.get("treated_state_fips", [])
    if not isinstance(treated_states, list):
        treated_states = []
    rows = [
        {
            "omitted_state_fips": "baseline",
            "status": str(baseline_summary.get("status", "unknown")),
            "n_obs": int(baseline_summary.get("n_obs", 0)),
            "n_states": int(baseline_summary.get("n_states", 0)),
            "shock_state_count": int(baseline_summary.get("shock_state_count", 0)),
            "first_stage_f_stat": float((baseline_summary.get("first_stage_price") or {}).get("f_stat", float("nan"))),
            "iv_supply_elasticity_provider_density": baseline_provider,
            "iv_supply_elasticity_employer_establishment_density": baseline_employer,
            "delta_provider_vs_baseline": 0.0,
            "delta_employer_vs_baseline": 0.0,
        }
    ]
    for omitted_state in sorted(set(str(value).zfill(2) for value in treated_states)):
        county_restricted = _normalize_state_year(county_panel)
        county_restricted = county_restricted.loc[~county_restricted["state_fips"].eq(omitted_state)].copy()
        shock_restricted = core.shock_panel.loc[~core.shock_panel["state_fips"].eq(omitted_state)].copy()
        summary, _ = _fit_iv_summary(county_restricted, shock_restricted)
        provider = float(summary.get("iv_supply_elasticity_provider_density", float("nan")))
        employer = float(summary.get("iv_supply_elasticity_employer_establishment_density", float("nan")))
        rows.append(
            {
                "omitted_state_fips": omitted_state,
                "status": str(summary.get("status", "unknown")),
                "n_obs": int(summary.get("n_obs", 0)),
                "n_states": int(summary.get("n_states", 0)),
                "shock_state_count": int(summary.get("shock_state_count", 0)),
                "first_stage_f_stat": float((summary.get("first_stage_price") or {}).get("f_stat", float("nan"))),
                "iv_supply_elasticity_provider_density": provider,
                "iv_supply_elasticity_employer_establishment_density": employer,
                "delta_provider_vs_baseline": provider - baseline_provider
                if np.isfinite(provider) and np.isfinite(baseline_provider)
                else float("nan"),
                "delta_employer_vs_baseline": employer - baseline_employer
                if np.isfinite(employer) and np.isfinite(baseline_employer)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=LEAVE_ONE_OUT_COLUMNS).sort_values(
        ["omitted_state_fips"], kind="stable"
    ).reset_index(drop=True)


def _build_backend_summary(
    event_study_results: pd.DataFrame,
    iv_results: pd.DataFrame,
    first_stage_diagnostics: pd.DataFrame,
    treatment_timing: pd.DataFrame,
    leave_one_state_out: pd.DataFrame,
    iv_usability_summary: pd.DataFrame,
) -> dict[str, object]:
    iv_statuses = (
        sorted(iv_results["status"].dropna().astype(str).unique().tolist())
        if not iv_results.empty and "status" in iv_results.columns
        else []
    )
    event_statuses = (
        sorted(event_study_results["status"].dropna().astype(str).unique().tolist())
        if not event_study_results.empty and "status" in event_study_results.columns
        else []
    )
    treated_states = (
        int(pd.to_numeric(treatment_timing.get("ever_treated"), errors="coerce").fillna(0).astype(bool).sum())
        if not treatment_timing.empty and "ever_treated" in treatment_timing.columns
        else 0
    )
    summary: dict[str, object] = {
        "event_study_row_count": int(len(event_study_results)),
        "iv_result_row_count": int(len(iv_results)),
        "first_stage_row_count": int(len(first_stage_diagnostics)),
        "treatment_timing_row_count": int(len(treatment_timing)),
        "leave_one_state_out_row_count": int(len(leave_one_state_out)),
        "treated_state_count": treated_states,
        "iv_statuses": iv_statuses,
        "event_study_statuses": event_statuses,
        "status": iv_statuses[0] if len(iv_statuses) == 1 else ("mixed" if iv_statuses else "missing"),
    }
    if not iv_results.empty:
        provider_row = iv_results.loc[iv_results["outcome"].eq("provider_density")]
        if not provider_row.empty:
            summary["provider_density_iv_estimate"] = float(provider_row.iloc[0]["estimate"])
    if not first_stage_diagnostics.empty:
        summary["first_stage_strength_flag"] = str(first_stage_diagnostics.iloc[0]["first_stage_strength_flag"])
        summary["first_stage_f_stat"] = float(first_stage_diagnostics.iloc[0]["first_stage_f_stat"])
    if not iv_usability_summary.empty:
        summary["iv_usability_counts"] = (
            iv_usability_summary["recommended_use_tier"]
            .fillna("diagnostics_only")
            .astype(str)
            .value_counts()
            .sort_index()
            .to_dict()
        )
        headline_ready = iv_usability_summary["usable_for_headline"].fillna(False).astype(bool)
        summary["iv_headline_usable_outcome_count"] = int(headline_ready.sum())
        summary["iv_usability_summary_rows"] = iv_usability_summary.to_dict(orient="records")
    return summary


def _build_iv_usability_summary(
    iv_results: pd.DataFrame,
    first_stage_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    if iv_results.empty:
        return pd.DataFrame(columns=IV_USABILITY_COLUMNS)
    working = iv_results.copy()
    if "first_stage_n_clusters" not in working.columns:
        if not first_stage_diagnostics.empty and "first_stage_n_clusters" in first_stage_diagnostics.columns:
            cluster_count = int(first_stage_diagnostics.iloc[0]["first_stage_n_clusters"])
        else:
            cluster_count = 0
        working["first_stage_n_clusters"] = cluster_count
    working["estimate_finite"] = pd.to_numeric(working.get("estimate"), errors="coerce").map(np.isfinite)
    rows = []
    for row in working.to_dict(orient="records"):
        status = str(row.get("status", "unknown"))
        first_stage_f_stat = float(pd.to_numeric(pd.Series([row.get("first_stage_f_stat")]), errors="coerce").iloc[0])
        first_stage_strength_tier = str(row.get("first_stage_strength_tier", _first_stage_strength_tier(first_stage_f_stat)))
        treated_state_count = int(pd.to_numeric(pd.Series([row.get("treated_state_count", 0)]), errors="coerce").fillna(0).iloc[0])
        treated_state_count_flag = str(
            row.get(
                "treated_state_count_sufficiency_flag",
                _treated_state_count_sufficiency_flag(treated_state_count),
            )
        )
        cluster_count = int(pd.to_numeric(pd.Series([row.get("first_stage_n_clusters", 0)]), errors="coerce").fillna(0).iloc[0])
        cluster_warning = bool(row.get("cluster_count_warning_flag", _cluster_count_warning_flag(cluster_count)))
        estimate_finite = bool(row.get("estimate_finite", False))
        recommended_use_tier = str(
            row.get(
                "recommended_use_tier",
                _recommended_use_tier(
                    status=status,
                    estimate_finite=estimate_finite,
                    first_stage_strength_tier=first_stage_strength_tier,
                    treated_state_count_sufficiency_flag=treated_state_count_flag,
                    cluster_count_warning_flag=cluster_warning,
                ),
            )
        )
        usable_for_headline = bool(row.get("usable_for_headline", recommended_use_tier == "headline"))
        rows.append(
            {
                "outcome": str(row.get("outcome", "unknown")),
                "status": status,
                "estimate_finite": estimate_finite,
                "first_stage_f_stat": first_stage_f_stat,
                "first_stage_strength_tier": first_stage_strength_tier,
                "treated_state_count": treated_state_count,
                "treated_state_count_sufficiency_flag": treated_state_count_flag,
                "first_stage_n_clusters": cluster_count,
                "cluster_count_warning_flag": cluster_warning,
                "usable_for_headline": usable_for_headline,
                "recommended_use_tier": recommended_use_tier,
            }
        )
    return pd.DataFrame(rows, columns=IV_USABILITY_COLUMNS).sort_values(
        ["recommended_use_tier", "outcome"], kind="stable"
    ).reset_index(drop=True)


def build_licensing_iv_backend_outputs(
    harmonized_rules: pd.DataFrame | None,
    stringency_index: pd.DataFrame,
    county_panel: pd.DataFrame,
    state_panel: pd.DataFrame | None = None,
    *,
    index_column: str | None = None,
    event_window: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3),
    reference_period: int = -1,
) -> dict[str, object]:
    treatment_timing = build_licensing_treatment_timing(
        harmonized_rules,
        stringency_index,
        index_column=index_column,
    )
    event_study_results = build_licensing_event_study_results(
        harmonized_rules,
        stringency_index,
        county_panel,
        state_panel,
        index_column=index_column,
        event_window=event_window,
        reference_period=reference_period,
    )
    iv_results = build_licensing_iv_results(
        harmonized_rules,
        stringency_index,
        county_panel,
        state_panel,
        index_column=index_column,
    )
    first_stage_diagnostics = build_licensing_first_stage_diagnostics(
        harmonized_rules,
        stringency_index,
        county_panel,
        state_panel,
        index_column=index_column,
    )
    leave_one_state_out = build_licensing_leave_one_state_out(
        harmonized_rules,
        stringency_index,
        county_panel,
        state_panel,
        index_column=index_column,
    )
    iv_usability_summary = _build_iv_usability_summary(
        iv_results=iv_results,
        first_stage_diagnostics=first_stage_diagnostics,
    )
    summary = _build_backend_summary(
        event_study_results=event_study_results,
        iv_results=iv_results,
        first_stage_diagnostics=first_stage_diagnostics,
        treatment_timing=treatment_timing,
        leave_one_state_out=leave_one_state_out,
        iv_usability_summary=iv_usability_summary,
    )
    return {
        "event_study_results": event_study_results,
        "iv_results": iv_results,
        "first_stage_diagnostics": first_stage_diagnostics,
        "treatment_timing": treatment_timing,
        "leave_one_state_out": leave_one_state_out,
        "iv_usability_summary": iv_usability_summary,
        "summary": summary,
    }
