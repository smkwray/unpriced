from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from unpriced.models.common import design_matrix, weighted_least_squares
from unpriced.storage import write_json, write_parquet

EXOG_FEATURES = [
    "parent_employment_rate",
    "single_parent_share",
    "median_income",
    "unemployment_rate",
]
SPECIFICATION_PROFILES = {
    "full_controls": EXOG_FEATURES,
    "household_parsimonious": ["parent_employment_rate", "single_parent_share"],
    "labor_parsimonious": ["parent_employment_rate", "unemployment_rate"],
    "instrument_only": [],
}
DEFAULT_SPECIFICATION_PROFILE = "full_controls"
CANONICAL_OBSERVED_SPECIFICATION_PROFILE = "household_parsimonious"
CANONICAL_COMPARISON_SPECIFICATION_PROFILE = CANONICAL_OBSERVED_SPECIFICATION_PROFILE

SAMPLE_MODE_ALIASES = {
    "baseline": "broad_complete",
    "strict_observed": "observed_core",
    "broad_complete": "broad_complete",
    "observed_core": "observed_core",
    "observed_core_low_impute": "observed_core_low_impute",
}
CANONICAL_SAMPLE_MODES = [
    "broad_complete",
    "observed_core",
    "observed_core_low_impute",
]
LOW_IMPUTE_THRESHOLD_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
HIGH_LABOR_SUPPORT_THRESHOLD_SWEEP = [0.95, 0.96, 0.98, 0.99, 1.00]
HEADLINE_SAMPLE_ORDER = ["observed_core_low_impute", "observed_core"]
HEADLINE_MIN_OBS = 100
HEADLINE_MIN_STATES = 15
HEADLINE_MIN_YEARS = 6


def _regression_weights(dataset: pd.DataFrame) -> np.ndarray:
    if "atus_weight_sum" not in dataset.columns:
        return np.ones(len(dataset), dtype=float)
    weights = pd.to_numeric(dataset["atus_weight_sum"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    weights = np.where(weights > 0, weights, 1.0)
    return weights


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return float(values.mean())
    return float(np.average(values, weights=weights))


def _weighted_r2(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    mean_actual = _weighted_mean(actual, weights)
    sse = float((weights * (actual - predicted) ** 2).sum())
    sst = float((weights * (actual - mean_actual) ** 2).sum())
    if sst <= 0:
        return float("nan")
    return 1.0 - sse / sst


def _weighted_rmse(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return float(np.sqrt((weights * (actual - predicted) ** 2).sum() / total))


def normalize_demand_mode(mode: str) -> str:
    normalized = SAMPLE_MODE_ALIASES.get(mode)
    if normalized is None:
        raise ValueError(f"unsupported childcare demand mode: {mode}")
    return normalized


def normalize_specification_profile(profile: str | None) -> str:
    normalized = profile or DEFAULT_SPECIFICATION_PROFILE
    if normalized not in SPECIFICATION_PROFILES:
        raise ValueError(f"unsupported childcare demand specification profile: {profile}")
    return normalized


def _drop_to_complete_cases(
    frame: pd.DataFrame,
    instrument: str,
    exog_features: list[str],
) -> pd.DataFrame:
    required = ["unpaid_childcare_hours", "state_price_index", instrument] + exog_features
    dataset = frame.dropna(subset=required).copy()
    return dataset.loc[pd.to_numeric(dataset["state_price_index"], errors="coerce").gt(0)].reset_index(drop=True)


def filter_childcare_demand_sample(
    frame: pd.DataFrame,
    instrument: str = "outside_option_wage",
    mode: str = "broad_complete",
    specification_profile: str | None = None,
) -> pd.DataFrame:
    normalized = normalize_demand_mode(mode)
    normalized_spec = normalize_specification_profile(specification_profile)
    dataset = _drop_to_complete_cases(frame, instrument, SPECIFICATION_PROFILES[normalized_spec])
    if normalized == "broad_complete":
        return dataset
    eligibility_column = f"eligible_{normalized}"
    if eligibility_column not in dataset.columns:
        return dataset.iloc[0:0].copy()
    mask = dataset[eligibility_column].fillna(False).astype(bool)
    return dataset.loc[mask].reset_index(drop=True)


def _run_2sls(
    dataset: pd.DataFrame,
    instrument: str = "outside_option_wage",
    exog_features: list[str] | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    controls = EXOG_FEATURES if exog_features is None else exog_features
    z = design_matrix(
        [dataset[instrument].to_numpy(dtype=float)]
        + [dataset[col].to_numpy(dtype=float) for col in controls]
    )
    p = dataset["state_price_index"].to_numpy(dtype=float)
    y = dataset["unpaid_childcare_hours"].to_numpy(dtype=float)
    x_exog = [dataset[col].to_numpy(dtype=float) for col in controls]
    weights = _regression_weights(dataset)

    stage1_beta = weighted_least_squares(z, p, weights=weights)
    predicted_price = z @ stage1_beta

    x_second = design_matrix([predicted_price] + x_exog)
    stage2_beta = weighted_least_squares(x_second, y, weights=weights)
    fitted_hours = x_second @ stage2_beta

    first_stage_r2 = _weighted_r2(p, predicted_price, weights)
    elasticity_at_mean = float(
        stage2_beta[1]
        * _weighted_mean(dataset["state_price_index"].to_numpy(dtype=float), weights)
        / _weighted_mean(dataset["unpaid_childcare_hours"].to_numpy(dtype=float), weights)
    )

    summary = {
        "n_obs": int(len(dataset)),
        "first_stage_r2": first_stage_r2,
        "price_coefficient": float(stage2_beta[1]),
        "elasticity_at_mean": elasticity_at_mean,
        "avg_fitted_hours": _weighted_mean(fitted_hours, weights),
    }
    return summary, predicted_price, fitted_hours


def _leave_one_out_cv(
    dataset: pd.DataFrame,
    group_col: str,
    instrument: str = "outside_option_wage",
    exog_features: list[str] | None = None,
) -> dict[str, float]:
    controls = EXOG_FEATURES if exog_features is None else exog_features
    groups = sorted(dataset[group_col].unique().tolist())
    if len(groups) < 3:
        return {
            f"loo_{group_col}_groups": len(groups),
            f"loo_{group_col}_rmse": float("nan"),
            f"loo_{group_col}_r2": float("nan"),
        }

    all_actual: list[float] = []
    all_predicted: list[float] = []
    all_weights: list[float] = []
    for holdout in groups:
        train = dataset.loc[dataset[group_col] != holdout].copy()
        test = dataset.loc[dataset[group_col] == holdout].copy()
        if len(train) < 5 or len(test) == 0:
            continue
        train_weights = _regression_weights(train)
        test_weights = _regression_weights(test)

        z_train = design_matrix(
            [train[instrument].to_numpy(dtype=float)]
            + [train[col].to_numpy(dtype=float) for col in controls]
        )
        p_train = train["state_price_index"].to_numpy(dtype=float)
        stage1_beta = weighted_least_squares(z_train, p_train, weights=train_weights)

        z_test = design_matrix(
            [test[instrument].to_numpy(dtype=float)]
            + [test[col].to_numpy(dtype=float) for col in controls]
        )
        predicted_price_test = z_test @ stage1_beta

        predicted_price_train = z_train @ stage1_beta
        x_train = design_matrix([predicted_price_train] + [train[col].to_numpy(dtype=float) for col in controls])
        y_train = train["unpaid_childcare_hours"].to_numpy(dtype=float)
        stage2_beta = weighted_least_squares(x_train, y_train, weights=train_weights)

        x_test = design_matrix([predicted_price_test] + [test[col].to_numpy(dtype=float) for col in controls])
        y_test = test["unpaid_childcare_hours"].to_numpy(dtype=float)
        fitted_test = x_test @ stage2_beta

        all_actual.extend(y_test.tolist())
        all_predicted.extend(fitted_test.tolist())
        all_weights.extend(test_weights.tolist())

    if not all_actual:
        return {
            f"loo_{group_col}_groups": len(groups),
            f"loo_{group_col}_rmse": float("nan"),
            f"loo_{group_col}_r2": float("nan"),
        }

    actual = np.array(all_actual)
    predicted = np.array(all_predicted)
    weights = np.array(all_weights, dtype=float)
    rmse = _weighted_rmse(actual, predicted, weights)
    r2 = _weighted_r2(actual, predicted, weights)

    return {
        f"loo_{group_col}_groups": len(groups),
        f"loo_{group_col}_rmse": round(rmse, 6),
        f"loo_{group_col}_r2": round(r2, 6),
    }


def _empty_summary(mode: str) -> dict[str, float | int | bool]:
    normalized = normalize_demand_mode(mode)
    headline_rank = HEADLINE_SAMPLE_ORDER.index(normalized) + 1 if normalized in HEADLINE_SAMPLE_ORDER else 3
    return {
        "n_obs": 0,
        "first_stage_r2": float("nan"),
        "price_coefficient": float("nan"),
        "elasticity_at_mean": float("nan"),
        "avg_fitted_hours": float("nan"),
        "mode": normalized,
        "n_states": 0,
        "n_years": 0,
        "year_min": 0,
        "year_max": 0,
        "headline_eligible": False,
        "headline_rank": headline_rank,
        "negative_price_response": False,
        "economically_admissible": False,
    }


def _headline_eligible(summary: dict[str, float | int | bool]) -> bool:
    return (
        int(summary.get("n_obs", 0)) >= HEADLINE_MIN_OBS
        and int(summary.get("n_states", 0)) >= HEADLINE_MIN_STATES
        and int(summary.get("n_years", 0)) >= HEADLINE_MIN_YEARS
    )


def _economically_admissible(summary: dict[str, float | int | bool]) -> bool:
    price_coefficient = float(summary.get("price_coefficient", float("nan")))
    elasticity = float(summary.get("elasticity_at_mean", float("nan")))
    if not np.isfinite(price_coefficient) or not np.isfinite(elasticity):
        return False
    return price_coefficient <= 0 and elasticity <= 0


def estimate_childcare_demand_summary(
    frame: pd.DataFrame,
    instrument: str = "outside_option_wage",
    mode: str = "baseline",
    specification_profile: str | None = None,
) -> tuple[dict[str, float | int | bool], pd.DataFrame]:
    normalized = normalize_demand_mode(mode)
    normalized_spec = normalize_specification_profile(specification_profile)
    exog_features = SPECIFICATION_PROFILES[normalized_spec]
    dataset = filter_childcare_demand_sample(
        frame,
        instrument=instrument,
        mode=normalized,
        specification_profile=normalized_spec,
    )

    if len(dataset) == 0:
        summary = _empty_summary(normalized)
        summary["instrument"] = instrument
        summary["specification_profile"] = normalized_spec
        summary["exog_features"] = exog_features
        return summary, dataset

    summary, predicted_price, fitted_hours = _run_2sls(dataset, instrument, exog_features=exog_features)
    dataset["predicted_state_price"] = predicted_price
    summary["instrument"] = instrument
    summary["mode"] = normalized
    summary["specification_profile"] = normalized_spec
    summary["exog_features"] = exog_features
    summary["n_states"] = int(dataset["state_fips"].nunique()) if "state_fips" in dataset.columns else 0
    summary["n_years"] = int(dataset["year"].nunique()) if "year" in dataset.columns else 0
    summary["year_min"] = int(dataset["year"].min()) if "year" in dataset.columns else 0
    summary["year_max"] = int(dataset["year"].max()) if "year" in dataset.columns else 0
    summary["headline_rank"] = HEADLINE_SAMPLE_ORDER.index(normalized) + 1 if normalized in HEADLINE_SAMPLE_ORDER else 3
    summary["negative_price_response"] = float(summary.get("price_coefficient", float("nan"))) <= 0
    summary["economically_admissible"] = _economically_admissible(summary)
    summary["headline_eligible"] = (
        normalized in HEADLINE_SAMPLE_ORDER
        and _headline_eligible(summary)
        and bool(summary["economically_admissible"])
    )

    if "state_fips" in dataset.columns and len(dataset) >= 10:
        summary.update(_leave_one_out_cv(dataset, "state_fips", instrument, exog_features=exog_features))
    if "year" in dataset.columns and len(dataset) >= 10:
        summary.update(_leave_one_out_cv(dataset, "year", instrument, exog_features=exog_features))

    return summary, dataset


def fit_childcare_demand_iv(
    frame: pd.DataFrame,
    output_json: Path,
    output_panel: Path,
    instrument: str = "outside_option_wage",
    mode: str = "baseline",
    specification_profile: str | None = None,
) -> dict[str, float | int | bool]:
    summary, dataset = estimate_childcare_demand_summary(
        frame,
        instrument=instrument,
        mode=mode,
        specification_profile=specification_profile,
    )
    write_parquet(dataset, output_panel)
    write_json(summary, output_json)
    return summary


def select_headline_sample(
    sample_summaries: dict[str, dict[str, float | int | bool]],
) -> tuple[str | None, str]:
    for sample_name in HEADLINE_SAMPLE_ORDER:
        summary = sample_summaries.get(sample_name, {})
        if _headline_eligible(summary) and bool(summary.get("economically_admissible", False)):
            return sample_name, f"{sample_name}_passes_minimum_support"
    return None, "no_admissible_observed_core_sample_passes_minimum_support"


def build_childcare_demand_sample_comparison(
    frame: pd.DataFrame,
    output_json: Path | None = None,
    instrument: str = "outside_option_wage",
    specification_profile: str | None = None,
) -> dict[str, object]:
    normalized_spec = normalize_specification_profile(specification_profile or CANONICAL_COMPARISON_SPECIFICATION_PROFILE)
    samples: dict[str, dict[str, float | int | bool]] = {}
    for mode in CANONICAL_SAMPLE_MODES:
        summary, _ = estimate_childcare_demand_summary(
            frame,
            instrument=instrument,
            mode=mode,
            specification_profile=normalized_spec,
        )
        samples[mode] = summary

    selected_sample, selection_reason = select_headline_sample(samples)
    comparison = {
        "comparison_specification_profile": normalized_spec,
        "samples": samples,
        "selected_headline_sample": selected_sample,
        "selected_headline_reason": selection_reason,
    }
    if output_json is not None:
        write_json(comparison, output_json)
    return comparison


def build_childcare_imputation_sweep(
    frame: pd.DataFrame,
    output_json: Path | None = None,
    instrument: str = "outside_option_wage",
    thresholds: list[float] | None = None,
    specification_profile: str | None = None,
) -> dict[str, object]:
    threshold_values = thresholds or LOW_IMPUTE_THRESHOLD_SWEEP
    normalized_spec = normalize_specification_profile(specification_profile or CANONICAL_COMPARISON_SPECIFICATION_PROFILE)
    observed_core_summary, _ = estimate_childcare_demand_summary(
        frame,
        instrument=instrument,
        mode="observed_core",
        specification_profile=normalized_spec,
    )
    base = frame.copy()
    rows: list[dict[str, float | int | bool | None]] = []
    for threshold in threshold_values:
        threshold_mask = (
            base["eligible_observed_core"].fillna(False).astype(bool)
            & pd.to_numeric(base["state_ndcp_imputed_share"], errors="coerce").le(float(threshold))
        )
        candidate = base.copy()
        candidate["eligible_observed_core_low_impute"] = threshold_mask
        summary, _ = estimate_childcare_demand_summary(
            candidate,
            instrument=instrument,
            mode="observed_core_low_impute",
            specification_profile=normalized_spec,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "n_obs": int(summary.get("n_obs", 0)),
                "n_states": int(summary.get("n_states", 0)),
                "n_years": int(summary.get("n_years", 0)),
                "first_stage_r2": float(summary.get("first_stage_r2", float("nan"))),
                "loo_state_fips_r2": float(summary.get("loo_state_fips_r2", float("nan"))),
                "loo_year_r2": float(summary.get("loo_year_r2", float("nan"))),
                "elasticity_at_mean": float(summary.get("elasticity_at_mean", float("nan"))),
                "headline_eligible": bool(_headline_eligible(summary)),
                "beats_observed_core_loo_state": bool(
                    float(summary.get("loo_state_fips_r2", float("-inf")))
                    > float(observed_core_summary.get("loo_state_fips_r2", float("-inf")))
                ),
                "beats_observed_core_loo_year": bool(
                    float(summary.get("loo_year_r2", float("-inf")))
                    > float(observed_core_summary.get("loo_year_r2", float("-inf")))
                ),
            }
        )
    passing_thresholds = [row for row in rows if bool(row["headline_eligible"])]
    if passing_thresholds:
        best_headline_candidate = max(
            passing_thresholds,
            key=lambda row: (
                float(row["loo_year_r2"]),
                float(row["loo_state_fips_r2"]),
                -float(row["threshold"]),
            ),
        )
    else:
        best_headline_candidate = None
    sweep = {
        "current_headline_sample": "observed_core",
        "specification_profile": normalized_spec,
        "current_headline_loo_state_fips_r2": float(observed_core_summary.get("loo_state_fips_r2", float("nan"))),
        "current_headline_loo_year_r2": float(observed_core_summary.get("loo_year_r2", float("nan"))),
        "current_headline_n_obs": int(observed_core_summary.get("n_obs", 0)),
        "current_headline_n_states": int(observed_core_summary.get("n_states", 0)),
        "current_headline_n_years": int(observed_core_summary.get("n_years", 0)),
        "thresholds": rows,
        "best_headline_eligible_threshold": (
            {
                "threshold": best_headline_candidate["threshold"],
                "loo_state_fips_r2": best_headline_candidate["loo_state_fips_r2"],
                "loo_year_r2": best_headline_candidate["loo_year_r2"],
                "n_obs": best_headline_candidate["n_obs"],
                "n_states": best_headline_candidate["n_states"],
                "n_years": best_headline_candidate["n_years"],
            }
            if best_headline_candidate is not None
            else None
        ),
    }
    if output_json is not None:
        write_json(sweep, output_json)
    return sweep


def build_childcare_labor_support_sweep(
    frame: pd.DataFrame,
    output_json: Path | None = None,
    instrument: str = "outside_option_wage",
    thresholds: list[float] | None = None,
    specification_profile: str | None = None,
) -> dict[str, object]:
    threshold_values = thresholds or HIGH_LABOR_SUPPORT_THRESHOLD_SWEEP
    normalized_spec = normalize_specification_profile(specification_profile or CANONICAL_COMPARISON_SPECIFICATION_PROFILE)
    observed_core_summary, _ = estimate_childcare_demand_summary(
        frame,
        instrument=instrument,
        mode="observed_core",
        specification_profile=normalized_spec,
    )
    base = frame.copy()
    rows: list[dict[str, float | int | bool]] = []
    for threshold in threshold_values:
        threshold_mask = (
            base["eligible_observed_core"].fillna(False).astype(bool)
            & pd.to_numeric(base["state_qcew_labor_observed_share"], errors="coerce").ge(float(threshold))
        )
        candidate = base.copy()
        candidate["eligible_observed_core_low_impute"] = threshold_mask
        summary, _ = estimate_childcare_demand_summary(
            candidate,
            instrument=instrument,
            mode="observed_core_low_impute",
            specification_profile=normalized_spec,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "n_obs": int(summary.get("n_obs", 0)),
                "n_states": int(summary.get("n_states", 0)),
                "n_years": int(summary.get("n_years", 0)),
                "first_stage_r2": float(summary.get("first_stage_r2", float("nan"))),
                "loo_state_fips_r2": float(summary.get("loo_state_fips_r2", float("nan"))),
                "loo_year_r2": float(summary.get("loo_year_r2", float("nan"))),
                "elasticity_at_mean": float(summary.get("elasticity_at_mean", float("nan"))),
                "headline_eligible": bool(_headline_eligible(summary)),
                "beats_observed_core_loo_state": bool(
                    float(summary.get("loo_state_fips_r2", float("-inf")))
                    > float(observed_core_summary.get("loo_state_fips_r2", float("-inf")))
                ),
                "beats_observed_core_loo_year": bool(
                    float(summary.get("loo_year_r2", float("-inf")))
                    > float(observed_core_summary.get("loo_year_r2", float("-inf")))
                ),
            }
        )
    passing_thresholds = [row for row in rows if bool(row["headline_eligible"])]
    if passing_thresholds:
        best_headline_candidate = max(
            passing_thresholds,
            key=lambda row: (
                float(row["loo_year_r2"]),
                float(row["loo_state_fips_r2"]),
                -float(row["threshold"]),
            ),
        )
    else:
        best_headline_candidate = None
    sweep = {
        "current_headline_sample": "observed_core",
        "specification_profile": normalized_spec,
        "current_headline_loo_state_fips_r2": float(observed_core_summary.get("loo_state_fips_r2", float("nan"))),
        "current_headline_loo_year_r2": float(observed_core_summary.get("loo_year_r2", float("nan"))),
        "current_headline_n_obs": int(observed_core_summary.get("n_obs", 0)),
        "current_headline_n_states": int(observed_core_summary.get("n_states", 0)),
        "current_headline_n_years": int(observed_core_summary.get("n_years", 0)),
        "thresholds": rows,
        "best_headline_eligible_threshold": (
            {
                "threshold": best_headline_candidate["threshold"],
                "loo_state_fips_r2": best_headline_candidate["loo_state_fips_r2"],
                "loo_year_r2": best_headline_candidate["loo_year_r2"],
                "n_obs": best_headline_candidate["n_obs"],
                "n_states": best_headline_candidate["n_states"],
                "n_years": best_headline_candidate["n_years"],
            }
            if best_headline_candidate is not None
            else None
        ),
    }
    if output_json is not None:
        write_json(sweep, output_json)
    return sweep


def build_childcare_specification_sweep(
    frame: pd.DataFrame,
    output_json: Path | None = None,
    instrument: str = "outside_option_wage",
    mode: str = "observed_core",
    profiles: list[str] | None = None,
    current_profile: str | None = None,
) -> dict[str, object]:
    profile_names = profiles or list(SPECIFICATION_PROFILES.keys())
    normalized_mode = normalize_demand_mode(mode)
    reference_profile = normalize_specification_profile(
        current_profile
        or (CANONICAL_OBSERVED_SPECIFICATION_PROFILE if normalized_mode in HEADLINE_SAMPLE_ORDER else DEFAULT_SPECIFICATION_PROFILE)
    )
    results: dict[str, dict[str, float | int | bool | list[str] | str]] = {}
    for profile in profile_names:
        summary, _ = estimate_childcare_demand_summary(
            frame,
            instrument=instrument,
            mode=mode,
            specification_profile=profile,
        )
        results[profile] = summary
    current = results[reference_profile]
    beating = [
        profile
        for profile, summary in results.items()
        if profile != reference_profile
        and bool(summary.get("economically_admissible", False))
        and float(summary.get("loo_state_fips_r2", float("-inf"))) > float(current.get("loo_state_fips_r2", float("-inf")))
        and float(summary.get("loo_year_r2", float("-inf"))) > float(current.get("loo_year_r2", float("-inf")))
    ]
    preferred = None
    if beating:
        preferred = max(
            beating,
            key=lambda name: (
                float(results[name].get("loo_year_r2", float("-inf"))),
                float(results[name].get("loo_state_fips_r2", float("-inf"))),
                -len(SPECIFICATION_PROFILES[name]),
            ),
        )
    sweep = {
        "mode": normalized_mode,
        "current_profile": reference_profile,
        "profiles": results,
        "profiles_beating_current_on_both_holdouts": beating,
        "preferred_holdout_profile": preferred,
    }
    if output_json is not None:
        write_json(sweep, output_json)
    return sweep
