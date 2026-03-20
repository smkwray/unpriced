from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from unpriced.models.common import design_matrix, r2_score, weighted_least_squares
from unpriced.storage import write_json


STATUS_BASELINE = "observed"
STATUS_LEVELS = [
    ("national_avg_not_reported", "not_reported"),
    ("national_avg_nonmetro", "nonmetro"),
]


def _add_noaa_status_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    result = dataset.copy()
    for raw_status, label in STATUS_LEVELS:
        result[f"noaa_match_status_{label}"] = result["noaa_match_status"].eq(raw_status).astype(float)
    result["noaa_match_status_other_fallback"] = ~result["noaa_match_status"].isin(
        {STATUS_BASELINE, *(status for status, _ in STATUS_LEVELS)}
    )
    result["noaa_match_status_other_fallback"] = result["noaa_match_status_other_fallback"].astype(float)
    return result


def _status_specific_weather_effects(beta: np.ndarray) -> dict[str, float]:
    storm_observed = float(beta[4])
    precip_observed = float(beta[5])
    return {
        "storm_exposure_effect_observed": storm_observed,
        "storm_exposure_effect_not_reported": storm_observed + float(beta[10]),
        "storm_exposure_effect_nonmetro": storm_observed + float(beta[11]),
        "storm_exposure_effect_other_fallback": storm_observed + float(beta[12]),
        "precip_event_days_effect_observed": precip_observed,
        "precip_event_days_effect_not_reported": precip_observed + float(beta[13]),
        "precip_event_days_effect_nonmetro": precip_observed + float(beta[14]),
        "precip_event_days_effect_other_fallback": precip_observed + float(beta[15]),
        "storm_exposure_interaction_not_reported": float(beta[10]),
        "storm_exposure_interaction_nonmetro": float(beta[11]),
        "storm_exposure_interaction_other_fallback": float(beta[12]),
        "precip_event_days_interaction_not_reported": float(beta[13]),
        "precip_event_days_interaction_nonmetro": float(beta[14]),
        "precip_event_days_interaction_other_fallback": float(beta[15]),
    }


def fit_home_switching(frame: pd.DataFrame, output_json: Path) -> dict[str, float]:
    dataset = frame.copy()
    if "noaa_match_status" not in dataset.columns:
        dataset["noaa_match_status"] = "observed"
    if "precip_event_days" not in dataset.columns:
        dataset["precip_event_days"] = 0.0
    if "storm_event_count" not in dataset.columns:
        dataset["storm_event_count"] = 0.0
    if "storm_property_damage" not in dataset.columns:
        dataset["storm_property_damage"] = 0.0
    dataset["precip_event_days"] = pd.to_numeric(dataset["precip_event_days"], errors="coerce").fillna(0.0)
    dataset["storm_event_count"] = pd.to_numeric(dataset["storm_event_count"], errors="coerce").fillna(0.0)
    dataset["storm_property_damage"] = pd.to_numeric(dataset["storm_property_damage"], errors="coerce").fillna(0.0)
    dataset["log_storm_property_damage"] = np.log1p(dataset["storm_property_damage"])
    dataset["noaa_match_status"] = dataset["noaa_match_status"].fillna("observed").astype(str)
    dataset = _add_noaa_status_columns(dataset)
    dataset = dataset.dropna(
        subset=[
            "job_diy",
            "predicted_job_cost",
            "household_income",
            "home_value",
            "storm_exposure",
            "cbsa_unemployment_rate",
        ]
    ).copy()
    x = design_matrix(
        [
            dataset["predicted_job_cost"].to_numpy(dtype=float),
            dataset["household_income"].to_numpy(dtype=float),
            dataset["home_value"].to_numpy(dtype=float),
            dataset["storm_exposure"].to_numpy(dtype=float),
            dataset["precip_event_days"].to_numpy(dtype=float),
            dataset["cbsa_unemployment_rate"].to_numpy(dtype=float),
            dataset["noaa_match_status_not_reported"].to_numpy(dtype=float),
            dataset["noaa_match_status_nonmetro"].to_numpy(dtype=float),
            dataset["noaa_match_status_other_fallback"].to_numpy(dtype=float),
            (
                dataset["storm_exposure"].to_numpy(dtype=float)
                * dataset["noaa_match_status_not_reported"].to_numpy(dtype=float)
            ),
            (
                dataset["storm_exposure"].to_numpy(dtype=float)
                * dataset["noaa_match_status_nonmetro"].to_numpy(dtype=float)
            ),
            (
                dataset["storm_exposure"].to_numpy(dtype=float)
                * dataset["noaa_match_status_other_fallback"].to_numpy(dtype=float)
            ),
            (
                dataset["precip_event_days"].to_numpy(dtype=float)
                * dataset["noaa_match_status_not_reported"].to_numpy(dtype=float)
            ),
            (
                dataset["precip_event_days"].to_numpy(dtype=float)
                * dataset["noaa_match_status_nonmetro"].to_numpy(dtype=float)
            ),
            (
                dataset["precip_event_days"].to_numpy(dtype=float)
                * dataset["noaa_match_status_other_fallback"].to_numpy(dtype=float)
            ),
            dataset["storm_event_count"].to_numpy(dtype=float),
            dataset["log_storm_property_damage"].to_numpy(dtype=float),
        ]
    )
    y = dataset["job_diy"].to_numpy(dtype=float)
    beta = weighted_least_squares(x, y)
    predicted = np.clip(x @ beta, 0.0, 1.0)
    status_counts = dataset["noaa_match_status"].value_counts().to_dict()
    summary = {
        "n_obs": int(len(dataset)),
        "price_effect": float(beta[1]),
        "storm_exposure_effect": float(beta[4]),
        "precip_event_days_effect": float(beta[5]),
        "unemployment_effect": float(beta[6]),
        "noaa_not_reported_effect": float(beta[7]),
        "noaa_nonmetro_effect": float(beta[8]),
        "noaa_other_fallback_effect": float(beta[9]),
        "storm_event_count_effect": float(beta[16]),
        "log_storm_property_damage_effect": float(beta[17]),
        "model_specification": "weather_status_interactions_plus_storm_load",
        "r2_in_sample": r2_score(y, predicted),
        "noaa_observed_rows": int(status_counts.get("observed", 0)),
        "noaa_not_reported_rows": int(status_counts.get("national_avg_not_reported", 0)),
        "noaa_nonmetro_rows": int(status_counts.get("national_avg_nonmetro", 0)),
        "noaa_other_fallback_rows": int(
            len(dataset)
            - int(status_counts.get("observed", 0))
            - int(status_counts.get("national_avg_not_reported", 0))
            - int(status_counts.get("national_avg_nonmetro", 0))
        ),
        "mean_predicted_diy_probability": float(predicted.mean()),
    }
    summary.update(_status_specific_weather_effects(beta))
    write_json(summary, output_json)
    return summary
