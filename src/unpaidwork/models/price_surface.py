from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from unpaidwork.models.common import design_matrix, rmse, r2_score, weighted_least_squares
from unpaidwork.storage import write_json, write_parquet

DEFAULT_FEATURES = [
    "childcare_worker_wage",
    "provider_density",
    "median_income",
    "under5_population",
    "unemployment_rate",
    "head_start_capacity",
    "public_school_option_index",
]


def fit_price_surface(
    frame: pd.DataFrame,
    output_json: Path,
    output_panel: Path,
    target: str = "annual_price",
    features: list[str] | None = None,
) -> dict[str, float | int | dict[str, float]]:
    features = features or DEFAULT_FEATURES
    dataset = frame.dropna(subset=[target] + features).copy()
    holdout_year = int(dataset["year"].max())
    train = dataset.loc[dataset["year"] < holdout_year].copy()
    test = dataset.loc[dataset["year"] == holdout_year].copy()
    if train.empty or test.empty:
        train = dataset.copy()
        test = dataset.copy()

    x_train = design_matrix([train[col].to_numpy(dtype=float) for col in features])
    y_train = train[target].to_numpy(dtype=float)
    weights = train.get("under5_population", pd.Series(1.0, index=train.index)).to_numpy(dtype=float)
    beta = weighted_least_squares(x_train, y_train, weights=weights, ridge=1e-5)

    def predict(block: pd.DataFrame) -> np.ndarray:
        x = design_matrix([block[col].to_numpy(dtype=float) for col in features])
        return x @ beta

    train_pred = predict(train)
    test_pred = predict(test)
    dataset["predicted_price"] = predict(dataset)
    write_parquet(dataset, output_panel)

    summary = {
        "holdout_year": holdout_year,
        "n_obs": int(len(dataset)),
        "rmse_train": rmse(y_train, train_pred),
        "rmse_test": rmse(test[target].to_numpy(dtype=float), test_pred),
        "r2_train": r2_score(y_train, train_pred),
        "r2_test": r2_score(test[target].to_numpy(dtype=float), test_pred),
        "coefficients": {
            "intercept": float(beta[0]),
            **{feature: float(beta[idx + 1]) for idx, feature in enumerate(features)},
        },
    }
    write_json(summary, output_json)
    return summary
