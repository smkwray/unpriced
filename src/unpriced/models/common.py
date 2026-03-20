from __future__ import annotations

import numpy as np


def design_matrix(values: list[np.ndarray]) -> np.ndarray:
    columns = [np.ones(len(values[0]))] + values
    return np.column_stack(columns)


def weighted_least_squares(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None, ridge: float = 1e-8
) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(y))
    w = np.sqrt(weights)[:, None]
    xw = x * w
    yw = y * w.ravel()
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    system = xw.T @ xw + penalty
    rhs = xw.T @ yw
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(system) @ rhs


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    sse = float(((y - yhat) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    if sst == 0:
        return 1.0
    return 1.0 - sse / sst


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))
