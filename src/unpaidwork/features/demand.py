from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_average(group: pd.DataFrame, value_col: str) -> float:
    values = pd.to_numeric(group[value_col], errors="coerce")
    weights = pd.to_numeric(group["under5_population"], errors="coerce").fillna(0.0)
    valid = values.notna() & weights.gt(0)
    if valid.any():
        return float(np.average(values.loc[valid], weights=weights.loc[valid]))
    if values.notna().any():
        return float(values.loc[values.notna()].mean())
    return float("nan")


def aggregate_counties_to_state(frame: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(["state_fips", "year"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    col: _weighted_average(g, col)
                    for col in value_cols
                }
                | {"under5_population": float(g["under5_population"].sum())}
            )
        )
        .reset_index(drop=True)
    )
