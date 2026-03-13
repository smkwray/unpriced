from __future__ import annotations

import pandas as pd


def apply_replacement_cost(frame: pd.DataFrame, hours_col: str, wage_col: str) -> pd.DataFrame:
    valued = frame.copy()
    valued["benchmark_replacement_cost"] = valued[hours_col] * valued[wage_col]
    return valued
