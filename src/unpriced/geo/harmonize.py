from __future__ import annotations

import pandas as pd


def annualize(frame: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    standardized = frame.copy()
    standardized[year_col] = standardized[year_col].astype(int)
    return standardized


def add_sensitivity_flag(frame: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    flagged = frame.copy()
    flagged["is_sensitivity_year"] = flagged[year_col].eq(2020)
    return flagged
