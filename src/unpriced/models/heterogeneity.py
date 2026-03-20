from __future__ import annotations

import pandas as pd


def summarize_by_group(frame: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    return frame.groupby(group_col, as_index=False)[value_col].mean()
