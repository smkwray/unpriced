from __future__ import annotations

import pandas as pd

from unpaidwork.errors import DataSchemaError


def require_columns(frame: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataSchemaError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )
