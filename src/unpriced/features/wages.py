from __future__ import annotations

import pandas as pd


def add_benchmark_wages(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["benchmark_childcare_wage"] = enriched["childcare_worker_wage"]
    return enriched
