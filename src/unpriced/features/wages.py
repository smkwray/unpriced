from __future__ import annotations

import pandas as pd


def add_benchmark_wages(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["benchmark_childcare_wage"] = enriched["childcare_worker_wage"]
    preschool_teacher_wage = pd.to_numeric(enriched.get("oews_preschool_teacher_wage"), errors="coerce")
    if not isinstance(preschool_teacher_wage, pd.Series):
        preschool_teacher_wage = pd.Series([preschool_teacher_wage] * len(enriched), index=enriched.index, dtype="float64")
    enriched["specialist_childcare_wage"] = preschool_teacher_wage.fillna(
        pd.to_numeric(enriched["benchmark_childcare_wage"], errors="coerce")
    )
    return enriched
