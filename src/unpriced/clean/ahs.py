from __future__ import annotations

import pandas as pd

from unpriced.config import ProjectPaths
from unpriced.storage import read_parquet, write_parquet
from unpriced.validate import require_columns


def build_job_panel(paths: ProjectPaths) -> pd.DataFrame:
    frame = read_parquet(paths.interim / "ahs" / "ahs.parquet")
    require_columns(
        frame,
        [
            "job_id",
            "cbsa_code",
            "year",
            "job_type",
            "job_cost",
            "job_diy",
            "household_income",
            "home_value",
            "storm_exposure",
        ],
        "AHS",
    )
    frame["job_diy"] = frame["job_diy"].astype(int)
    frame["cbsa_code"] = frame["cbsa_code"].astype(str).str.zfill(5)
    write_parquet(frame, paths.processed / "ahs_jobs_clean.parquet")
    return frame
