from __future__ import annotations

import numpy as np
import pandas as pd

from unpaidwork.config import ProjectPaths
from unpaidwork.geo.harmonize import annualize
from unpaidwork.storage import read_parquet, write_parquet
from unpaidwork.validate import require_columns


def build_county_year_panel(paths: ProjectPaths) -> pd.DataFrame:
    frame = read_parquet(paths.interim / "ndcp" / "ndcp.parquet")
    require_columns(
        frame,
        ["county_fips", "state_fips", "year", "annual_price", "imputed_flag", "sample_weight"],
        "NDCP",
    )
    frame = annualize(frame)
    grouped = (
        frame.groupby(["county_fips", "state_fips", "year"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "annual_price": float(np.average(g["annual_price"], weights=g["sample_weight"])),
                    "imputed_share": float(np.average(g["imputed_flag"], weights=g["sample_weight"])),
                    "price_rows": int(len(g)),
                }
            )
        )
        .reset_index(drop=True)
    )
    write_parquet(grouped, paths.processed / "ndcp_county_year_clean.parquet")
    return grouped
