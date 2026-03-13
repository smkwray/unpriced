from __future__ import annotations

import pandas as pd

from unpaidwork.ingest.licensing import ingest as ingest_licensing
from unpaidwork.storage import read_parquet


def test_licensing_sample_ingest_writes_expected_columns(project_paths):
    result = ingest_licensing(project_paths, sample=True, refresh=True)

    assert result.normalized_path.exists()
    frame = read_parquet(result.normalized_path)
    assert {
        "state_fips",
        "year",
        "center_infant_ratio",
        "center_toddler_ratio",
        "center_infant_group_size",
        "center_toddler_group_size",
        "shock_label",
    } <= set(frame.columns)


def test_licensing_real_ingest_normalizes_curated_csv(project_paths):
    raw_path = project_paths.raw / "licensing" / "licensing_supply_shocks.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "state_fips": "6",
                "year": 2021,
                "center_infant_ratio": 3,
                "center_toddler_ratio": 6,
                "center_infant_group_size": 6,
                "center_toddler_group_size": 12,
                "shock_label": "demo",
            }
        ]
    ).to_csv(raw_path, index=False)

    result = ingest_licensing(project_paths, sample=False, refresh=True)

    frame = read_parquet(result.normalized_path)
    assert result.normalized_path.name == "licensing_supply_shocks.parquet"
    assert frame.loc[0, "state_fips"] == "06"
    assert int(frame.loc[0, "year"]) == 2021
