from __future__ import annotations

import pandas as pd

from unpriced.config import ProjectPaths
from unpriced.storage import read_parquet, write_parquet
from unpriced.validate import require_columns


def _nearest_year_map(target_years: list[int], available_years: list[int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    ordered = sorted(set(available_years))
    for target in sorted(set(target_years)):
        mapping[target] = min(ordered, key=lambda year: (abs(year - target), year))
    return mapping


def build_county_labor_panel(paths: ProjectPaths) -> pd.DataFrame:
    frame = read_parquet(paths.interim / "qcew" / "qcew.parquet")
    require_columns(
        frame,
        [
            "county_fips",
            "state_fips",
            "year",
            "childcare_worker_wage",
            "outside_option_wage",
            "employment",
        ],
        "QCEW",
    )
    keep = [
        "county_fips",
        "state_fips",
        "year",
        "childcare_worker_wage",
        "outside_option_wage",
        "employment",
    ]
    panel = frame[keep].copy()
    panel["childcare_worker_wage_source"] = panel["childcare_worker_wage"].notna().map(
        lambda observed: "qcew_county_observed" if observed else "missing"
    )
    panel["outside_option_wage_source"] = panel["outside_option_wage"].notna().map(
        lambda observed: "input_series" if observed else "missing"
    )
    oews_path = paths.interim / "oews" / "oews.parquet"
    if oews_path.exists():
        oews = read_parquet(oews_path)
        require_columns(
            oews,
            ["state_fips", "year", "oews_childcare_worker_wage", "oews_outside_option_wage"],
            "OEWS",
        )
        panel = panel.merge(
            oews[["state_fips", "year", "oews_childcare_worker_wage", "oews_outside_option_wage"]],
            on=["state_fips", "year"],
            how="left",
        )
        panel["childcare_worker_wage"] = panel["childcare_worker_wage"].fillna(
            panel["oews_childcare_worker_wage"]
        )
        panel.loc[
            panel["childcare_worker_wage_source"].eq("missing") & panel["oews_childcare_worker_wage"].notna(),
            "childcare_worker_wage_source",
        ] = "oews_state_observed"
        panel["outside_option_wage"] = panel["oews_outside_option_wage"].where(
            panel["oews_outside_option_wage"].notna(),
            panel["outside_option_wage"],
        )
        panel.loc[panel["oews_outside_option_wage"].notna(), "outside_option_wage_source"] = "oews_state_observed"
        oews_ratio_rows = panel.loc[
            panel["oews_childcare_worker_wage"].notna()
            & panel["oews_outside_option_wage"].notna()
            & panel["oews_childcare_worker_wage"].gt(0)
        ].copy()
        if not oews_ratio_rows.empty:
            oews_ratio_rows["oews_outside_option_ratio"] = (
                pd.to_numeric(oews_ratio_rows["oews_outside_option_wage"], errors="coerce")
                / pd.to_numeric(oews_ratio_rows["oews_childcare_worker_wage"], errors="coerce")
            )
            ratio_map = oews_ratio_rows.loc[:, ["state_fips", "year", "oews_outside_option_ratio"]].drop_duplicates()
            panel = panel.merge(ratio_map, on=["state_fips", "year"], how="left")
            same_year_missing = panel["outside_option_wage"].isna() & panel["oews_outside_option_ratio"].notna()
            panel.loc[same_year_missing, "outside_option_wage"] = (
                pd.to_numeric(panel.loc[same_year_missing, "childcare_worker_wage"], errors="coerce")
                * pd.to_numeric(panel.loc[same_year_missing, "oews_outside_option_ratio"], errors="coerce")
            )
            panel.loc[same_year_missing, "outside_option_wage_source"] = "oews_state_ratio_same_year"

            target_years = (
                pd.to_numeric(panel.loc[panel["outside_option_wage"].isna(), "year"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            available_years = (
                pd.to_numeric(ratio_map["year"], errors="coerce").dropna().astype(int).unique().tolist()
            )
            if target_years and available_years:
                nearest = _nearest_year_map(target_years, available_years)
                nearest_frame = pd.DataFrame(
                    {"year": list(nearest.keys()), "oews_ratio_year": list(nearest.values())}
                )
                ratio_by_nearest = nearest_frame.merge(
                    ratio_map.rename(
                        columns={"year": "oews_ratio_year", "oews_outside_option_ratio": "oews_outside_option_ratio_nearest"}
                    ),
                    on="oews_ratio_year",
                    how="left",
                ).drop(columns=["oews_ratio_year"])
                panel = panel.merge(ratio_by_nearest, on=["state_fips", "year"], how="left")
                nearest_missing = (
                    panel["outside_option_wage"].isna()
                    & panel["oews_outside_option_ratio_nearest"].notna()
                )
                panel.loc[nearest_missing, "outside_option_wage"] = (
                    pd.to_numeric(panel.loc[nearest_missing, "childcare_worker_wage"], errors="coerce")
                    * pd.to_numeric(panel.loc[nearest_missing, "oews_outside_option_ratio_nearest"], errors="coerce")
                )
                panel.loc[nearest_missing, "outside_option_wage_source"] = "oews_state_ratio_nearest_year"

            national_ratio = float(pd.to_numeric(oews_ratio_rows["oews_outside_option_ratio"], errors="coerce").median())
            national_missing = panel["outside_option_wage"].isna() & panel["childcare_worker_wage"].notna()
            panel.loc[national_missing, "outside_option_wage"] = (
                pd.to_numeric(panel.loc[national_missing, "childcare_worker_wage"], errors="coerce")
                * national_ratio
            )
            panel.loc[national_missing, "outside_option_wage_source"] = "oews_national_ratio_median"
            panel = panel.drop(
                columns=[
                    col
                    for col in ("oews_outside_option_ratio", "oews_outside_option_ratio_nearest")
                    if col in panel.columns
                ]
            )
    write_parquet(panel, paths.processed / "county_labor_panel.parquet")
    return panel
