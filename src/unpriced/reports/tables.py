from __future__ import annotations

import pandas as pd


def summarize_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        column
        for column in (
            "demand_sample_name",
            "demand_specification_profile",
            "state_fips",
            "year",
            "state_price_observation_status",
            "state_price_nowcast",
        )
        if column in frame.columns
    ]
    keep += [
        "p_baseline",
        "p_baseline_direct_care",
        "p_baseline_non_direct_care",
        "p_shadow_marginal",
        "p_shadow_marginal_lower",
        "p_shadow_marginal_upper",
        "p_shadow_marginal_direct_care",
        "wage_shadow_implied",
        "alpha",
        "p_alpha",
        "p_alpha_lower",
        "p_alpha_upper",
        "p_alpha_direct_care",
        "p_alpha_non_direct_care",
        "wage_alpha_implied",
    ]
    keep = [column for column in keep if column in frame.columns]
    return frame[keep].sort_values(["state_fips", "year", "alpha"]).reset_index(drop=True)
