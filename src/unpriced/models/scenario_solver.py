from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from unpriced.models.demand_iv import estimate_childcare_demand_summary, normalize_demand_mode
from unpriced.models.supply_curve import calibrate_supply_elasticity


@dataclass(frozen=True)
class ScenarioResult:
    alpha: float
    price: float


@dataclass(frozen=True)
class ScenarioInterval:
    point: float
    lower: float
    upper: float


REQUIRED_SCENARIO_COLUMNS = [
    "state_fips",
    "year",
    "state_price_index",
    "market_quantity_proxy",
    "unpaid_quantity_proxy",
    "benchmark_replacement_cost",
]
MARGINAL_ALPHA = 1e-6


def _demand_quantity(price: float, baseline_price: float, market_quantity: float, elasticity: float) -> float:
    return market_quantity * (price / baseline_price) ** (-elasticity)


def _supply_quantity(price: float, baseline_price: float, market_quantity: float, elasticity: float) -> float:
    return market_quantity * (price / baseline_price) ** elasticity


def _supply_quantity_piecewise(
    price: float,
    baseline_price: float,
    market_quantity: float,
    elasticity_below: float,
    elasticity_above: float,
) -> float:
    elasticity = elasticity_below if price <= baseline_price else elasticity_above
    return market_quantity * (price / baseline_price) ** elasticity


def resolve_solver_demand_elasticity(demand_summary: dict[str, float | int | bool] | None) -> tuple[float, float]:
    if demand_summary is None:
        raise ValueError("missing demand summary for childcare scenarios")
    signed = float(demand_summary.get("elasticity_at_mean", float("nan")))
    admissible = bool(demand_summary.get("economically_admissible", np.isfinite(signed) and signed <= 0))
    if not np.isfinite(signed):
        raise ValueError("missing finite childcare demand elasticity")
    if not admissible or signed > 0:
        raise ValueError(f"inadmissible childcare demand fit with positive price response ({signed:.6f})")
    return signed, max(abs(signed), 1e-6)


def solve_price(
    baseline_price: float,
    market_quantity: float,
    unpaid_quantity: float,
    demand_elasticity: float,
    supply_elasticity: float,
    alpha: float,
) -> float:
    low = max(1e-6, baseline_price * 0.25)
    high = baseline_price * (1.5 + alpha * max(1.0, unpaid_quantity / max(market_quantity, 1e-6)))
    for _ in range(80):
        mid = 0.5 * (low + high)
        excess = (
            _demand_quantity(mid, baseline_price, market_quantity, demand_elasticity)
            + alpha * unpaid_quantity
            - _supply_quantity(mid, baseline_price, market_quantity, supply_elasticity)
        )
        if excess > 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def solve_alpha_grid(
    baseline_price: float,
    market_quantity: float,
    unpaid_quantity: float,
    demand_elasticity: float,
    supply_elasticity: float,
    alphas: list[float],
) -> list[ScenarioResult]:
    return [
        ScenarioResult(
            alpha=alpha,
            price=solve_price(
                baseline_price,
                market_quantity,
                unpaid_quantity,
                demand_elasticity,
                supply_elasticity,
                alpha,
            ),
        )
        for alpha in alphas
    ]


def solve_price_piecewise_supply(
    baseline_price: float,
    market_quantity: float,
    unpaid_quantity: float,
    demand_elasticity: float,
    supply_elasticity_below: float,
    supply_elasticity_above: float,
    alpha: float,
) -> float:
    low = max(1e-6, baseline_price * 0.25)
    high = baseline_price * (1.5 + alpha * max(1.0, unpaid_quantity / max(market_quantity, 1e-6)))
    for _ in range(80):
        mid = 0.5 * (low + high)
        excess = (
            _demand_quantity(mid, baseline_price, market_quantity, demand_elasticity)
            + alpha * unpaid_quantity
            - _supply_quantity_piecewise(
                mid,
                baseline_price,
                market_quantity,
                supply_elasticity_below,
                supply_elasticity_above,
            )
        )
        if excess > 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def solve_alpha_grid_piecewise_supply(
    baseline_price: float,
    market_quantity: float,
    unpaid_quantity: float,
    demand_elasticity: float,
    supply_elasticity_below: float,
    supply_elasticity_above: float,
    alphas: list[float],
) -> list[ScenarioResult]:
    return [
        ScenarioResult(
            alpha=alpha,
            price=solve_price_piecewise_supply(
                baseline_price,
                market_quantity,
                unpaid_quantity,
                demand_elasticity,
                supply_elasticity_below,
                supply_elasticity_above,
                alpha,
            ),
        )
        for alpha in alphas
    ]


def prepare_childcare_scenario_inputs(state_frame: pd.DataFrame) -> pd.DataFrame:
    if state_frame.empty:
        return state_frame.copy()

    prepared = state_frame.copy()
    numeric_cols = [
        "state_price_index",
        "market_quantity_proxy",
        "unpaid_quantity_proxy",
        "benchmark_replacement_cost",
    ]
    for column in numeric_cols:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    valid = prepared.dropna(subset=REQUIRED_SCENARIO_COLUMNS).copy()
    valid = valid.loc[
        valid["state_price_index"].gt(0)
        & valid["market_quantity_proxy"].gt(0)
        & valid["unpaid_quantity_proxy"].ge(0)
    ].copy()
    return valid.reset_index(drop=True)


def _cluster_bootstrap_by_state(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    states = sorted(frame["state_fips"].dropna().astype(str).unique().tolist())
    if not states:
        return frame.copy()
    sampled = rng.choice(states, size=len(states), replace=True)
    draws = []
    for draw_number, state_fips in enumerate(sampled):
        subset = frame.loc[frame["state_fips"].astype(str) == state_fips].copy()
        subset["_bootstrap_draw"] = draw_number
        subset["_bootstrap_state"] = str(state_fips)
        draws.append(subset)
    if not draws:
        return frame.copy()
    return pd.concat(draws, ignore_index=True)


def bootstrap_childcare_intervals(
    state_frame: pd.DataFrame,
    county_frame: pd.DataFrame,
    scenarios: pd.DataFrame,
    demand_mode: str = "broad_complete",
    demand_specification_profile: str | None = None,
    n_boot: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    if scenarios.empty:
        return scenarios.copy()

    rng = np.random.default_rng(seed)
    state = state_frame.dropna(
        subset=[
            "unpaid_childcare_hours",
            "state_price_index",
            "outside_option_wage",
            "parent_employment_rate",
            "single_parent_share",
            "median_income",
            "unemployment_rate",
        ]
    ).reset_index(drop=True)
    county = county_frame.dropna(subset=["provider_density", "annual_price"]).reset_index(drop=True)
    if len(state) < 10 or len(county) < 10:
        enriched = scenarios.copy()
        for column in (
            "p_shadow_marginal_lower",
            "p_shadow_marginal_upper",
            "p_alpha_lower",
            "p_alpha_upper",
        ):
            enriched[column] = enriched["p_alpha"] if "p_alpha" in column else enriched["p_shadow_marginal"]
        return enriched

    shadow_draws = np.full((len(scenarios), n_boot), np.nan)
    alpha_draws = np.full((len(scenarios), n_boot), np.nan)
    accepted_draws = 0
    for draw in range(n_boot):
        state_sample = _cluster_bootstrap_by_state(state, rng)
        county_sample = _cluster_bootstrap_by_state(county, rng)
        demand_summary, _ = estimate_childcare_demand_summary(
            state_sample.copy(),
            mode=normalize_demand_mode(demand_mode),
            specification_profile=demand_specification_profile,
        )
        try:
            _, demand_elasticity = resolve_solver_demand_elasticity(demand_summary)
        except ValueError:
            continue
        supply_elasticity = calibrate_supply_elasticity(county_sample.copy())
        accepted_draws += 1
        for row_number, row in enumerate(scenarios.itertuples(index=False)):
            shadow_draws[row_number, draw] = solve_price(
                row.p_baseline,
                row.market_quantity_proxy,
                row.unpaid_quantity_proxy,
                demand_elasticity,
                supply_elasticity,
                MARGINAL_ALPHA,
            )
            alpha_draws[row_number, draw] = solve_price(
                row.p_baseline,
                row.market_quantity_proxy,
                row.unpaid_quantity_proxy,
                demand_elasticity,
                supply_elasticity,
                row.alpha,
            )

    enriched = scenarios.copy()
    if accepted_draws == 0:
        enriched["p_shadow_marginal_lower"] = enriched["p_shadow_marginal"]
        enriched["p_shadow_marginal_upper"] = enriched["p_shadow_marginal"]
        enriched["p_alpha_lower"] = enriched["p_alpha"]
        enriched["p_alpha_upper"] = enriched["p_alpha"]
        return enriched
    enriched["p_shadow_marginal_lower"] = np.nanquantile(shadow_draws, 0.1, axis=1)
    enriched["p_shadow_marginal_upper"] = np.nanquantile(shadow_draws, 0.9, axis=1)
    enriched["p_alpha_lower"] = np.nanquantile(alpha_draws, 0.1, axis=1)
    enriched["p_alpha_upper"] = np.nanquantile(alpha_draws, 0.9, axis=1)
    return enriched


def summarize_childcare_scenario_diagnostics(
    scenarios: pd.DataFrame,
    skipped_state_rows: int = 0,
    demand_summary: dict[str, float] | None = None,
    demand_sample_name: str | None = None,
    demand_sample_selection_reason: str | None = None,
) -> dict[str, float | int | bool]:
    if scenarios.empty:
        return {
            "scenario_rows": 0,
            "scenario_states": 0,
            "scenario_year_min": 0,
            "scenario_year_max": 0,
            "skipped_state_rows": int(skipped_state_rows),
            "zero_width_shadow_count": 0,
            "zero_width_alpha_count": 0,
            "zero_unpaid_quantity_count": 0,
            "shadow_width_p10": 0.0,
            "shadow_width_p50": 0.0,
            "shadow_width_p90": 0.0,
            "alpha_width_p10": 0.0,
            "alpha_width_p50": 0.0,
            "alpha_width_p90": 0.0,
            "demand_first_stage_r2": float(demand_summary.get("first_stage_r2", 0.0)) if demand_summary else 0.0,
            "demand_first_stage_near_perfect": bool(
                demand_summary and float(demand_summary.get("first_stage_r2", 0.0)) >= 0.999
            ),
            "demand_mode": str(demand_summary.get("mode", "baseline")) if demand_summary else "baseline",
            "demand_n_states": int(demand_summary.get("n_states", 0)) if demand_summary else 0,
            "demand_sample_name": demand_sample_name or (str(demand_summary.get("mode", "broad_complete")) if demand_summary else "broad_complete"),
            "demand_sample_n_obs": int(demand_summary.get("n_obs", 0)) if demand_summary else 0,
            "demand_sample_n_states": int(demand_summary.get("n_states", 0)) if demand_summary else 0,
            "demand_sample_n_years": int(demand_summary.get("n_years", 0)) if demand_summary else 0,
            "demand_sample_selection_reason": demand_sample_selection_reason or "none",
            "demand_specification_profile": str(demand_summary.get("specification_profile", "full_controls")) if demand_summary else "full_controls",
            "demand_elasticity_at_mean": float(demand_summary.get("elasticity_at_mean", float("nan"))) if demand_summary else float("nan"),
            "demand_economically_admissible": bool(demand_summary.get("economically_admissible", False)) if demand_summary else False,
            "solver_demand_elasticity_magnitude": 0.0,
            "demand_loo_state_fips_r2": float(demand_summary.get("loo_state_fips_r2", float("nan"))) if demand_summary else float("nan"),
            "demand_loo_year_r2": float(demand_summary.get("loo_year_r2", float("nan"))) if demand_summary else float("nan"),
            "scenario_price_observed_support_rows": 0,
            "scenario_price_pre_support_rows": 0,
            "scenario_price_nowcast_rows": 0,
            "scenario_contains_nowcast_rows": False,
            "scenario_price_nowcast_years": [],
            "baseline_price_p50": 0.0,
            "baseline_direct_care_price_p50": 0.0,
            "baseline_non_direct_care_price_p50": 0.0,
            "baseline_implied_wage_p50": 0.0,
            "baseline_direct_care_labor_share_p50": 0.0,
            "baseline_direct_care_clip_binding_row_share": 0.0,
            "alpha_50_price_p50": 0.0,
            "alpha_50_direct_care_price_p50": 0.0,
            "alpha_50_non_direct_care_price_p50": 0.0,
            "alpha_50_implied_wage_p50": 0.0,
            "bootstrap_resampling_unit": "state_fips_cluster",
        }

    shadow_width = pd.to_numeric(
        scenarios["p_shadow_marginal_upper"] - scenarios["p_shadow_marginal_lower"],
        errors="coerce",
    )
    alpha_width = pd.to_numeric(
        scenarios["p_alpha_upper"] - scenarios["p_alpha_lower"],
        errors="coerce",
    )
    zero_unpaid = pd.to_numeric(scenarios["unpaid_quantity_proxy"], errors="coerce").fillna(0.0).eq(0)
    first_stage_r2 = float(demand_summary.get("first_stage_r2", 0.0)) if demand_summary else 0.0
    price_status = scenarios.get("state_price_observation_status")
    if price_status is None:
        observed_support_rows = 0
        pre_support_rows = 0
        nowcast_rows = 0
    else:
        price_status = price_status.fillna("missing").astype(str)
        observed_support_rows = int(price_status.eq("observed_ndcp_support").sum())
        pre_support_rows = int(price_status.eq("pre_ndcp_support_gap").sum())
        nowcast_rows = int(price_status.eq("post_ndcp_nowcast").sum())
    nowcast_years = []
    if nowcast_rows > 0:
        nowcast_mask = scenarios.get("state_price_nowcast")
        if nowcast_mask is None:
            nowcast_mask = scenarios["state_price_observation_status"].eq("post_ndcp_nowcast")
        nowcast_years = sorted(
            pd.to_numeric(scenarios.loc[nowcast_mask.fillna(False).astype(bool), "year"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    baseline_price = pd.to_numeric(scenarios.get("p_baseline"), errors="coerce")
    baseline_direct_care_price = pd.to_numeric(scenarios.get("p_baseline_direct_care"), errors="coerce")
    baseline_non_direct_care_price = pd.to_numeric(scenarios.get("p_baseline_non_direct_care"), errors="coerce")
    baseline_implied_wage = pd.to_numeric(scenarios.get("wage_baseline_implied"), errors="coerce")
    baseline_direct_care_labor_share_series = scenarios.get("direct_care_labor_share")
    baseline_direct_care_labor_share = (
        pd.to_numeric(baseline_direct_care_labor_share_series, errors="coerce")
        if baseline_direct_care_labor_share_series is not None
        else pd.Series(dtype=float)
    )
    baseline_direct_care_clip_binding_series = scenarios.get("direct_care_price_clip_binding")
    baseline_direct_care_clip_binding = (
        pd.to_numeric(baseline_direct_care_clip_binding_series, errors="coerce")
        if baseline_direct_care_clip_binding_series is not None
        else pd.Series(dtype=float)
    )
    alpha_half = scenarios.loc[pd.to_numeric(scenarios["alpha"], errors="coerce").round(4).eq(0.5)].copy()
    alpha_half_price = pd.to_numeric(alpha_half.get("p_alpha"), errors="coerce")
    alpha_half_direct_care_price = pd.to_numeric(alpha_half.get("p_alpha_direct_care"), errors="coerce")
    alpha_half_non_direct_care_price = pd.to_numeric(alpha_half.get("p_alpha_non_direct_care"), errors="coerce")
    alpha_half_implied_wage = pd.to_numeric(alpha_half.get("wage_alpha_implied"), errors="coerce")
    solver_series = scenarios.get("solver_demand_elasticity_magnitude")
    solver_demand_elasticity = (
        pd.to_numeric(solver_series, errors="coerce") if solver_series is not None else pd.Series(dtype=float)
    )

    return {
        "scenario_rows": int(len(scenarios)),
        "scenario_states": int(scenarios["state_fips"].nunique()),
        "scenario_year_min": int(pd.to_numeric(scenarios["year"], errors="coerce").min()),
        "scenario_year_max": int(pd.to_numeric(scenarios["year"], errors="coerce").max()),
        "skipped_state_rows": int(skipped_state_rows),
        "zero_width_shadow_count": int(shadow_width.fillna(0.0).eq(0).sum()),
        "zero_width_alpha_count": int(alpha_width.fillna(0.0).eq(0).sum()),
        "zero_unpaid_quantity_count": int(zero_unpaid.sum()),
        "shadow_width_p10": float(shadow_width.quantile(0.1)),
        "shadow_width_p50": float(shadow_width.quantile(0.5)),
        "shadow_width_p90": float(shadow_width.quantile(0.9)),
        "alpha_width_p10": float(alpha_width.quantile(0.1)),
        "alpha_width_p50": float(alpha_width.quantile(0.5)),
        "alpha_width_p90": float(alpha_width.quantile(0.9)),
        "demand_first_stage_r2": first_stage_r2,
        "demand_first_stage_near_perfect": first_stage_r2 >= 0.999,
        "demand_mode": str(demand_summary.get("mode", "baseline")) if demand_summary else "baseline",
        "demand_n_states": int(demand_summary.get("n_states", 0)) if demand_summary else 0,
        "demand_sample_name": demand_sample_name or (str(demand_summary.get("mode", "broad_complete")) if demand_summary else "broad_complete"),
        "demand_sample_n_obs": int(demand_summary.get("n_obs", 0)) if demand_summary else 0,
        "demand_sample_n_states": int(demand_summary.get("n_states", 0)) if demand_summary else 0,
        "demand_sample_n_years": int(demand_summary.get("n_years", 0)) if demand_summary else 0,
        "demand_sample_selection_reason": demand_sample_selection_reason or "none",
        "demand_specification_profile": str(demand_summary.get("specification_profile", "full_controls")) if demand_summary else "full_controls",
        "demand_elasticity_at_mean": float(demand_summary.get("elasticity_at_mean", float("nan"))) if demand_summary else float("nan"),
        "demand_economically_admissible": bool(demand_summary.get("economically_admissible", False)) if demand_summary else False,
        "solver_demand_elasticity_magnitude": float(solver_demand_elasticity.median()) if solver_demand_elasticity.notna().any() else 0.0,
        "demand_loo_state_fips_r2": float(demand_summary.get("loo_state_fips_r2", float("nan"))) if demand_summary else float("nan"),
        "demand_loo_year_r2": float(demand_summary.get("loo_year_r2", float("nan"))) if demand_summary else float("nan"),
        "scenario_price_observed_support_rows": observed_support_rows,
        "scenario_price_pre_support_rows": pre_support_rows,
        "scenario_price_nowcast_rows": nowcast_rows,
        "scenario_contains_nowcast_rows": nowcast_rows > 0,
        "scenario_price_nowcast_years": nowcast_years,
        "baseline_price_p50": float(baseline_price.median()) if baseline_price.notna().any() else 0.0,
        "baseline_direct_care_price_p50": float(baseline_direct_care_price.median()) if baseline_direct_care_price.notna().any() else 0.0,
        "baseline_non_direct_care_price_p50": float(baseline_non_direct_care_price.median()) if baseline_non_direct_care_price.notna().any() else 0.0,
        "baseline_implied_wage_p50": float(baseline_implied_wage.median()) if baseline_implied_wage.notna().any() else 0.0,
        "baseline_direct_care_labor_share_p50": float(baseline_direct_care_labor_share.median()) if baseline_direct_care_labor_share.notna().any() else 0.0,
        "baseline_direct_care_clip_binding_row_share": float(baseline_direct_care_clip_binding.fillna(0.0).mean()) if baseline_direct_care_clip_binding.notna().any() else 0.0,
        "alpha_50_price_p50": float(alpha_half_price.median()) if alpha_half_price.notna().any() else 0.0,
        "alpha_50_direct_care_price_p50": float(alpha_half_direct_care_price.median()) if alpha_half_direct_care_price.notna().any() else 0.0,
        "alpha_50_non_direct_care_price_p50": float(alpha_half_non_direct_care_price.median()) if alpha_half_non_direct_care_price.notna().any() else 0.0,
        "alpha_50_implied_wage_p50": float(alpha_half_implied_wage.median()) if alpha_half_implied_wage.notna().any() else 0.0,
        "bootstrap_resampling_unit": "state_fips_cluster",
    }
