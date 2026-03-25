from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from unpriced.models.demand_iv import (
    SPECIFICATION_PROFILES,
    estimate_childcare_demand_summary,
    normalize_demand_mode,
    normalize_specification_profile,
)
from unpriced.models.supply_curve import calibrate_supply_elasticity


@dataclass(frozen=True)
class ScenarioResult:
    alpha: float
    price: float
    solver_status: str = "ok"
    solver_iterations: int = 0
    solver_expansion_steps: int = 0
    solver_low: float = float("nan")
    solver_high: float = float("nan")


@dataclass(frozen=True)
class SolverMetadata:
    price: float
    status: str
    iterations: int
    expansion_steps: int
    bracket_low: float
    bracket_high: float
    low_excess: float
    high_excess: float


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
SOLVER_BRACKET_EXPANSION_FACTOR = 2.0
SOLVER_MAX_BRACKET_STEPS = 12
SOLVER_ITERATIONS = 80
PRICE_EPSILON = 1e-6
MAX_INITIAL_PRICE_MULTIPLIER = 128.0


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


def _supply_quantity_dual_shift(
    price: float,
    baseline_price: float,
    market_quantity: float,
    alpha: float,
    elasticity: float,
    kappa_q: float,
    kappa_c: float,
) -> float:
    quantity_shift = float(np.exp(kappa_q * alpha))
    cost_shift = float(np.exp(kappa_c * alpha))
    adjusted_baseline_price = baseline_price * cost_shift
    return market_quantity * quantity_shift * (price / adjusted_baseline_price) ** elasticity


def dual_shift_zero_price_frontier(
    market_quantity: float,
    unpaid_quantity: float,
    supply_elasticity: float,
    kappa_c: float,
) -> float:
    if market_quantity <= 0:
        raise ValueError("market_quantity must be positive")
    return float(unpaid_quantity / market_quantity + supply_elasticity * kappa_c)


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


def _solve_equilibrium_with_metadata(
    baseline_price: float,
    market_quantity: float,
    alpha: float,
    demand_function,
    supply_function,
    return_price_only: bool = False,
) -> SolverMetadata | float:
    if baseline_price <= 0:
        raise ValueError("baseline_price must be positive")
    if market_quantity <= 0:
        raise ValueError("market_quantity must be positive")

    def excess_at(price: float) -> float:
        return float(demand_function(price) - supply_function(price))

    initial_low = max(PRICE_EPSILON, baseline_price * 0.25)
    initial_high = baseline_price * (1.5 + alpha * 2.0)
    low = initial_low
    high = initial_high
    expansion_steps = 0
    low_excess = float(excess_at(low))
    high_excess = float(excess_at(high))
    if low_excess == 0.0:
        metadata = SolverMetadata(
            price=low,
            status="root_at_low",
            iterations=0,
            expansion_steps=0,
            bracket_low=low,
            bracket_high=low,
            low_excess=low_excess,
            high_excess=low_excess,
        )
        return low if return_price_only else metadata
    if high_excess == 0.0:
        metadata = SolverMetadata(
            price=high,
            status="root_at_high",
            iterations=0,
            expansion_steps=0,
            bracket_low=high,
            bracket_high=high,
            low_excess=high_excess,
            high_excess=high_excess,
        )
        return high if return_price_only else metadata

    max_high = max(initial_high, baseline_price * MAX_INITIAL_PRICE_MULTIPLIER)
    for expansion_steps in range(SOLVER_MAX_BRACKET_STEPS + 1):
        if low_excess * high_excess < 0:
            break
        if expansion_steps == SOLVER_MAX_BRACKET_STEPS:
            raise RuntimeError(
                f"solver failed to bracket root for baseline_price={baseline_price}, "
                f"market_quantity={market_quantity}, alpha={alpha}, low_excess={low_excess:.6g}, "
                f"high_excess={high_excess:.6g}"
            )
        next_low = max(PRICE_EPSILON, low / SOLVER_BRACKET_EXPANSION_FACTOR)
        next_high = min(high * SOLVER_BRACKET_EXPANSION_FACTOR, max_high)
        if next_low == low and next_high == high:
            break
        low, high = next_low, next_high
        low_excess = float(excess_at(low))
        high_excess = float(excess_at(high))

    # If high maxed out and still not bracketed, treat as failed before expensive bisection.
    if low_excess * high_excess > 0:
        raise RuntimeError(
            f"solver failed to bracket root for baseline_price={baseline_price}, market_quantity={market_quantity}, alpha={alpha}"
        )

    for _ in range(SOLVER_ITERATIONS):
        mid = 0.5 * (low + high)
        mid_excess = float(excess_at(mid))
        if mid_excess > 0:
            low = mid
            low_excess = mid_excess
        else:
            high = mid
            high_excess = mid_excess

    price = 0.5 * (low + high)
    metadata = SolverMetadata(
        price=float(price),
        status="converged",
        iterations=SOLVER_ITERATIONS,
        expansion_steps=expansion_steps,
        bracket_low=float(low),
        bracket_high=float(high),
        low_excess=float(low_excess),
        high_excess=float(high_excess),
    )
    return price if return_price_only else metadata


def _solve_with_metadata(
    baseline_price: float,
    market_quantity: float,
    unpaid_quantity: float,
    demand_elasticity: float,
    supply_function,
    alpha: float,
    return_price_only: bool = False,
) -> SolverMetadata | float:
    if not np.isfinite(unpaid_quantity):
        raise ValueError("unpaid_quantity must be finite")
    def demand_function(price):
        return _demand_quantity(price, baseline_price, market_quantity, demand_elasticity) + alpha * unpaid_quantity

    return _solve_equilibrium_with_metadata(
        baseline_price,
        market_quantity,
        alpha,
        demand_function,
        supply_function,
        return_price_only=return_price_only,
    )


def solve_price(
    baseline_price: float,
    market_quantity: float,
    unpaid_quantity: float,
    demand_elasticity: float,
    supply_elasticity: float,
    alpha: float,
    return_metadata: bool = False,
) -> float | SolverMetadata:
    return _solve_with_metadata(
        baseline_price,
        market_quantity,
        unpaid_quantity,
        demand_elasticity,
        lambda price: _supply_quantity(price, baseline_price, market_quantity, supply_elasticity),
        alpha,
        return_price_only=not return_metadata,
    )


def solve_alpha_grid(
    baseline_price: float,
    market_quantity: float,
    unpaid_quantity: float,
    demand_elasticity: float,
    supply_elasticity: float,
    alphas: list[float],
    return_metadata: bool = False,
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for alpha in alphas:
        solve_result = solve_price(
            baseline_price,
            market_quantity,
            unpaid_quantity,
            demand_elasticity,
            supply_elasticity,
            alpha,
            return_metadata=return_metadata,
        )
        if isinstance(solve_result, SolverMetadata):
            results.append(
                ScenarioResult(
                    alpha=alpha,
                    price=solve_result.price,
                    solver_status=solve_result.status,
                    solver_iterations=solve_result.iterations,
                    solver_expansion_steps=solve_result.expansion_steps,
                    solver_low=solve_result.bracket_low,
                    solver_high=solve_result.bracket_high,
                )
            )
        else:
            results.append(
                ScenarioResult(
                    alpha=alpha,
                    price=float(solve_result),
                )
            )
    return results


def solve_price_piecewise_supply(
    baseline_price: float,
    market_quantity: float,
    unpaid_quantity: float,
    demand_elasticity: float,
    supply_elasticity_below: float,
    supply_elasticity_above: float,
    alpha: float,
    return_metadata: bool = False,
) -> float | SolverMetadata:
    return _solve_with_metadata(
        baseline_price,
        market_quantity,
        unpaid_quantity,
        demand_elasticity,
        lambda price: _supply_quantity_piecewise(
            price,
            baseline_price,
            market_quantity,
            supply_elasticity_below,
            supply_elasticity_above,
        ),
        alpha,
        return_price_only=not return_metadata,
    )


def solve_alpha_grid_piecewise_supply(
    baseline_price: float,
    market_quantity: float,
    unpaid_quantity: float,
    demand_elasticity: float,
    supply_elasticity_below: float,
    supply_elasticity_above: float,
    alphas: list[float],
    return_metadata: bool = False,
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for alpha in alphas:
        solve_result = solve_price_piecewise_supply(
            baseline_price,
            market_quantity,
            unpaid_quantity,
            demand_elasticity,
            supply_elasticity_below,
            supply_elasticity_above,
            alpha,
            return_metadata=return_metadata,
        )
        if isinstance(solve_result, SolverMetadata):
            results.append(
                ScenarioResult(
                    alpha=alpha,
                    price=solve_result.price,
                    solver_status=solve_result.status,
                    solver_iterations=solve_result.iterations,
                    solver_expansion_steps=solve_result.expansion_steps,
                    solver_low=solve_result.bracket_low,
                    solver_high=solve_result.bracket_high,
                )
            )
        else:
            results.append(
                ScenarioResult(
                    alpha=alpha,
                    price=float(solve_result),
                )
            )
    return results


def solve_price_dual_shift(
    baseline_price: float,
    market_quantity: float,
    unpaid_quantity: float,
    demand_elasticity: float,
    supply_elasticity: float,
    alpha: float,
    kappa_q: float,
    kappa_c: float,
    return_metadata: bool = False,
) -> float | SolverMetadata:
    if not np.isfinite(unpaid_quantity):
        raise ValueError("unpaid_quantity must be finite")
    def demand_function(price):
        return _demand_quantity(price, baseline_price, market_quantity, demand_elasticity) + alpha * unpaid_quantity

    def supply_function(price):
        return _supply_quantity_dual_shift(
            price=price,
            baseline_price=baseline_price,
            market_quantity=market_quantity,
            alpha=alpha,
            elasticity=supply_elasticity,
            kappa_q=kappa_q,
            kappa_c=kappa_c,
        )
    return _solve_equilibrium_with_metadata(
        baseline_price,
        market_quantity,
        alpha,
        demand_function,
        supply_function,
        return_price_only=not return_metadata,
    )


def solve_alpha_grid_dual_shift(
    baseline_price: float,
    market_quantity: float,
    unpaid_quantity: float,
    demand_elasticity: float,
    supply_elasticity: float,
    alphas: list[float],
    kappa_q: float,
    kappa_c: float,
    return_metadata: bool = False,
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for alpha in alphas:
        solve_result = solve_price_dual_shift(
            baseline_price=baseline_price,
            market_quantity=market_quantity,
            unpaid_quantity=unpaid_quantity,
            demand_elasticity=demand_elasticity,
            supply_elasticity=supply_elasticity,
            alpha=alpha,
            kappa_q=kappa_q,
            kappa_c=kappa_c,
            return_metadata=return_metadata,
        )
        if isinstance(solve_result, SolverMetadata):
            results.append(
                ScenarioResult(
                    alpha=alpha,
                    price=solve_result.price,
                    solver_status=solve_result.status,
                    solver_iterations=solve_result.iterations,
                    solver_expansion_steps=solve_result.expansion_steps,
                    solver_low=solve_result.bracket_low,
                    solver_high=solve_result.bracket_high,
                )
            )
        else:
            results.append(ScenarioResult(alpha=alpha, price=float(solve_result)))
    return results


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
) -> tuple[pd.DataFrame, dict[str, float | int | bool | dict[str, int] | str]]:
    bootstrap_meta: dict[str, float | int | bool | dict[str, int] | str] = {
        "bootstrap_draws_requested": int(n_boot),
        "bootstrap_draws_accepted": 0,
        "bootstrap_draws_rejected": 0,
        "bootstrap_acceptance_rate": 0.0,
        "bootstrap_failed": False,
        "bootstrap_rejection_reasons": {},
    }
    rejection_reasons: dict[str, int] = {}

    def record_rejection(reason: str) -> None:
        rejection_reasons[reason] = int(rejection_reasons.get(reason, 0)) + 1
        bootstrap_meta["bootstrap_draws_rejected"] = int(bootstrap_meta["bootstrap_draws_rejected"]) + 1

    if scenarios.empty:
        bootstrap_meta["bootstrap_rejection_reasons"] = rejection_reasons
        bootstrap_meta["bootstrap_acceptance_rate"] = 0.0
        bootstrap_meta["bootstrap_failed"] = False
        return scenarios.copy(), bootstrap_meta

    if n_boot <= 0:
        enriched = scenarios.copy()
        for column in (
            "p_shadow_marginal_lower",
            "p_shadow_marginal_upper",
            "p_alpha_lower",
            "p_alpha_upper",
        ):
            enriched[column] = enriched["p_alpha"] if "p_alpha" in column else enriched["p_shadow_marginal"]
        bootstrap_meta["bootstrap_rejection_reasons"] = rejection_reasons
        return enriched, bootstrap_meta

    rng = np.random.default_rng(seed)
    normalized_profile = normalize_specification_profile(demand_specification_profile)
    state_required = ["unpaid_childcare_hours", "state_price_index", "outside_option_wage"] + SPECIFICATION_PROFILES[normalized_profile]
    state = state_frame.dropna(subset=state_required).reset_index(drop=True)
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
        record_rejection("insufficient_bootstrap_inputs")
        bootstrap_meta["bootstrap_rejection_reasons"] = rejection_reasons
        bootstrap_meta["bootstrap_draws_rejected"] = int(bootstrap_meta["bootstrap_draws_requested"])
        bootstrap_meta["bootstrap_acceptance_rate"] = 0.0
        bootstrap_meta["bootstrap_failed"] = True
        return enriched, bootstrap_meta

    shadow_draws = np.full((len(scenarios), n_boot), np.nan)
    alpha_draws = np.full((len(scenarios), n_boot), np.nan)
    accepted_draws = 0
    for draw in range(n_boot):
        state_sample = _cluster_bootstrap_by_state(state, rng)
        county_sample = _cluster_bootstrap_by_state(county, rng)
        try:
            demand_summary, _ = estimate_childcare_demand_summary(
                state_sample.copy(),
                mode=normalize_demand_mode(demand_mode),
                specification_profile=demand_specification_profile,
            )
            _, demand_elasticity = resolve_solver_demand_elasticity(demand_summary)
        except Exception as exc:
            record_rejection(f"demand_fit:{type(exc).__name__}")
            continue
        try:
            supply_elasticity = calibrate_supply_elasticity(county_sample.copy())
        except Exception as exc:
            record_rejection(f"supply_fit:{type(exc).__name__}")
            continue

        draw_failed = False
        for row_number, row in enumerate(scenarios.itertuples(index=False)):
            try:
                shadow_result = solve_price(
                    row.p_baseline,
                    row.market_quantity_proxy,
                    row.unpaid_quantity_proxy,
                    demand_elasticity,
                    supply_elasticity,
                    MARGINAL_ALPHA,
                    return_metadata=True,
                )
                if not isinstance(shadow_result, SolverMetadata):
                    raise RuntimeError("missing solver metadata")
                alpha_result = solve_price(
                    row.p_baseline,
                    row.market_quantity_proxy,
                    row.unpaid_quantity_proxy,
                    demand_elasticity,
                    supply_elasticity,
                    row.alpha,
                    return_metadata=True,
                )
                if not isinstance(alpha_result, SolverMetadata):
                    raise RuntimeError("missing solver metadata")
            except Exception as exc:
                draw_failed = True
                record_rejection(f"solver:{type(exc).__name__}")
                break

            shadow_draws[row_number, draw] = shadow_result.price
            alpha_draws[row_number, draw] = alpha_result.price
        if draw_failed:
            continue
        accepted_draws += 1

    bootstrap_meta["bootstrap_draws_accepted"] = accepted_draws
    bootstrap_meta["bootstrap_draws_rejected"] = int(bootstrap_meta["bootstrap_draws_requested"]) - accepted_draws
    bootstrap_meta["bootstrap_rejection_reasons"] = rejection_reasons
    if int(bootstrap_meta["bootstrap_draws_requested"]) > 0:
        bootstrap_meta["bootstrap_acceptance_rate"] = float(accepted_draws / int(bootstrap_meta["bootstrap_draws_requested"]))
    else:
        bootstrap_meta["bootstrap_acceptance_rate"] = 0.0
    bootstrap_meta["bootstrap_failed"] = accepted_draws == 0

    enriched = scenarios.copy()
    if accepted_draws == 0:
        enriched["p_shadow_marginal_lower"] = enriched["p_shadow_marginal"]
        enriched["p_shadow_marginal_upper"] = enriched["p_shadow_marginal"]
        enriched["p_alpha_lower"] = enriched["p_alpha"]
        enriched["p_alpha_upper"] = enriched["p_alpha"]
        return enriched, bootstrap_meta
    enriched["p_shadow_marginal_lower"] = np.nanquantile(shadow_draws, 0.1, axis=1)
    enriched["p_shadow_marginal_upper"] = np.nanquantile(shadow_draws, 0.9, axis=1)
    enriched["p_alpha_lower"] = np.nanquantile(alpha_draws, 0.1, axis=1)
    enriched["p_alpha_upper"] = np.nanquantile(alpha_draws, 0.9, axis=1)
    return enriched, bootstrap_meta


def bootstrap_childcare_dual_shift_headline_table(
    state_frame: pd.DataFrame,
    county_frame: pd.DataFrame,
    scenarios: pd.DataFrame,
    demand_mode: str = "broad_complete",
    demand_specification_profile: str | None = None,
    n_boot: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, float | int | bool | dict[str, int] | str]]:
    bootstrap_meta: dict[str, float | int | bool | dict[str, int] | str] = {
        "bootstrap_draws_requested": int(n_boot),
        "bootstrap_draws_accepted": 0,
        "bootstrap_draws_rejected": 0,
        "bootstrap_acceptance_rate": 0.0,
        "bootstrap_failed": False,
        "bootstrap_rejection_reasons": {},
    }
    rejection_reasons: dict[str, int] = {}

    def record_rejection(reason: str) -> None:
        rejection_reasons[reason] = int(rejection_reasons.get(reason, 0)) + 1
        bootstrap_meta["bootstrap_draws_rejected"] = int(bootstrap_meta["bootstrap_draws_rejected"]) + 1

    def summarize_headline_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "kappa_q",
                    "kappa_c",
                    "row_count",
                    "median_baseline_price",
                    "median_p_alpha",
                    "median_p_alpha_pct_change",
                    "share_price_increase",
                    "share_price_decrease",
                ]
            )
        return (
            frame.groupby(["kappa_q", "kappa_c"], as_index=False)
            .agg(
                row_count=("state_fips", "size"),
                median_baseline_price=("p_baseline", "median"),
                median_p_alpha=("p_alpha", "median"),
                median_p_alpha_pct_change=("p_alpha_pct_change_vs_baseline", "median"),
                share_price_increase=("p_alpha_delta_vs_baseline", lambda values: float(pd.Series(values).gt(0).mean())),
                share_price_decrease=("p_alpha_delta_vs_baseline", lambda values: float(pd.Series(values).lt(0).mean())),
            )
            .sort_values(["kappa_q", "kappa_c"], kind="stable")
            .reset_index(drop=True)
        )

    if scenarios.empty:
        bootstrap_meta["bootstrap_rejection_reasons"] = rejection_reasons
        return summarize_headline_rows(scenarios), bootstrap_meta

    summarized = summarize_headline_rows(scenarios)

    if n_boot <= 0:
        output = summarized.copy()
        for column in (
            "median_p_alpha_lower",
            "median_p_alpha_upper",
            "median_p_alpha_pct_change_lower",
            "median_p_alpha_pct_change_upper",
        ):
            source = "median_p_alpha" if "median_p_alpha_pct_change" not in column else "median_p_alpha_pct_change"
            output[column] = output[source]
        output["share_price_increase_lower"] = output["share_price_increase"]
        output["share_price_increase_upper"] = output["share_price_increase"]
        output["share_price_decrease_lower"] = output["share_price_decrease"]
        output["share_price_decrease_upper"] = output["share_price_decrease"]
        bootstrap_meta["bootstrap_rejection_reasons"] = rejection_reasons
        return output, bootstrap_meta

    normalized_profile = normalize_specification_profile(demand_specification_profile)
    state_required = ["unpaid_childcare_hours", "state_price_index", "outside_option_wage"] + SPECIFICATION_PROFILES[normalized_profile]
    state = state_frame.dropna(subset=state_required).reset_index(drop=True)
    county = county_frame.dropna(subset=["provider_density", "annual_price"]).reset_index(drop=True)
    if len(state) < 10 or len(county) < 10:
        output = summarized.copy()
        for column in (
            "median_p_alpha_lower",
            "median_p_alpha_upper",
            "median_p_alpha_pct_change_lower",
            "median_p_alpha_pct_change_upper",
        ):
            source = "median_p_alpha" if "median_p_alpha_pct_change" not in column else "median_p_alpha_pct_change"
            output[column] = output[source]
        output["share_price_increase_lower"] = output["share_price_increase"]
        output["share_price_increase_upper"] = output["share_price_increase"]
        output["share_price_decrease_lower"] = output["share_price_decrease"]
        output["share_price_decrease_upper"] = output["share_price_decrease"]
        record_rejection("insufficient_bootstrap_inputs")
        bootstrap_meta["bootstrap_rejection_reasons"] = rejection_reasons
        bootstrap_meta["bootstrap_draws_rejected"] = int(bootstrap_meta["bootstrap_draws_requested"])
        bootstrap_meta["bootstrap_acceptance_rate"] = 0.0
        bootstrap_meta["bootstrap_failed"] = True
        return output, bootstrap_meta

    working = scenarios.copy().reset_index(drop=True)
    working["kappa_q"] = pd.to_numeric(working["kappa_q"], errors="coerce")
    working["kappa_c"] = pd.to_numeric(working["kappa_c"], errors="coerce")
    pair_keys = ["kappa_q", "kappa_c"]
    group_positions = {
        (
            float(pair["kappa_q"]),
            float(pair["kappa_c"]),
        ): working.index[
            np.isclose(working["kappa_q"], float(pair["kappa_q"]))
            & np.isclose(working["kappa_c"], float(pair["kappa_c"]))
        ].to_numpy()
        for pair in working[pair_keys].drop_duplicates().sort_values(pair_keys, kind="stable").to_dict(orient="records")
    }
    rng = np.random.default_rng(seed)
    accepted_draws = 0
    draw_metrics: dict[tuple[float, float], dict[str, list[float]]] = {
        key: {
            "median_price": [],
            "median_pct_change": [],
            "share_price_increase": [],
            "share_price_decrease": [],
        }
        for key in group_positions
    }
    for _ in range(n_boot):
        state_sample = _cluster_bootstrap_by_state(state, rng)
        county_sample = _cluster_bootstrap_by_state(county, rng)
        try:
            demand_summary, _ = estimate_childcare_demand_summary(
                state_sample.copy(),
                mode=normalize_demand_mode(demand_mode),
                specification_profile=demand_specification_profile,
            )
            _, demand_elasticity = resolve_solver_demand_elasticity(demand_summary)
        except Exception as exc:
            record_rejection(f"demand_fit:{type(exc).__name__}")
            continue
        try:
            supply_elasticity = calibrate_supply_elasticity(county_sample.copy())
        except Exception as exc:
            record_rejection(f"supply_fit:{type(exc).__name__}")
            continue

        prices = np.full(len(working), np.nan)
        draw_failed = False
        for row_number, row in enumerate(working.itertuples(index=False)):
            try:
                result = solve_price_dual_shift(
                    baseline_price=float(row.p_baseline),
                    market_quantity=float(row.market_quantity_proxy),
                    unpaid_quantity=float(row.unpaid_quantity_proxy),
                    demand_elasticity=demand_elasticity,
                    supply_elasticity=supply_elasticity,
                    alpha=float(row.alpha),
                    kappa_q=float(row.kappa_q),
                    kappa_c=float(row.kappa_c),
                    return_metadata=True,
                )
                if not isinstance(result, SolverMetadata):
                    raise RuntimeError("missing solver metadata")
            except Exception as exc:
                draw_failed = True
                record_rejection(f"solver:{type(exc).__name__}")
                break
            prices[row_number] = result.price
        if draw_failed:
            continue

        accepted_draws += 1
        baseline_prices = pd.to_numeric(working["p_baseline"], errors="coerce").to_numpy(dtype=float)
        for key, positions in group_positions.items():
            pair_prices = prices[positions]
            pair_baseline = baseline_prices[positions]
            pair_pct = np.divide(
                pair_prices - pair_baseline,
                pair_baseline,
                out=np.zeros_like(pair_prices),
                where=pair_baseline != 0,
            )
            draw_metrics[key]["median_price"].append(float(np.nanmedian(pair_prices)))
            draw_metrics[key]["median_pct_change"].append(float(np.nanmedian(pair_pct)))
            draw_metrics[key]["share_price_increase"].append(float(np.nanmean(pair_prices > pair_baseline)))
            draw_metrics[key]["share_price_decrease"].append(float(np.nanmean(pair_prices < pair_baseline)))

    bootstrap_meta["bootstrap_draws_accepted"] = accepted_draws
    bootstrap_meta["bootstrap_draws_rejected"] = int(bootstrap_meta["bootstrap_draws_requested"]) - accepted_draws
    bootstrap_meta["bootstrap_rejection_reasons"] = rejection_reasons
    bootstrap_meta["bootstrap_acceptance_rate"] = (
        float(accepted_draws / int(bootstrap_meta["bootstrap_draws_requested"]))
        if int(bootstrap_meta["bootstrap_draws_requested"]) > 0
        else 0.0
    )
    bootstrap_meta["bootstrap_failed"] = accepted_draws == 0

    output = summarized.copy().reset_index(drop=True)
    if accepted_draws == 0:
        output["median_p_alpha_lower"] = output["median_p_alpha"]
        output["median_p_alpha_upper"] = output["median_p_alpha"]
        output["median_p_alpha_pct_change_lower"] = output["median_p_alpha_pct_change"]
        output["median_p_alpha_pct_change_upper"] = output["median_p_alpha_pct_change"]
        output["share_price_increase_lower"] = output["share_price_increase"]
        output["share_price_increase_upper"] = output["share_price_increase"]
        output["share_price_decrease_lower"] = output["share_price_decrease"]
        output["share_price_decrease_upper"] = output["share_price_decrease"]
        return output, bootstrap_meta

    lower_median_price = []
    upper_median_price = []
    lower_median_pct = []
    upper_median_pct = []
    lower_increase = []
    upper_increase = []
    lower_decrease = []
    upper_decrease = []
    for row in output.itertuples(index=False):
        key = (float(row.kappa_q), float(row.kappa_c))
        metrics = draw_metrics[key]
        lower_median_price.append(float(np.nanquantile(metrics["median_price"], 0.1)))
        upper_median_price.append(float(np.nanquantile(metrics["median_price"], 0.9)))
        lower_median_pct.append(float(np.nanquantile(metrics["median_pct_change"], 0.1)))
        upper_median_pct.append(float(np.nanquantile(metrics["median_pct_change"], 0.9)))
        lower_increase.append(float(np.nanquantile(metrics["share_price_increase"], 0.1)))
        upper_increase.append(float(np.nanquantile(metrics["share_price_increase"], 0.9)))
        lower_decrease.append(float(np.nanquantile(metrics["share_price_decrease"], 0.1)))
        upper_decrease.append(float(np.nanquantile(metrics["share_price_decrease"], 0.9)))
    output["median_p_alpha_lower"] = lower_median_price
    output["median_p_alpha_upper"] = upper_median_price
    output["median_p_alpha_pct_change_lower"] = lower_median_pct
    output["median_p_alpha_pct_change_upper"] = upper_median_pct
    output["share_price_increase_lower"] = lower_increase
    output["share_price_increase_upper"] = upper_increase
    output["share_price_decrease_lower"] = lower_decrease
    output["share_price_decrease_upper"] = upper_decrease
    return output, bootstrap_meta


def summarize_childcare_scenario_diagnostics(
    scenarios: pd.DataFrame,
    skipped_state_rows: int = 0,
    demand_summary: dict[str, float] | None = None,
    demand_sample_name: str | None = None,
    demand_sample_selection_reason: str | None = None,
    bootstrap_meta: dict[str, object] | None = None,
) -> dict[str, float | int | bool | dict[str, int]]:
    bootstrap_meta = bootstrap_meta or {}
    bootstrap_rejection_reasons = bootstrap_meta.get("bootstrap_rejection_reasons", {})
    if not isinstance(bootstrap_rejection_reasons, dict):
        bootstrap_rejection_reasons = {}
    solver_status_counts: dict[str, int] = {}
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
            "demand_instrument": str(demand_summary.get("instrument", "outside_option_wage")) if demand_summary else "outside_option_wage",
            "demand_elasticity_at_mean": float(demand_summary.get("elasticity_at_mean", float("nan"))) if demand_summary else float("nan"),
            "demand_economically_admissible": bool(demand_summary.get("economically_admissible", False)) if demand_summary else False,
            "solver_demand_elasticity_magnitude": 0.0,
            "solver_status_counts": solver_status_counts,
            "solver_nonconverged_rows": 0,
            "solver_root_at_boundary_rows": 0,
            "solver_max_iterations": 0,
            "solver_max_expansion_steps": 0,
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
            "bootstrap_draws_requested": int(bootstrap_meta.get("bootstrap_draws_requested", 0)),
            "bootstrap_draws_accepted": int(bootstrap_meta.get("bootstrap_draws_accepted", 0)),
            "bootstrap_draws_rejected": int(bootstrap_meta.get("bootstrap_draws_rejected", 0)),
            "bootstrap_acceptance_rate": float(bootstrap_meta.get("bootstrap_acceptance_rate", 0.0)),
            "bootstrap_failed": bool(bootstrap_meta.get("bootstrap_failed", False)),
            "bootstrap_rejection_reasons": bootstrap_rejection_reasons,
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
    solver_status_series = scenarios.get("solver_status")
    if solver_status_series is None:
        solver_status = pd.Series(dtype=str)
    else:
        solver_status = solver_status_series.fillna("unknown").astype(str)
    if not solver_status.empty:
        solver_status_counts = {
            str(status): int(count)
            for status, count in solver_status.value_counts(dropna=False).to_dict().items()
        }
    solver_iteration_series = scenarios.get("solver_iterations")
    solver_iterations = (
        pd.to_numeric(solver_iteration_series, errors="coerce") if solver_iteration_series is not None else pd.Series(dtype=float)
    )
    solver_expansion_series = scenarios.get("solver_expansion_steps")
    solver_expansion_steps = (
        pd.to_numeric(solver_expansion_series, errors="coerce")
        if solver_expansion_series is not None
        else pd.Series(dtype=float)
    )
    boundary_rows = int(solver_status.isin(["root_at_low", "root_at_high"]).sum()) if not solver_status.empty else 0
    nonconverged_rows = (
        int((~solver_status.isin(["converged", "root_at_low", "root_at_high"])).sum())
        if not solver_status.empty
        else 0
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
        "demand_instrument": str(demand_summary.get("instrument", "outside_option_wage")) if demand_summary else "outside_option_wage",
        "demand_elasticity_at_mean": float(demand_summary.get("elasticity_at_mean", float("nan"))) if demand_summary else float("nan"),
        "demand_economically_admissible": bool(demand_summary.get("economically_admissible", False)) if demand_summary else False,
        "solver_demand_elasticity_magnitude": float(solver_demand_elasticity.median()) if solver_demand_elasticity.notna().any() else 0.0,
        "solver_status_counts": solver_status_counts,
        "solver_nonconverged_rows": nonconverged_rows,
        "solver_root_at_boundary_rows": boundary_rows,
        "solver_max_iterations": int(solver_iterations.max()) if solver_iterations.notna().any() else 0,
        "solver_max_expansion_steps": int(solver_expansion_steps.max()) if solver_expansion_steps.notna().any() else 0,
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
        "bootstrap_draws_requested": int(bootstrap_meta.get("bootstrap_draws_requested", 0)),
        "bootstrap_draws_accepted": int(bootstrap_meta.get("bootstrap_draws_accepted", 0)),
        "bootstrap_draws_rejected": int(bootstrap_meta.get("bootstrap_draws_rejected", 0)),
        "bootstrap_acceptance_rate": float(bootstrap_meta.get("bootstrap_acceptance_rate", 0.0)),
        "bootstrap_failed": bool(bootstrap_meta.get("bootstrap_failed", False)),
        "bootstrap_rejection_reasons": bootstrap_rejection_reasons,
    }
