from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from unpaidwork.models.scenario_solver import (
    MARGINAL_ALPHA,
    resolve_solver_demand_elasticity,
    solve_alpha_grid,
    solve_price,
)

CHANNEL_ORDER = ["private_unsubsidized", "private_subsidized", "public_admin"]
BASELINE_QUANTITY_COLUMNS = {
    "private_unsubsidized": "solver_private_unsubsidized_slots",
    "private_subsidized": "solver_private_subsidized_slots",
    "public_admin": "solver_public_admin_slots",
}
STATE_REQUIRED_COLUMNS = {
    "state_fips",
    "year",
    "state_price_index",
    "unpaid_quantity_proxy",
    "benchmark_replacement_cost",
}
BASELINE_REQUIRED_COLUMNS = {
    "state_fips",
    "year",
    "solver_private_unsubsidized_slots",
    "solver_private_subsidized_slots",
    "solver_public_admin_slots",
}
ELASTICITY_REQUIRED_COLUMNS = {
    "solver_channel",
    "active_in_price_solver",
}
STATE_METADATA_COLUMNS = (
    "state_price_observation_status",
    "state_price_nowcast",
    "state_price_support_window",
    "state_ndcp_imputed_share",
    "is_sensitivity_year",
)


def _normalize_state_year(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["state_fips"] = working["state_fips"].astype(str).str.zfill(2)
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype("Int64")
    working = working.dropna(subset=["year"]).copy()
    working["year"] = working["year"].astype(int)
    return working


def _validate_required(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {', '.join(missing)}")


def _weight_price(frame: pd.DataFrame, price_column: str) -> float:
    quantity = pd.to_numeric(frame["market_quantity_proxy"], errors="coerce").fillna(0.0)
    total_quantity = float(quantity.sum())
    if total_quantity <= 0:
        return 0.0
    price = pd.to_numeric(frame[price_column], errors="coerce").fillna(0.0)
    return float((price * quantity).sum() / total_quantity)


def build_segmented_scenario_inputs(
    state_frame: pd.DataFrame,
    solver_baseline_state_year: pd.DataFrame,
    solver_elasticity_mapping: pd.DataFrame,
    solver_policy_controls_state_year: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _validate_required(state_frame, STATE_REQUIRED_COLUMNS, "state frame")
    _validate_required(solver_baseline_state_year, BASELINE_REQUIRED_COLUMNS, "solver baseline state-year")
    _validate_required(solver_elasticity_mapping, ELASTICITY_REQUIRED_COLUMNS, "solver elasticity mapping")

    state = _normalize_state_year(state_frame)
    baseline = _normalize_state_year(solver_baseline_state_year)
    elasticity = solver_elasticity_mapping.copy()
    elasticity["solver_channel"] = elasticity["solver_channel"].astype(str)
    elasticity["price_responsive"] = elasticity["active_in_price_solver"].fillna(False).astype(bool)
    if "elasticity_family" not in elasticity.columns:
        elasticity["elasticity_family"] = "pooled_childcare_demand"
    elasticity = elasticity[["solver_channel", "price_responsive", "elasticity_family"]].drop_duplicates(
        ["solver_channel"], keep="last"
    )

    merged = state.merge(baseline, on=["state_fips", "year"], how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "solver_channel",
                "price_responsive",
                "elasticity_family",
                "state_price_index",
                "market_quantity_proxy",
                "unpaid_quantity_proxy",
                "benchmark_replacement_cost",
            ]
        )

    for column in BASELINE_REQUIRED_COLUMNS - {"state_fips", "year"}:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    merged["state_price_index"] = pd.to_numeric(merged["state_price_index"], errors="coerce")
    merged["unpaid_quantity_proxy"] = pd.to_numeric(merged["unpaid_quantity_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    merged["benchmark_replacement_cost"] = pd.to_numeric(
        merged["benchmark_replacement_cost"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    merged["solver_total_private_slots"] = (
        merged["solver_private_unsubsidized_slots"] + merged["solver_private_subsidized_slots"]
    ).clip(lower=0.0)
    private_denom = merged["solver_total_private_slots"].replace({0.0: pd.NA})
    merged["private_unsubsidized_share"] = (
        merged["solver_private_unsubsidized_slots"].div(private_denom).fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    merged["private_subsidized_share"] = (
        merged["solver_private_subsidized_slots"].div(private_denom).fillna(0.0).clip(lower=0.0, upper=1.0)
    )

    rows: list[dict[str, Any]] = []
    optional_state_metadata = [column for column in STATE_METADATA_COLUMNS if column in merged.columns]
    for row in merged.to_dict(orient="records"):
        unpaid_total = float(row.get("unpaid_quantity_proxy", 0.0))
        unpaid_private_unsubsidized = unpaid_total * float(row.get("private_unsubsidized_share", 0.0))
        unpaid_private_subsidized = unpaid_total * float(row.get("private_subsidized_share", 0.0))
        unpaid_by_channel = {
            "private_unsubsidized": max(unpaid_private_unsubsidized, 0.0),
            "private_subsidized": max(unpaid_private_subsidized, 0.0),
            "public_admin": 0.0,
        }
        for solver_channel in CHANNEL_ORDER:
            market_quantity = float(row.get(BASELINE_QUANTITY_COLUMNS[solver_channel], 0.0))
            record = {
                "state_fips": str(row["state_fips"]).zfill(2),
                "year": int(row["year"]),
                "solver_channel": solver_channel,
                "state_price_index": float(row["state_price_index"]),
                "market_quantity_proxy": max(market_quantity, 0.0),
                "unpaid_quantity_proxy": float(unpaid_by_channel[solver_channel]),
                "benchmark_replacement_cost": float(row["benchmark_replacement_cost"]),
                "solver_total_private_slots": float(row.get("solver_total_private_slots", 0.0)),
                "solver_total_paid_slots": float(
                    row.get("solver_private_unsubsidized_slots", 0.0)
                    + row.get("solver_private_subsidized_slots", 0.0)
                    + row.get("solver_public_admin_slots", 0.0)
                ),
                "private_unsubsidized_share": float(row.get("private_unsubsidized_share", 0.0)),
                "private_subsidized_share": float(row.get("private_subsidized_share", 0.0)),
            }
            for column in optional_state_metadata:
                record[column] = row.get(column)
            rows.append(record)
    channel_inputs = pd.DataFrame(rows)
    channel_inputs = channel_inputs.merge(elasticity, on="solver_channel", how="left")
    channel_inputs["price_responsive"] = channel_inputs["price_responsive"].fillna(
        channel_inputs["solver_channel"].isin({"private_unsubsidized", "private_subsidized"})
    )
    default_family = pd.Series(
        np.where(channel_inputs["price_responsive"], "pooled_childcare_demand", "exogenous_non_price"),
        index=channel_inputs.index,
    )
    channel_inputs["elasticity_family"] = channel_inputs["elasticity_family"].where(
        channel_inputs["elasticity_family"].notna(),
        default_family,
    )

    if solver_policy_controls_state_year is not None and not solver_policy_controls_state_year.empty:
        controls = _normalize_state_year(solver_policy_controls_state_year)
        control_columns = [
            "state_fips",
            "year",
            *[column for column in controls.columns if column.startswith("ccdf_control_")],
            *[
                column
                for column in (
                    "ccdf_policy_control_count",
                    "ccdf_policy_control_support_status",
                    "ccdf_policy_promoted_controls_selected",
                    "ccdf_policy_promoted_control_rule",
                    "ccdf_policy_promoted_min_state_year_coverage",
                )
                if column in controls.columns
            ],
        ]
        controls = controls[control_columns].drop_duplicates(["state_fips", "year"], keep="last")
        channel_inputs = channel_inputs.merge(controls, on=["state_fips", "year"], how="left")

    channel_inputs["solver_channel"] = pd.Categorical(
        channel_inputs["solver_channel"], categories=CHANNEL_ORDER, ordered=True
    )
    channel_inputs = channel_inputs.sort_values(
        ["state_fips", "year", "solver_channel"], kind="stable"
    ).reset_index(drop=True)
    channel_inputs["solver_channel"] = channel_inputs["solver_channel"].astype(str)
    return channel_inputs


def simulate_segmented_childcare_scenarios(
    channel_inputs: pd.DataFrame,
    alphas: list[float],
    demand_summary: Mapping[str, Any],
    supply_summary: Mapping[str, Any],
    childcare_assumptions: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    del childcare_assumptions
    if channel_inputs.empty:
        return pd.DataFrame()
    required = {
        "state_fips",
        "year",
        "solver_channel",
        "price_responsive",
        "state_price_index",
        "market_quantity_proxy",
        "unpaid_quantity_proxy",
    }
    _validate_required(channel_inputs, required, "channel inputs")
    demand_elasticity_signed, solver_demand_elasticity = resolve_solver_demand_elasticity(dict(demand_summary))
    supply_elasticity = float(pd.to_numeric(pd.Series([supply_summary.get("supply_elasticity")]), errors="coerce").iloc[0])
    if not np.isfinite(supply_elasticity) or supply_elasticity <= 0:
        raise ValueError("supply summary missing finite positive supply_elasticity")

    rows: list[dict[str, Any]] = []
    for record in channel_inputs.to_dict(orient="records"):
        baseline = float(pd.to_numeric(pd.Series([record.get("state_price_index")]), errors="coerce").fillna(0.0).iloc[0])
        market_quantity = float(pd.to_numeric(pd.Series([record.get("market_quantity_proxy")]), errors="coerce").fillna(0.0).iloc[0])
        unpaid_quantity = float(pd.to_numeric(pd.Series([record.get("unpaid_quantity_proxy")]), errors="coerce").fillna(0.0).iloc[0])
        price_responsive = bool(record.get("price_responsive", False))
        if price_responsive and market_quantity > 0 and baseline > 0:
            shadow_price = solve_price(
                baseline,
                market_quantity,
                unpaid_quantity,
                solver_demand_elasticity,
                supply_elasticity,
                MARGINAL_ALPHA,
            )
            alpha_results = solve_alpha_grid(
                baseline,
                market_quantity,
                unpaid_quantity,
                solver_demand_elasticity,
                supply_elasticity,
                [float(alpha) for alpha in alphas],
            )
            alpha_to_price = {float(result.alpha): float(result.price) for result in alpha_results}
        else:
            shadow_price = baseline
            alpha_to_price = {float(alpha): baseline for alpha in alphas}

        for alpha in [float(alpha) for alpha in alphas]:
            row = dict(record)
            row["alpha"] = float(alpha)
            row["p_baseline"] = baseline
            row["p_shadow_marginal"] = float(shadow_price)
            row["p_alpha"] = float(alpha_to_price[float(alpha)])
            row["demand_elasticity_signed"] = float(demand_elasticity_signed)
            row["solver_demand_elasticity_magnitude"] = float(solver_demand_elasticity)
            row["supply_elasticity"] = float(supply_elasticity)
            rows.append(row)
    scenarios = pd.DataFrame(rows)
    scenarios["solver_channel"] = pd.Categorical(scenarios["solver_channel"], categories=CHANNEL_ORDER, ordered=True)
    scenarios = scenarios.sort_values(["state_fips", "year", "solver_channel", "alpha"], kind="stable").reset_index(drop=True)
    scenarios["solver_channel"] = scenarios["solver_channel"].astype(str)
    return scenarios


def build_segmented_state_year_summary(channel_scenarios: pd.DataFrame) -> pd.DataFrame:
    if channel_scenarios.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "alpha",
                "market_quantity_total",
                "unpaid_quantity_total",
                "private_market_quantity_total",
                "public_market_quantity_total",
                "private_unpaid_quantity_total",
                "public_unpaid_quantity_total",
                "quantity_weighted_p_baseline",
                "quantity_weighted_p_shadow_marginal",
                "quantity_weighted_p_alpha",
            ]
        )
    required = {
        "state_fips",
        "year",
        "alpha",
        "solver_channel",
        "market_quantity_proxy",
        "unpaid_quantity_proxy",
        "p_baseline",
        "p_shadow_marginal",
        "p_alpha",
    }
    _validate_required(channel_scenarios, required, "channel scenarios")

    summaries: list[dict[str, Any]] = []
    for (state_fips, year, alpha), subset in channel_scenarios.groupby(["state_fips", "year", "alpha"], as_index=False):
        market_quantity = pd.to_numeric(subset["market_quantity_proxy"], errors="coerce").fillna(0.0)
        unpaid_quantity = pd.to_numeric(subset["unpaid_quantity_proxy"], errors="coerce").fillna(0.0)
        private_mask = subset["solver_channel"].astype(str).isin({"private_unsubsidized", "private_subsidized"})
        public_mask = subset["solver_channel"].astype(str).eq("public_admin")
        summaries.append(
            {
                "state_fips": str(state_fips).zfill(2),
                "year": int(year),
                "alpha": float(alpha),
                "market_quantity_total": float(market_quantity.sum()),
                "unpaid_quantity_total": float(unpaid_quantity.sum()),
                "private_market_quantity_total": float(market_quantity.loc[private_mask].sum()),
                "public_market_quantity_total": float(market_quantity.loc[public_mask].sum()),
                "private_unpaid_quantity_total": float(unpaid_quantity.loc[private_mask].sum()),
                "public_unpaid_quantity_total": float(unpaid_quantity.loc[public_mask].sum()),
                "quantity_weighted_p_baseline": _weight_price(subset, "p_baseline"),
                "quantity_weighted_p_shadow_marginal": _weight_price(subset, "p_shadow_marginal"),
                "quantity_weighted_p_alpha": _weight_price(subset, "p_alpha"),
            }
        )
    return pd.DataFrame(summaries).sort_values(["state_fips", "year", "alpha"], kind="stable").reset_index(drop=True)


def build_segmented_scenario_diagnostics(
    channel_scenarios: pd.DataFrame,
    demand_summary: Mapping[str, Any],
    supply_summary: Mapping[str, Any],
) -> pd.DataFrame:
    if channel_scenarios.empty:
        return pd.DataFrame(
            [
                {
                    "scenario_rows": 0,
                    "scenario_states": 0,
                    "scenario_year_min": 0,
                    "scenario_year_max": 0,
                    "alpha_count": 0,
                    "price_responsive_rows": 0,
                    "non_price_responsive_rows": 0,
                    "public_admin_invariant_prices": True,
                    "unpaid_public_quantity_total": 0.0,
                    "demand_elasticity_at_mean": float(demand_summary.get("elasticity_at_mean", float("nan"))),
                    "supply_elasticity": float(supply_summary.get("supply_elasticity", float("nan"))),
                }
            ]
        )
    public_rows = channel_scenarios.loc[channel_scenarios["solver_channel"].astype(str).eq("public_admin")].copy()
    public_invariant = True
    if not public_rows.empty:
        public_invariant = bool(
            np.isclose(
                pd.to_numeric(public_rows["p_alpha"], errors="coerce"),
                pd.to_numeric(public_rows["p_baseline"], errors="coerce"),
                equal_nan=True,
            ).all()
            and np.isclose(
                pd.to_numeric(public_rows["p_shadow_marginal"], errors="coerce"),
                pd.to_numeric(public_rows["p_baseline"], errors="coerce"),
                equal_nan=True,
            ).all()
        )
    diagnostics = {
        "scenario_rows": int(len(channel_scenarios)),
        "scenario_states": int(channel_scenarios["state_fips"].astype(str).nunique()),
        "scenario_year_min": int(pd.to_numeric(channel_scenarios["year"], errors="coerce").min()),
        "scenario_year_max": int(pd.to_numeric(channel_scenarios["year"], errors="coerce").max()),
        "alpha_count": int(pd.to_numeric(channel_scenarios["alpha"], errors="coerce").nunique()),
        "price_responsive_rows": int(channel_scenarios["price_responsive"].fillna(False).astype(bool).sum()),
        "non_price_responsive_rows": int((~channel_scenarios["price_responsive"].fillna(False).astype(bool)).sum()),
        "public_admin_invariant_prices": bool(public_invariant),
        "unpaid_public_quantity_total": float(
            pd.to_numeric(public_rows.get("unpaid_quantity_proxy"), errors="coerce").fillna(0.0).sum()
        ),
        "demand_elasticity_at_mean": float(demand_summary.get("elasticity_at_mean", float("nan"))),
        "solver_demand_elasticity_magnitude": float(
            max(abs(float(demand_summary.get("elasticity_at_mean", float("nan")))), 1e-6)
        )
        if np.isfinite(float(demand_summary.get("elasticity_at_mean", float("nan"))))
        else float("nan"),
        "supply_elasticity": float(supply_summary.get("supply_elasticity", float("nan"))),
    }
    return pd.DataFrame([diagnostics])


def build_childcare_segmented_scenarios(
    state_frame: pd.DataFrame,
    solver_baseline_state_year: pd.DataFrame,
    solver_elasticity_mapping: pd.DataFrame,
    solver_policy_controls_state_year: pd.DataFrame | None,
    alphas: list[float],
    demand_summary: Mapping[str, Any],
    supply_summary: Mapping[str, Any],
    childcare_assumptions: Mapping[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    segmented_channel_inputs = build_segmented_scenario_inputs(
        state_frame=state_frame,
        solver_baseline_state_year=solver_baseline_state_year,
        solver_elasticity_mapping=solver_elasticity_mapping,
        solver_policy_controls_state_year=solver_policy_controls_state_year,
    )
    segmented_channel_scenarios = simulate_segmented_childcare_scenarios(
        channel_inputs=segmented_channel_inputs,
        alphas=alphas,
        demand_summary=demand_summary,
        supply_summary=supply_summary,
        childcare_assumptions=childcare_assumptions,
    )
    segmented_state_year_summary = build_segmented_state_year_summary(segmented_channel_scenarios)
    segmented_state_year_diagnostics = build_segmented_scenario_diagnostics(
        channel_scenarios=segmented_channel_scenarios,
        demand_summary=demand_summary,
        supply_summary=supply_summary,
    )
    return {
        "segmented_channel_inputs": segmented_channel_inputs,
        "segmented_channel_scenarios": segmented_channel_scenarios,
        "segmented_state_year_summary": segmented_state_year_summary,
        "segmented_state_year_diagnostics": segmented_state_year_diagnostics,
    }
