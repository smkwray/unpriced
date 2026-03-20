from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CHANNEL_ORDER = ["private_unsubsidized", "private_subsidized", "public_admin"]


def _validate_required(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {', '.join(missing)}")


def _normalize_state_year(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["state_fips"] = working["state_fips"].astype(str).str.zfill(2)
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype("Int64")
    working = working.dropna(subset=["year"]).copy()
    working["year"] = working["year"].astype(int)
    return working


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    total_weight = float(numeric_weights.sum())
    if total_weight > 0:
        return float((numeric_values.fillna(0.0) * numeric_weights).sum() / total_weight)
    if numeric_values.notna().any():
        return float(numeric_values.mean())
    return 0.0


def _resolve_headline_alpha(values: pd.Series | list[float]) -> float | None:
    if len(values) == 0:
        return None
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return None
    unique_sorted = sorted(float(value) for value in numeric.unique())
    exact = [value for value in unique_sorted if np.isclose(value, 0.5)]
    if exact:
        return float(exact[0])
    return float(min(unique_sorted, key=lambda value: (abs(value - 0.5), value)))


def _select_preferred_alpha_row(group: pd.DataFrame, preferred_alpha: float | None) -> pd.Series:
    if preferred_alpha is None:
        return group.sort_values(["alpha"], kind="stable").iloc[0]
    alpha = pd.to_numeric(group["alpha"], errors="coerce")
    close = np.isclose(alpha, preferred_alpha)
    if bool(close.any()):
        return group.loc[close].sort_values(["alpha"], kind="stable").iloc[0]
    scored = group.assign(_alpha_distance=(alpha - preferred_alpha).abs())
    return scored.sort_values(["_alpha_distance", "alpha"], kind="stable").iloc[0]


def _format_float(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "nan"
    return f"{float(numeric):.6f}"


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def build_segmented_channel_response_summary(segmented_channel_scenarios: pd.DataFrame) -> pd.DataFrame:
    required = {
        "state_fips",
        "year",
        "solver_channel",
        "alpha",
        "p_baseline",
        "p_shadow_marginal",
        "p_alpha",
    }
    _validate_required(segmented_channel_scenarios, required, "segmented channel scenarios")

    output_columns = [
        "state_fips",
        "year",
        "solver_channel",
        "alpha",
        "market_quantity_proxy",
        "unpaid_quantity_proxy",
        "price_responsive",
        "p_baseline",
        "p_shadow_marginal",
        "p_alpha",
        "p_shadow_delta_vs_baseline",
        "p_shadow_pct_change_vs_baseline",
        "p_alpha_delta_vs_baseline",
        "p_alpha_pct_change_vs_baseline",
        "public_admin_price_invariant",
    ]
    if segmented_channel_scenarios.empty:
        return pd.DataFrame(columns=output_columns)

    working = _normalize_state_year(segmented_channel_scenarios)
    working["solver_channel"] = working["solver_channel"].astype(str)
    working = working.loc[working["solver_channel"].isin(CHANNEL_ORDER)].copy()
    for column in ("alpha", "p_baseline", "p_shadow_marginal", "p_alpha"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    if "market_quantity_proxy" not in working.columns:
        working["market_quantity_proxy"] = 0.0
    if "unpaid_quantity_proxy" not in working.columns:
        working["unpaid_quantity_proxy"] = 0.0
    if "price_responsive" not in working.columns:
        working["price_responsive"] = working["solver_channel"].isin({"private_unsubsidized", "private_subsidized"})
    working["market_quantity_proxy"] = pd.to_numeric(working["market_quantity_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    working["unpaid_quantity_proxy"] = pd.to_numeric(working["unpaid_quantity_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    working["price_responsive"] = working["price_responsive"].fillna(False).astype(bool)

    rows: list[dict[str, Any]] = []
    grouped = working.groupby(["state_fips", "year", "solver_channel", "alpha"], as_index=False, sort=True)
    for (state_fips, year, solver_channel, alpha), subset in grouped:
        weights = pd.to_numeric(subset["market_quantity_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
        p_baseline = _weighted_average(subset["p_baseline"], weights)
        p_shadow = _weighted_average(subset["p_shadow_marginal"], weights)
        p_alpha = _weighted_average(subset["p_alpha"], weights)
        p_shadow_delta = float(p_shadow - p_baseline)
        p_alpha_delta = float(p_alpha - p_baseline)
        p_shadow_pct = float(p_shadow_delta / p_baseline) if p_baseline != 0 else 0.0
        p_alpha_pct = float(p_alpha_delta / p_baseline) if p_baseline != 0 else 0.0
        public_invariant = bool(
            solver_channel == "public_admin"
            and np.isclose(p_alpha, p_baseline, equal_nan=True)
            and np.isclose(p_shadow, p_baseline, equal_nan=True)
        )
        rows.append(
            {
                "state_fips": str(state_fips).zfill(2),
                "year": int(year),
                "solver_channel": str(solver_channel),
                "alpha": float(alpha),
                "market_quantity_proxy": float(weights.sum()),
                "unpaid_quantity_proxy": float(
                    pd.to_numeric(subset["unpaid_quantity_proxy"], errors="coerce").fillna(0.0).sum()
                ),
                "price_responsive": bool(subset["price_responsive"].astype(bool).any()),
                "p_baseline": p_baseline,
                "p_shadow_marginal": p_shadow,
                "p_alpha": p_alpha,
                "p_shadow_delta_vs_baseline": p_shadow_delta,
                "p_shadow_pct_change_vs_baseline": p_shadow_pct,
                "p_alpha_delta_vs_baseline": p_alpha_delta,
                "p_alpha_pct_change_vs_baseline": p_alpha_pct,
                "public_admin_price_invariant": public_invariant,
            }
        )

    output = pd.DataFrame(rows)
    output["solver_channel"] = pd.Categorical(output["solver_channel"], categories=CHANNEL_ORDER, ordered=True)
    output = output.sort_values(["state_fips", "year", "solver_channel", "alpha"], kind="stable").reset_index(drop=True)
    output["solver_channel"] = output["solver_channel"].astype(str)
    return output[output_columns]


def build_segmented_state_fallback_summary(
    segmented_state_year_summary: pd.DataFrame,
    state_year_support_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"state_fips", "year", "alpha"}
    _validate_required(segmented_state_year_summary, required, "segmented state-year summary")

    output_defaults: dict[str, Any] = {
        "any_segment_allocation_fallback": False,
        "any_private_allocation_fallback": False,
        "ccdf_support_flag": "missing",
        "ccdf_admin_support_status": "missing",
        "q0_support_flag": "missing",
        "public_program_support_status": "missing",
        "ccdf_policy_control_support_status": "missing",
        "ccdf_policy_control_count": 0,
        "promoted_control_observed": False,
        "explicit_ccdf_row_count": 0,
        "inferred_ccdf_row_count": 0,
        "proxy_ccdf_row_count": 0,
        "missing_ccdf_row_count": 0,
        "component_sum_gap": 0.0,
    }
    if segmented_state_year_summary.empty:
        columns = ["state_fips", "year", "headline_alpha", "has_support_row", *output_defaults.keys()]
        return pd.DataFrame(columns=columns)

    state_year = _normalize_state_year(segmented_state_year_summary)
    state_year["alpha"] = pd.to_numeric(state_year["alpha"], errors="coerce")
    preferred_alpha = _resolve_headline_alpha(state_year["alpha"])

    selected_rows: list[pd.Series] = []
    for _, group in state_year.groupby(["state_fips", "year"], sort=True):
        selected_rows.append(_select_preferred_alpha_row(group, preferred_alpha))
    selected = pd.DataFrame(selected_rows).copy()
    selected = selected.rename(
        columns={
            "alpha": "headline_alpha",
            "quantity_weighted_p_alpha": "quantity_weighted_p_headline_alpha",
        }
    )

    if state_year_support_summary is not None and not state_year_support_summary.empty:
        support_required = {"state_fips", "year"}
        _validate_required(state_year_support_summary, support_required, "state-year support summary")
        support = _normalize_state_year(state_year_support_summary).drop_duplicates(["state_fips", "year"], keep="last")
        fallback = selected.merge(support, on=["state_fips", "year"], how="left")
        fallback["has_support_row"] = fallback.apply(
            lambda row: bool(not pd.isna(row.get("any_segment_allocation_fallback")) or not pd.isna(row.get("ccdf_support_flag"))),
            axis=1,
        )
    else:
        fallback = selected.copy()
        fallback["has_support_row"] = False

    for column, default in output_defaults.items():
        if column not in fallback.columns:
            fallback[column] = default
            continue
        if isinstance(default, bool):
            fallback[column] = fallback[column].fillna(default).astype(bool)
        elif isinstance(default, int):
            fallback[column] = pd.to_numeric(fallback[column], errors="coerce").fillna(default).astype(int)
        elif isinstance(default, float):
            fallback[column] = pd.to_numeric(fallback[column], errors="coerce").fillna(default)
        else:
            fallback[column] = fallback[column].fillna(default).astype(str)

    fallback = fallback.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)
    return fallback


def build_segmented_headline_summary(
    segmented_channel_scenarios: pd.DataFrame,
    segmented_state_year_summary: pd.DataFrame,
    segmented_state_year_diagnostics: pd.DataFrame,
    state_year_support_summary: pd.DataFrame | None = None,
) -> dict[str, object]:
    channel_response = build_segmented_channel_response_summary(segmented_channel_scenarios)
    fallback_summary = build_segmented_state_fallback_summary(
        segmented_state_year_summary=segmented_state_year_summary,
        state_year_support_summary=state_year_support_summary,
    )

    available_alphas = sorted(float(value) for value in pd.to_numeric(channel_response.get("alpha"), errors="coerce").dropna().unique())
    headline_alpha = _resolve_headline_alpha(available_alphas)
    if headline_alpha is None:
        alpha_slice = channel_response.iloc[0:0].copy()
    else:
        alpha_slice = channel_response.loc[np.isclose(channel_response["alpha"], headline_alpha)].copy()

    channel_price_response: dict[str, dict[str, float | int | bool]] = {}
    for channel in CHANNEL_ORDER:
        subset = alpha_slice.loc[alpha_slice["solver_channel"].astype(str).eq(channel)].copy()
        channel_price_response[channel] = {
            "state_year_count": int(len(subset)),
            "mean_p_baseline": float(pd.to_numeric(subset["p_baseline"], errors="coerce").mean()) if not subset.empty else 0.0,
            "mean_p_alpha": float(pd.to_numeric(subset["p_alpha"], errors="coerce").mean()) if not subset.empty else 0.0,
            "mean_p_shadow_marginal": float(pd.to_numeric(subset["p_shadow_marginal"], errors="coerce").mean()) if not subset.empty else 0.0,
            "mean_p_alpha_delta_vs_baseline": float(
                pd.to_numeric(subset["p_alpha_delta_vs_baseline"], errors="coerce").mean()
            )
            if not subset.empty
            else 0.0,
            "mean_p_alpha_pct_change_vs_baseline": float(
                pd.to_numeric(subset["p_alpha_pct_change_vs_baseline"], errors="coerce").mean()
            )
            if not subset.empty
            else 0.0,
            "price_responsive": bool(subset["price_responsive"].fillna(False).astype(bool).any()) if not subset.empty else bool(
                channel in {"private_unsubsidized", "private_subsidized"}
            ),
        }

    public_invariant = False
    if segmented_state_year_diagnostics is not None and not segmented_state_year_diagnostics.empty:
        public_invariant = bool(
            segmented_state_year_diagnostics.get("public_admin_invariant_prices", pd.Series([False])).iloc[0]
        )
    elif not channel_response.empty:
        public_rows = channel_response.loc[channel_response["solver_channel"].astype(str).eq("public_admin")]
        public_invariant = bool(public_rows["public_admin_price_invariant"].fillna(False).all()) if not public_rows.empty else False

    fallback_mask = (
        fallback_summary.get("any_segment_allocation_fallback", pd.Series(dtype=bool)).fillna(False).astype(bool)
        | fallback_summary.get("any_private_allocation_fallback", pd.Series(dtype=bool)).fillna(False).astype(bool)
    )
    fallback_state_year_count = int(fallback_mask.sum())
    fallback_state_count = int(fallback_summary.loc[fallback_mask, "state_fips"].astype(str).nunique()) if not fallback_summary.empty else 0
    promoted_count = int(
        fallback_summary.get("promoted_control_observed", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
    )
    proxy_ccdf_state_year_count = int(
        fallback_summary.get("proxy_ccdf_row_count", pd.Series(dtype=float)).fillna(0).gt(0).sum()
    )

    state_year_count = (
        int(
            _normalize_state_year(segmented_state_year_summary)[["state_fips", "year"]]
            .drop_duplicates(["state_fips", "year"], keep="first")
            .shape[0]
        )
        if not segmented_state_year_summary.empty
        else int(channel_response[["state_fips", "year"]].drop_duplicates(["state_fips", "year"], keep="first").shape[0])
    )

    return {
        "headline_alpha": headline_alpha,
        "available_alphas": available_alphas,
        "state_year_count": state_year_count,
        "public_admin_invariant_prices": bool(public_invariant),
        "fallback_state_year_count": fallback_state_year_count,
        "fallback_state_count": fallback_state_count,
        "promoted_control_observed_state_year_count": promoted_count,
        "proxy_ccdf_state_year_count": proxy_ccdf_state_year_count,
        "channel_price_response": channel_price_response,
    }


def build_segmented_report_markdown(
    headline_summary: dict[str, object],
    channel_response_summary: pd.DataFrame,
    fallback_summary: pd.DataFrame,
) -> str:
    headline_alpha = headline_summary.get("headline_alpha")
    alpha_text = "nan" if headline_alpha is None else _format_float(headline_alpha)
    lines = [
        "# segmented_childcare_report",
        f"headline_alpha: {alpha_text}",
        f"state_year_count: {int(headline_summary.get('state_year_count', 0))}",
        f"fallback_state_year_count: {int(headline_summary.get('fallback_state_year_count', 0))}",
        f"public_admin_invariant_prices: {_format_bool(headline_summary.get('public_admin_invariant_prices', False))}",
        "## channel_response_headline_alpha",
        "|solver_channel|state_year_count|price_responsive|mean_p_baseline|mean_p_alpha|mean_p_shadow_marginal|mean_p_alpha_delta_vs_baseline|mean_p_alpha_pct_change_vs_baseline|",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]

    channel_price_response = headline_summary.get("channel_price_response", {})
    for channel in CHANNEL_ORDER:
        metrics = channel_price_response.get(channel, {}) if isinstance(channel_price_response, dict) else {}
        lines.append(
            "|"
            + "|".join(
                [
                    channel,
                    str(int(metrics.get("state_year_count", 0))),
                    _format_bool(metrics.get("price_responsive", False)),
                    _format_float(metrics.get("mean_p_baseline", 0.0)),
                    _format_float(metrics.get("mean_p_alpha", 0.0)),
                    _format_float(metrics.get("mean_p_shadow_marginal", 0.0)),
                    _format_float(metrics.get("mean_p_alpha_delta_vs_baseline", 0.0)),
                    _format_float(metrics.get("mean_p_alpha_pct_change_vs_baseline", 0.0)),
                ]
            )
            + "|"
        )

    lines.extend(
        [
            "## fallback_state_years",
            "|state_fips|year|headline_alpha|any_segment_allocation_fallback|any_private_allocation_fallback|ccdf_support_flag|promoted_control_observed|",
            "|---|---:|---:|---|---|---|---|",
        ]
    )
    if fallback_summary.empty:
        lines.append("|none|0|nan|false|false|missing|false|")
    else:
        rows = fallback_summary.sort_values(["state_fips", "year"], kind="stable")
        for _, row in rows.iterrows():
            lines.append(
                "|"
                + "|".join(
                    [
                        str(row.get("state_fips", "")),
                        str(int(pd.to_numeric(pd.Series([row.get("year")]), errors="coerce").fillna(0).iloc[0])),
                        _format_float(row.get("headline_alpha")),
                        _format_bool(row.get("any_segment_allocation_fallback", False)),
                        _format_bool(row.get("any_private_allocation_fallback", False)),
                        str(row.get("ccdf_support_flag", "missing")),
                        _format_bool(row.get("promoted_control_observed", False)),
                    ]
                )
                + "|"
            )

    if not channel_response_summary.empty:
        lines.append(f"channel_response_rows: {int(len(channel_response_summary))}")
    else:
        lines.append("channel_response_rows: 0")
    return "\n".join(lines)


def build_childcare_segmented_reports(
    segmented_channel_scenarios: pd.DataFrame,
    segmented_state_year_summary: pd.DataFrame,
    segmented_state_year_diagnostics: pd.DataFrame,
    state_year_support_summary: pd.DataFrame | None = None,
) -> dict[str, object]:
    segmented_channel_response_summary = build_segmented_channel_response_summary(segmented_channel_scenarios)
    segmented_state_fallback_summary = build_segmented_state_fallback_summary(
        segmented_state_year_summary=segmented_state_year_summary,
        state_year_support_summary=state_year_support_summary,
    )
    segmented_headline_summary = build_segmented_headline_summary(
        segmented_channel_scenarios=segmented_channel_scenarios,
        segmented_state_year_summary=segmented_state_year_summary,
        segmented_state_year_diagnostics=segmented_state_year_diagnostics,
        state_year_support_summary=state_year_support_summary,
    )
    segmented_report_markdown = build_segmented_report_markdown(
        headline_summary=segmented_headline_summary,
        channel_response_summary=segmented_channel_response_summary,
        fallback_summary=segmented_state_fallback_summary,
    )
    return {
        "segmented_channel_response_summary": segmented_channel_response_summary,
        "segmented_state_fallback_summary": segmented_state_fallback_summary,
        "segmented_headline_summary": segmented_headline_summary,
        "segmented_report_markdown": segmented_report_markdown,
    }
