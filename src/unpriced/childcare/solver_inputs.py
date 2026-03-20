from __future__ import annotations

import pandas as pd

COMPONENT_TO_SOLVER_CHANNEL = {
    "private_unsubsidized": {
        "solver_channel": "private_unsubsidized",
        "price_responsive": True,
    },
    "private_subsidized": {
        "solver_channel": "private_subsidized",
        "price_responsive": True,
    },
    "public_admin": {
        "solver_channel": "public_admin",
        "price_responsive": False,
    },
}

SOLVER_CHANNEL_ORDER = ["private_unsubsidized", "private_subsidized", "public_admin"]
POLICY_CONTROL_METADATA_COLUMNS = [
    "ccdf_policy_control_count",
    "ccdf_policy_control_support_status",
    "ccdf_policy_promoted_controls_selected",
    "ccdf_policy_promoted_control_rule",
    "ccdf_policy_promoted_min_state_year_coverage",
]


def _collapse_text(values: pd.Series) -> str:
    seen: list[str] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        if text not in seen:
            seen.append(text)
    return "|".join(seen) if seen else "missing"


def _normalize_state_year(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["state_fips"] = working["state_fips"].astype(str).str.zfill(2)
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype("Int64")
    working = working.dropna(subset=["year"]).copy()
    working["year"] = working["year"].astype(int)
    return working


def _segment_price_support_state_year(ndcp_segment_prices: pd.DataFrame) -> pd.DataFrame:
    required = {"state_fips", "year"}
    missing = sorted(required - set(ndcp_segment_prices.columns))
    if missing:
        raise KeyError(f"Segment price frame missing required columns: {', '.join(missing)}")
    working = _normalize_state_year(ndcp_segment_prices)
    if "segment_channel" in working.columns:
        working = working.loc[working["segment_channel"].astype(str).eq("private")].copy()
    if "segment_id" in working.columns:
        grouped = (
            working.groupby(["state_fips", "year"], as_index=False)
            .agg(ndcp_private_segment_count=("segment_id", "nunique"))
        )
    else:
        grouped = (
            working.groupby(["state_fips", "year"], as_index=False)
            .agg(ndcp_private_segment_count=("state_fips", "size"))
        )
    grouped["ndcp_price_panel_support_status"] = grouped["ndcp_private_segment_count"].map(
        lambda count: "observed_private_segments" if int(count) > 0 else "missing_private_segments"
    )
    return grouped.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def build_solver_channel_quantities(q0_segmented: pd.DataFrame) -> pd.DataFrame:
    required = {"state_fips", "year", "quantity_component", "quantity_slots"}
    missing = sorted(required - set(q0_segmented.columns))
    if missing:
        raise KeyError(f"q0 segmented frame missing required columns: {', '.join(missing)}")

    working = _normalize_state_year(q0_segmented)
    working["quantity_component"] = working["quantity_component"].astype(str)
    working["quantity_slots"] = pd.to_numeric(working["quantity_slots"], errors="coerce").fillna(0.0).clip(lower=0.0)
    working = working.loc[working["quantity_component"].isin(COMPONENT_TO_SOLVER_CHANNEL)].copy()

    output_columns = [
        "state_fips",
        "year",
        "solver_channel",
        "quantity_slots",
        "price_responsive",
        "source_quantity_component",
        "public_program_support_status",
        "ccdf_support_flag",
        "ccdf_admin_support_status",
        "q0_support_flag",
        "any_segment_allocation_fallback",
        "any_ndcp_segment_support",
        "source_row_count",
    ]
    if working.empty:
        return pd.DataFrame(columns=output_columns)

    group_keys = ["state_fips", "year", "quantity_component"]
    grouped = (
        working.groupby(group_keys, as_index=False)
        .agg(
            quantity_slots=("quantity_slots", "sum"),
            source_row_count=("quantity_slots", "size"),
        )
    )

    text_metadata = [
        "public_program_support_status",
        "ccdf_support_flag",
        "ccdf_admin_support_status",
        "q0_support_flag",
    ]
    for column in text_metadata:
        if column not in working.columns:
            grouped[column] = "missing"
            continue
        collapsed = (
            working.groupby(group_keys)[column]
            .agg(_collapse_text)
            .reset_index(name=column)
        )
        grouped = grouped.merge(collapsed, on=group_keys, how="left")

    bool_metadata = [
        ("segment_allocation_fallback", "any_segment_allocation_fallback"),
        ("ndcp_segment_support", "any_ndcp_segment_support"),
    ]
    for source_column, output_column in bool_metadata:
        if source_column not in working.columns:
            grouped[output_column] = False
            continue
        collapsed = (
            working.groupby(group_keys)[source_column]
            .agg(lambda values: bool(pd.to_numeric(values, errors="coerce").fillna(0.0).gt(0).any()))
            .reset_index(name=output_column)
        )
        grouped = grouped.merge(collapsed, on=group_keys, how="left")

    grouped["solver_channel"] = grouped["quantity_component"].map(
        lambda value: COMPONENT_TO_SOLVER_CHANNEL[str(value)]["solver_channel"]
    )
    grouped["price_responsive"] = grouped["quantity_component"].map(
        lambda value: bool(COMPONENT_TO_SOLVER_CHANNEL[str(value)]["price_responsive"])
    )
    grouped = grouped.rename(columns={"quantity_component": "source_quantity_component"})
    grouped["solver_channel"] = pd.Categorical(grouped["solver_channel"], categories=SOLVER_CHANNEL_ORDER, ordered=True)
    grouped = grouped.sort_values(["state_fips", "year", "solver_channel"], kind="stable").reset_index(drop=True)
    grouped["solver_channel"] = grouped["solver_channel"].astype(str)
    return grouped[output_columns]


def build_solver_baseline_state_year(solver_channel_quantities: pd.DataFrame) -> pd.DataFrame:
    required = {"state_fips", "year", "solver_channel", "quantity_slots"}
    missing = sorted(required - set(solver_channel_quantities.columns))
    if missing:
        raise KeyError(f"Solver channel quantities missing required columns: {', '.join(missing)}")

    working = _normalize_state_year(solver_channel_quantities)
    working["solver_channel"] = working["solver_channel"].astype(str)
    working["quantity_slots"] = pd.to_numeric(working["quantity_slots"], errors="coerce").fillna(0.0).clip(lower=0.0)
    working = working.loc[working["solver_channel"].isin(SOLVER_CHANNEL_ORDER)].copy()

    output_columns = [
        "state_fips",
        "year",
        "solver_private_unsubsidized_slots",
        "solver_private_subsidized_slots",
        "solver_public_admin_slots",
        "solver_total_private_slots",
        "solver_total_paid_slots",
        "solver_exogenous_public_admin_slots",
    ]
    if working.empty:
        return pd.DataFrame(columns=output_columns)

    grouped = (
        working.groupby(["state_fips", "year", "solver_channel"], as_index=False)
        .agg(quantity_slots=("quantity_slots", "sum"))
    )
    pivoted = grouped.pivot(index=["state_fips", "year"], columns="solver_channel", values="quantity_slots").reset_index()
    pivoted.columns.name = None
    pivoted["solver_private_unsubsidized_slots"] = pd.to_numeric(
        pivoted.get("private_unsubsidized"), errors="coerce"
    ).fillna(0.0)
    pivoted["solver_private_subsidized_slots"] = pd.to_numeric(
        pivoted.get("private_subsidized"), errors="coerce"
    ).fillna(0.0)
    pivoted["solver_public_admin_slots"] = pd.to_numeric(
        pivoted.get("public_admin"), errors="coerce"
    ).fillna(0.0)
    pivoted["solver_total_private_slots"] = (
        pivoted["solver_private_unsubsidized_slots"] + pivoted["solver_private_subsidized_slots"]
    )
    pivoted["solver_total_paid_slots"] = pivoted["solver_total_private_slots"] + pivoted["solver_public_admin_slots"]
    pivoted["solver_exogenous_public_admin_slots"] = pivoted["solver_public_admin_slots"]
    result = pivoted.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)
    return result[output_columns]


def build_solver_elasticity_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "solver_channel": "private_unsubsidized",
                "elasticity_family": "pooled_childcare_demand",
                "active_in_price_solver": True,
                "inheritance_rule": "inherits_from_pooled_childcare_demand",
            },
            {
                "solver_channel": "private_subsidized",
                "elasticity_family": "pooled_childcare_demand",
                "active_in_price_solver": True,
                "inheritance_rule": "inherits_from_pooled_childcare_demand",
            },
            {
                "solver_channel": "public_admin",
                "elasticity_family": "exogenous_non_price",
                "active_in_price_solver": False,
                "inheritance_rule": "public_admin_exogenous_non_price_responsive",
            },
        ]
    )


def build_solver_policy_controls_state_year(
    promoted_controls: pd.DataFrame | None,
    solver_channel_quantities: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base_state_year = pd.DataFrame(columns=["state_fips", "year"])
    if solver_channel_quantities is not None and not solver_channel_quantities.empty:
        base_state_year = _normalize_state_year(solver_channel_quantities)[["state_fips", "year"]].drop_duplicates(
            ["state_fips", "year"], keep="first"
        )

    if promoted_controls is None or promoted_controls.empty:
        result = base_state_year.copy()
        result["ccdf_policy_control_count"] = 0
        result["ccdf_policy_control_support_status"] = "missing_policy_promoted_controls"
        result["ccdf_policy_promoted_controls_selected"] = ""
        result["ccdf_policy_promoted_control_rule"] = "state_year_coverage_gte_threshold"
        result["ccdf_policy_promoted_min_state_year_coverage"] = pd.NA
        return result.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)

    required = {"state_fips", "year"}
    missing = sorted(required - set(promoted_controls.columns))
    if missing:
        raise KeyError(f"Promoted controls frame missing required columns: {', '.join(missing)}")
    controls = _normalize_state_year(promoted_controls)
    control_columns = sorted([column for column in controls.columns if column.startswith("ccdf_control_")])
    metadata_columns = [
        column
        for column in POLICY_CONTROL_METADATA_COLUMNS
        if column in controls.columns
    ]
    keep_columns = ["state_fips", "year"] + metadata_columns + control_columns
    controls = controls[keep_columns].drop_duplicates(["state_fips", "year"], keep="last")

    if not base_state_year.empty:
        controls = base_state_year.merge(controls, on=["state_fips", "year"], how="left")

    if "ccdf_policy_control_count" in controls.columns:
        controls["ccdf_policy_control_count"] = pd.to_numeric(
            controls["ccdf_policy_control_count"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        controls["ccdf_policy_control_count"] = 0
    if "ccdf_policy_control_support_status" not in controls.columns:
        controls["ccdf_policy_control_support_status"] = "missing_policy_promoted_controls"
    controls["ccdf_policy_control_support_status"] = controls["ccdf_policy_control_support_status"].fillna(
        "missing_policy_promoted_controls"
    )
    if "ccdf_policy_promoted_controls_selected" not in controls.columns:
        controls["ccdf_policy_promoted_controls_selected"] = ""
    controls["ccdf_policy_promoted_controls_selected"] = controls[
        "ccdf_policy_promoted_controls_selected"
    ].fillna("")
    if "ccdf_policy_promoted_control_rule" not in controls.columns:
        controls["ccdf_policy_promoted_control_rule"] = "state_year_coverage_gte_threshold"
    controls["ccdf_policy_promoted_control_rule"] = controls["ccdf_policy_promoted_control_rule"].fillna(
        "state_year_coverage_gte_threshold"
    )
    if "ccdf_policy_promoted_min_state_year_coverage" not in controls.columns:
        controls["ccdf_policy_promoted_min_state_year_coverage"] = pd.NA

    ordered_columns = (
        ["state_fips", "year"]
        + [column for column in POLICY_CONTROL_METADATA_COLUMNS if column in controls.columns]
        + control_columns
    )
    result = controls.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)
    return result[ordered_columns]


def build_childcare_solver_inputs(
    q0_segmented: pd.DataFrame,
    promoted_controls: pd.DataFrame | None = None,
    ccdf_policy_controls_state_year: pd.DataFrame | None = None,
    ndcp_segment_prices: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    control_frame = promoted_controls if promoted_controls is not None else ccdf_policy_controls_state_year
    solver_channel_quantities = build_solver_channel_quantities(q0_segmented)

    if ndcp_segment_prices is not None and not ndcp_segment_prices.empty and not solver_channel_quantities.empty:
        segment_support = _segment_price_support_state_year(ndcp_segment_prices)
        solver_channel_quantities = solver_channel_quantities.merge(
            segment_support,
            on=["state_fips", "year"],
            how="left",
        )
    else:
        solver_channel_quantities["ndcp_private_segment_count"] = 0
        solver_channel_quantities["ndcp_price_panel_support_status"] = "missing_segment_price_panel"

    if "ndcp_private_segment_count" in solver_channel_quantities.columns:
        solver_channel_quantities["ndcp_private_segment_count"] = pd.to_numeric(
            solver_channel_quantities["ndcp_private_segment_count"], errors="coerce"
        ).fillna(0).astype(int)
    if "ndcp_price_panel_support_status" in solver_channel_quantities.columns:
        solver_channel_quantities["ndcp_price_panel_support_status"] = solver_channel_quantities[
            "ndcp_price_panel_support_status"
        ].fillna("missing_segment_price_panel")

    solver_baseline_state_year = build_solver_baseline_state_year(solver_channel_quantities)
    solver_elasticity_mapping = build_solver_elasticity_mapping()
    solver_policy_controls_state_year = build_solver_policy_controls_state_year(
        promoted_controls=control_frame,
        solver_channel_quantities=solver_channel_quantities,
    )
    return {
        "solver_channel_quantities": solver_channel_quantities.sort_values(
            ["state_fips", "year", "solver_channel"], kind="stable"
        ).reset_index(drop=True),
        "solver_baseline_state_year": solver_baseline_state_year,
        "solver_elasticity_mapping": solver_elasticity_mapping,
        "solver_policy_controls_state_year": solver_policy_controls_state_year,
    }
