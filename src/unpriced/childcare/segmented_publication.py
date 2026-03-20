from __future__ import annotations

from typing import Any, Mapping

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


def _format_float(value: Any, digits: int = 6) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "nan"
    return f"{float(numeric):.{digits}f}"


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| none |\n| --- |\n| none |"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_bool_dtype(display[column]):
            display[column] = display[column].map(_format_bool)
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider] + rows)


def _support_quality_tier(row: pd.Series) -> str:
    has_support_row = bool(row.get("has_support_row", False))
    segment_fallback = bool(row.get("any_segment_allocation_fallback", False))
    private_fallback = bool(row.get("any_private_allocation_fallback", False))
    proxy_ccdf_rows = int(pd.to_numeric(pd.Series([row.get("proxy_ccdf_row_count")]), errors="coerce").fillna(0).iloc[0])
    ccdf_support_flag = str(row.get("ccdf_support_flag", "missing")).strip().lower()
    ccdf_admin_support_status = str(row.get("ccdf_admin_support_status", "missing")).strip().lower()

    if not has_support_row or ccdf_support_flag in {"", "missing", "nan"}:
        return "missing_support"
    if segment_fallback or private_fallback or proxy_ccdf_rows > 0 or "proxy" in ccdf_support_flag or "proxy" in ccdf_admin_support_status:
        return "proxy_or_fallback_support"
    if "inferred" in ccdf_support_flag or "inferred" in ccdf_admin_support_status:
        return "inferred_split_support"
    if "explicit" in ccdf_support_flag or "explicit" in ccdf_admin_support_status:
        return "explicit_split_support"
    return "supported_other"


def build_segmented_publication_channel_table(
    segmented_channel_response_summary: pd.DataFrame,
    headline_summary: Mapping[str, Any],
) -> pd.DataFrame:
    required = {
        "state_fips",
        "year",
        "solver_channel",
        "alpha",
        "market_quantity_proxy",
        "unpaid_quantity_proxy",
        "price_responsive",
        "p_baseline",
        "p_alpha",
        "p_alpha_delta_vs_baseline",
        "p_alpha_pct_change_vs_baseline",
        "p_shadow_marginal",
        "public_admin_price_invariant",
    }
    _validate_required(segmented_channel_response_summary, required, "segmented channel response summary")
    if segmented_channel_response_summary.empty:
        return pd.DataFrame(columns=list(required))

    headline_alpha = pd.to_numeric(pd.Series([headline_summary.get("headline_alpha")]), errors="coerce").iloc[0]
    working = _normalize_state_year(segmented_channel_response_summary)
    working["solver_channel"] = working["solver_channel"].astype(str)
    working["alpha"] = pd.to_numeric(working["alpha"], errors="coerce")
    if not pd.isna(headline_alpha):
        working = working.loc[np.isclose(working["alpha"], float(headline_alpha))].copy()
    working = working.loc[working["solver_channel"].isin(CHANNEL_ORDER)].copy()
    working["solver_channel"] = pd.Categorical(working["solver_channel"], categories=CHANNEL_ORDER, ordered=True)
    working = working.sort_values(["state_fips", "year", "solver_channel"], kind="stable").reset_index(drop=True)
    working["solver_channel"] = working["solver_channel"].astype(str)
    return working[
        [
            "state_fips",
            "year",
            "solver_channel",
            "alpha",
            "market_quantity_proxy",
            "unpaid_quantity_proxy",
            "price_responsive",
            "p_baseline",
            "p_alpha",
            "p_alpha_delta_vs_baseline",
            "p_alpha_pct_change_vs_baseline",
            "p_shadow_marginal",
            "public_admin_price_invariant",
        ]
    ]


def build_segmented_publication_fallback_table(segmented_state_fallback_summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "state_fips",
        "year",
        "headline_alpha",
        "has_support_row",
        "any_segment_allocation_fallback",
        "any_private_allocation_fallback",
        "ccdf_support_flag",
        "ccdf_admin_support_status",
        "q0_support_flag",
        "public_program_support_status",
        "promoted_control_observed",
        "proxy_ccdf_row_count",
    }
    _validate_required(segmented_state_fallback_summary, required, "segmented state fallback summary")
    if segmented_state_fallback_summary.empty:
        return pd.DataFrame(columns=list(required))

    working = _normalize_state_year(segmented_state_fallback_summary)
    working["headline_alpha"] = pd.to_numeric(working["headline_alpha"], errors="coerce")
    for column in ("any_segment_allocation_fallback", "any_private_allocation_fallback", "has_support_row", "promoted_control_observed"):
        working[column] = working[column].fillna(False).astype(bool)
    working["proxy_ccdf_row_count"] = pd.to_numeric(working["proxy_ccdf_row_count"], errors="coerce").fillna(0).astype(int)
    working["support_quality_tier"] = working.apply(_support_quality_tier, axis=1)
    working["fallback_rank"] = (
        working["any_segment_allocation_fallback"].astype(int) + working["any_private_allocation_fallback"].astype(int)
    )
    working = working.sort_values(
        ["fallback_rank", "proxy_ccdf_row_count", "support_quality_tier", "state_fips", "year"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return working[
        [
            "state_fips",
            "year",
            "headline_alpha",
            "has_support_row",
            "any_segment_allocation_fallback",
            "any_private_allocation_fallback",
            "ccdf_support_flag",
            "ccdf_admin_support_status",
            "q0_support_flag",
            "public_program_support_status",
            "promoted_control_observed",
            "proxy_ccdf_row_count",
            "support_quality_tier",
        ]
    ]


def build_segmented_publication_support_quality_summary(publication_fallback_table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "support_quality_tier",
        "has_support_row",
        "any_segment_allocation_fallback",
        "any_private_allocation_fallback",
        "promoted_control_observed",
        "proxy_ccdf_row_count",
    }
    _validate_required(publication_fallback_table, required, "segmented publication fallback table")
    if publication_fallback_table.empty:
        return pd.DataFrame(
            columns=[
                "support_quality_tier",
                "state_year_count",
                "state_year_share",
                "fallback_state_year_count",
                "promoted_control_observed_state_year_count",
                "proxy_ccdf_state_year_count",
            ]
        )

    working = publication_fallback_table.copy()
    total_state_years = max(int(len(working)), 1)
    for column in ("has_support_row", "any_segment_allocation_fallback", "any_private_allocation_fallback", "promoted_control_observed"):
        working[column] = working[column].fillna(False).astype(bool)
    working["proxy_ccdf_row_count"] = pd.to_numeric(working["proxy_ccdf_row_count"], errors="coerce").fillna(0).astype(int)
    working["fallback_flag"] = working["any_segment_allocation_fallback"] | working["any_private_allocation_fallback"]
    summary = (
        working.groupby("support_quality_tier", dropna=False, sort=True)
        .agg(
            state_year_count=("support_quality_tier", "size"),
            fallback_state_year_count=("fallback_flag", "sum"),
            promoted_control_observed_state_year_count=("promoted_control_observed", "sum"),
            proxy_ccdf_state_year_count=("proxy_ccdf_row_count", lambda series: int(series.gt(0).sum())),
        )
        .reset_index()
    )
    summary["state_year_share"] = summary["state_year_count"] / float(total_state_years)
    summary = summary[
        [
            "support_quality_tier",
            "state_year_count",
            "state_year_share",
            "fallback_state_year_count",
            "promoted_control_observed_state_year_count",
            "proxy_ccdf_state_year_count",
        ]
    ]
    return summary.sort_values(["state_year_count", "support_quality_tier"], ascending=[False, True], kind="stable").reset_index(drop=True)


def build_segmented_publication_priority_states(publication_fallback_table: pd.DataFrame) -> pd.DataFrame:
    required = {"state_fips", "support_quality_tier", "promoted_control_observed"}
    _validate_required(publication_fallback_table, required, "segmented publication fallback table")
    if publication_fallback_table.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "state_year_count",
                "proxy_or_fallback_state_year_count",
                "inferred_split_state_year_count",
                "missing_support_state_year_count",
                "promoted_control_observed_state_year_count",
                "primary_support_issue",
            ]
        )

    working = publication_fallback_table.copy()
    working["state_fips"] = working["state_fips"].astype(str).str.zfill(2)
    working["promoted_control_observed"] = working["promoted_control_observed"].fillna(False).astype(bool)
    grouped = (
        working.groupby("state_fips", sort=True)
        .agg(
            state_year_count=("state_fips", "size"),
            proxy_or_fallback_state_year_count=("support_quality_tier", lambda series: int(series.eq("proxy_or_fallback_support").sum())),
            inferred_split_state_year_count=("support_quality_tier", lambda series: int(series.eq("inferred_split_support").sum())),
            missing_support_state_year_count=("support_quality_tier", lambda series: int(series.eq("missing_support").sum())),
            promoted_control_observed_state_year_count=("promoted_control_observed", "sum"),
        )
        .reset_index()
    )

    def _primary_issue(row: pd.Series) -> str:
        if int(row["proxy_or_fallback_state_year_count"]) > 0:
            return "proxy_or_fallback_support"
        if int(row["missing_support_state_year_count"]) > 0:
            return "missing_support"
        if int(row["inferred_split_state_year_count"]) > 0:
            return "inferred_split_support"
        return "stable_supported"

    grouped["primary_support_issue"] = grouped.apply(_primary_issue, axis=1)
    grouped = grouped.sort_values(
        [
            "proxy_or_fallback_state_year_count",
            "missing_support_state_year_count",
            "inferred_split_state_year_count",
            "state_year_count",
            "state_fips",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return grouped[
        [
            "state_fips",
            "state_year_count",
            "proxy_or_fallback_state_year_count",
            "inferred_split_state_year_count",
            "missing_support_state_year_count",
            "promoted_control_observed_state_year_count",
            "primary_support_issue",
        ]
    ]


def build_segmented_publication_priority_summary(
    support_quality_summary: pd.DataFrame,
    priority_states: pd.DataFrame,
) -> dict[str, object]:
    total_state_year_count = int(pd.to_numeric(support_quality_summary.get("state_year_count"), errors="coerce").fillna(0).sum())
    weak_tiers = {"proxy_or_fallback_support", "missing_support", "inferred_split_support"}
    weak_support_summary = support_quality_summary.loc[support_quality_summary["support_quality_tier"].isin(weak_tiers)].copy()
    weak_support_state_year_count = int(pd.to_numeric(weak_support_summary.get("state_year_count"), errors="coerce").fillna(0).sum())
    priority_state_rows = priority_states.loc[
        priority_states["primary_support_issue"].ne("stable_supported")
    ].copy()
    return {
        "total_state_year_count": total_state_year_count,
        "total_state_count": int(priority_states["state_fips"].nunique()) if not priority_states.empty else 0,
        "weak_support_state_year_count": weak_support_state_year_count,
        "weak_support_state_year_share": float(weak_support_state_year_count / total_state_year_count) if total_state_year_count else 0.0,
        "priority_state_count": int(priority_state_rows["state_fips"].nunique()) if not priority_state_rows.empty else 0,
        "support_quality_tiers": support_quality_summary.to_dict(orient="records"),
        "priority_states": priority_state_rows.head(10).to_dict(orient="records"),
    }


def build_segmented_publication_issue_breakdown(publication_fallback_table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "support_quality_tier",
        "state_fips",
        "ccdf_support_flag",
        "ccdf_admin_support_status",
        "q0_support_flag",
        "public_program_support_status",
        "promoted_control_observed",
    }
    _validate_required(publication_fallback_table, required, "segmented publication fallback table")
    if publication_fallback_table.empty:
        return pd.DataFrame(
            columns=[
                "support_quality_tier",
                "ccdf_support_flag",
                "ccdf_admin_support_status",
                "q0_support_flag",
                "public_program_support_status",
                "state_year_count",
                "state_count",
                "promoted_control_observed_state_year_count",
            ]
        )

    working = publication_fallback_table.copy()
    for column in ("ccdf_support_flag", "ccdf_admin_support_status", "q0_support_flag", "public_program_support_status"):
        working[column] = working[column].fillna("missing").astype(str)
    working["state_fips"] = working["state_fips"].astype(str).str.zfill(2)
    working["promoted_control_observed"] = working["promoted_control_observed"].fillna(False).astype(bool)
    issue_breakdown = (
        working.groupby(
            [
                "support_quality_tier",
                "ccdf_support_flag",
                "ccdf_admin_support_status",
                "q0_support_flag",
                "public_program_support_status",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            state_year_count=("state_fips", "size"),
            state_count=("state_fips", "nunique"),
            promoted_control_observed_state_year_count=("promoted_control_observed", "sum"),
        )
        .reset_index()
    )
    return issue_breakdown.sort_values(
        ["state_year_count", "state_count", "support_quality_tier", "ccdf_support_flag"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def build_segmented_publication_parser_focus_areas(
    support_quality_summary: pd.DataFrame,
    ccdf_admin_long: pd.DataFrame | None = None,
    ccdf_admin_state_year: pd.DataFrame | None = None,
) -> pd.DataFrame:
    weak_tiers = {"proxy_or_fallback_support", "missing_support", "inferred_split_support"}
    weak_support_state_year_count = int(
        pd.to_numeric(
            support_quality_summary.loc[support_quality_summary["support_quality_tier"].isin(weak_tiers), "state_year_count"],
            errors="coerce",
        ).fillna(0).sum()
    )
    inferred_state_year_count = int(
        pd.to_numeric(
            support_quality_summary.loc[
                support_quality_summary["support_quality_tier"].eq("inferred_split_support"), "state_year_count"
            ],
            errors="coerce",
        ).fillna(0).sum()
    )
    observed_columns: list[str] = []
    if ccdf_admin_long is not None and not ccdf_admin_long.empty and "column_name" in ccdf_admin_long.columns:
        observed_columns = sorted({str(value) for value in ccdf_admin_long["column_name"].dropna().astype(str)})
    expenditures_missing = False
    if ccdf_admin_state_year is not None and not ccdf_admin_state_year.empty and "ccdf_expenditures_support_status" in ccdf_admin_state_year.columns:
        expenditures_missing = bool(
            ccdf_admin_state_year["ccdf_expenditures_support_status"].fillna("").astype(str).str.contains("missing", case=False).any()
        )

    focus_specs = [
        {
            "focus_area": "children_served_columns",
            "keywords": ("children_served", "served_average_monthly"),
            "priority_tier": "low",
            "reason": "Baseline children-served coverage anchor for CCDF state-year mapping.",
            "recommended_keywords": "children served, average monthly served",
        },
        {
            "focus_area": "subsidized_private_split_columns",
            "keywords": ("subsid", "certificate", "voucher", "private"),
            "priority_tier": "high" if weak_support_state_year_count else "medium",
            "reason": "Weak segmented support is currently driven by proxy or inferred private/public splits.",
            "recommended_keywords": "subsidized private, certificate, voucher, care type",
        },
        {
            "focus_area": "public_admin_split_columns",
            "keywords": ("public", "school", "prek", "pre-k", "head_start", "admin"),
            "priority_tier": "high" if weak_support_state_year_count else "medium",
            "reason": "Public-admin quantities still rely on fallback or complements in weak-support tiers.",
            "recommended_keywords": "public admin, school-based, pre-k, head start",
        },
        {
            "focus_area": "split_share_or_percentage_columns",
            "keywords": ("share", "percent", "percentage"),
            "priority_tier": "high" if inferred_state_year_count or weak_support_state_year_count else "medium",
            "reason": "Share-based tables can reduce complement inference when explicit slot totals are absent.",
            "recommended_keywords": "share, percent, percentage",
        },
        {
            "focus_area": "expenditure_columns",
            "keywords": ("expend", "disburse", "obligation"),
            "priority_tier": "high" if expenditures_missing else "medium",
            "reason": "Current CCDF admin state-year outputs still carry missing expenditure support.",
            "recommended_keywords": "expenditures, disbursements, obligations",
        },
    ]
    rows: list[dict[str, object]] = []
    for spec in focus_specs:
        matched = [column for column in observed_columns if any(keyword in column.lower() for keyword in spec["keywords"])]
        rows.append(
            {
                "focus_area": spec["focus_area"],
                "priority_tier": spec["priority_tier"],
                "current_signal_present": bool(matched),
                "matched_column_count": int(len(matched)),
                "matched_column_examples": "; ".join(matched[:6]),
                "recommended_keywords": spec["recommended_keywords"],
                "reason": spec["reason"],
            }
        )
    focus_areas = pd.DataFrame(rows)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    focus_areas["priority_order"] = focus_areas["priority_tier"].map(priority_order).fillna(9)
    return (
        focus_areas.sort_values(
            ["priority_order", "current_signal_present", "focus_area"],
            ascending=[True, True, True],
            kind="stable",
        )
        .drop(columns=["priority_order"])
        .reset_index(drop=True)
    )


def build_segmented_publication_admin_sheet_targets(
    parser_focus_areas: pd.DataFrame,
    ccdf_parse_inventory: pd.DataFrame | None = None,
    ccdf_admin_long: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "filename",
        "source_sheet",
        "table_year",
        "parse_status",
        "parsed_row_count",
        "matched_focus_areas",
        "missing_priority_focus_areas",
        "parser_target_recommendation",
    ]
    if ccdf_parse_inventory is None or ccdf_parse_inventory.empty:
        return pd.DataFrame(columns=columns)

    admin_inventory = ccdf_parse_inventory.loc[ccdf_parse_inventory["source_component"].astype(str).eq("admin")].copy()
    if admin_inventory.empty:
        return pd.DataFrame(columns=columns)

    signal_map = {
        "children_served_columns": ("children_served", "served_average_monthly"),
        "subsidized_private_split_columns": ("subsid", "certificate", "voucher", "private"),
        "public_admin_split_columns": ("public", "school", "prek", "pre-k", "head_start", "admin"),
        "split_share_or_percentage_columns": ("share", "percent", "percentage"),
        "expenditure_columns": ("expend", "disburse", "obligation"),
    }
    high_priority_focuses = set(
        parser_focus_areas.loc[parser_focus_areas["priority_tier"].astype(str).eq("high"), "focus_area"].astype(str)
    )
    long_columns = pd.DataFrame(columns=["filename", "source_sheet", "column_name"])
    if ccdf_admin_long is not None and not ccdf_admin_long.empty:
        long_columns = ccdf_admin_long[["filename", "source_sheet", "column_name"]].copy()
        for column in ("filename", "source_sheet", "column_name"):
            long_columns[column] = long_columns[column].astype(str)

    rows: list[dict[str, object]] = []
    for row in admin_inventory.to_dict(orient="records"):
        filename = str(row.get("filename", ""))
        source_sheet = str(row.get("source_sheet", ""))
        subset = long_columns.loc[
            long_columns["filename"].eq(filename) & long_columns["source_sheet"].eq(source_sheet)
        ]
        matched_focuses: list[str] = []
        columns_here = subset["column_name"].tolist()
        for focus_area, keywords in signal_map.items():
            if any(any(keyword in column.lower() for keyword in keywords) for column in columns_here):
                matched_focuses.append(focus_area)
        missing_priority = sorted(high_priority_focuses - set(matched_focuses))
        recommendation = (
            f"expand parser coverage for {', '.join(missing_priority[:3])}"
            if missing_priority
            else "current sheet covers configured high-priority parser focuses"
        )
        rows.append(
            {
                "filename": filename,
                "source_sheet": source_sheet,
                "table_year": row.get("table_year"),
                "parse_status": row.get("parse_status"),
                "parsed_row_count": row.get("parsed_row_count"),
                "matched_focus_areas": "; ".join(matched_focuses),
                "missing_priority_focus_areas": "; ".join(missing_priority),
                "parser_target_recommendation": recommendation,
            }
        )
    targets = pd.DataFrame(rows)
    return targets.sort_values(
        ["missing_priority_focus_areas", "table_year", "filename", "source_sheet"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _classify_admin_inventory_scope(admin_sheet_targets: pd.DataFrame) -> tuple[str, str]:
    if admin_sheet_targets.empty:
        return (
            "no_admin_inventory",
            "No parsed CCDF admin sheets are available in this run; parser recommendations are blocked on additional files.",
        )
    unique_files = int(admin_sheet_targets.get("filename", pd.Series(dtype=object)).dropna().astype(str).nunique())
    parsed_rows = pd.to_numeric(admin_sheet_targets.get("parsed_row_count"), errors="coerce").fillna(0)
    parsed_row_total = float(parsed_rows.sum()) if not parsed_rows.empty else 0.0
    if unique_files <= 1 and parsed_row_total <= 200:
        return (
            "sample_or_limited_inventory",
            "Available CCDF admin inventory appears sample-sized or single-sheet; action ranking is provisional.",
        )
    if unique_files < 3:
        return (
            "limited_inventory",
            "Available CCDF admin inventory is limited; expand file coverage before broad parser conclusions.",
        )
    return (
        "broader_inventory",
        "Available CCDF admin inventory spans multiple sheets/files; parser action ranking uses current evidence.",
    )


def build_segmented_publication_parser_action_plan(
    priority_states: pd.DataFrame,
    issue_breakdown: pd.DataFrame,
    parser_focus_areas: pd.DataFrame,
    admin_sheet_targets: pd.DataFrame,
    max_actions: int = 6,
) -> pd.DataFrame:
    _validate_required(
        priority_states,
        {
            "state_fips",
            "state_year_count",
            "primary_support_issue",
        },
        "segmented publication priority states",
    )
    _validate_required(
        issue_breakdown,
        {
            "support_quality_tier",
            "state_year_count",
        },
        "segmented publication issue breakdown",
    )
    _validate_required(
        parser_focus_areas,
        {
            "focus_area",
            "priority_tier",
            "current_signal_present",
            "recommended_keywords",
            "reason",
        },
        "segmented publication parser focus areas",
    )
    _validate_required(
        admin_sheet_targets,
        {
            "filename",
            "source_sheet",
            "missing_priority_focus_areas",
            "matched_focus_areas",
            "parser_target_recommendation",
        },
        "segmented publication admin sheet targets",
    )

    columns = [
        "action_rank",
        "focus_area",
        "priority_tier",
        "current_signal_present",
        "candidate_target_count",
        "candidate_target_files",
        "candidate_target_sheets",
        "top_priority_states",
        "dominant_support_issue",
        "inventory_scope",
        "inventory_scope_note",
        "recommended_next_task",
        "rationale",
    ]
    if parser_focus_areas.empty:
        return pd.DataFrame(columns=columns)

    inventory_scope, inventory_scope_note = _classify_admin_inventory_scope(admin_sheet_targets)

    dominant_issue = "unknown"
    if not issue_breakdown.empty:
        issue_rows = issue_breakdown.copy()
        issue_rows["state_year_count"] = pd.to_numeric(issue_rows["state_year_count"], errors="coerce").fillna(0)
        dominant_issue = str(
            issue_rows.sort_values(["state_year_count", "support_quality_tier"], ascending=[False, True], kind="stable")
            .iloc[0]
            .get("support_quality_tier", "unknown")
        )

    top_states_rows = priority_states.loc[priority_states["primary_support_issue"].astype(str).ne("stable_supported")].copy()
    top_states_rows["state_year_count"] = pd.to_numeric(top_states_rows["state_year_count"], errors="coerce").fillna(0)
    top_states_rows = top_states_rows.sort_values(["state_year_count", "state_fips"], ascending=[False, True], kind="stable")
    top_priority_states = ", ".join(top_states_rows["state_fips"].astype(str).head(5).tolist()) if not top_states_rows.empty else "none"

    focus_working = parser_focus_areas.copy()
    focus_working["priority_tier"] = focus_working["priority_tier"].fillna("medium").astype(str)
    focus_working["current_signal_present"] = focus_working["current_signal_present"].fillna(False).astype(bool)
    focus_working["priority_order"] = focus_working["priority_tier"].map({"high": 0, "medium": 1, "low": 2}).fillna(9)
    focus_working = focus_working.sort_values(
        ["priority_order", "current_signal_present", "focus_area"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    targets = admin_sheet_targets.copy()
    for column in ("missing_priority_focus_areas", "matched_focus_areas", "filename", "source_sheet"):
        targets[column] = targets[column].fillna("").astype(str)

    rows: list[dict[str, object]] = []
    for _, focus_row in focus_working.iterrows():
        focus_area = str(focus_row["focus_area"])
        missing_match = targets["missing_priority_focus_areas"].str.contains(focus_area, regex=False)
        matched_match = targets["matched_focus_areas"].str.contains(focus_area, regex=False)
        candidates = targets.loc[missing_match | matched_match].copy()

        target_files = sorted(candidates["filename"].loc[candidates["filename"].ne("")].unique().tolist())
        target_sheets = sorted(
            {
                f"{filename}:{sheet}"
                for filename, sheet in zip(candidates["filename"], candidates["source_sheet"])
                if filename and sheet
            }
        )
        signal_present = bool(focus_row["current_signal_present"])
        if not signal_present and not candidates.empty:
            next_task = f"Expand parser coverage for {focus_area} on available admin sheet targets."
        elif not signal_present:
            if inventory_scope in {"no_admin_inventory", "sample_or_limited_inventory", "limited_inventory"}:
                next_task = f"Acquire additional CCDF admin tables for {focus_area} before extending parser mappings."
            else:
                next_task = f"Add parser mappings for {focus_area} in newly parsed admin sheets."
        else:
            next_task = f"Harden and regression-test existing mapping coverage for {focus_area}."

        rows.append(
            {
                "focus_area": focus_area,
                "priority_tier": str(focus_row["priority_tier"]),
                "current_signal_present": signal_present,
                "candidate_target_count": int(len(candidates)),
                "candidate_target_files": "; ".join(target_files),
                "candidate_target_sheets": "; ".join(target_sheets),
                "top_priority_states": top_priority_states,
                "dominant_support_issue": dominant_issue,
                "inventory_scope": inventory_scope,
                "inventory_scope_note": inventory_scope_note,
                "recommended_next_task": next_task,
                "rationale": str(focus_row["reason"]),
            }
        )

    action_plan = pd.DataFrame(rows).head(max_actions).copy()
    action_plan["action_rank"] = range(1, len(action_plan) + 1)
    return action_plan[columns]


def build_segmented_publication_parser_action_plan_summary(parser_action_plan: pd.DataFrame) -> dict[str, object]:
    if parser_action_plan.empty:
        return {
            "row_count": 0,
            "priority_tier_counts": {},
            "inventory_scope_counts": {},
            "rows": [],
        }
    priority_counts = (
        parser_action_plan["priority_tier"].fillna("missing").astype(str).value_counts(dropna=False).to_dict()
        if "priority_tier" in parser_action_plan.columns
        else {}
    )
    inventory_scope_counts = (
        parser_action_plan["inventory_scope"].fillna("missing").astype(str).value_counts(dropna=False).to_dict()
        if "inventory_scope" in parser_action_plan.columns
        else {}
    )
    return {
        "row_count": int(len(parser_action_plan)),
        "priority_tier_counts": {str(key): int(value) for key, value in priority_counts.items()},
        "inventory_scope_counts": {str(key): int(value) for key, value in inventory_scope_counts.items()},
        "rows": parser_action_plan.to_dict(orient="records"),
    }


def build_segmented_publication_readout(
    headline_summary: Mapping[str, Any],
    publication_channel_table: pd.DataFrame,
    publication_fallback_table: pd.DataFrame,
    support_quality_summary: pd.DataFrame,
    priority_states: pd.DataFrame,
    issue_breakdown: pd.DataFrame,
    parser_focus_areas: pd.DataFrame,
    parser_action_plan: pd.DataFrame,
) -> str:
    channel_metrics = headline_summary.get("channel_price_response", {})
    lines = [
        "# Childcare Segmented Headline Readout",
        "",
        f"- headline alpha: {_format_float(headline_summary.get('headline_alpha'))}",
        f"- state-years covered: {int(headline_summary.get('state_year_count', 0))}",
        f"- fallback state-years: {int(headline_summary.get('fallback_state_year_count', 0))}",
        f"- promoted-control observed state-years: {int(headline_summary.get('promoted_control_observed_state_year_count', 0))}",
        f"- public-admin prices invariant: {_format_bool(headline_summary.get('public_admin_invariant_prices', False))}",
        "",
        "## Channel headline summary",
    ]
    table_rows = []
    for channel in CHANNEL_ORDER:
        metrics = channel_metrics.get(channel, {}) if isinstance(channel_metrics, Mapping) else {}
        table_rows.append(
            {
                "solver_channel": channel,
                "state_year_count": int(metrics.get("state_year_count", 0)),
                "price_responsive": _format_bool(metrics.get("price_responsive", False)),
                "mean_p_baseline": _format_float(metrics.get("mean_p_baseline", 0.0)),
                "mean_p_alpha": _format_float(metrics.get("mean_p_alpha", 0.0)),
                "mean_p_alpha_pct_change_vs_baseline": _format_float(
                    metrics.get("mean_p_alpha_pct_change_vs_baseline", 0.0)
                ),
            }
        )
    lines.append(_markdown_table(pd.DataFrame(table_rows)))
    lines.extend(
        [
            "",
            "## Fallback coverage",
            f"- state-year rows in publication table: {len(publication_channel_table)}",
            f"- fallback rows retained for publication: {len(publication_fallback_table)}",
            "",
            "## Support quality",
            _markdown_table(
                support_quality_summary.assign(
                    state_year_share=support_quality_summary["state_year_share"].map(lambda value: _format_float(value, digits=4))
                )
            ),
            "",
            "## Priority states",
            _markdown_table(priority_states.head(10)),
            "",
            "## Support issue mix",
            _markdown_table(issue_breakdown.head(10)),
            "",
            "## Parser focus areas",
            _markdown_table(parser_focus_areas.head(10)),
            "",
            "## Parser action plan",
            _markdown_table(parser_action_plan.head(10)),
        ]
    )
    return "\n".join(lines)


def build_segmented_publication_report(
    headline_summary: Mapping[str, Any],
    publication_channel_table: pd.DataFrame,
    publication_fallback_table: pd.DataFrame,
    support_quality_summary: pd.DataFrame,
    priority_states: pd.DataFrame,
    issue_breakdown: pd.DataFrame,
    parser_focus_areas: pd.DataFrame,
    admin_sheet_targets: pd.DataFrame,
    parser_action_plan: pd.DataFrame,
) -> str:
    channel_metrics = headline_summary.get("channel_price_response", {})
    summary_rows = []
    for channel in CHANNEL_ORDER:
        metrics = channel_metrics.get(channel, {}) if isinstance(channel_metrics, Mapping) else {}
        summary_rows.append(
            {
                "solver_channel": channel,
                "state_year_count": int(metrics.get("state_year_count", 0)),
                "price_responsive": bool(metrics.get("price_responsive", False)),
                "mean_p_baseline": _format_float(metrics.get("mean_p_baseline", 0.0)),
                "mean_p_alpha": _format_float(metrics.get("mean_p_alpha", 0.0)),
                "mean_p_shadow_marginal": _format_float(metrics.get("mean_p_shadow_marginal", 0.0)),
                "mean_p_alpha_pct_change_vs_baseline": _format_float(
                    metrics.get("mean_p_alpha_pct_change_vs_baseline", 0.0)
                ),
            }
        )

    fallback_preview = publication_fallback_table.loc[
        publication_fallback_table["any_segment_allocation_fallback"] | publication_fallback_table["any_private_allocation_fallback"]
    ].head(12)
    if fallback_preview.empty:
        fallback_preview = publication_fallback_table.head(12)

    lines = [
        "# childcare_segmented_report",
        "",
        "## Scope",
        "- This report is additive-only and does not modify the pooled childcare MVP report path.",
        "- It summarizes the segmented childcare scenario scaffolding using the promoted copayment control surface and CCDF support diagnostics currently available in the repo.",
        "",
        "## Headline",
        f"- headline alpha: {_format_float(headline_summary.get('headline_alpha'))}",
        f"- state-years covered: {int(headline_summary.get('state_year_count', 0))}",
        f"- fallback state-years: {int(headline_summary.get('fallback_state_year_count', 0))}",
        f"- fallback states: {int(headline_summary.get('fallback_state_count', 0))}",
        f"- proxy-CCDF state-years: {int(headline_summary.get('proxy_ccdf_state_year_count', 0))}",
        f"- public-admin price invariant: {_format_bool(headline_summary.get('public_admin_invariant_prices', False))}",
        "",
        "## Channel response at headline alpha",
        _markdown_table(pd.DataFrame(summary_rows)),
        "",
        "## Published channel table preview",
        _markdown_table(publication_channel_table.head(12)),
        "",
        "## Fallback and support preview",
        _markdown_table(fallback_preview),
        "",
        "## Support quality summary",
        _markdown_table(
            support_quality_summary.assign(
                state_year_share=support_quality_summary["state_year_share"].map(lambda value: _format_float(value, digits=4))
            )
        ),
        "",
        "## Parser-target priority states",
        _markdown_table(priority_states.head(12)),
        "",
        "## Support issue breakdown",
        _markdown_table(issue_breakdown.head(12)),
        "",
        "## Parser focus areas",
        _markdown_table(parser_focus_areas.head(12)),
        "",
        "## Admin sheet targets",
        _markdown_table(admin_sheet_targets.head(12)),
        "",
        "## Parser action plan",
        _markdown_table(parser_action_plan.head(12)),
    ]
    return "\n".join(lines)


def build_childcare_segmented_publication_outputs(
    segmented_headline_summary: Mapping[str, Any],
    segmented_channel_response_summary: pd.DataFrame,
    segmented_state_fallback_summary: pd.DataFrame,
    ccdf_parse_inventory: pd.DataFrame | None = None,
    ccdf_admin_long: pd.DataFrame | None = None,
    ccdf_admin_state_year: pd.DataFrame | None = None,
) -> dict[str, object]:
    publication_channel_table = build_segmented_publication_channel_table(
        segmented_channel_response_summary=segmented_channel_response_summary,
        headline_summary=segmented_headline_summary,
    )
    publication_fallback_table = build_segmented_publication_fallback_table(segmented_state_fallback_summary)
    support_quality_summary = build_segmented_publication_support_quality_summary(publication_fallback_table)
    priority_states = build_segmented_publication_priority_states(publication_fallback_table)
    priority_summary = build_segmented_publication_priority_summary(support_quality_summary, priority_states)
    issue_breakdown = build_segmented_publication_issue_breakdown(publication_fallback_table)
    parser_focus_areas = build_segmented_publication_parser_focus_areas(
        support_quality_summary=support_quality_summary,
        ccdf_admin_long=ccdf_admin_long,
        ccdf_admin_state_year=ccdf_admin_state_year,
    )
    admin_sheet_targets = build_segmented_publication_admin_sheet_targets(
        parser_focus_areas=parser_focus_areas,
        ccdf_parse_inventory=ccdf_parse_inventory,
        ccdf_admin_long=ccdf_admin_long,
    )
    parser_action_plan = build_segmented_publication_parser_action_plan(
        priority_states=priority_states,
        issue_breakdown=issue_breakdown,
        parser_focus_areas=parser_focus_areas,
        admin_sheet_targets=admin_sheet_targets,
    )
    parser_action_plan_summary = build_segmented_publication_parser_action_plan_summary(parser_action_plan)
    publication_readout_markdown = build_segmented_publication_readout(
        headline_summary=segmented_headline_summary,
        publication_channel_table=publication_channel_table,
        publication_fallback_table=publication_fallback_table,
        support_quality_summary=support_quality_summary,
        priority_states=priority_states,
        issue_breakdown=issue_breakdown,
        parser_focus_areas=parser_focus_areas,
        parser_action_plan=parser_action_plan,
    )
    publication_report_markdown = build_segmented_publication_report(
        headline_summary=segmented_headline_summary,
        publication_channel_table=publication_channel_table,
        publication_fallback_table=publication_fallback_table,
        support_quality_summary=support_quality_summary,
        priority_states=priority_states,
        issue_breakdown=issue_breakdown,
        parser_focus_areas=parser_focus_areas,
        admin_sheet_targets=admin_sheet_targets,
        parser_action_plan=parser_action_plan,
    )
    return {
        "segmented_publication_headline_summary": dict(segmented_headline_summary),
        "segmented_publication_channel_table": publication_channel_table,
        "segmented_publication_fallback_table": publication_fallback_table,
        "segmented_publication_support_quality_summary": support_quality_summary,
        "segmented_publication_priority_states": priority_states,
        "segmented_publication_priority_summary": priority_summary,
        "segmented_publication_issue_breakdown": issue_breakdown,
        "segmented_publication_parser_focus_areas": parser_focus_areas,
        "segmented_publication_admin_sheet_targets": admin_sheet_targets,
        "segmented_publication_parser_action_plan": parser_action_plan,
        "segmented_publication_parser_action_plan_summary": parser_action_plan_summary,
        "segmented_publication_readout_markdown": publication_readout_markdown,
        "segmented_publication_report_markdown": publication_report_markdown,
    }
