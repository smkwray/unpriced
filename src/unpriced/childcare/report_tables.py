from __future__ import annotations

from typing import Iterable

import pandas as pd

LOCKED_CHANNEL_MAP = {
    "private_unsubsidized": "private_unsubsidized",
    "private_subsidized": "private_subsidized",
    "public_admin": "public_admin",
}

LOCKED_CHANNEL_ORDER = ["private_unsubsidized", "private_subsidized", "public_admin"]

POLICY_METADATA_COLUMNS = [
    "ccdf_policy_control_count",
    "ccdf_policy_control_support_status",
    "ccdf_policy_promoted_controls_selected",
    "ccdf_policy_promoted_control_rule",
    "ccdf_policy_promoted_min_state_year_coverage",
]


def _first_non_missing_text(values: Iterable[object], default: str = "missing") -> str:
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "missing":
            return text
    return default


def _normalize_q0_segmented(q0_segmented: pd.DataFrame) -> pd.DataFrame:
    required = {"state_fips", "year", "quantity_component", "quantity_slots"}
    missing = sorted(required - set(q0_segmented.columns))
    if missing:
        raise KeyError(f"q0_segmented missing required columns: {', '.join(missing)}")

    working = q0_segmented.copy()
    working["state_fips"] = working["state_fips"].astype(str).str.zfill(2)
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype("Int64")
    working = working.dropna(subset=["year"]).copy()
    working["quantity_component"] = working["quantity_component"].astype(str)
    working = working.loc[working["quantity_component"].isin(LOCKED_CHANNEL_MAP)].copy()
    working["solver_channel"] = working["quantity_component"].map(LOCKED_CHANNEL_MAP)
    working["quantity_slots"] = pd.to_numeric(working["quantity_slots"], errors="coerce").fillna(0.0).clip(lower=0.0)

    default_columns: dict[str, object] = {
        "segment_allocation_fallback": False,
        "ccdf_support_flag": "missing",
        "ccdf_admin_support_status": "missing",
        "q0_support_flag": "missing",
        "public_program_support_status": "missing",
        "total_paid_slots_target": 0.0,
        "reconciled_paid_slots": 0.0,
        "accounting_gap_from_target": 0.0,
    }
    for column, default in default_columns.items():
        if column not in working.columns:
            working[column] = default
    working["segment_allocation_fallback"] = working["segment_allocation_fallback"].fillna(False).astype(bool)
    for column in ("ccdf_support_flag", "ccdf_admin_support_status", "q0_support_flag", "public_program_support_status"):
        working[column] = working[column].fillna("missing").astype(str)
    for column in ("total_paid_slots_target", "reconciled_paid_slots", "accounting_gap_from_target"):
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)

    return working


def _normalize_promoted_controls(promoted_controls: pd.DataFrame | None) -> pd.DataFrame:
    if promoted_controls is None or promoted_controls.empty:
        columns = ["state_fips", "year"] + POLICY_METADATA_COLUMNS + ["ccdf_control_copayment_required"]
        return pd.DataFrame(columns=columns)
    required = {"state_fips", "year"}
    missing = sorted(required - set(promoted_controls.columns))
    if missing:
        raise KeyError(f"promoted controls missing required columns: {', '.join(missing)}")
    controls = promoted_controls.copy()
    controls["state_fips"] = controls["state_fips"].astype(str).str.zfill(2)
    controls["year"] = pd.to_numeric(controls["year"], errors="coerce").astype("Int64")
    controls = controls.dropna(subset=["year"]).copy()
    keep = ["state_fips", "year"]
    keep.extend(column for column in controls.columns if column.startswith("ccdf_control_"))
    keep.extend(column for column in POLICY_METADATA_COLUMNS if column in controls.columns)
    deduped = list(dict.fromkeys(keep))
    controls = controls[deduped].copy()
    if "ccdf_control_copayment_required" not in controls.columns:
        controls["ccdf_control_copayment_required"] = pd.NA
    if "ccdf_policy_control_count" not in controls.columns:
        controls["ccdf_policy_control_count"] = 0
    if "ccdf_policy_control_support_status" not in controls.columns:
        controls["ccdf_policy_control_support_status"] = "missing"
    return controls


def build_state_year_channel_summary(q0_segmented: pd.DataFrame) -> pd.DataFrame:
    q0 = _normalize_q0_segmented(q0_segmented)
    if q0.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "solver_channel",
                "quantity_slots",
                "channel_share_of_paid",
                "private_slots_total",
                "public_slots_total",
                "private_share_of_paid",
                "public_share_of_paid",
                "any_segment_allocation_fallback",
                "ccdf_support_flag",
                "ccdf_admin_support_status",
                "q0_support_flag",
                "public_program_support_status",
            ]
        )

    grouped = (
        q0.groupby(["state_fips", "year", "solver_channel"], as_index=False)
        .agg(
            quantity_slots=("quantity_slots", "sum"),
            any_segment_allocation_fallback=("segment_allocation_fallback", "max"),
            ccdf_support_flag=("ccdf_support_flag", _first_non_missing_text),
            ccdf_admin_support_status=("ccdf_admin_support_status", _first_non_missing_text),
            q0_support_flag=("q0_support_flag", _first_non_missing_text),
            public_program_support_status=("public_program_support_status", _first_non_missing_text),
        )
    )

    totals = (
        grouped.groupby(["state_fips", "year"], as_index=False)
        .agg(total_paid_slots=("quantity_slots", "sum"))
    )
    private_totals = (
        grouped.loc[grouped["solver_channel"].isin({"private_unsubsidized", "private_subsidized"})]
        .groupby(["state_fips", "year"], as_index=False)
        .agg(private_slots_total=("quantity_slots", "sum"))
    )
    public_totals = (
        grouped.loc[grouped["solver_channel"].eq("public_admin")]
        .groupby(["state_fips", "year"], as_index=False)
        .agg(public_slots_total=("quantity_slots", "sum"))
    )
    grouped = grouped.merge(totals, on=["state_fips", "year"], how="left")
    grouped = grouped.merge(private_totals, on=["state_fips", "year"], how="left")
    grouped = grouped.merge(public_totals, on=["state_fips", "year"], how="left")
    grouped["private_slots_total"] = pd.to_numeric(grouped["private_slots_total"], errors="coerce").fillna(0.0)
    grouped["public_slots_total"] = pd.to_numeric(grouped["public_slots_total"], errors="coerce").fillna(0.0)
    denominator = grouped["total_paid_slots"].replace({0.0: pd.NA})
    grouped["channel_share_of_paid"] = grouped["quantity_slots"].div(denominator).fillna(0.0).clip(lower=0.0)
    grouped["private_share_of_paid"] = grouped["private_slots_total"].div(denominator).fillna(0.0).clip(lower=0.0)
    grouped["public_share_of_paid"] = grouped["public_slots_total"].div(denominator).fillna(0.0).clip(lower=0.0)
    grouped["solver_channel"] = pd.Categorical(grouped["solver_channel"], categories=LOCKED_CHANNEL_ORDER, ordered=True)
    grouped = grouped.sort_values(["state_fips", "year", "solver_channel"], kind="stable").reset_index(drop=True)
    grouped["solver_channel"] = grouped["solver_channel"].astype(str)
    return grouped


def build_state_year_policy_quantity_summary(
    channel_summary: pd.DataFrame,
    promoted_controls: pd.DataFrame | None,
) -> pd.DataFrame:
    required = {"state_fips", "year", "solver_channel", "quantity_slots"}
    missing = sorted(required - set(channel_summary.columns))
    if missing:
        raise KeyError(f"channel_summary missing required columns: {', '.join(missing)}")
    base = channel_summary.copy()
    base["state_fips"] = base["state_fips"].astype(str).str.zfill(2)
    base["year"] = pd.to_numeric(base["year"], errors="coerce").astype("Int64")
    base = base.dropna(subset=["year"]).copy()

    controls = _normalize_promoted_controls(promoted_controls)
    merged = base.merge(controls, on=["state_fips", "year"], how="left")
    merged["ccdf_policy_control_count"] = pd.to_numeric(
        merged.get("ccdf_policy_control_count"), errors="coerce"
    ).fillna(0).astype(int)
    merged["ccdf_policy_control_support_status"] = (
        merged.get("ccdf_policy_control_support_status").fillna("missing").astype(str)
    )
    for column in POLICY_METADATA_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
    if "ccdf_control_copayment_required" not in merged.columns:
        merged["ccdf_control_copayment_required"] = pd.NA
    merged["promoted_control_observed"] = (
        merged["ccdf_control_copayment_required"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.len()
        .gt(0)
    )
    return merged.sort_values(["state_fips", "year", "solver_channel"], kind="stable").reset_index(drop=True)


def build_state_year_support_summary(
    q0_segmented: pd.DataFrame,
    utilization_diagnostics: pd.DataFrame | None = None,
    promoted_controls: pd.DataFrame | None = None,
) -> pd.DataFrame:
    q0 = _normalize_q0_segmented(q0_segmented)
    if q0.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "quantity_slots_total",
                "explicit_ccdf_row_count",
                "inferred_ccdf_row_count",
                "proxy_ccdf_row_count",
                "missing_ccdf_row_count",
            ]
        )

    ccdf_flag = q0["ccdf_support_flag"].fillna("missing").astype(str).str.lower()
    q0 = q0.assign(
        _ccdf_is_explicit=ccdf_flag.str.contains("explicit", na=False),
        _ccdf_is_inferred=ccdf_flag.str.contains("inferred", na=False),
        _ccdf_is_proxy=ccdf_flag.str.contains("proxy", na=False),
        _ccdf_is_missing=ccdf_flag.eq("missing"),
    )
    support = (
        q0.groupby(["state_fips", "year"], as_index=False)
        .agg(
            quantity_slots_total=("quantity_slots", "sum"),
            total_paid_slots_target=("total_paid_slots_target", "max"),
            reconciled_paid_slots=("reconciled_paid_slots", "max"),
            accounting_gap_from_target_max_abs=("accounting_gap_from_target", lambda values: float(pd.to_numeric(values, errors="coerce").abs().max())),
            any_segment_allocation_fallback=("segment_allocation_fallback", "max"),
            ccdf_support_flag=("ccdf_support_flag", _first_non_missing_text),
            ccdf_admin_support_status=("ccdf_admin_support_status", _first_non_missing_text),
            q0_support_flag=("q0_support_flag", _first_non_missing_text),
            public_program_support_status=("public_program_support_status", _first_non_missing_text),
            explicit_ccdf_row_count=("_ccdf_is_explicit", "sum"),
            inferred_ccdf_row_count=("_ccdf_is_inferred", "sum"),
            proxy_ccdf_row_count=("_ccdf_is_proxy", "sum"),
            missing_ccdf_row_count=("_ccdf_is_missing", "sum"),
        )
    )

    if utilization_diagnostics is not None and not utilization_diagnostics.empty:
        diagnostics = utilization_diagnostics.copy()
        required = {"state_fips", "year"}
        missing = sorted(required - set(diagnostics.columns))
        if missing:
            raise KeyError(f"utilization diagnostics missing required columns: {', '.join(missing)}")
        diagnostics["state_fips"] = diagnostics["state_fips"].astype(str).str.zfill(2)
        diagnostics["year"] = pd.to_numeric(diagnostics["year"], errors="coerce").astype("Int64")
        diagnostics = diagnostics.dropna(subset=["year"]).copy()
        keep_columns = [
            "state_fips",
            "year",
            "component_sum_gap",
            "any_private_allocation_fallback",
            "any_negative_quantity",
        ]
        keep = [column for column in keep_columns if column in diagnostics.columns]
        support = support.merge(diagnostics[keep], on=["state_fips", "year"], how="left")

    controls = _normalize_promoted_controls(promoted_controls)
    support = support.merge(controls, on=["state_fips", "year"], how="left")
    support["ccdf_policy_control_count"] = pd.to_numeric(
        support.get("ccdf_policy_control_count"), errors="coerce"
    ).fillna(0).astype(int)
    support["ccdf_policy_control_support_status"] = (
        support.get("ccdf_policy_control_support_status").fillna("missing").astype(str)
    )
    support["ccdf_control_copayment_required"] = support.get("ccdf_control_copayment_required")
    support["promoted_control_observed"] = (
        support["ccdf_control_copayment_required"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.len()
        .gt(0)
    )
    return support.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def build_childcare_report_tables(
    q0_segmented: pd.DataFrame,
    promoted_controls: pd.DataFrame | None = None,
    ccdf_policy_controls_state_year: pd.DataFrame | None = None,
    utilization_diagnostics: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    control_frame = promoted_controls if promoted_controls is not None else ccdf_policy_controls_state_year
    channel_summary = build_state_year_channel_summary(q0_segmented)
    policy_quantity_summary = build_state_year_policy_quantity_summary(
        channel_summary=channel_summary,
        promoted_controls=control_frame,
    )
    support_summary = build_state_year_support_summary(
        q0_segmented=q0_segmented,
        utilization_diagnostics=utilization_diagnostics,
        promoted_controls=control_frame,
    )
    return {
        "state_year_channel_summary": channel_summary,
        "state_year_policy_quantity_summary": policy_quantity_summary,
        "state_year_support_summary": support_summary,
    }
