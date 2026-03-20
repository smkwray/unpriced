from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

PAYMENT_PROXY_LARGE_GAP_SUPPORT_FLAG = "ccdf_split_proxy_from_payment_method_shares_large_gap"
PAYMENT_PROXY_LARGE_GAP_DOWNGRADED_SUPPORT_FLAG = (
    "ccdf_split_proxy_from_payment_method_shares_large_gap_downgraded_to_children_served_proxy"
)
PAYMENT_PROXY_LARGE_GAP_DOWNGRADED_SUPPORT_STATUS = (
    "observed_long_payment_method_share_proxy_large_gap_downgraded_to_children_served_proxy"
)


def _normalize_state_year_under5(acs_frame: pd.DataFrame) -> pd.DataFrame:
    required = {"state_fips", "year", "under5_population"}
    missing = sorted(required - set(acs_frame.columns))
    if missing:
        raise KeyError(f"ACS frame missing required columns: {', '.join(missing)}")
    working = acs_frame.copy()
    working["state_fips"] = working["state_fips"].astype(str).str.zfill(2)
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype("Int64")
    working["under5_population"] = pd.to_numeric(working["under5_population"], errors="coerce").fillna(0.0)
    state_year = (
        working.dropna(subset=["year"])
        .groupby(["state_fips", "year"], as_index=False)
        .agg(under5_population=("under5_population", "sum"))
    )
    state_year["under5_population"] = state_year["under5_population"].clip(lower=0.0)
    return state_year.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def build_public_program_slots_state_year(
    acs_frame: pd.DataFrame,
    head_start_frame: pd.DataFrame,
    ccdf_state_year: pd.DataFrame | None = None,
    head_start_reference_year: int | None = None,
) -> pd.DataFrame:
    state_year = _normalize_state_year_under5(acs_frame)
    result = state_year.copy()
    result["head_start_slots"] = 0.0
    result["ccdf_children_served"] = 0.0
    result["ccdf_public_admin_slots"] = 0.0
    result["ccdf_subsidized_private_slots"] = 0.0
    result["ccdf_total_expenditures"] = 0.0
    result["ccdf_grants_contracts_share"] = 0.0
    result["ccdf_certificates_share"] = 0.0
    result["ccdf_cash_share"] = 0.0
    result["ccdf_payment_method_total_children"] = 0.0
    result["ccdf_payment_method_gap_vs_children_served"] = 0.0
    result["ccdf_payment_method_ratio_vs_children_served"] = 0.0
    result["public_admin_slots"] = 0.0
    result["head_start_observed_by_year"] = False
    result["head_start_carry_forward"] = False
    result["head_start_reference_year"] = pd.NA
    result["public_program_support_status"] = "missing"
    result["head_start_support_flag"] = "missing"
    result["ccdf_support_flag"] = "missing"
    result["ccdf_admin_support_status"] = "missing"

    if not head_start_frame.empty:
        required = {"state_fips", "head_start_slots"}
        missing = sorted(required - set(head_start_frame.columns))
        if missing:
            raise KeyError(f"Head Start frame missing required columns: {', '.join(missing)}")

        head_start = head_start_frame.copy()
        head_start["state_fips"] = head_start["state_fips"].astype(str).str.zfill(2)
        head_start["head_start_slots"] = pd.to_numeric(head_start["head_start_slots"], errors="coerce").fillna(0.0)

        if "year" in head_start.columns:
            head_start["year"] = pd.to_numeric(head_start["year"], errors="coerce").astype("Int64")
            by_year = (
                head_start.dropna(subset=["year"])
                .groupby(["state_fips", "year"], as_index=False)
                .agg(head_start_slots=("head_start_slots", "sum"))
            )
            result = result.merge(by_year, on=["state_fips", "year"], how="left", suffixes=("", "_src"))
            result["head_start_slots"] = pd.to_numeric(result["head_start_slots_src"], errors="coerce").fillna(0.0)
            result = result.drop(columns=["head_start_slots_src"])
            observed = result["head_start_slots"].gt(0)
            result.loc[observed, "head_start_observed_by_year"] = True
            result.loc[observed, "public_program_support_status"] = "observed_yearly"
            result.loc[observed, "head_start_support_flag"] = "head_start_exact_year"
            result.loc[observed, "head_start_reference_year"] = result.loc[observed, "year"]
        else:
            cross_section = (
                head_start.groupby(["state_fips"], as_index=False)
                .agg(head_start_slots=("head_start_slots", "sum"))
            )
            result = result.merge(cross_section, on=["state_fips"], how="left", suffixes=("", "_src"))
            result["head_start_slots"] = pd.to_numeric(result["head_start_slots_src"], errors="coerce").fillna(0.0)
            result = result.drop(columns=["head_start_slots_src"])
            reference_year = head_start_reference_year
            if reference_year is None and not result.empty:
                reference_year = int(pd.to_numeric(result["year"], errors="coerce").max())
            carried = result["head_start_slots"].gt(0)
            result.loc[carried, "head_start_carry_forward"] = True
            result.loc[carried, "public_program_support_status"] = "cross_section_carry_forward"
            result.loc[carried, "head_start_support_flag"] = "head_start_cross_section_carryforward"
            if reference_year is not None:
                result.loc[carried, "head_start_reference_year"] = int(reference_year)

    if ccdf_state_year is not None and not ccdf_state_year.empty:
        required_ccdf = {"state_fips", "year"}
        missing_ccdf = sorted(required_ccdf - set(ccdf_state_year.columns))
        if missing_ccdf:
            raise KeyError(f"CCDF state-year frame missing required columns: {', '.join(missing_ccdf)}")
        ccdf = ccdf_state_year.copy()
        ccdf["state_fips"] = ccdf["state_fips"].astype(str).str.zfill(2)
        ccdf["year"] = pd.to_numeric(ccdf["year"], errors="coerce").astype("Int64")
        for column in (
            "ccdf_children_served",
            "ccdf_public_admin_slots",
            "ccdf_subsidized_private_slots",
            "ccdf_total_expenditures",
            "ccdf_grants_contracts_share",
            "ccdf_certificates_share",
            "ccdf_cash_share",
            "ccdf_payment_method_total_children",
            "ccdf_payment_method_gap_vs_children_served",
            "ccdf_payment_method_ratio_vs_children_served",
        ):
            if column not in ccdf.columns:
                ccdf[column] = 0.0
            ccdf[column] = pd.to_numeric(ccdf[column], errors="coerce").fillna(0.0)
        for column in (
            "ccdf_children_served",
            "ccdf_public_admin_slots",
            "ccdf_subsidized_private_slots",
            "ccdf_total_expenditures",
            "ccdf_grants_contracts_share",
            "ccdf_certificates_share",
            "ccdf_cash_share",
            "ccdf_payment_method_total_children",
        ):
            ccdf[column] = ccdf[column].clip(lower=0.0)
        if "ccdf_support_flag" not in ccdf.columns:
            ccdf["ccdf_support_flag"] = np.where(
                ccdf["ccdf_public_admin_slots"].gt(0),
                "ccdf_observed_state_year",
                "missing",
            )
        if "ccdf_admin_support_status" not in ccdf.columns:
            ccdf["ccdf_admin_support_status"] = np.where(
                ccdf["ccdf_public_admin_slots"].gt(0),
                "observed_state_year",
                "missing",
            )
        large_gap_proxy = ccdf["ccdf_support_flag"].astype(str).str.startswith(PAYMENT_PROXY_LARGE_GAP_SUPPORT_FLAG)
        if bool(large_gap_proxy.any()):
            # Defensive downgrade in case pre-built CCDF layers still carry the legacy large-gap split label.
            ccdf.loc[large_gap_proxy, "ccdf_subsidized_private_slots"] = (
                pd.to_numeric(ccdf.loc[large_gap_proxy, "ccdf_children_served"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )
            ccdf.loc[large_gap_proxy, "ccdf_public_admin_slots"] = 0.0
            ccdf.loc[large_gap_proxy, "ccdf_support_flag"] = PAYMENT_PROXY_LARGE_GAP_DOWNGRADED_SUPPORT_FLAG
            ccdf.loc[large_gap_proxy, "ccdf_admin_support_status"] = PAYMENT_PROXY_LARGE_GAP_DOWNGRADED_SUPPORT_STATUS
        keep_columns = [
            "state_fips",
            "year",
            "ccdf_children_served",
            "ccdf_public_admin_slots",
            "ccdf_subsidized_private_slots",
            "ccdf_total_expenditures",
            "ccdf_grants_contracts_share",
            "ccdf_certificates_share",
            "ccdf_cash_share",
            "ccdf_payment_method_total_children",
            "ccdf_payment_method_gap_vs_children_served",
            "ccdf_payment_method_ratio_vs_children_served",
            "ccdf_support_flag",
            "ccdf_admin_support_status",
        ]
        result = result.merge(ccdf[keep_columns], on=["state_fips", "year"], how="left", suffixes=("", "_src"))
        for column in (
            "ccdf_children_served",
            "ccdf_public_admin_slots",
            "ccdf_subsidized_private_slots",
            "ccdf_total_expenditures",
            "ccdf_grants_contracts_share",
            "ccdf_certificates_share",
            "ccdf_cash_share",
            "ccdf_payment_method_total_children",
            "ccdf_payment_method_gap_vs_children_served",
            "ccdf_payment_method_ratio_vs_children_served",
        ):
            src_column = f"{column}_src"
            if src_column in result.columns:
                result[column] = pd.to_numeric(result[src_column], errors="coerce").fillna(0.0)
                result = result.drop(columns=[src_column])
        for column in (
            "ccdf_children_served",
            "ccdf_public_admin_slots",
            "ccdf_subsidized_private_slots",
            "ccdf_total_expenditures",
            "ccdf_grants_contracts_share",
            "ccdf_certificates_share",
            "ccdf_cash_share",
            "ccdf_payment_method_total_children",
        ):
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0).clip(lower=0.0)
        for column in ("ccdf_support_flag", "ccdf_admin_support_status"):
            src_column = f"{column}_src"
            if src_column in result.columns:
                result[column] = result[src_column].fillna(result[column]).astype(str)
                result = result.drop(columns=[src_column])

    result["head_start_slots"] = pd.to_numeric(result["head_start_slots"], errors="coerce").fillna(0.0).clip(lower=0.0)
    result["public_admin_slots"] = (
        pd.to_numeric(result["head_start_slots"], errors="coerce").fillna(0.0)
        + pd.to_numeric(result["ccdf_public_admin_slots"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    ccdf_observed = pd.to_numeric(result["ccdf_public_admin_slots"], errors="coerce").gt(0)
    result.loc[ccdf_observed, "public_program_support_status"] = np.where(
        result.loc[ccdf_observed, "head_start_slots"].gt(0),
        "head_start_plus_ccdf_observed",
        "ccdf_observed",
    )
    result["public_admin_slot_share_under5"] = (
        result["public_admin_slots"]
        .div(result["under5_population"].replace({0.0: pd.NA}))
        .fillna(0.0)
        .clip(lower=0.0)
    )
    return result.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def _select_sipp_under5_rate_by_year(sipp_frame: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "subgroup", "any_paid_childcare_rate"}
    missing = sorted(required - set(sipp_frame.columns))
    if missing:
        raise KeyError(f"SIPP frame missing required columns: {', '.join(missing)}")
    working = sipp_frame.copy()
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype("Int64")
    working["any_paid_childcare_rate"] = pd.to_numeric(
        working["any_paid_childcare_rate"], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    for column in (
        "center_care_rate",
        "family_daycare_rate",
        "nonrelative_care_rate",
        "head_start_rate",
        "nursery_preschool_rate",
    ):
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    rates = (
        working.loc[working["subgroup"].astype(str).eq("under5_ref_parents")]
        .dropna(subset=["year", "any_paid_childcare_rate"])
        .sort_values("year", kind="stable")
        .drop_duplicates("year", keep="last")
        .reset_index(drop=True)
    )
    if rates.empty:
        raise ValueError("SIPP frame has no usable under5_ref_parents paid-care rates")
    return rates


def _resolve_sipp_rate(target_year: int, rates_by_year: pd.DataFrame) -> tuple[int, pd.Series, bool, str]:
    exact = rates_by_year.loc[rates_by_year["year"].eq(target_year)]
    if not exact.empty:
        row = exact.iloc[0]
        return int(row["year"]), row, False, "sipp_under5_exact_year"

    previous = rates_by_year.loc[rates_by_year["year"].lt(target_year)]
    if not previous.empty:
        row = previous.iloc[-1]
        return int(row["year"]), row, True, "sipp_under5_carry_forward"

    row = rates_by_year.iloc[0]
    return int(row["year"]), row, True, "sipp_under5_carry_backward"


def build_survey_paid_use_targets(acs_frame: pd.DataFrame, sipp_frame: pd.DataFrame) -> pd.DataFrame:
    state_year = _normalize_state_year_under5(acs_frame)
    rates_by_year = _select_sipp_under5_rate_by_year(sipp_frame)
    unique_years = sorted(pd.to_numeric(state_year["year"], errors="coerce").dropna().astype(int).unique().tolist())
    resolved_rows = []
    for year in unique_years:
        used_year, row, carry_forward, source = _resolve_sipp_rate(year, rates_by_year)
        any_paid_rate = float(pd.to_numeric(pd.Series([row["any_paid_childcare_rate"]]), errors="coerce").fillna(0.0).iloc[0])
        center_rate = float(pd.to_numeric(pd.Series([row.get("center_care_rate", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        family_rate = float(pd.to_numeric(pd.Series([row.get("family_daycare_rate", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        nonrelative_rate = float(pd.to_numeric(pd.Series([row.get("nonrelative_care_rate", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        head_start_rate = float(pd.to_numeric(pd.Series([row.get("head_start_rate", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        nursery_rate = float(pd.to_numeric(pd.Series([row.get("nursery_preschool_rate", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        resolved_rows.append(
            {
                "year": int(year),
                "survey_rate_year_used": used_year,
                "sipp_any_paid_childcare_rate": any_paid_rate,
                "survey_rate_carry_forward": bool(carry_forward),
                "survey_rate_source": source,
                "survey_rate_support_flag": (
                    f"sipp_under5_exact_{used_year}"
                    if source == "sipp_under5_exact_year"
                    else (
                        f"sipp_under5_carryforward_{used_year}"
                        if source == "sipp_under5_carry_forward"
                        else f"sipp_under5_carryback_{used_year}"
                    )
                ),
                "center_care_rate": center_rate,
                "family_daycare_rate": family_rate,
                "nonrelative_care_rate": nonrelative_rate,
                "head_start_rate": head_start_rate,
                "nursery_preschool_rate": nursery_rate,
            }
        )
    mapping = pd.DataFrame(resolved_rows)
    result = state_year.merge(mapping, on="year", how="left")
    result["total_paid_slots_target"] = (
        pd.to_numeric(result["under5_population"], errors="coerce").fillna(0.0)
        * pd.to_numeric(result["sipp_any_paid_childcare_rate"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    result["total_paid_slots_from_surveys"] = result["total_paid_slots_target"]
    result["any_paid_childcare_rate"] = result["sipp_any_paid_childcare_rate"]
    result["survey_target_support_status"] = np.where(
        result["survey_rate_carry_forward"].fillna(False),
        "carry_forward",
        "exact_year",
    )
    return result.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def _private_segment_shares(ndcp_segment_prices: pd.DataFrame) -> pd.DataFrame:
    required = {"state_fips", "year", "segment_id"}
    missing = sorted(required - set(ndcp_segment_prices.columns))
    if missing:
        raise KeyError(f"Segment price frame missing required columns: {', '.join(missing)}")
    working = ndcp_segment_prices.copy()
    working["state_fips"] = working["state_fips"].astype(str).str.zfill(2)
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype("Int64")
    if "segment_channel" in working.columns:
        working = working.loc[working["segment_channel"].astype(str).eq("private")].copy()
    else:
        working["segment_channel"] = "private"
    if "segment_label" not in working.columns:
        working["segment_label"] = working["segment_id"].astype(str)
    if "segment_weight_sum" in working.columns:
        weight_series = pd.to_numeric(working["segment_weight_sum"], errors="coerce").fillna(0.0)
    elif "segment_observation_count" in working.columns:
        weight_series = pd.to_numeric(working["segment_observation_count"], errors="coerce").fillna(0.0)
    else:
        weight_series = pd.Series(1.0, index=working.index)
    working["allocation_weight"] = weight_series.clip(lower=0.0)
    grouped = (
        working.dropna(subset=["year"])
        .groupby(["state_fips", "year", "segment_id", "segment_label", "segment_channel"], as_index=False)
        .agg(allocation_weight=("allocation_weight", "sum"))
    )
    totals = grouped.groupby(["state_fips", "year"], as_index=False).agg(
        total_weight=("allocation_weight", "sum"),
        n_segments=("segment_id", "nunique"),
    )
    merged = grouped.merge(totals, on=["state_fips", "year"], how="left")
    merged["segment_share"] = np.where(
        pd.to_numeric(merged["total_weight"], errors="coerce").gt(0),
        pd.to_numeric(merged["allocation_weight"], errors="coerce")
        .div(pd.to_numeric(merged["total_weight"], errors="coerce")),
        1.0 / pd.to_numeric(merged["n_segments"], errors="coerce").replace({0: pd.NA}),
    )
    merged["segment_share"] = pd.to_numeric(merged["segment_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return merged[
        [
            "state_fips",
            "year",
            "segment_id",
            "segment_label",
            "segment_channel",
            "segment_share",
        ]
    ].sort_values(["state_fips", "year", "segment_id"], kind="stable").reset_index(drop=True)


def _series_or_default(frame: pd.DataFrame, column: str, default: float | str = 0.0) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def build_segmented_quantity_panel(
    public_program_slots_state_year: pd.DataFrame,
    survey_paid_use_targets: pd.DataFrame,
    ndcp_segment_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required_public = {"state_fips", "year", "public_admin_slots"}
    required_survey = {"state_fips", "year"}
    public_frame = public_program_slots_state_year
    survey_frame = survey_paid_use_targets
    if not required_public.issubset(set(public_frame.columns)) and required_public.issubset(
        set(survey_frame.columns)
    ):
        # Backward compatibility for call sites that pass survey/public in the opposite order.
        public_frame = survey_paid_use_targets
        survey_frame = public_program_slots_state_year

    missing_public = sorted(required_public - set(public_frame.columns))
    if missing_public:
        raise KeyError(f"Public-program frame missing required columns: {', '.join(missing_public)}")
    missing_survey = sorted(required_survey - set(survey_frame.columns))
    if missing_survey:
        raise KeyError(f"Survey-target frame missing required columns: {', '.join(missing_survey)}")
    if "total_paid_slots_target" not in survey_frame.columns and "total_paid_slots_from_surveys" not in survey_frame.columns:
        raise KeyError("Survey-target frame missing required columns: total_paid_slots_target")

    survey = survey_frame.copy()
    if "total_paid_slots_target" not in survey.columns:
        survey["total_paid_slots_target"] = pd.to_numeric(
            survey.get("total_paid_slots_from_surveys"), errors="coerce"
        )
    public_columns = [
        "state_fips",
        "year",
        "public_admin_slots",
        "ccdf_public_admin_slots",
        "ccdf_subsidized_private_slots",
        "ccdf_children_served",
        "ccdf_support_flag",
        "ccdf_admin_support_status",
        "public_program_support_status",
        "head_start_observed_by_year",
        "head_start_carry_forward",
        "head_start_reference_year",
    ]
    available_public_columns = [column for column in public_columns if column in public_frame.columns]
    merged = survey.merge(public_frame[available_public_columns], on=["state_fips", "year"], how="outer")
    merged["state_fips"] = merged["state_fips"].astype(str).str.zfill(2)
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("Int64")
    merged = merged.dropna(subset=["year"]).reset_index(drop=True)
    merged["public_admin_slots"] = pd.to_numeric(merged["public_admin_slots"], errors="coerce").fillna(0.0).clip(lower=0.0)
    merged["ccdf_public_admin_slots"] = pd.to_numeric(
        _series_or_default(merged, "ccdf_public_admin_slots"), errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    merged["ccdf_subsidized_private_slots"] = pd.to_numeric(
        _series_or_default(merged, "ccdf_subsidized_private_slots"), errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    merged["ccdf_children_served"] = pd.to_numeric(
        _series_or_default(merged, "ccdf_children_served"), errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    merged["total_paid_slots_target"] = pd.to_numeric(
        merged["total_paid_slots_target"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    merged["subsidized_private_slots"] = merged["ccdf_subsidized_private_slots"].clip(lower=0.0)
    merged["private_paid_slots"] = (
        merged["total_paid_slots_target"] - merged["public_admin_slots"] - merged["subsidized_private_slots"]
    ).clip(lower=0.0)
    merged["reconciled_paid_slots"] = (
        merged["public_admin_slots"] + merged["subsidized_private_slots"] + merged["private_paid_slots"]
    )
    merged["accounting_gap_from_target"] = merged["reconciled_paid_slots"] - merged["total_paid_slots_target"]

    shares = pd.DataFrame(
        columns=["state_fips", "year", "segment_id", "segment_label", "segment_channel", "segment_share"]
    )
    if ndcp_segment_prices is not None and not ndcp_segment_prices.empty:
        shares = _private_segment_shares(ndcp_segment_prices)

    rows: list[dict[str, Any]] = []
    for base in merged.to_dict(orient="records"):
        state_fips = str(base["state_fips"]).zfill(2)
        year = int(base["year"])
        unsubsidized_private_slots = float(base["private_paid_slots"])
        subsidized_private_slots = float(base.get("subsidized_private_slots", 0.0))
        share_subset = shares.loc[
            shares["state_fips"].astype(str).eq(state_fips)
            & pd.to_numeric(shares["year"], errors="coerce").eq(year)
        ]
        used_fallback = share_subset.empty
        if used_fallback:
            share_subset = pd.DataFrame(
                [
                    {
                        "segment_id": "private_all",
                        "segment_label": "private / all",
                        "segment_channel": "private",
                        "segment_share": 1.0,
                    }
                ]
            )
        for private_segment in share_subset.to_dict(orient="records"):
            share = float(private_segment.get("segment_share", 0.0))
            subsidized_quantity = max(subsidized_private_slots * share, 0.0)
            unsubsidized_quantity = max(unsubsidized_private_slots * share, 0.0)
            if subsidized_quantity > 0:
                rows.append(
                    {
                        "state_fips": state_fips,
                        "year": year,
                        "segment_id": str(private_segment["segment_id"]),
                        "segment_label": str(private_segment.get("segment_label", private_segment["segment_id"])),
                        "segment_channel": "subsidized_private",
                        "quantity_component": "private_subsidized",
                        "quantity_slots": subsidized_quantity,
                        "q0_slots": subsidized_quantity,
                        "segment_allocation_source": "ndcp_segment_weights"
                        if not used_fallback
                        else "fallback_single_private_segment",
                        "segment_allocation_fallback": bool(used_fallback),
                        "ndcp_segment_support": bool(not used_fallback),
                        "total_paid_slots_target": float(base["total_paid_slots_target"]),
                        "public_admin_slots": float(base["public_admin_slots"]),
                        "ccdf_public_admin_slots": float(base.get("ccdf_public_admin_slots", 0.0)),
                        "ccdf_subsidized_private_slots": float(base.get("ccdf_subsidized_private_slots", 0.0)),
                        "ccdf_children_served": float(base.get("ccdf_children_served", 0.0)),
                        "subsidized_private_slots": subsidized_private_slots,
                        "private_paid_slots": unsubsidized_private_slots,
                        "reconciled_paid_slots": float(base["reconciled_paid_slots"]),
                        "accounting_gap_from_target": float(base["accounting_gap_from_target"]),
                        "public_program_support_status": str(base.get("public_program_support_status", "missing")),
                        "head_start_observed_by_year": bool(base.get("head_start_observed_by_year", False)),
                        "head_start_carry_forward": bool(base.get("head_start_carry_forward", False)),
                        "head_start_reference_year": base.get("head_start_reference_year", pd.NA),
                        "survey_rate_source": str(base.get("survey_rate_source", "missing")),
                        "survey_rate_carry_forward": bool(base.get("survey_rate_carry_forward", False)),
                        "q0_support_flag": (
                            "ccdf_subsidized_private_plus_private_unallocated_fallback"
                            if used_fallback
                            else "ccdf_subsidized_private_allocated_by_observed_segment_weights"
                        ),
                    }
                )
            rows.append(
                {
                    "state_fips": state_fips,
                    "year": year,
                    "segment_id": str(private_segment["segment_id"]),
                    "segment_label": str(private_segment.get("segment_label", private_segment["segment_id"])),
                    "segment_channel": "private",
                    "quantity_component": "private_unsubsidized",
                    "quantity_slots": unsubsidized_quantity,
                    "q0_slots": unsubsidized_quantity,
                    "segment_allocation_source": "ndcp_segment_weights"
                    if not used_fallback
                    else "fallback_single_private_segment",
                    "segment_allocation_fallback": bool(used_fallback),
                    "ndcp_segment_support": bool(not used_fallback),
                    "total_paid_slots_target": float(base["total_paid_slots_target"]),
                    "public_admin_slots": float(base["public_admin_slots"]),
                    "ccdf_public_admin_slots": float(base.get("ccdf_public_admin_slots", 0.0)),
                    "ccdf_subsidized_private_slots": float(base.get("ccdf_subsidized_private_slots", 0.0)),
                    "ccdf_children_served": float(base.get("ccdf_children_served", 0.0)),
                    "subsidized_private_slots": subsidized_private_slots,
                    "private_paid_slots": unsubsidized_private_slots,
                    "reconciled_paid_slots": float(base["reconciled_paid_slots"]),
                    "accounting_gap_from_target": float(base["accounting_gap_from_target"]),
                    "public_program_support_status": str(base.get("public_program_support_status", "missing")),
                    "head_start_observed_by_year": bool(base.get("head_start_observed_by_year", False)),
                    "head_start_carry_forward": bool(base.get("head_start_carry_forward", False)),
                    "head_start_reference_year": base.get("head_start_reference_year", pd.NA),
                    "survey_rate_source": str(base.get("survey_rate_source", "missing")),
                    "survey_rate_carry_forward": bool(base.get("survey_rate_carry_forward", False)),
                    "q0_support_flag": (
                        "ccdf_plus_private_unallocated_fallback"
                        if used_fallback and float(base.get("ccdf_public_admin_slots", 0.0)) > 0
                        else (
                            "ccdf_plus_observed_segment_weights"
                            if float(base.get("ccdf_public_admin_slots", 0.0)) > 0
                            else (
                                "private_unallocated_fallback"
                                if used_fallback
                                else "observed_segment_weights"
                            )
                        )
                    ),
                }
            )
        public_quantity = float(base["public_admin_slots"])
        rows.append(
            {
                "state_fips": state_fips,
                "year": year,
                "segment_id": "public_head_start",
                "segment_label": "public / head_start",
                "segment_channel": "public",
                "quantity_component": "public_admin",
                "quantity_slots": public_quantity,
                "q0_slots": public_quantity,
                "segment_allocation_source": "public_admin_observed_or_carry_forward",
                "segment_allocation_fallback": False,
                "ndcp_segment_support": False,
                "total_paid_slots_target": float(base["total_paid_slots_target"]),
                "public_admin_slots": float(base["public_admin_slots"]),
                "ccdf_public_admin_slots": float(base.get("ccdf_public_admin_slots", 0.0)),
                "ccdf_subsidized_private_slots": float(base.get("ccdf_subsidized_private_slots", 0.0)),
                "ccdf_children_served": float(base.get("ccdf_children_served", 0.0)),
                "subsidized_private_slots": subsidized_private_slots,
                "private_paid_slots": unsubsidized_private_slots,
                "reconciled_paid_slots": float(base["reconciled_paid_slots"]),
                "accounting_gap_from_target": float(base["accounting_gap_from_target"]),
                "public_program_support_status": str(base.get("public_program_support_status", "missing")),
                "head_start_observed_by_year": bool(base.get("head_start_observed_by_year", False)),
                "head_start_carry_forward": bool(base.get("head_start_carry_forward", False)),
                "head_start_reference_year": base.get("head_start_reference_year", pd.NA),
                "survey_rate_source": str(base.get("survey_rate_source", "missing")),
                "survey_rate_carry_forward": bool(base.get("survey_rate_carry_forward", False)),
                "q0_support_flag": (
                    str(base.get("ccdf_support_flag"))
                    if float(base.get("ccdf_public_admin_slots", 0.0)) > 0
                    else str(base.get("head_start_support_flag", "missing"))
                ),
            }
        )
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    panel["quantity_slots"] = pd.to_numeric(panel["quantity_slots"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return panel.sort_values(
        ["state_fips", "year", "quantity_component", "segment_channel", "segment_id"], kind="stable"
    ).reset_index(drop=True)


def _build_reconciliation_diagnostics(
    segmented_quantity_panel: pd.DataFrame,
    ccdf_policy_features_state_year: pd.DataFrame | None = None,
    ccdf_policy_controls_state_year: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if segmented_quantity_panel.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "total_paid_slots_target",
                "public_admin_slots",
                "subsidized_private_slots",
                "private_paid_slots",
                "reconciled_paid_slots",
                "component_slots_sum",
                "component_sum_gap",
                "any_private_allocation_fallback",
                "any_negative_quantity",
            ]
        )
    grouped = (
        segmented_quantity_panel.groupby(["state_fips", "year"], as_index=False)
        .agg(
            total_paid_slots_target=("total_paid_slots_target", "max"),
            public_admin_slots=("public_admin_slots", "max"),
            subsidized_private_slots=("subsidized_private_slots", "max"),
            private_paid_slots=("private_paid_slots", "max"),
            reconciled_paid_slots=("reconciled_paid_slots", "max"),
            component_slots_sum=("quantity_slots", "sum"),
            any_private_allocation_fallback=("segment_allocation_fallback", "max"),
            any_negative_quantity=("quantity_slots", lambda values: bool(pd.to_numeric(values, errors="coerce").lt(0).any())),
        )
    )
    grouped["component_sum_gap"] = grouped["component_slots_sum"] - grouped["reconciled_paid_slots"]
    if ccdf_policy_controls_state_year is not None and not ccdf_policy_controls_state_year.empty:
        controls = ccdf_policy_controls_state_year.copy()
        required = {"state_fips", "year"}
        missing = sorted(required - set(controls.columns))
        if missing:
            raise KeyError(f"CCDF policy controls frame missing required columns: {', '.join(missing)}")
        controls["state_fips"] = controls["state_fips"].astype(str).str.zfill(2)
        controls["year"] = pd.to_numeric(controls["year"], errors="coerce").astype("Int64")
        controls = controls.dropna(subset=["year"]).copy()
        grouped = grouped.merge(controls, on=["state_fips", "year"], how="left")
    if ccdf_policy_features_state_year is not None and not ccdf_policy_features_state_year.empty:
        policy = ccdf_policy_features_state_year.copy()
        required = {"state_fips", "year"}
        missing = sorted(required - set(policy.columns))
        if missing:
            raise KeyError(f"CCDF policy feature frame missing required columns: {', '.join(missing)}")
        policy["state_fips"] = policy["state_fips"].astype(str).str.zfill(2)
        policy["year"] = pd.to_numeric(policy["year"], errors="coerce").astype("Int64")
        policy = policy.dropna(subset=["year"]).copy()
        grouped = grouped.merge(policy, on=["state_fips", "year"], how="left")
    return grouped.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def _filter_year_window(frame: pd.DataFrame, config: Mapping[str, Any] | None) -> pd.DataFrame:
    if config is None or frame.empty or "year" not in frame.columns:
        return frame.copy()
    year_window = config.get("year_window", {})
    start = year_window.get("start")
    end = year_window.get("end")
    result = frame.copy()
    years = pd.to_numeric(result["year"], errors="coerce")
    if start is not None:
        result = result.loc[years.ge(int(start))].copy()
        years = pd.to_numeric(result["year"], errors="coerce")
    if end is not None:
        result = result.loc[years.le(int(end))].copy()
    return result.reset_index(drop=True)


def build_utilization_stack(
    acs_frame: pd.DataFrame,
    sipp_frame: pd.DataFrame,
    head_start_frame: pd.DataFrame,
    ccdf_state_year: pd.DataFrame | None = None,
    ccdf_policy_features_state_year: pd.DataFrame | None = None,
    ccdf_policy_controls_state_year: pd.DataFrame | None = None,
    ndcp_segment_prices: pd.DataFrame | None = None,
    head_start_reference_year: int | None = None,
) -> dict[str, pd.DataFrame]:
    public_program_slots = build_public_program_slots_state_year(
        acs_frame=acs_frame,
        head_start_frame=head_start_frame,
        ccdf_state_year=ccdf_state_year,
        head_start_reference_year=head_start_reference_year,
    )
    survey_targets = build_survey_paid_use_targets(
        acs_frame=acs_frame,
        sipp_frame=sipp_frame,
    )
    segmented_quantity = build_segmented_quantity_panel(
        public_program_slots_state_year=public_program_slots,
        survey_paid_use_targets=survey_targets,
        ndcp_segment_prices=ndcp_segment_prices,
    )
    diagnostics = _build_reconciliation_diagnostics(
        segmented_quantity,
        ccdf_policy_features_state_year=ccdf_policy_features_state_year,
        ccdf_policy_controls_state_year=ccdf_policy_controls_state_year,
    )
    return {
        "public_program_slots_state_year": public_program_slots,
        "survey_paid_use_targets": survey_targets,
        "q0_segmented": segmented_quantity,
        "utilization_reconciliation_diagnostics": diagnostics,
    }


def build_childcare_utilization_outputs(
    acs_frame: pd.DataFrame,
    sipp_frame: pd.DataFrame,
    head_start_frame: pd.DataFrame,
    segment_price_panel: pd.DataFrame,
    ccdf_state_year: pd.DataFrame | None = None,
    ccdf_policy_features_state_year: pd.DataFrame | None = None,
    ccdf_policy_controls_state_year: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    filtered_acs = _filter_year_window(acs_frame, config)
    filtered_sipp = _filter_year_window(sipp_frame, config)
    filtered_segments = _filter_year_window(segment_price_panel, config)
    filtered_ccdf = _filter_year_window(ccdf_state_year, config) if ccdf_state_year is not None else None
    filtered_ccdf_policy = (
        _filter_year_window(ccdf_policy_features_state_year, config)
        if ccdf_policy_features_state_year is not None
        else None
    )
    filtered_ccdf_policy_controls = (
        _filter_year_window(ccdf_policy_controls_state_year, config)
        if ccdf_policy_controls_state_year is not None
        else None
    )
    outputs = build_utilization_stack(
        acs_frame=filtered_acs,
        sipp_frame=filtered_sipp,
        head_start_frame=head_start_frame,
        ccdf_state_year=filtered_ccdf,
        ccdf_policy_features_state_year=filtered_ccdf_policy,
        ccdf_policy_controls_state_year=filtered_ccdf_policy_controls,
        ndcp_segment_prices=filtered_segments,
    )
    return {
        "public_program_slots": outputs["public_program_slots_state_year"],
        "survey_paid_use_targets": outputs["survey_paid_use_targets"],
        "quantity_by_segment": outputs["q0_segmented"],
        "reconciliation_diagnostics": outputs["utilization_reconciliation_diagnostics"],
    }
