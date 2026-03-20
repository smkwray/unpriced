from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"state_fips", "year"}

RULE_SPECS: tuple[dict[str, str], ...] = (
    {
        "rule_column": "center_infant_ratio",
        "provider_type": "center",
        "age_group": "infant",
        "rule_family": "max_children_per_staff",
        "strictness_direction": "lower_is_stricter",
    },
    {
        "rule_column": "center_toddler_ratio",
        "provider_type": "center",
        "age_group": "toddler",
        "rule_family": "max_children_per_staff",
        "strictness_direction": "lower_is_stricter",
    },
    {
        "rule_column": "center_infant_group_size",
        "provider_type": "center",
        "age_group": "infant",
        "rule_family": "max_group_size",
        "strictness_direction": "lower_is_stricter",
    },
    {
        "rule_column": "center_toddler_group_size",
        "provider_type": "center",
        "age_group": "toddler",
        "rule_family": "max_group_size",
        "strictness_direction": "lower_is_stricter",
    },
    {
        "rule_column": "center_labor_intensity_index",
        "provider_type": "center",
        "age_group": "all",
        "rule_family": "labor_intensity_index",
        "strictness_direction": "higher_is_stricter",
    },
)

SOURCE_METADATA_COLUMNS = ("shock_label", "effective_date", "source_url", "source_note")


def _normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _normalize_licensing_shocks(licensing_shocks: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(licensing_shocks.columns))
    if missing:
        raise KeyError(f"licensing shocks missing required columns: {', '.join(missing)}")

    frame = licensing_shocks.copy()
    frame["state_fips"] = frame["state_fips"].astype(str).str.zfill(2)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["state_fips", "year"]).copy()

    numeric_columns = [spec["rule_column"] for spec in RULE_SPECS]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in SOURCE_METADATA_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype(str)

    frame["_input_row_number"] = np.arange(1, len(frame) + 1)
    frame = frame.sort_values(["state_fips", "year", "_input_row_number"], kind="stable").reset_index(drop=True)
    return frame


def _strictness_sign(value: str) -> float:
    return 1.0 if value == "higher_is_stricter" else -1.0


def _is_rule_level_contract(frame: pd.DataFrame) -> bool:
    required = {"provider_type", "age_group", "rule_family"}
    return required.issubset(frame.columns) and (
        "rule_value" in frame.columns or "rule_value_observed" in frame.columns
    )


def _build_rule_level_rows(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "rule_value_observed" in working.columns:
        working["rule_value_observed"] = pd.to_numeric(working["rule_value_observed"], errors="coerce")
    elif "rule_value" in working.columns:
        working["rule_value_observed"] = pd.to_numeric(working["rule_value"], errors="coerce")
    else:
        working["rule_value_observed"] = np.nan

    if "strictness_direction" not in working.columns:
        default_directions = {
            "max_children_per_staff": "lower_is_stricter",
            "max_group_size": "lower_is_stricter",
            "labor_intensity_index": "higher_is_stricter",
        }
        working["strictness_direction"] = (
            working["rule_family"].astype(str).map(default_directions).fillna("lower_is_stricter")
        )

    if "rule_column_source" not in working.columns:
        working["rule_column_source"] = working.get("rule_name", "rule_value")

    if "rule_id" not in working.columns:
        working["rule_id"] = (
            working["provider_type"].astype(str)
            + "__"
            + working["age_group"].astype(str)
            + "__"
            + working["rule_family"].astype(str)
        )

    working["strictness_sign"] = working["strictness_direction"].astype(str).map(_strictness_sign)
    for column in SOURCE_METADATA_COLUMNS:
        if column not in working.columns:
            working[column] = pd.NA

    keep = [
        "state_fips",
        "year",
        "provider_type",
        "age_group",
        "rule_family",
        "rule_id",
        "rule_column_source",
        "strictness_direction",
        "strictness_sign",
        "rule_value_observed",
        "shock_label",
        "effective_date",
        "source_url",
        "source_note",
    ]
    return working[keep].copy()


def _build_wide_rule_rows(frame: pd.DataFrame) -> pd.DataFrame:
    observed_rows: list[dict[str, Any]] = []
    for spec in RULE_SPECS:
        source_column = spec["rule_column"]
        for _, row in frame.iterrows():
            observed_rows.append(
                {
                    "state_fips": row["state_fips"],
                    "year": int(row["year"]),
                    "provider_type": spec["provider_type"],
                    "age_group": spec["age_group"],
                    "rule_family": spec["rule_family"],
                    "rule_id": f"{spec['provider_type']}__{spec['age_group']}__{spec['rule_family']}",
                    "rule_column_source": source_column,
                    "strictness_direction": spec["strictness_direction"],
                    "strictness_sign": _strictness_sign(spec["strictness_direction"]),
                    "rule_value_observed": (
                        float(row[source_column])
                        if source_column in frame.columns and pd.notna(row[source_column])
                        else np.nan
                    ),
                    "shock_label": row.get("shock_label"),
                    "effective_date": row.get("effective_date"),
                    "source_url": row.get("source_url"),
                    "source_note": row.get("source_note"),
                }
            )
    return pd.DataFrame(observed_rows)


def build_licensing_rules_raw_audit(licensing_shocks: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_licensing_shocks(licensing_shocks)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "source_row_number",
                "raw_column_name",
                "value_text",
                "value_numeric",
                "value_kind",
                "source_structure_status",
            ]
        )

    records: list[dict[str, Any]] = []
    ordered_columns = list(licensing_shocks.columns)
    for _, row in frame.iterrows():
        for column in ordered_columns:
            value = row.get(column)
            value_numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            value_text = None if value is None or pd.isna(value) else str(value)
            value_kind = (
                "missing"
                if value_text is None
                else "numeric"
                if not pd.isna(value_numeric)
                else "text"
            )
            records.append(
                {
                    "state_fips": row["state_fips"],
                    "year": int(row["year"]),
                    "source_row_number": int(row["_input_row_number"]),
                    "raw_column_name": str(column),
                    "value_text": value_text,
                    "value_numeric": None if pd.isna(value_numeric) else float(value_numeric),
                    "value_kind": value_kind,
                    "source_structure_status": "observed_raw_column",
                }
            )
    return pd.DataFrame(records).sort_values(
        ["state_fips", "year", "source_row_number", "raw_column_name"],
        kind="stable",
    ).reset_index(drop=True)


def build_licensing_harmonized_rules(licensing_shocks: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_licensing_shocks(licensing_shocks)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "provider_type",
                "age_group",
                "rule_family",
                "rule_id",
                "rule_column_source",
                "strictness_direction",
                "rule_value_observed",
                "rule_value",
                "licensing_rule_missing_original",
                "licensing_rule_carry_forward_applied",
                "licensing_rule_reference_year",
                "licensing_rule_support_status",
            ]
        )

    states = sorted(frame["state_fips"].astype(str).unique().tolist())
    year_min = int(frame["year"].min())
    year_max = int(frame["year"].max())
    years = list(range(year_min, year_max + 1))

    observed = (
        _build_rule_level_rows(frame) if _is_rule_level_contract(frame) else _build_wide_rule_rows(frame)
    )
    full_grid = pd.MultiIndex.from_product(
        [
            states,
            years,
            sorted(observed["rule_id"].unique().tolist()),
        ],
        names=["state_fips", "year", "rule_id"],
    ).to_frame(index=False)
    merged = full_grid.merge(
        observed,
        on=["state_fips", "year", "rule_id"],
        how="left",
    )
    merged = merged.sort_values(["state_fips", "rule_id", "year"], kind="stable").reset_index(drop=True)

    group_keys = ["state_fips", "rule_id"]
    merged["rule_value"] = merged.groupby(group_keys, dropna=False)["rule_value_observed"].ffill()
    merged["licensing_rule_reference_year"] = (
        merged["year"].where(merged["rule_value_observed"].notna()).groupby([merged["state_fips"], merged["rule_id"]]).ffill()
    )
    for column in SOURCE_METADATA_COLUMNS:
        if column in merged.columns:
            merged[column] = merged.groupby(group_keys, dropna=False)[column].ffill()

    merged["licensing_rule_missing_original"] = merged["rule_value_observed"].isna()
    merged["licensing_rule_carry_forward_applied"] = (
        merged["licensing_rule_missing_original"] & merged["rule_value"].notna()
    )
    merged["licensing_rule_support_status"] = np.select(
        [
            merged["rule_value_observed"].notna(),
            merged["licensing_rule_carry_forward_applied"],
        ],
        [
            "observed_rule",
            "carry_forward_rule",
        ],
        default="missing_rule",
    )
    merged["licensing_rule_reference_year"] = (
        pd.to_numeric(merged["licensing_rule_reference_year"], errors="coerce").astype("Int64")
    )
    merged["rule_value_strictness_signed"] = pd.to_numeric(merged["rule_value"], errors="coerce") * pd.to_numeric(
        merged["strictness_sign"], errors="coerce"
    )

    keep_columns = [
        "state_fips",
        "year",
        "provider_type",
        "age_group",
        "rule_family",
        "rule_id",
        "rule_column_source",
        "strictness_direction",
        "strictness_sign",
        "rule_value_observed",
        "rule_value",
        "rule_value_strictness_signed",
        "licensing_rule_missing_original",
        "licensing_rule_carry_forward_applied",
        "licensing_rule_reference_year",
        "licensing_rule_support_status",
        "shock_label",
        "effective_date",
        "source_url",
        "source_note",
    ]
    for column in SOURCE_METADATA_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged[keep_columns].sort_values(
        ["state_fips", "year", "provider_type", "age_group", "rule_family"],
        kind="stable",
    ).reset_index(drop=True)


def build_licensing_stringency_index(harmonized_rules: pd.DataFrame) -> pd.DataFrame:
    required = {
        "state_fips",
        "year",
        "rule_id",
        "rule_value_strictness_signed",
        "licensing_rule_support_status",
    }
    missing = sorted(required - set(harmonized_rules.columns))
    if missing:
        raise KeyError(f"harmonized rules missing required columns: {', '.join(missing)}")

    frame = harmonized_rules.copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "stringency_equal_weight_index",
                "stringency_pca_like_index",
                "stringency_pca_method",
                "stringency_feature_count_used",
                "stringency_total_feature_count",
                "stringency_feature_coverage_rate",
                "observed_rule_count",
                "carry_forward_rule_count",
                "missing_rule_count",
            ]
        )

    frame["state_fips"] = frame["state_fips"].astype(str).str.zfill(2)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["rule_value_strictness_signed"] = pd.to_numeric(frame["rule_value_strictness_signed"], errors="coerce")

    wide = frame.pivot_table(
        index=["state_fips", "year"],
        columns="rule_id",
        values="rule_value_strictness_signed",
        aggfunc="last",
    ).sort_index()
    feature_columns = list(wide.columns)
    if not feature_columns:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "stringency_equal_weight_index",
                "stringency_pca_like_index",
                "stringency_pca_method",
                "stringency_feature_count_used",
                "stringency_total_feature_count",
                "stringency_feature_coverage_rate",
                "observed_rule_count",
                "carry_forward_rule_count",
                "missing_rule_count",
            ]
        )

    standardized = wide.copy()
    for column in feature_columns:
        series = pd.to_numeric(standardized[column], errors="coerce")
        mean = float(series.mean(skipna=True)) if series.notna().any() else 0.0
        std = float(series.std(skipna=True, ddof=0)) if series.notna().any() else 0.0
        if std > 0:
            standardized[column] = (series - mean) / std
        else:
            standardized[column] = series * 0.0

    feature_count_used = standardized.notna().sum(axis=1).astype(int)
    total_feature_count = len(feature_columns)
    equal_weight = standardized.mean(axis=1, skipna=True)

    matrix = standardized.fillna(0.0).to_numpy(dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    pca_supported = centered.shape[0] >= 3 and centered.shape[1] >= 2 and float(np.nanvar(centered)) > 0.0
    if pca_supported:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        loadings = vt[0]
        pca_scores = centered @ loadings
        equal_weight_values = equal_weight.to_numpy(dtype=float)
        if np.isfinite(equal_weight_values).sum() > 1 and np.isfinite(pca_scores).sum() > 1:
            correlation = np.corrcoef(np.nan_to_num(equal_weight_values), np.nan_to_num(pca_scores))[0, 1]
            if np.isfinite(correlation) and correlation < 0:
                pca_scores = -pca_scores
        pca_method = "pca_first_component"
    else:
        pca_scores = equal_weight.to_numpy(dtype=float)
        pca_method = "equal_weight_fallback_sparse"

    support_counts = (
        frame.groupby(["state_fips", "year", "licensing_rule_support_status"], as_index=False)
        .size()
        .pivot_table(
            index=["state_fips", "year"],
            columns="licensing_rule_support_status",
            values="size",
            fill_value=0,
            aggfunc="sum",
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )
    for column in ("observed_rule", "carry_forward_rule", "missing_rule"):
        if column not in support_counts.columns:
            support_counts[column] = 0

    result = standardized.reset_index()[["state_fips", "year"]].copy()
    result["stringency_equal_weight_index"] = equal_weight.to_numpy(dtype=float)
    result["stringency_pca_like_index"] = pca_scores
    result["stringency_pca_method"] = pca_method
    result["stringency_feature_count_used"] = feature_count_used.to_numpy(dtype=int)
    result["stringency_total_feature_count"] = int(total_feature_count)
    denominator = float(total_feature_count) if total_feature_count > 0 else np.nan
    result["stringency_feature_coverage_rate"] = (
        pd.to_numeric(result["stringency_feature_count_used"], errors="coerce") / denominator
    ).fillna(0.0)
    result = result.merge(support_counts, on=["state_fips", "year"], how="left")
    result["observed_rule_count"] = pd.to_numeric(result["observed_rule"], errors="coerce").fillna(0).astype(int)
    result["carry_forward_rule_count"] = (
        pd.to_numeric(result["carry_forward_rule"], errors="coerce").fillna(0).astype(int)
    )
    result["missing_rule_count"] = pd.to_numeric(result["missing_rule"], errors="coerce").fillna(0).astype(int)
    return result[
        [
            "state_fips",
            "year",
            "stringency_equal_weight_index",
            "stringency_pca_like_index",
            "stringency_pca_method",
            "stringency_feature_count_used",
            "stringency_total_feature_count",
            "stringency_feature_coverage_rate",
            "observed_rule_count",
            "carry_forward_rule_count",
            "missing_rule_count",
        ]
    ].sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def build_licensing_harmonization_summary(
    harmonized_rules: pd.DataFrame,
    stringency_index: pd.DataFrame,
) -> pd.DataFrame:
    required = {"state_fips", "year", "licensing_rule_support_status", "rule_id"}
    missing = sorted(required - set(harmonized_rules.columns))
    if missing:
        raise KeyError(f"harmonized rules missing required columns: {', '.join(missing)}")

    if harmonized_rules.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "harmonized_rule_count",
                "observed_rule_count",
                "carry_forward_rule_count",
                "missing_rule_count",
                "carry_forward_rule_share",
                "missing_rule_share",
                "stringency_equal_weight_index",
                "stringency_pca_like_index",
                "stringency_pca_method",
                "stringency_feature_coverage_rate",
                "harmonization_support_status",
            ]
        )

    grouped = (
        harmonized_rules.groupby(["state_fips", "year", "licensing_rule_support_status"], as_index=False)
        .size()
        .pivot_table(
            index=["state_fips", "year"],
            columns="licensing_rule_support_status",
            values="size",
            fill_value=0,
            aggfunc="sum",
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )
    for column in ("observed_rule", "carry_forward_rule", "missing_rule"):
        if column not in grouped.columns:
            grouped[column] = 0
    grouped["observed_rule_count"] = pd.to_numeric(grouped["observed_rule"], errors="coerce").fillna(0).astype(int)
    grouped["carry_forward_rule_count"] = (
        pd.to_numeric(grouped["carry_forward_rule"], errors="coerce").fillna(0).astype(int)
    )
    grouped["missing_rule_count"] = pd.to_numeric(grouped["missing_rule"], errors="coerce").fillna(0).astype(int)
    grouped["harmonized_rule_count"] = (
        grouped["observed_rule_count"] + grouped["carry_forward_rule_count"] + grouped["missing_rule_count"]
    )
    denominator = grouped["harmonized_rule_count"].replace({0: pd.NA})
    grouped["carry_forward_rule_share"] = grouped["carry_forward_rule_count"].div(denominator).fillna(0.0)
    grouped["missing_rule_share"] = grouped["missing_rule_count"].div(denominator).fillna(0.0)
    grouped["harmonization_support_status"] = np.select(
        [
            grouped["missing_rule_count"].gt(0),
            grouped["carry_forward_rule_count"].gt(0),
        ],
        [
            "incomplete_rules",
            "carry_forward_rules_present",
        ],
        default="fully_observed_rules",
    )

    keep_index = [
        "state_fips",
        "year",
        "stringency_equal_weight_index",
        "stringency_pca_like_index",
        "stringency_pca_method",
        "stringency_feature_coverage_rate",
    ]
    merged = grouped.merge(stringency_index[keep_index], on=["state_fips", "year"], how="left")
    return merged[
        [
            "state_fips",
            "year",
            "harmonized_rule_count",
            "observed_rule_count",
            "carry_forward_rule_count",
            "missing_rule_count",
            "carry_forward_rule_share",
            "missing_rule_share",
            "stringency_equal_weight_index",
            "stringency_pca_like_index",
            "stringency_pca_method",
            "stringency_feature_coverage_rate",
            "harmonization_support_status",
        ]
    ].sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def build_licensing_backend_outputs(licensing_shocks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw_audit = build_licensing_rules_raw_audit(licensing_shocks)
    harmonized_rules = build_licensing_rules_harmonized(licensing_shocks)
    stringency_index = build_licensing_stringency_index(harmonized_rules)
    summary = build_licensing_harmonization_summary(harmonized_rules, stringency_index)
    return {
        "licensing_rules_raw_audit": raw_audit,
        "licensing_rules_harmonized": harmonized_rules,
        "licensing_stringency_index": stringency_index,
        "licensing_harmonization_summary": summary,
    }


def build_licensing_raw_audit_table(licensing_shocks: pd.DataFrame) -> pd.DataFrame:
    return build_licensing_rules_raw_audit(licensing_shocks)


def build_licensing_rules_harmonized(licensing_shocks: pd.DataFrame) -> pd.DataFrame:
    return build_licensing_harmonized_rules(licensing_shocks)


def build_licensing_harmonization_outputs(licensing_shocks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return build_licensing_backend_outputs(licensing_shocks)
