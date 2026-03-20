from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RELEASE_NAME = "childcare_backend"
RELEASE_SCOPE = "backend_only"
DEFAULT_COLUMNS = {
    "state_fips": "00",
    "year": 0,
}
SOURCE_READINESS_COLUMNS = [
    "source_name",
    "source_tier",
    "source_path",
    "path_kind",
    "exists",
    "manual_download",
    "real_release_required",
    "readiness_status",
    "current_mode",
    "description",
]
MANUAL_REQUIREMENTS_COLUMNS = [
    "source_name",
    "source_tier",
    "source_path",
    "manual_download",
    "real_release_required",
    "readiness_status",
    "action_needed",
    "priority",
    "description",
]
CONTRACT_ARTIFACT_COLUMNS = [
    "artifact_name",
    "artifact_group",
    "artifact_kind",
    "artifact_status",
    "frontend_priority",
    "artifact_grain",
    "primary_key_columns",
    "join_key_columns",
    "headline_safe",
    "description",
]
COLUMN_DICTIONARY_COLUMNS = [
    "artifact_name",
    "artifact_group",
    "artifact_status",
    "frontend_priority",
    "column_name",
    "column_position",
    "pandas_dtype",
    "nullable",
    "semantic_type",
    "unit",
    "description",
]
BUNDLE_INDEX_COLUMNS = [
    "artifact_name",
    "artifact_group",
    "artifact_kind",
    "artifact_status",
    "frontend_priority",
    "publication_tier",
    "current_mode",
    "sample_available",
    "real_available",
    "producer_command",
    "output_path",
    "provenance_path",
    "file_exists",
    "provenance_exists",
    "sha256",
    "upstream_dependencies",
]

ARTIFACT_STATUS_OVERRIDES = {
    "release_headline_summary": "canonical",
    "release_headline_summary_json": "canonical",
    "methods_summary": "canonical",
    "methods_markdown": "canonical",
    "support_quality_summary": "canonical",
    "ccdf_support_mix_summary": "canonical",
    "policy_coverage_summary": "canonical",
    "release_frontend_handoff_summary": "canonical",
    "release_source_readiness_table": "diagnostic",
    "release_source_readiness_summary": "diagnostic",
    "release_manual_requirements_table": "diagnostic",
    "release_manual_requirements_summary": "diagnostic",
    "release_manual_requirements_markdown": "diagnostic",
    "ccdf_proxy_gap_summary": "diagnostic",
    "ccdf_proxy_gap_state_years": "diagnostic",
    "segmented_comparison": "additive",
    "licensing_rules_raw_audit": "diagnostic",
    "licensing_iv_results": "canonical",
    "licensing_iv_usability_summary": "diagnostic",
    "licensing_first_stage_diagnostics": "diagnostic",
    "licensing_treatment_timing": "diagnostic",
    "licensing_leave_one_state_out": "diagnostic",
}

FRONTEND_PRIORITY_OVERRIDES = {
    "release_headline_summary": "high",
    "release_headline_summary_json": "high",
    "methods_summary": "high",
    "methods_markdown": "medium",
    "support_quality_summary": "high",
    "ccdf_support_mix_summary": "high",
    "policy_coverage_summary": "medium",
    "release_frontend_handoff_summary": "high",
    "ccdf_proxy_gap_summary": "medium",
    "ccdf_proxy_gap_state_years": "low",
    "segmented_comparison": "medium",
    "licensing_iv_results": "high",
    "licensing_iv_usability_summary": "high",
    "licensing_first_stage_diagnostics": "medium",
    "licensing_treatment_timing": "medium",
    "licensing_leave_one_state_out": "low",
    "release_source_readiness_table": "medium",
    "release_source_readiness_summary": "medium",
    "release_manual_requirements_table": "medium",
    "release_manual_requirements_summary": "medium",
    "release_manual_requirements_markdown": "low",
}


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ").replace("\n", " ").strip().lower()
    return " ".join(text.split())


def _mapping_value(mapping: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        return value
    return default


def _frame_or_empty(frame: pd.DataFrame | None) -> pd.DataFrame:
    return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _series_or_default(frame: pd.DataFrame | None, column: str, default: object) -> pd.Series:
    if frame is None:
        return pd.Series(dtype="object")
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype="object")
    series = frame[column]
    if isinstance(series, pd.Series):
        return series
    return pd.Series([default] * len(frame), index=frame.index, dtype="object")


def _row_count(frame: pd.DataFrame | None) -> int:
    return int(len(frame)) if isinstance(frame, pd.DataFrame) else 0


def _resolve_source_path(path: object, repo_root: Path | None) -> Path:
    source_path = Path(str(path))
    if source_path.is_absolute() or repo_root is None:
        return source_path
    return repo_root / source_path


def _support_tier(value: object) -> str:
    text = _normalize_text(value)
    if not text or text in {"missing", "nan"}:
        return "missing_support"
    if "proxy" in text or "fallback" in text:
        return "proxy_or_fallback_support"
    if "inferred" in text:
        return "inferred_split_support"
    if "explicit" in text or "observed" in text:
        return "explicit_split_support"
    return "supported_other"


def _support_quality_counts(frame: pd.DataFrame, *, label: str, status_columns: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            [
                {
                    "component": label,
                    "support_tier": "missing_support",
                    "state_year_count": 0,
                    "support_share": 0.0,
                    "notes": "empty input",
                }
            ]
        )

    status_column = next((column for column in status_columns if column in frame.columns), None)
    if status_column is None:
        tier_series = pd.Series(["supported_other"] * len(frame), index=frame.index)
    else:
        tier_series = frame[status_column].map(_support_tier)
    summary = (
        tier_series.value_counts(dropna=False)
        .rename_axis("support_tier")
        .reset_index(name="state_year_count")
        .sort_values(["state_year_count", "support_tier"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    summary.insert(0, "component", label)
    summary["support_share"] = summary["state_year_count"] / float(max(len(frame), 1))
    summary["notes"] = ""
    return summary[["component", "support_tier", "state_year_count", "support_share", "notes"]]


def _ccdf_support_bucket(row: Mapping[str, Any]) -> str:
    support_flag = _normalize_text(row.get("ccdf_support_flag"))
    support_status = _normalize_text(row.get("ccdf_admin_support_status"))
    combined = f"{support_flag} {support_status}".strip()
    if "downgraded_to_children_served_proxy" in combined:
        return "downgraded_proxy"
    if "proxy" in combined:
        return "retained_proxy"
    if "inferred" in combined:
        return "inferred"
    if "explicit" in combined or "observed" in combined:
        return "explicit"
    if not combined or combined in {"missing", "nan"}:
        return "missing"
    return "other"


def build_childcare_release_ccdf_support_mix_table(
    ccdf_admin_state_year: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "support_bucket",
        "state_year_count",
        "support_share",
        "is_proxy",
        "is_downgraded_proxy",
        "headline_decomposition_eligible",
        "publication_treatment",
        "notes",
    ]
    if ccdf_admin_state_year is None or ccdf_admin_state_year.empty:
        return pd.DataFrame(columns=columns)
    working = ccdf_admin_state_year.copy()
    if "ccdf_support_flag" not in working.columns:
        working["ccdf_support_flag"] = pd.NA
    if "ccdf_admin_support_status" not in working.columns:
        working["ccdf_admin_support_status"] = pd.NA
    rows = []
    total = float(max(len(working), 1))
    for bucket in ["explicit", "inferred", "retained_proxy", "downgraded_proxy", "missing", "other"]:
        count = int(
            sum(
                1
                for row in working[["ccdf_support_flag", "ccdf_admin_support_status"]].to_dict(orient="records")
                if _ccdf_support_bucket(row) == bucket
            )
        )
        if count == 0:
            continue
        rows.append(
            {
                "support_bucket": bucket,
                "state_year_count": count,
                "support_share": count / total,
                "is_proxy": bucket in {"retained_proxy", "downgraded_proxy"},
                "is_downgraded_proxy": bucket == "downgraded_proxy",
                "headline_decomposition_eligible": bucket in {"explicit", "inferred"},
                "publication_treatment": (
                    "headline_eligible"
                    if bucket in {"explicit", "inferred"}
                    else "diagnostic_only"
                ),
                "notes": (
                    "Exclude from headline-grade decomposition claims; retain in diagnostics."
                    if bucket in {"retained_proxy", "downgraded_proxy"}
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["state_year_count", "support_bucket"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def _policy_coverage_counts(frame: pd.DataFrame | None, *, label: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            [
                {
                    "component": label,
                    "item_name": "missing",
                    "item_kind": "missing",
                    "observed_state_year_rows": 0,
                    "missing_state_year_rows": 0,
                    "coverage_rate": 0.0,
                    "coverage_support_status": "missing",
                    "promoted_state_year_rows": 0,
                    "promoted_share_of_observed": 0.0,
                    "notes": "empty input",
                }
            ]
        )

    working = frame.copy()
    if "control_name" in working.columns:
        control_name = working["control_name"].astype(str)
    else:
        control_name = pd.Series([label] * len(working), index=working.index)
    if "control_kind" in working.columns:
        control_kind = working["control_kind"].astype(str)
    else:
        control_kind = pd.Series(["unknown"] * len(working), index=working.index)

    observed = pd.to_numeric(_series_or_default(working, "observed_state_year_rows", 0), errors="coerce").fillna(0).astype(int)
    missing = pd.to_numeric(_series_or_default(working, "missing_state_year_rows", 0), errors="coerce").fillna(0).astype(int)
    coverage = pd.to_numeric(_series_or_default(working, "state_year_coverage_rate", pd.NA), errors="coerce")
    if coverage.isna().all():
        denominator = observed + missing
        coverage = observed.div(denominator.replace({0: pd.NA})).fillna(0.0)
    promoted = pd.to_numeric(_series_or_default(working, "promoted_state_year_rows", 0), errors="coerce").fillna(0).astype(int)
    support_status = _series_or_default(working, "coverage_support_status", "observed")
    summary = pd.DataFrame(
        {
            "component": label,
            "item_name": control_name,
            "item_kind": control_kind,
            "observed_state_year_rows": observed,
            "missing_state_year_rows": missing,
            "coverage_rate": coverage.fillna(0.0).astype(float),
            "coverage_support_status": support_status.astype(str),
            "promoted_state_year_rows": promoted,
        }
    )
    summary["promoted_share_of_observed"] = summary["promoted_state_year_rows"].div(
        summary["observed_state_year_rows"].replace({0: pd.NA})
    ).fillna(0.0)
    summary["notes"] = ""
    return summary[
        [
            "component",
            "item_name",
            "item_kind",
            "observed_state_year_rows",
            "missing_state_year_rows",
            "coverage_rate",
            "coverage_support_status",
            "promoted_state_year_rows",
            "promoted_share_of_observed",
            "notes",
        ]
    ].sort_values(["component", "item_name"], kind="stable").reset_index(drop=True)


def _support_quality_summary_from_frames(
    *,
    ccdf_admin_state_year: pd.DataFrame | None = None,
    segmented_support_quality_summary: pd.DataFrame | None = None,
    licensing_harmonized_rules: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if ccdf_admin_state_year is not None:
        pieces.append(
            _support_quality_counts(
                ccdf_admin_state_year,
                label="ccdf_admin",
                status_columns=("ccdf_admin_support_status", "ccdf_support_flag"),
            )
        )
    if segmented_support_quality_summary is not None:
        segmented = segmented_support_quality_summary.copy()
        if not segmented.empty:
            if "support_quality_tier" in segmented.columns:
                summary = (
                    segmented.groupby("support_quality_tier", dropna=False, sort=True)
                    .size()
                    .rename("state_year_count")
                    .reset_index()
                )
                summary.insert(0, "component", "segmented")
                summary["support_share"] = summary["state_year_count"] / float(max(len(segmented), 1))
                summary["notes"] = ""
                pieces.append(summary[["component", "support_quality_tier", "state_year_count", "support_share", "notes"]].rename(columns={"support_quality_tier": "support_tier"}))
            else:
                pieces.append(
                    _support_quality_counts(
                        segmented,
                        label="segmented",
                        status_columns=("support_quality_tier",),
                    )
                )
    if licensing_harmonized_rules is not None:
        pieces.append(
            _support_quality_counts(
                licensing_harmonized_rules,
                label="licensing",
                status_columns=("licensing_rule_support_status", "rule_support_status", "support_status"),
            )
        )
    if not pieces:
        return pd.DataFrame(columns=["component", "support_tier", "state_year_count", "support_share", "notes"])
    return pd.concat(pieces, ignore_index=True)


def _policy_coverage_summary_from_frames(
    *,
    ccdf_policy_controls_coverage: pd.DataFrame | None = None,
    ccdf_policy_promoted_controls_state_year: pd.DataFrame | None = None,
    licensing_harmonized_rules: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if ccdf_policy_controls_coverage is not None:
        coverage = ccdf_policy_controls_coverage.copy()
        if not coverage.empty:
            coverage = coverage.rename(
                columns={
                    "control_name": "item_name",
                    "control_kind": "item_kind",
                    "state_year_coverage_rate": "coverage_rate",
                }
            )
            if "item_name" not in coverage.columns:
                coverage["item_name"] = "ccdf_control"
            if "item_kind" not in coverage.columns:
                coverage["item_kind"] = "unknown"
            if "observed_state_year_rows" not in coverage.columns:
                coverage["observed_state_year_rows"] = 0
            if "missing_state_year_rows" not in coverage.columns:
                coverage["missing_state_year_rows"] = 0
            if "coverage_rate" not in coverage.columns:
                observed = pd.to_numeric(coverage["observed_state_year_rows"], errors="coerce").fillna(0)
                missing = pd.to_numeric(coverage["missing_state_year_rows"], errors="coerce").fillna(0)
                coverage["coverage_rate"] = observed.div((observed + missing).replace({0: pd.NA})).fillna(0.0)
            if "coverage_support_status" not in coverage.columns:
                coverage["coverage_support_status"] = "observed"
            if "promoted_state_year_rows" not in coverage.columns:
                coverage["promoted_state_year_rows"] = 0
            pieces.append(
                coverage[
                    [
                        "item_name",
                        "item_kind",
                        "observed_state_year_rows",
                        "missing_state_year_rows",
                        "coverage_rate",
                        "coverage_support_status",
                        "promoted_state_year_rows",
                    ]
                ]
                .assign(component="ccdf_policy_controls")
                .rename(columns={"item_name": "item_name", "item_kind": "item_kind"})
            )
    if ccdf_policy_promoted_controls_state_year is not None and not ccdf_policy_promoted_controls_state_year.empty:
        promoted = ccdf_policy_promoted_controls_state_year.copy()
        promoted_columns = [column for column in promoted.columns if column.startswith("ccdf_control_")]
        promoted_count = int(len(promoted))
        pieces.append(
            pd.DataFrame(
                [
                    {
                        "component": "ccdf_promoted_controls",
                        "item_name": ",".join(promoted_columns) if promoted_columns else "ccdf_promoted_controls",
                        "item_kind": "promoted_control_state_year",
                        "observed_state_year_rows": promoted_count,
                        "missing_state_year_rows": 0,
                        "coverage_rate": 1.0 if promoted_count else 0.0,
                        "coverage_support_status": _mapping_value(
                            promoted.iloc[0].to_dict(),
                            "ccdf_policy_control_support_status",
                            default="observed",
                        ),
                        "promoted_state_year_rows": promoted_count,
                    }
                ]
            )
        )
    if licensing_harmonized_rules is not None and not licensing_harmonized_rules.empty:
        support_column = next(
            (
                column
                for column in ("licensing_rule_support_status", "rule_support_status", "support_status")
                if column in licensing_harmonized_rules.columns
            ),
            None,
        )
        item_column = next(
            (column for column in ("rule_id", "rule_family", "raw_column_name") if column in licensing_harmonized_rules.columns),
            None,
        )
        if support_column is not None and item_column is not None:
            working = licensing_harmonized_rules.copy()
            working[item_column] = working[item_column].astype(str)
            support_status = working[support_column].fillna("missing").astype(str)
            grouped = (
                working.assign(
                    _observed=support_status.ne("missing_rule").astype(int),
                    _missing=support_status.eq("missing_rule").astype(int),
                )
                .groupby(item_column, as_index=False)
                .agg(
                    observed_state_year_rows=("_observed", "sum"),
                    missing_state_year_rows=("_missing", "sum"),
                )
                .rename(columns={item_column: "item_name"})
            )
            grouped["component"] = "licensing_rules"
            grouped["item_kind"] = "licensing_rule"
            grouped["coverage_rate"] = pd.to_numeric(grouped["observed_state_year_rows"], errors="coerce").div(
                (
                    pd.to_numeric(grouped["observed_state_year_rows"], errors="coerce")
                    + pd.to_numeric(grouped["missing_state_year_rows"], errors="coerce")
                ).replace({0: pd.NA})
            ).fillna(0.0)
            grouped["coverage_support_status"] = "rule_support_observed"
            grouped["promoted_state_year_rows"] = 0
            grouped["promoted_share_of_observed"] = 0.0
            grouped["notes"] = ""
            pieces.append(grouped)
    if not pieces:
        return pd.DataFrame(
            columns=[
                "component",
                "item_name",
                "item_kind",
                "observed_state_year_rows",
                "missing_state_year_rows",
                "coverage_rate",
                "coverage_support_status",
                "promoted_state_year_rows",
                "promoted_share_of_observed",
                "notes",
            ]
        )
    coverage = pd.concat(pieces, ignore_index=True, sort=False)
    if "promoted_share_of_observed" not in coverage.columns:
        coverage["promoted_share_of_observed"] = pd.to_numeric(
            coverage.get("promoted_state_year_rows"), errors="coerce"
        ).fillna(0).div(
            pd.to_numeric(coverage.get("observed_state_year_rows"), errors="coerce").replace({0: pd.NA})
        ).fillna(0.0)
    if "notes" not in coverage.columns:
        coverage["notes"] = ""
    return coverage[
        [
            "component",
            "item_name",
            "item_kind",
            "observed_state_year_rows",
            "missing_state_year_rows",
            "coverage_rate",
            "coverage_support_status",
            "promoted_state_year_rows",
            "promoted_share_of_observed",
            "notes",
        ]
    ].sort_values(["component", "item_name"], kind="stable").reset_index(drop=True)


def build_childcare_release_headline_summary(
    *,
    pooled_headline_summary: Mapping[str, Any] | None = None,
    ccdf_admin_state_year: pd.DataFrame | None = None,
    ccdf_policy_controls_coverage: pd.DataFrame | None = None,
    ccdf_policy_promoted_controls_state_year: pd.DataFrame | None = None,
    segmented_state_year_summary: pd.DataFrame | None = None,
    segmented_state_fallback_summary: pd.DataFrame | None = None,
    licensing_rules_raw_audit: pd.DataFrame | None = None,
    licensing_iv_summary: Mapping[str, Any] | None = None,
    licensing_rules_harmonized: pd.DataFrame | None = None,
    licensing_stringency_index: pd.DataFrame | None = None,
    release_source_readiness: pd.DataFrame | None = None,
    release_manual_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pooled = dict(pooled_headline_summary or {})
    ccdf_support_rows = _row_count(ccdf_admin_state_year)
    ccdf_support_status = "missing"
    if ccdf_admin_state_year is not None and not ccdf_admin_state_year.empty:
        status_series = _series_or_default(ccdf_admin_state_year, "ccdf_admin_support_status", "missing").fillna("missing").astype(str)
        explicit = int(status_series.str.contains("explicit", case=False, na=False).sum())
        inferred = int(status_series.str.contains("inferred", case=False, na=False).sum())
        proxy = int(status_series.str.contains("proxy", case=False, na=False).sum())
        missing = int(status_series.str.contains("missing", case=False, na=False).sum())
        ccdf_support_status = "observed_real_ccdf_admin"
    else:
        explicit = inferred = proxy = missing = 0
    proxy_gap_tables = build_childcare_release_ccdf_proxy_gap_tables(ccdf_admin_state_year)
    proxy_gap_summary = proxy_gap_tables["proxy_gap_summary"]
    proxy_gap_record = proxy_gap_summary.iloc[0].to_dict() if not proxy_gap_summary.empty else {}
    support_mix = build_childcare_release_ccdf_support_mix_table(ccdf_admin_state_year)
    support_mix_map = {
        str(row["support_bucket"]): int(row["state_year_count"])
        for row in support_mix.to_dict(orient="records")
    }
    headline_eligible_rows = int(
        pd.to_numeric(
            support_mix.loc[support_mix["headline_decomposition_eligible"].fillna(False), "state_year_count"],
            errors="coerce",
        ).fillna(0).sum()
    ) if not support_mix.empty else 0
    headline_excluded_rows = int(
        pd.to_numeric(
            support_mix.loc[~support_mix["headline_decomposition_eligible"].fillna(False), "state_year_count"],
            errors="coerce",
        ).fillna(0).sum()
    ) if not support_mix.empty else 0
    if segmented_state_fallback_summary is not None and not segmented_state_fallback_summary.empty:
        segmented_rows = _row_count(segmented_state_fallback_summary)
        segmented_tiers = sorted(
            pd.Series(segmented_state_fallback_summary.get("support_quality_tier"), dtype="object")
            .fillna("missing_support")
            .astype(str)
            .value_counts()
            .index.tolist()
        )
    elif segmented_state_year_summary is not None:
        segmented_rows = _row_count(segmented_state_year_summary)
        segmented_tiers = []
    else:
        segmented_rows = 0
        segmented_tiers = []
    licensing_status = _mapping_value(licensing_iv_summary, "status", default="missing")
    iv_use_counts = dict(_mapping_value(licensing_iv_summary, "iv_usability_counts", default={}) or {})
    iv_headline_usable_outcome_count = int(
        _mapping_value(licensing_iv_summary, "iv_headline_usable_outcome_count", default=0) or 0
    )
    if iv_headline_usable_outcome_count > 0:
        licensing_iv_recommended_use = "headline"
    elif int(iv_use_counts.get("appendix", 0) or 0) > 0:
        licensing_iv_recommended_use = "appendix"
    elif iv_use_counts or licensing_status != "missing":
        licensing_iv_recommended_use = "diagnostics_only"
    else:
        licensing_iv_recommended_use = "missing"
    licensing_raw_rows = _row_count(licensing_rules_raw_audit)
    licensing_rows = _row_count(licensing_rules_harmonized)
    stringency_rows = _row_count(licensing_stringency_index)
    policy_rows = _row_count(ccdf_policy_controls_coverage)
    promoted_rows = _row_count(ccdf_policy_promoted_controls_state_year)
    backend_complete = bool(ccdf_support_rows > 0 and licensing_rows > 0 and stringency_rows > 0)
    source_readiness = _frame_or_empty(release_source_readiness)
    manual_requirements = dict(release_manual_requirements or {})
    manual_requirements_summary = dict(manual_requirements.get("summary") or manual_requirements.get("json") or {})
    real_release_blocker_count = int(
        _mapping_value(manual_requirements_summary, "real_release_blocker_count", default=0) or 0
    )
    manual_action_count = int(
        _mapping_value(manual_requirements_summary, "manual_action_count", default=0) or 0
    )
    missing_required_sources = (
        bool(source_readiness["readiness_status"].eq("missing_required").any())
        if not source_readiness.empty and "readiness_status" in source_readiness.columns
        else False
    )
    if missing_required_sources:
        source_readiness_status = "missing_required_sources"
    elif real_release_blocker_count > 0:
        source_readiness_status = "real_release_blocked"
    else:
        source_readiness_status = "ready"
    current_mode = (
        str(source_readiness["current_mode"].iloc[0])
        if not source_readiness.empty and "current_mode" in source_readiness.columns
        else _mapping_value(manual_requirements_summary, "current_mode", default="unknown")
    )
    headline_json = {
        "release_name": RELEASE_NAME,
        "release_scope": RELEASE_SCOPE,
        "backend_complete": backend_complete,
        "current_mode": current_mode,
        "pooled_sample_name": _mapping_value(pooled, "selected_headline_sample", "demand_sample_name", "mode", default="missing"),
        "pooled_mode": _mapping_value(pooled, "mode", default="missing"),
        "pooled_n_obs": int(_mapping_value(pooled, "n_obs", default=0) or 0),
        "pooled_n_states": int(_mapping_value(pooled, "n_states", default=0) or 0),
        "pooled_year_min": _mapping_value(pooled, "year_min", default=None),
        "pooled_year_max": _mapping_value(pooled, "year_max", default=None),
        "ccdf_admin_state_year_rows": ccdf_support_rows,
        "ccdf_admin_support_status": ccdf_support_status,
        "ccdf_admin_explicit_rows": explicit,
        "ccdf_admin_inferred_rows": inferred,
        "ccdf_admin_proxy_rows": proxy,
        "ccdf_admin_retained_proxy_rows": int(support_mix_map.get("retained_proxy", 0)),
        "ccdf_admin_downgraded_proxy_rows": int(support_mix_map.get("downgraded_proxy", 0)),
        "ccdf_headline_decomposition_eligible_rows": headline_eligible_rows,
        "ccdf_headline_decomposition_excluded_rows": headline_excluded_rows,
        "ccdf_headline_decomposition_safe": bool(headline_excluded_rows == 0),
        "ccdf_admin_missing_rows": missing,
        "ccdf_proxy_gap_rows": int(_mapping_value(proxy_gap_record, "proxy_support_rows", default=0) or 0),
        "ccdf_proxy_gap_downgraded_rows": int(
            _mapping_value(proxy_gap_record, "proxy_downgraded_children_served_rows", default=0) or 0
        ),
        "ccdf_proxy_gap_mean_ratio_vs_children_served": float(
            _mapping_value(proxy_gap_record, "mean_payment_method_ratio_vs_children_served", default=0.0) or 0.0
        ),
        "ccdf_proxy_gap_max_abs_vs_children_served": float(
            _mapping_value(proxy_gap_record, "max_abs_payment_method_gap_vs_children_served", default=0.0) or 0.0
        ),
        "ccdf_policy_controls_rows": policy_rows,
        "ccdf_policy_promoted_rows": promoted_rows,
        "source_readiness_status": source_readiness_status,
        "source_readiness_missing_count": int((~source_readiness["exists"]).sum()) if not source_readiness.empty and "exists" in source_readiness.columns else 0,
        "manual_action_count": manual_action_count,
        "real_release_blocker_count": real_release_blocker_count,
        "segmented_support_rows": segmented_rows,
        "segmented_support_tiers": segmented_tiers,
        "licensing_raw_audit_rows": licensing_raw_rows,
        "licensing_rule_rows": licensing_rows,
        "licensing_stringency_rows": stringency_rows,
        "licensing_iv_status": licensing_status,
        "licensing_iv_recommended_use": licensing_iv_recommended_use,
        "licensing_iv_headline_usable_outcome_count": iv_headline_usable_outcome_count,
        "licensing_iv_rows": _row_count(
            _frame_or_empty(
                pd.DataFrame() if licensing_iv_summary is None else pd.DataFrame([dict(licensing_iv_summary)])
            )
        ),
    }
    headline_json["release_ready"] = bool(
        headline_json["ccdf_admin_state_year_rows"] > 0
        and headline_json["ccdf_policy_controls_rows"] >= 0
        and headline_json["licensing_raw_audit_rows"] >= 0
        and headline_json["licensing_rule_rows"] > 0
        and headline_json["real_release_blocker_count"] == 0
    )
    headline_json["real_release_ready"] = headline_json["release_ready"]
    headline_json["support_rows_present"] = bool(headline_json["ccdf_admin_state_year_rows"] > 0)
    headline_json["notes"] = [
        "backend-only release package",
        "frontend/figure work delegated",
        "segmented outputs remain additive-only",
        "pooled childcare path remains canonical",
        "real CCDF policy coverage may be partial",
    ]
    if headline_json["ccdf_proxy_gap_downgraded_rows"] > 0:
        headline_json["notes"].append(
            "large-gap payment-method proxy rows are downgraded to a children-served fallback"
        )
    if headline_excluded_rows > 0:
        headline_json["notes"].append(
            "headline CCDF decomposition claims should use explicit and inferred support rows only; retained and downgraded proxy rows are diagnostics-only"
        )
    if licensing_iv_recommended_use != "headline" and licensing_status != "missing":
        headline_json["notes"].append(
            f"licensing IV outputs are currently best treated as {licensing_iv_recommended_use.replace('_', ' ')}"
        )
    headline_table = pd.DataFrame([headline_json])
    return {"json": headline_json, "table": headline_table}


def build_childcare_release_support_tables(
    *,
    ccdf_admin_state_year: pd.DataFrame | None = None,
    segmented_support_quality_summary: pd.DataFrame | None = None,
    licensing_harmonized_rules: pd.DataFrame | None = None,
    ccdf_policy_controls_coverage: pd.DataFrame | None = None,
    ccdf_policy_promoted_controls_state_year: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    support_quality_summary = _support_quality_summary_from_frames(
        ccdf_admin_state_year=ccdf_admin_state_year,
        segmented_support_quality_summary=segmented_support_quality_summary,
        licensing_harmonized_rules=licensing_harmonized_rules,
    )
    policy_coverage_summary = _policy_coverage_summary_from_frames(
        ccdf_policy_controls_coverage=ccdf_policy_controls_coverage,
        ccdf_policy_promoted_controls_state_year=ccdf_policy_promoted_controls_state_year,
        licensing_harmonized_rules=licensing_harmonized_rules,
    )
    ccdf_support_mix_summary = build_childcare_release_ccdf_support_mix_table(
        ccdf_admin_state_year=ccdf_admin_state_year
    )
    return {
        "support_quality_summary": support_quality_summary,
        "policy_coverage_summary": policy_coverage_summary,
        "ccdf_support_mix_summary": ccdf_support_mix_summary,
    }


def build_childcare_release_ccdf_proxy_gap_tables(
    ccdf_admin_state_year: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    summary_columns = [
        "proxy_support_rows",
        "proxy_close_gap_rows",
        "proxy_moderate_gap_rows",
        "proxy_large_gap_rows",
        "proxy_unknown_gap_rows",
        "proxy_downgraded_children_served_rows",
        "rows_with_payment_method_counts",
        "rows_without_payment_method_counts",
        "mean_payment_method_ratio_vs_children_served",
        "median_payment_method_ratio_vs_children_served",
        "mean_abs_payment_method_gap_vs_children_served",
        "max_abs_payment_method_gap_vs_children_served",
    ]
    top_columns = [
        "state_fips",
        "year",
        "ccdf_support_flag",
        "ccdf_admin_support_status",
        "ccdf_children_served",
        "ccdf_payment_method_total_children",
        "ccdf_payment_method_gap_vs_children_served",
        "ccdf_payment_method_ratio_vs_children_served",
        "ccdf_grants_contracts_share",
        "ccdf_certificates_share",
        "ccdf_cash_share",
        "proxy_reliability_tier",
        "proxy_split_treatment",
    ]
    if ccdf_admin_state_year is None or ccdf_admin_state_year.empty:
        return {
            "proxy_gap_summary": pd.DataFrame(columns=summary_columns),
            "proxy_gap_state_years": pd.DataFrame(columns=top_columns),
        }

    working = ccdf_admin_state_year.copy()
    if "ccdf_support_flag" not in working.columns:
        working["ccdf_support_flag"] = "missing"
    if "ccdf_admin_support_status" not in working.columns:
        working["ccdf_admin_support_status"] = "missing"
    for column in top_columns:
        if column in {"state_fips", "year", "ccdf_support_flag", "ccdf_admin_support_status"}:
            continue
        if column not in working.columns:
            working[column] = pd.NA
        working[column] = pd.to_numeric(working[column], errors="coerce")

    proxy_rows = working.loc[
        working["ccdf_support_flag"].astype(str).str.startswith("ccdf_split_proxy_from_payment_method_shares", na=False)
    ].copy()
    if proxy_rows.empty:
        return {
            "proxy_gap_summary": pd.DataFrame([{column: 0.0 for column in summary_columns}]).astype(
                {
                    "proxy_support_rows": "int64",
                    "rows_with_payment_method_counts": "int64",
                    "rows_without_payment_method_counts": "int64",
                }
            ),
            "proxy_gap_state_years": pd.DataFrame(columns=top_columns),
        }

    ratio = pd.to_numeric(proxy_rows["ccdf_payment_method_ratio_vs_children_served"], errors="coerce")
    gap = pd.to_numeric(proxy_rows["ccdf_payment_method_gap_vs_children_served"], errors="coerce").abs()
    total = pd.to_numeric(proxy_rows["ccdf_payment_method_total_children"], errors="coerce")
    support_flag = proxy_rows["ccdf_support_flag"].fillna("missing").astype(str)
    proxy_rows["proxy_reliability_tier"] = np.select(
        [
            support_flag.str.contains("close_gap", case=False, na=False),
            support_flag.str.contains("moderate_gap", case=False, na=False),
            support_flag.str.contains("large_gap", case=False, na=False),
            support_flag.str.contains("unknown_gap", case=False, na=False),
        ],
        [
            "close_gap",
            "moderate_gap",
            "large_gap",
            "unknown_gap",
        ],
        default="payment_method_proxy",
    )
    proxy_rows["proxy_split_treatment"] = np.where(
        support_flag.str.contains("downgraded_to_children_served_proxy", case=False, na=False),
        "children_served_proxy_downgraded",
        "payment_method_split_proxy_retained",
    )
    summary = pd.DataFrame(
        [
            {
                "proxy_support_rows": int(len(proxy_rows)),
                "proxy_close_gap_rows": int(proxy_rows["proxy_reliability_tier"].eq("close_gap").sum()),
                "proxy_moderate_gap_rows": int(proxy_rows["proxy_reliability_tier"].eq("moderate_gap").sum()),
                "proxy_large_gap_rows": int(
                    proxy_rows["proxy_reliability_tier"].eq("large_gap").sum()
                    - proxy_rows["proxy_split_treatment"].eq("children_served_proxy_downgraded").sum()
                ),
                "proxy_unknown_gap_rows": int(proxy_rows["proxy_reliability_tier"].eq("unknown_gap").sum()),
                "proxy_downgraded_children_served_rows": int(
                    proxy_rows["proxy_split_treatment"].eq("children_served_proxy_downgraded").sum()
                ),
                "rows_with_payment_method_counts": int(total.notna().sum()),
                "rows_without_payment_method_counts": int(total.isna().sum()),
                "mean_payment_method_ratio_vs_children_served": float(ratio.mean()) if ratio.notna().any() else 0.0,
                "median_payment_method_ratio_vs_children_served": float(ratio.median()) if ratio.notna().any() else 0.0,
                "mean_abs_payment_method_gap_vs_children_served": float(gap.mean()) if gap.notna().any() else 0.0,
                "max_abs_payment_method_gap_vs_children_served": float(gap.max()) if gap.notna().any() else 0.0,
            }
        ]
    )
    proxy_rows = proxy_rows.sort_values(
        ["proxy_split_treatment", "ccdf_payment_method_ratio_vs_children_served", "ccdf_payment_method_gap_vs_children_served"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return {
        "proxy_gap_summary": summary[summary_columns],
        "proxy_gap_state_years": proxy_rows[top_columns],
    }


def build_childcare_release_source_readiness(
    *,
    required_sources: list[Mapping[str, Any]] | None = None,
    optional_sources: list[Mapping[str, Any]] | None = None,
    sample_mode: bool = False,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for tier, source_specs in (("required", required_sources or []), ("optional", optional_sources or [])):
        for spec in source_specs:
            name = str(spec.get("name", "unnamed_source"))
            source_path = _resolve_source_path(spec.get("path", ""), repo_root=repo_root)
            exists = source_path.exists()
            manual_download = bool(spec.get("manual_download", False))
            real_release_required = bool(spec.get("real_release_required", False))
            description = str(spec.get("description", ""))
            if exists:
                readiness_status = "ready"
            elif tier == "required" and not (sample_mode and real_release_required):
                readiness_status = "missing_required"
            elif sample_mode and real_release_required:
                readiness_status = "not_required_in_sample"
            else:
                readiness_status = "missing_optional"
            records.append(
                {
                    "source_name": name,
                    "source_tier": tier,
                    "source_path": str(source_path),
                    "path_kind": "directory" if source_path.suffix == "" else source_path.suffix.lstrip(".") or "file",
                    "exists": bool(exists),
                    "manual_download": manual_download,
                    "real_release_required": real_release_required,
                    "readiness_status": readiness_status,
                    "current_mode": "sample" if sample_mode else "real",
                    "description": description,
                }
            )
    if not records:
        return pd.DataFrame(columns=SOURCE_READINESS_COLUMNS)
    readiness = pd.DataFrame(records, columns=SOURCE_READINESS_COLUMNS)
    return readiness.sort_values(["source_tier", "source_name"], kind="stable").reset_index(drop=True)


def build_childcare_release_manual_requirements(
    source_readiness: pd.DataFrame | None,
    *,
    sample_mode: bool = False,
) -> dict[str, Any]:
    if source_readiness is None or source_readiness.empty:
        empty_table = pd.DataFrame(columns=MANUAL_REQUIREMENTS_COLUMNS)
        empty_summary = {
            "release_name": RELEASE_NAME,
            "current_mode": "sample" if sample_mode else "real",
            "missing_required_count": 0,
            "missing_optional_count": 0,
            "manual_action_count": 0,
            "real_release_blocker_count": 0,
            "status": "no_source_inventory",
        }
        return {
            "json": empty_summary,
            "summary": empty_summary,
            "markdown": "# Childcare backend manual requirements\n\nNo source inventory was provided.\n",
            "table": empty_table,
        }

    readiness = source_readiness.copy()
    readiness["manual_download"] = readiness["manual_download"].fillna(False).astype(bool)
    readiness["real_release_required"] = readiness["real_release_required"].fillna(False).astype(bool)
    readiness["readiness_status"] = readiness["readiness_status"].fillna("missing_optional").astype(str)
    actionable = readiness.loc[
        readiness["readiness_status"].isin({"missing_required", "missing_optional", "not_required_in_sample"})
    ].copy()
    actionable["action_needed"] = np.where(
        actionable["manual_download"],
        "add_manual_source",
        "rebuild_or_generate",
    )
    actionable["priority"] = np.select(
        [
            actionable["readiness_status"].eq("missing_required"),
            actionable["real_release_required"],
            actionable["readiness_status"].eq("missing_optional"),
        ],
        [
            "required_now",
            "real_release_blocker",
            "optional_improvement",
        ],
        default="informational",
    )
    if actionable.empty:
        actionable = pd.DataFrame(columns=MANUAL_REQUIREMENTS_COLUMNS)
    else:
        actionable = actionable[
            [
                "source_name",
                "source_tier",
                "source_path",
                "manual_download",
                "real_release_required",
                "readiness_status",
                "action_needed",
                "priority",
                "description",
            ]
        ].sort_values(["priority", "source_name"], kind="stable").reset_index(drop=True)

    missing_required_count = int(readiness["readiness_status"].eq("missing_required").sum())
    missing_optional_count = int(readiness["readiness_status"].eq("missing_optional").sum())
    manual_action_count = int(
        readiness.loc[
            readiness["manual_download"] & readiness["readiness_status"].ne("ready")
        ].shape[0]
    )
    real_release_blocker_count = int(
        readiness.loc[
            readiness["real_release_required"] & readiness["readiness_status"].ne("ready")
        ].shape[0]
    )
    if missing_required_count > 0:
        status = "missing_required_sources"
    elif real_release_blocker_count > 0:
        status = "real_release_blocked"
    else:
        status = "ready"
    summary_json = {
        "release_name": RELEASE_NAME,
        "current_mode": "sample" if sample_mode else "real",
        "missing_required_count": missing_required_count,
        "missing_optional_count": missing_optional_count,
        "manual_action_count": manual_action_count,
        "real_release_blocker_count": real_release_blocker_count,
        "status": status,
    }
    markdown_lines = [
        "# Childcare backend manual requirements",
        "",
        f"- current mode: {'sample' if sample_mode else 'real'}",
        f"- missing required sources: {missing_required_count}",
        f"- missing optional sources: {missing_optional_count}",
        f"- manual actions remaining: {manual_action_count}",
        f"- real-release blockers: {real_release_blocker_count}",
        "",
        "## Actions",
    ]
    if actionable.empty:
        markdown_lines.append("- No source actions are currently outstanding.")
    else:
        for row in actionable.itertuples(index=False):
            markdown_lines.append(
                f"- {row.source_name}: {row.priority} ({row.readiness_status}) -> {row.source_path}"
            )
    return {
        "json": summary_json,
        "summary": summary_json,
        "markdown": "\n".join(markdown_lines) + "\n",
        "table": actionable,
    }


def build_childcare_release_segmented_comparison(
    segmented_channel_scenarios: pd.DataFrame | None = None,
    segmented_state_fallback_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    scenarios = _frame_or_empty(segmented_channel_scenarios)
    if scenarios.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "solver_channel",
                "alpha",
                "market_quantity_proxy",
                "unpaid_quantity_proxy",
                "p_baseline",
                "p_shadow_marginal",
                "p_alpha",
                "p_shadow_delta_vs_baseline",
                "p_alpha_delta_vs_baseline",
                "p_shadow_pct_change_vs_baseline",
                "p_alpha_pct_change_vs_baseline",
                "public_admin_price_invariant",
                "comparison_role",
            ]
        )
    required = {
        "state_fips",
        "year",
        "solver_channel",
        "alpha",
        "market_quantity_proxy",
        "unpaid_quantity_proxy",
        "p_baseline",
        "p_shadow_marginal",
        "p_alpha",
    }
    missing = sorted(required - set(scenarios.columns))
    if missing:
        raise KeyError(f"segmented_channel_scenarios missing required columns: {', '.join(missing)}")

    working = scenarios.copy()
    working["state_fips"] = working["state_fips"].astype(str).str.zfill(2)
    working["year"] = pd.to_numeric(working["year"], errors="coerce").astype("Int64")
    working = working.dropna(subset=["year"]).copy()
    for column in ("alpha", "market_quantity_proxy", "unpaid_quantity_proxy", "p_baseline", "p_shadow_marginal", "p_alpha"):
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)
    if "price_responsive" not in working.columns:
        working["price_responsive"] = working["solver_channel"].astype(str).isin({"private_unsubsidized", "private_subsidized"})
    if "public_admin_price_invariant" not in working.columns:
        working["public_admin_price_invariant"] = working["solver_channel"].astype(str).eq("public_admin")
    working["p_shadow_delta_vs_baseline"] = working["p_shadow_marginal"] - working["p_baseline"]
    working["p_alpha_delta_vs_baseline"] = working["p_alpha"] - working["p_baseline"]
    working["p_shadow_pct_change_vs_baseline"] = working["p_shadow_delta_vs_baseline"].div(
        working["p_baseline"].replace({0.0: pd.NA})
    ).fillna(0.0)
    working["p_alpha_pct_change_vs_baseline"] = working["p_alpha_delta_vs_baseline"].div(
        working["p_baseline"].replace({0.0: pd.NA})
    ).fillna(0.0)
    if segmented_state_fallback_summary is not None and not segmented_state_fallback_summary.empty:
        support = segmented_state_fallback_summary.copy()
        support["state_fips"] = support["state_fips"].astype(str).str.zfill(2)
        support["year"] = pd.to_numeric(support["year"], errors="coerce").astype("Int64")
        support = support.dropna(subset=["year"]).copy()
        keep = [
            "state_fips",
            "year",
            "support_quality_tier",
            "ccdf_support_flag",
            "ccdf_admin_support_status",
            "q0_support_flag",
            "public_program_support_status",
            "promoted_control_observed",
            "proxy_ccdf_row_count",
            "any_segment_allocation_fallback",
            "any_private_allocation_fallback",
        ]
        support = support[[column for column in keep if column in support.columns]].drop_duplicates(
            ["state_fips", "year"], keep="last"
        )
        working = working.merge(support, on=["state_fips", "year"], how="left")
    working["comparison_role"] = np.where(
        np.isclose(working["alpha"], 0.5, equal_nan=False), "headline_alpha", "sensitivity_alpha"
    )
    working["solver_channel"] = pd.Categorical(
        working["solver_channel"], categories=["private_unsubsidized", "private_subsidized", "public_admin"], ordered=True
    )
    working = working.sort_values(["state_fips", "year", "solver_channel", "alpha"], kind="stable").reset_index(drop=True)
    working["solver_channel"] = working["solver_channel"].astype(str)
    return working[
        [
            "state_fips",
            "year",
            "solver_channel",
            "alpha",
            "market_quantity_proxy",
            "unpaid_quantity_proxy",
            "p_baseline",
            "p_shadow_marginal",
            "p_alpha",
            "p_shadow_delta_vs_baseline",
            "p_alpha_delta_vs_baseline",
            "p_shadow_pct_change_vs_baseline",
            "p_alpha_pct_change_vs_baseline",
            "public_admin_price_invariant",
            "comparison_role",
            *[column for column in (
                "support_quality_tier",
                "ccdf_support_flag",
                "ccdf_admin_support_status",
                "q0_support_flag",
                "public_program_support_status",
                "promoted_control_observed",
                "proxy_ccdf_row_count",
                "any_segment_allocation_fallback",
                "any_private_allocation_fallback",
            ) if column in working.columns],
        ]
    ]


def build_childcare_release_methods_summary(
    *,
    pooled_headline_summary: Mapping[str, Any] | None = None,
    ccdf_admin_state_year: pd.DataFrame | None = None,
    ccdf_policy_controls_coverage: pd.DataFrame | None = None,
    segmented_state_fallback_summary: pd.DataFrame | None = None,
    licensing_rules_raw_audit: pd.DataFrame | None = None,
    licensing_iv_summary: Mapping[str, Any] | None = None,
    licensing_rules_harmonized: pd.DataFrame | None = None,
    release_manual_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pooled = dict(pooled_headline_summary or {})
    ccdf_rows = _row_count(ccdf_admin_state_year)
    policy_rows = _row_count(ccdf_policy_controls_coverage)
    segmented_rows = _row_count(segmented_state_fallback_summary)
    licensing_raw_rows = _row_count(licensing_rules_raw_audit)
    licensing_rows = _row_count(licensing_rules_harmonized)
    iv_status = _mapping_value(licensing_iv_summary, "status", default="missing")
    iv_use_counts = dict(_mapping_value(licensing_iv_summary, "iv_usability_counts", default={}) or {})
    iv_headline_usable_outcome_count = int(
        _mapping_value(licensing_iv_summary, "iv_headline_usable_outcome_count", default=0) or 0
    )
    if iv_headline_usable_outcome_count > 0:
        licensing_iv_recommended_use = "headline"
    elif int(iv_use_counts.get("appendix", 0) or 0) > 0:
        licensing_iv_recommended_use = "appendix"
    elif iv_use_counts or iv_status != "missing":
        licensing_iv_recommended_use = "diagnostics_only"
    else:
        licensing_iv_recommended_use = "missing"
    manual_requirements = dict(release_manual_requirements or {})
    manual_requirements_summary = dict(manual_requirements.get("summary") or manual_requirements.get("json") or {})
    real_release_blocker_count = int(
        _mapping_value(manual_requirements_summary, "real_release_blocker_count", default=0) or 0
    )
    manual_action_count = int(
        _mapping_value(manual_requirements_summary, "manual_action_count", default=0) or 0
    )
    proxy_gap_tables = build_childcare_release_ccdf_proxy_gap_tables(ccdf_admin_state_year)
    proxy_gap_summary = proxy_gap_tables["proxy_gap_summary"]
    proxy_gap_record = proxy_gap_summary.iloc[0].to_dict() if not proxy_gap_summary.empty else {}
    proxy_support_rows = int(_mapping_value(proxy_gap_record, "proxy_support_rows", default=0) or 0)
    downgraded_proxy_rows = int(
        _mapping_value(proxy_gap_record, "proxy_downgraded_children_served_rows", default=0) or 0
    )
    support_mix = build_childcare_release_ccdf_support_mix_table(ccdf_admin_state_year)
    retained_proxy_rows = int(
        pd.to_numeric(
            support_mix.loc[support_mix["support_bucket"].eq("retained_proxy"), "state_year_count"],
            errors="coerce",
        ).fillna(0).sum()
    ) if not support_mix.empty else 0
    headline_excluded_rows = int(
        pd.to_numeric(
            support_mix.loc[~support_mix["headline_decomposition_eligible"].fillna(False), "state_year_count"],
            errors="coerce",
        ).fillna(0).sum()
    ) if not support_mix.empty else 0
    proxy_ratio_mean = float(
        _mapping_value(proxy_gap_record, "mean_payment_method_ratio_vs_children_served", default=0.0) or 0.0
    )
    assumptions = [
        "Pooled childcare remains the canonical benchmark.",
        "Segmented childcare outputs remain additive-only in this release.",
        "Frontend figure generation is intentionally out of scope for the backend package.",
    ]
    if ccdf_rows and policy_rows == 0:
        assumptions.append("Real CCDF admin data are present but policy controls remain limited or absent.")
    if licensing_raw_rows:
        assumptions.append("Licensing raw audit rows are retained for traceability alongside harmonized outputs.")
    if licensing_rows:
        assumptions.append("Licensing harmonization outputs are included as backend diagnostics, not presentation artifacts.")
    if proxy_support_rows:
        assumptions.append(
            f"Payment-method proxy-backed CCDF rows are retained with explicit gap metadata (mean payment-method ratio {proxy_ratio_mean:.3f})."
        )
    if downgraded_proxy_rows:
        assumptions.append(
            f"Large-gap payment-method proxy rows ({downgraded_proxy_rows}) are downgraded to a children-served fallback instead of contributing a payment-method split."
        )
    limitations = [
        "CCDF split support may still rely on proxy or inferred rows when direct sheet support is unavailable.",
        "Policy-control promotion is coverage-qualified and may be sparse if real policy files are missing.",
        "Licensing IV outputs should be interpreted together with their diagnostics and treatment-timing tables.",
    ]
    if retained_proxy_rows:
        limitations.append(
            f"{retained_proxy_rows} CCDF state-years still rely on retained proxy split support rather than explicit or inferred administrative decomposition."
        )
    if headline_excluded_rows:
        limitations.append(
            "Headline-grade CCDF decomposition claims should exclude retained-proxy and downgraded-proxy rows and rely on explicit or inferred support only."
        )
    if licensing_iv_recommended_use != "headline" and iv_status != "missing":
        limitations.append(
            f"Licensing IV outputs are currently best treated as {licensing_iv_recommended_use.replace('_', ' ')} rather than headline causal estimates."
        )
    if real_release_blocker_count:
        limitations.append(
            f"The real backend release still has {real_release_blocker_count} manual-source blocker(s) across {manual_action_count} outstanding manual action(s)."
        )
    markdown = "\n".join(
        [
            "# Childcare backend methods and limitations",
            "",
            "## Methods",
            "- Pooled childcare is retained as the canonical baseline for the release backend.",
            "- Segmented outputs are built additively and remain separate from the pooled path.",
            "- CCDF admin and policy data are tracked with explicit support and coverage metadata.",
            "- Licensing harmonization and IV outputs are included as backend diagnostics for defensibility.",
            "",
            "## Assumptions",
            *[f"- {item}" for item in assumptions],
            "",
            "## Limitations",
            *[f"- {item}" for item in limitations],
            "",
            f"- pooled sample/profile: {_mapping_value(pooled, 'selected_headline_sample', 'demand_sample_name', 'mode', default='missing')}",
            f"- ccdf admin rows: {ccdf_rows}",
            f"- segmented rows: {segmented_rows}",
            f"- licensing raw audit rows: {licensing_raw_rows}",
            f"- licensing rule rows: {licensing_rows}",
        f"- licensing iv status: {iv_status}",
        f"- manual actions remaining: {manual_action_count}",
        f"- real-release blockers: {real_release_blocker_count}",
        f"- proxy-backed ccdf rows with payment-method gap metadata: {proxy_support_rows}",
    ]
    )
    summary_json = {
        "release_name": RELEASE_NAME,
        "release_scope": RELEASE_SCOPE,
        "methods": [
            "pooled childcare benchmark retained",
            "segmented childcare additive only",
            "CCDF support metadata explicit",
            "licensing harmonization plus IV diagnostics included",
        ],
        "assumptions": assumptions,
        "limitations": limitations,
        "backend_only": True,
        "pooled_sample_name": _mapping_value(pooled, "selected_headline_sample", "demand_sample_name", "mode", default="missing"),
        "ccdf_admin_rows": ccdf_rows,
        "ccdf_policy_rows": policy_rows,
        "segmented_rows": segmented_rows,
        "licensing_raw_audit_rows": licensing_raw_rows,
        "licensing_rule_rows": licensing_rows,
        "licensing_iv_status": iv_status,
        "licensing_iv_recommended_use": licensing_iv_recommended_use,
        "licensing_iv_headline_usable_outcome_count": iv_headline_usable_outcome_count,
        "manual_action_count": manual_action_count,
        "real_release_blocker_count": real_release_blocker_count,
        "proxy_support_rows": proxy_support_rows,
        "retained_proxy_rows": retained_proxy_rows,
        "downgraded_proxy_rows": downgraded_proxy_rows,
        "headline_decomposition_excluded_rows": headline_excluded_rows,
        "mean_payment_method_ratio_vs_children_served": proxy_ratio_mean,
    }
    return {"markdown": markdown, "json": summary_json}


def build_childcare_release_manifest(artifacts: Mapping[str, object]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for artifact_name, value in artifacts.items():
        data = value
        role = "backend_artifact"
        description = ""
        if isinstance(value, Mapping) and ("data" in value or "frame" in value or "json" in value or "markdown" in value):
            data = value.get("data", value.get("frame", value.get("json", value.get("markdown"))))
            role = str(value.get("role", role))
            description = str(value.get("description", ""))
        if isinstance(data, pd.DataFrame):
            records.append(
                {
                    "artifact_name": artifact_name,
                    "artifact_group": _artifact_group(artifact_name),
                    "artifact_kind": "dataframe",
                    "artifact_role": role,
                    "row_count": int(len(data)),
                    "column_count": int(len(data.columns)),
                    "key_count": 0,
                    "text_length": 0,
                    "is_empty": bool(data.empty),
                    "description": description,
                }
            )
        elif isinstance(data, Mapping):
            records.append(
                {
                    "artifact_name": artifact_name,
                    "artifact_group": _artifact_group(artifact_name),
                    "artifact_kind": "mapping",
                    "artifact_role": role,
                    "row_count": 0,
                    "column_count": 0,
                    "key_count": int(len(data)),
                    "text_length": 0,
                    "is_empty": bool(len(data) == 0),
                    "description": description,
                }
            )
        elif isinstance(data, (str, Path)):
            text = str(data)
            records.append(
                {
                    "artifact_name": artifact_name,
                    "artifact_group": _artifact_group(artifact_name),
                    "artifact_kind": "text" if isinstance(data, str) else "path",
                    "artifact_role": role,
                    "row_count": 0,
                    "column_count": 0,
                    "key_count": 0,
                    "text_length": int(len(text)),
                    "is_empty": bool(len(text) == 0),
                    "description": description,
                }
            )
        else:
            records.append(
                {
                    "artifact_name": artifact_name,
                    "artifact_group": _artifact_group(artifact_name),
                    "artifact_kind": type(data).__name__,
                    "artifact_role": role,
                    "row_count": 0,
                    "column_count": 0,
                    "key_count": 0,
                    "text_length": 0,
                    "is_empty": data is None,
                    "description": description,
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "artifact_name",
            "artifact_group",
            "artifact_kind",
            "artifact_role",
            "row_count",
            "column_count",
            "key_count",
            "text_length",
            "is_empty",
            "description",
        ],
    ).sort_values(["artifact_group", "artifact_name"], kind="stable").reset_index(drop=True)


def build_childcare_release_schema_inventory(artifacts: Mapping[str, object]) -> dict[str, Any]:
    artifact_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    for artifact_name, value in artifacts.items():
        data = value
        role = "backend_artifact"
        description = ""
        if isinstance(value, Mapping) and ("data" in value or "frame" in value or "json" in value or "markdown" in value):
            data = value.get("data", value.get("frame", value.get("json", value.get("markdown"))))
            role = str(value.get("role", role))
            description = str(value.get("description", ""))
        artifact_group = _artifact_group(artifact_name)
        if isinstance(data, pd.DataFrame):
            artifact_rows.append(
                {
                    "artifact_name": artifact_name,
                    "artifact_group": artifact_group,
                    "artifact_kind": "dataframe",
                    "artifact_role": role,
                    "row_count": int(len(data)),
                    "column_count": int(len(data.columns)),
                    "description": description,
                }
            )
            for position, column_name in enumerate(data.columns, start=1):
                column_rows.append(
                    {
                        "artifact_name": artifact_name,
                        "artifact_group": artifact_group,
                        "artifact_role": role,
                        "column_name": str(column_name),
                        "column_position": int(position),
                        "pandas_dtype": str(data[column_name].dtype),
                        "non_null_rows": int(data[column_name].notna().sum()),
                    }
                )
        elif isinstance(data, Mapping):
            artifact_rows.append(
                {
                    "artifact_name": artifact_name,
                    "artifact_group": artifact_group,
                    "artifact_kind": "mapping",
                    "artifact_role": role,
                    "row_count": 0,
                    "column_count": int(len(data)),
                    "description": description,
                }
            )
        elif isinstance(data, str):
            artifact_rows.append(
                {
                    "artifact_name": artifact_name,
                    "artifact_group": artifact_group,
                    "artifact_kind": "text",
                    "artifact_role": role,
                    "row_count": int(len(data.splitlines())),
                    "column_count": 0,
                    "description": description,
                }
            )
        else:
            artifact_rows.append(
                {
                    "artifact_name": artifact_name,
                    "artifact_group": artifact_group,
                    "artifact_kind": type(data).__name__,
                    "artifact_role": role,
                    "row_count": 0,
                    "column_count": 0,
                    "description": description,
                }
            )
    artifact_summary = pd.DataFrame(
        artifact_rows,
        columns=[
            "artifact_name",
            "artifact_group",
            "artifact_kind",
            "artifact_role",
            "row_count",
            "column_count",
            "description",
        ],
    ).sort_values(["artifact_group", "artifact_name"], kind="stable").reset_index(drop=True)
    column_schema = pd.DataFrame(
        column_rows,
        columns=[
            "artifact_name",
            "artifact_group",
            "artifact_role",
            "column_name",
            "column_position",
            "pandas_dtype",
            "non_null_rows",
        ],
    ).sort_values(["artifact_group", "artifact_name", "column_position"], kind="stable").reset_index(drop=True)
    return {
        "artifact_summary": artifact_summary,
        "column_schema": column_schema,
        "json": {
            "artifact_summary": artifact_summary.to_dict(orient="records"),
            "column_schema": column_schema.to_dict(orient="records"),
        },
    }


def _artifact_group(artifact_name: str) -> str:
    lowered = artifact_name.lower()
    if "pooled" in lowered or "mvp" in lowered or "headline" in lowered:
        return "pooled"
    if "ccdf" in lowered:
        return "ccdf"
    if "licensing" in lowered or "iv" in lowered:
        return "licensing"
    if "segmented" in lowered:
        return "segmented"
    if "release" in lowered or "methods" in lowered:
        return "release"
    return "other"


def _artifact_status(artifact_name: str) -> str:
    return ARTIFACT_STATUS_OVERRIDES.get(artifact_name, "diagnostic")


def _frontend_priority(artifact_name: str) -> str:
    return FRONTEND_PRIORITY_OVERRIDES.get(artifact_name, "low")


def _artifact_grain(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "unknown"
    columns = set(frame.columns.astype(str))
    if {"state_fips", "year", "solver_channel", "alpha"} <= columns:
        return "state_year_channel_alpha"
    if {"state_fips", "year", "solver_channel"} <= columns:
        return "state_year_channel"
    if {"state_fips", "year"} <= columns:
        return "state_year"
    if {"component", "support_tier"} <= columns:
        return "component_support_tier"
    if {"component", "item_name"} <= columns:
        return "component_item"
    if {"source_name", "source_tier"} <= columns:
        return "source"
    if {"artifact_name", "artifact_group"} <= columns:
        return "artifact"
    return "unknown"


def _key_columns_for_frame(frame: pd.DataFrame | None) -> tuple[list[str], list[str]]:
    if frame is None or frame.empty:
        return ([], [])
    columns = [str(column) for column in frame.columns]
    column_set = set(columns)
    primary_candidates = [
        ["state_fips", "year", "solver_channel", "alpha"],
        ["state_fips", "year", "solver_channel"],
        ["state_fips", "year"],
        ["component", "support_tier"],
        ["component", "item_name"],
        ["source_name"],
        ["artifact_name", "column_name"],
        ["artifact_name"],
    ]
    for candidate in primary_candidates:
        if set(candidate) <= column_set:
            join_keys = [key for key in candidate if key in {"state_fips", "year", "solver_channel", "alpha"}]
            return (candidate, join_keys)
    return ([], [])


def _column_semantic_type(column_name: str, dtype: str) -> str:
    name = column_name.lower()
    if name in {"state_fips"}:
        return "geography_key"
    if name in {"year"}:
        return "time_key"
    if name.endswith("_status") or name.endswith("_flag") or "status" in name:
        return "support_status"
    if name.endswith("_path"):
        return "path"
    if name.endswith("_json"):
        return "json_payload"
    if name.endswith("_markdown") or name.endswith("_md"):
        return "markdown_text"
    if "share" in name or "rate" in name or "pct" in name or "percent" in name:
        return "share"
    if "price" in name or name.startswith("p_"):
        return "price"
    if "quantity" in name or "slots" in name or "children" in name or "count" in name or "rows" in name:
        return "count"
    if "checksum" in name or name == "sha256":
        return "checksum"
    if "command" in name:
        return "command"
    if "key" in name:
        return "key"
    if "bool" in dtype or name.startswith("is_") or name.startswith("any_") or name.endswith("_safe"):
        return "boolean"
    if "int" in dtype or "float" in dtype:
        return "numeric"
    return "text"


def _column_unit(column_name: str, semantic_type: str) -> str:
    name = column_name.lower()
    if semantic_type == "share":
        return "share_0_1"
    if semantic_type == "price":
        return "usd"
    if semantic_type == "count":
        if "rows" in name or "count" in name:
            return "count"
        if "children" in name or "slots" in name or "quantity" in name:
            return "children_or_slots"
        return "count"
    if semantic_type in {"geography_key", "time_key", "support_status", "path", "json_payload", "markdown_text", "checksum", "command", "key", "text", "boolean"}:
        return "n/a"
    return "n/a"


def _column_description(artifact_name: str, column_name: str, semantic_type: str) -> str:
    label = column_name.replace("_", " ")
    if semantic_type == "geography_key":
        return "State FIPS key for state-year joins."
    if semantic_type == "time_key":
        return "Calendar or fiscal year key for joins."
    if semantic_type == "support_status":
        return f"Support or quality classification for {artifact_name}."
    if semantic_type == "share":
        return f"Share-valued field for {artifact_name}: {label}."
    if semantic_type == "price":
        return f"Price-valued field for {artifact_name}: {label}."
    if semantic_type == "count":
        return f"Count or quantity field for {artifact_name}: {label}."
    return f"{label} field for {artifact_name}."


def build_childcare_release_contract(artifacts: Mapping[str, object]) -> dict[str, Any]:
    artifact_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    for artifact_name, value in artifacts.items():
        data = value
        description = ""
        artifact_kind = type(value).__name__
        if isinstance(value, Mapping) and ("data" in value or "frame" in value or "json" in value or "markdown" in value):
            data = value.get("data", value.get("frame", value.get("json", value.get("markdown"))))
            description = str(value.get("description", ""))
        if isinstance(data, pd.DataFrame):
            artifact_kind = "dataframe"
            primary_keys, join_keys = _key_columns_for_frame(data)
            artifact_rows.append(
                {
                    "artifact_name": artifact_name,
                    "artifact_group": _artifact_group(artifact_name),
                    "artifact_kind": artifact_kind,
                    "artifact_status": _artifact_status(artifact_name),
                    "frontend_priority": _frontend_priority(artifact_name),
                    "artifact_grain": _artifact_grain(data),
                    "primary_key_columns": ",".join(primary_keys),
                    "join_key_columns": ",".join(join_keys),
                    "headline_safe": bool(_frontend_priority(artifact_name) == "high" and _artifact_status(artifact_name) == "canonical"),
                    "description": description,
                }
            )
            for position, column_name in enumerate(data.columns, start=1):
                dtype = str(data[column_name].dtype)
                semantic_type = _column_semantic_type(str(column_name), dtype)
                column_rows.append(
                    {
                        "artifact_name": artifact_name,
                        "artifact_group": _artifact_group(artifact_name),
                        "artifact_status": _artifact_status(artifact_name),
                        "frontend_priority": _frontend_priority(artifact_name),
                        "column_name": str(column_name),
                        "column_position": int(position),
                        "pandas_dtype": dtype,
                        "nullable": bool(data[column_name].isna().any()),
                        "semantic_type": semantic_type,
                        "unit": _column_unit(str(column_name), semantic_type),
                        "description": _column_description(artifact_name, str(column_name), semantic_type),
                    }
                )
        elif isinstance(data, Mapping):
            artifact_kind = "mapping"
            artifact_rows.append(
                {
                    "artifact_name": artifact_name,
                    "artifact_group": _artifact_group(artifact_name),
                    "artifact_kind": artifact_kind,
                    "artifact_status": _artifact_status(artifact_name),
                    "frontend_priority": _frontend_priority(artifact_name),
                    "artifact_grain": "mapping",
                    "primary_key_columns": "",
                    "join_key_columns": "",
                    "headline_safe": bool(_frontend_priority(artifact_name) == "high" and _artifact_status(artifact_name) == "canonical"),
                    "description": description,
                }
            )
        elif isinstance(data, str):
            artifact_kind = "text"
            artifact_rows.append(
                {
                    "artifact_name": artifact_name,
                    "artifact_group": _artifact_group(artifact_name),
                    "artifact_kind": artifact_kind,
                    "artifact_status": _artifact_status(artifact_name),
                    "frontend_priority": _frontend_priority(artifact_name),
                    "artifact_grain": "document",
                    "primary_key_columns": "",
                    "join_key_columns": "",
                    "headline_safe": False,
                    "description": description,
                }
            )
    artifact_contracts = pd.DataFrame(artifact_rows, columns=CONTRACT_ARTIFACT_COLUMNS).sort_values(
        ["artifact_group", "artifact_name"], kind="stable"
    ).reset_index(drop=True)
    column_dictionary = pd.DataFrame(column_rows, columns=COLUMN_DICTIONARY_COLUMNS).sort_values(
        ["artifact_group", "artifact_name", "column_position"], kind="stable"
    ).reset_index(drop=True)
    return {
        "artifact_contracts": artifact_contracts,
        "column_dictionary": column_dictionary,
        "json": {
            "release_name": RELEASE_NAME,
            "release_scope": RELEASE_SCOPE,
            "artifact_contracts": artifact_contracts.to_dict(orient="records"),
            "column_dictionary": column_dictionary.to_dict(orient="records"),
        },
    }


def _sha256_for_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_childcare_release_bundle_index(
    *,
    published_artifacts: list[Mapping[str, Any]],
    current_mode: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for artifact in published_artifacts:
        output_path = Path(str(artifact.get("output_path", "")))
        provenance_path = Path(str(artifact.get("provenance_path", "")))
        output_exists = output_path.exists()
        provenance_exists = provenance_path.exists()
        rows.append(
            {
                "artifact_name": str(artifact.get("artifact_name", "")),
                "artifact_group": str(artifact.get("artifact_group", "")),
                "artifact_kind": str(artifact.get("artifact_kind", "")),
                "artifact_status": str(artifact.get("artifact_status", "diagnostic")),
                "frontend_priority": str(artifact.get("frontend_priority", "low")),
                "publication_tier": str(artifact.get("publication_tier", "appendix")),
                "current_mode": current_mode,
                "sample_available": bool(artifact.get("sample_available", True)),
                "real_available": bool(artifact.get("real_available", True)),
                "producer_command": str(artifact.get("producer_command", "")),
                "output_path": str(output_path),
                "provenance_path": str(provenance_path),
                "file_exists": output_exists,
                "provenance_exists": provenance_exists,
                "sha256": _sha256_for_path(output_path) if output_exists else "",
                "upstream_dependencies": "|".join(str(item) for item in artifact.get("upstream_dependencies", [])),
            }
        )
    table = pd.DataFrame(rows, columns=BUNDLE_INDEX_COLUMNS).sort_values(
        ["artifact_group", "artifact_name"], kind="stable"
    ).reset_index(drop=True)
    return {
        "table": table,
        "json": {
            "release_name": RELEASE_NAME,
            "release_scope": RELEASE_SCOPE,
            "current_mode": current_mode,
            "artifacts": table.to_dict(orient="records"),
        },
    }


def build_childcare_frontend_handoff_summary(
    *,
    published_artifacts: list[Mapping[str, Any]],
    headline_summary: Mapping[str, Any] | None = None,
    methods_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, str]]] = {
        "canonical_artifacts": [],
        "diagnostic_artifacts": [],
        "additive_artifacts": [],
    }
    for artifact in published_artifacts:
        status = str(artifact.get("artifact_status", "diagnostic"))
        row = {
            "artifact_name": str(artifact.get("artifact_name", "")),
            "output_path": str(artifact.get("output_path", "")),
            "publication_tier": str(artifact.get("publication_tier", "appendix")),
            "frontend_priority": str(artifact.get("frontend_priority", "low")),
        }
        if status == "canonical":
            grouped["canonical_artifacts"].append(row)
        elif status == "additive":
            grouped["additive_artifacts"].append(row)
        else:
            grouped["diagnostic_artifacts"].append(row)
    headline = dict(headline_summary or {})
    methods = dict(methods_summary or {})
    caveats = []
    for item in list(headline.get("notes", [])) + list(methods.get("limitations", [])):
        text = str(item).strip()
        if text and text not in caveats:
            caveats.append(text)
    return {
        "release_name": RELEASE_NAME,
        "release_scope": RELEASE_SCOPE,
        "canonical_backend": True,
        "pooled_path_canonical": True,
        "segmented_outputs_additive_only": True,
        "frontend_figure_work_out_of_scope": True,
        "real_release_ready": bool(headline.get("real_release_ready", False)),
        "publication_rules": [
            {
                "topic": "ccdf_headline_decomposition",
                "eligible_support_buckets": ["explicit", "inferred"],
                "diagnostic_only_support_buckets": ["retained_proxy", "downgraded_proxy"],
                "rule": "Use only explicit and inferred CCDF support rows for headline-grade decomposition claims.",
            },
            {
                "topic": "licensing_iv",
                "rule": "Treat licensing IV outputs as diagnostics-only unless the backend recommended_use_tier is headline.",
            },
        ],
        **grouped,
        "backend_caveats": caveats,
    }


def build_childcare_release_backend_outputs(
    *,
    pooled_headline_summary: Mapping[str, Any] | None = None,
    ccdf_admin_state_year: pd.DataFrame | None = None,
    ccdf_policy_controls_coverage: pd.DataFrame | None = None,
    ccdf_policy_promoted_controls_state_year: pd.DataFrame | None = None,
    segmented_state_year_summary: pd.DataFrame | None = None,
    segmented_state_fallback_summary: pd.DataFrame | None = None,
    segmented_channel_scenarios: pd.DataFrame | None = None,
    licensing_rules_raw_audit: pd.DataFrame | None = None,
    licensing_harmonized_rules: pd.DataFrame | None = None,
    licensing_stringency_index: pd.DataFrame | None = None,
    licensing_iv_summary: Mapping[str, Any] | None = None,
    licensing_iv_results: pd.DataFrame | None = None,
    licensing_iv_usability_summary: pd.DataFrame | None = None,
    licensing_first_stage_diagnostics: pd.DataFrame | None = None,
    licensing_treatment_timing: pd.DataFrame | None = None,
    licensing_leave_one_state_out: pd.DataFrame | None = None,
    release_source_readiness: pd.DataFrame | None = None,
    release_manual_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    headline_summary = build_childcare_release_headline_summary(
        pooled_headline_summary=pooled_headline_summary,
        ccdf_admin_state_year=ccdf_admin_state_year,
        ccdf_policy_controls_coverage=ccdf_policy_controls_coverage,
        ccdf_policy_promoted_controls_state_year=ccdf_policy_promoted_controls_state_year,
        segmented_state_year_summary=segmented_state_year_summary,
        segmented_state_fallback_summary=segmented_state_fallback_summary,
        licensing_rules_raw_audit=licensing_rules_raw_audit,
        licensing_iv_summary=licensing_iv_summary,
        licensing_rules_harmonized=licensing_harmonized_rules,
        licensing_stringency_index=licensing_stringency_index,
        release_source_readiness=release_source_readiness,
        release_manual_requirements=release_manual_requirements,
    )
    support_tables = build_childcare_release_support_tables(
        ccdf_admin_state_year=ccdf_admin_state_year,
        segmented_support_quality_summary=segmented_state_fallback_summary,
        licensing_harmonized_rules=licensing_harmonized_rules,
        ccdf_policy_controls_coverage=ccdf_policy_controls_coverage,
        ccdf_policy_promoted_controls_state_year=ccdf_policy_promoted_controls_state_year,
    )
    proxy_gap_tables = build_childcare_release_ccdf_proxy_gap_tables(ccdf_admin_state_year)
    segmented_comparison = build_childcare_release_segmented_comparison(
        segmented_channel_scenarios=segmented_channel_scenarios,
        segmented_state_fallback_summary=segmented_state_fallback_summary,
    )
    methods_summary = build_childcare_release_methods_summary(
        pooled_headline_summary=pooled_headline_summary,
        ccdf_admin_state_year=ccdf_admin_state_year,
        ccdf_policy_controls_coverage=ccdf_policy_controls_coverage,
        segmented_state_fallback_summary=segmented_state_fallback_summary,
        licensing_rules_raw_audit=licensing_rules_raw_audit,
        licensing_iv_summary=licensing_iv_summary,
        licensing_rules_harmonized=licensing_harmonized_rules,
        release_manual_requirements=release_manual_requirements,
    )
    readiness_table = (
        release_source_readiness.copy()
        if isinstance(release_source_readiness, pd.DataFrame)
        else pd.DataFrame(columns=SOURCE_READINESS_COLUMNS)
    )
    missing_required_count = (
        int(readiness_table["readiness_status"].eq("missing_required").sum())
        if not readiness_table.empty and "readiness_status" in readiness_table.columns
        else 0
    )
    real_release_blocker_count = (
        int(
            readiness_table.loc[
                readiness_table["real_release_required"].fillna(False).astype(bool)
                & readiness_table["readiness_status"].ne("ready")
            ].shape[0]
        )
        if not readiness_table.empty
        and "real_release_required" in readiness_table.columns
        and "readiness_status" in readiness_table.columns
        else 0
    )
    if missing_required_count > 0:
        readiness_status = "missing_required_sources"
    elif real_release_blocker_count > 0:
        readiness_status = "real_release_blocked"
    else:
        readiness_status = "ready"
    readiness_summary = {
        "release_name": RELEASE_NAME,
        "current_mode": (
            str(readiness_table["current_mode"].iloc[0])
            if not readiness_table.empty and "current_mode" in readiness_table.columns
            else "unknown"
        ),
        "ready_count": int(readiness_table["exists"].sum()) if not readiness_table.empty and "exists" in readiness_table.columns else 0,
        "missing_count": int((~readiness_table["exists"]).sum()) if not readiness_table.empty and "exists" in readiness_table.columns else 0,
        "missing_required_count": missing_required_count,
        "real_release_blocker_count": real_release_blocker_count,
        "status": readiness_status,
    }
    readiness_json = {
        "summary": readiness_summary,
        "rows": readiness_table.to_dict(orient="records"),
    }
    manual_requirements = dict(release_manual_requirements or {})
    manual_requirements_table = (
        manual_requirements.get("table", pd.DataFrame())
        if isinstance(manual_requirements.get("table", pd.DataFrame()), pd.DataFrame)
        else pd.DataFrame(columns=MANUAL_REQUIREMENTS_COLUMNS)
    )
    manual_requirements_summary = dict(manual_requirements.get("summary") or manual_requirements.get("json") or {})
    manual_requirements_json = dict(manual_requirements.get("json") or {})
    if not manual_requirements_json:
        manual_requirements_json = {
            "summary": manual_requirements_summary,
            "rows": manual_requirements_table.to_dict(orient="records"),
        }
    else:
        manual_requirements_json.setdefault("summary", manual_requirements_summary)
        manual_requirements_json.setdefault("rows", manual_requirements_table.to_dict(orient="records"))
    manual_requirements_markdown = str(manual_requirements.get("markdown", ""))
    release_artifacts = {
        "release_headline_summary": headline_summary["table"],
        "release_headline_summary_json": headline_summary["json"],
        "support_quality_summary": support_tables["support_quality_summary"],
        "ccdf_support_mix_summary": support_tables["ccdf_support_mix_summary"],
        "policy_coverage_summary": support_tables["policy_coverage_summary"],
        "ccdf_proxy_gap_summary": proxy_gap_tables["proxy_gap_summary"],
        "ccdf_proxy_gap_state_years": proxy_gap_tables["proxy_gap_state_years"],
        "segmented_comparison": segmented_comparison,
        "methods_summary": methods_summary["json"],
        "methods_markdown": methods_summary["markdown"],
        "release_source_readiness_table": readiness_table,
        "release_source_readiness_summary": readiness_summary,
        "release_manual_requirements_table": manual_requirements_table,
        "release_manual_requirements_summary": manual_requirements_summary,
        "release_manual_requirements_markdown": manual_requirements_markdown,
        "licensing_rules_raw_audit": licensing_rules_raw_audit if licensing_rules_raw_audit is not None else pd.DataFrame(),
        "licensing_iv_results": licensing_iv_results if licensing_iv_results is not None else pd.DataFrame(),
        "licensing_iv_usability_summary": (
            licensing_iv_usability_summary if licensing_iv_usability_summary is not None else pd.DataFrame()
        ),
        "licensing_first_stage_diagnostics": licensing_first_stage_diagnostics if licensing_first_stage_diagnostics is not None else pd.DataFrame(),
        "licensing_treatment_timing": licensing_treatment_timing if licensing_treatment_timing is not None else pd.DataFrame(),
        "licensing_leave_one_state_out": licensing_leave_one_state_out if licensing_leave_one_state_out is not None else pd.DataFrame(),
    }
    schema_inventory = build_childcare_release_schema_inventory(release_artifacts)
    contract_inventory = build_childcare_release_contract(release_artifacts)
    artifact_index = build_childcare_release_manifest(
        release_artifacts
        | {
            "release_schema_artifact_summary": schema_inventory["artifact_summary"],
            "release_schema_column_schema": schema_inventory["column_schema"],
            "release_artifact_contracts": contract_inventory["artifact_contracts"],
            "release_column_dictionary": contract_inventory["column_dictionary"],
        }
    )
    return {
        "release_headline_summary": headline_summary,
        "release_support_tables": support_tables,
        "release_ccdf_proxy_gap_tables": proxy_gap_tables,
        "release_segmented_comparison": segmented_comparison,
        "release_methods_summary": methods_summary,
        "release_source_readiness": {
            "summary": readiness_summary,
            "json": readiness_json,
            "table": readiness_table,
        },
        "release_manual_requirements": {
            "summary": manual_requirements_summary,
            "json": manual_requirements_json,
            "table": manual_requirements_table,
            "markdown": manual_requirements_markdown,
        },
        "release_schema_inventory": schema_inventory,
        "release_contract_inventory": contract_inventory,
        "release_manifest": artifact_index,
    }
