from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

SEGMENT_DIMENSIONS = ("child_age", "provider_type", "channel")
SEGMENT_DEFINITION_COLUMNS = [
    "segment_id",
    "segment_label",
    "segment_order",
    "child_age",
    "provider_type",
    "channel",
]


def load_segment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"segment config must parse to a mapping: {config_path}")
    segments = loaded.get("segments")
    if isinstance(segments, list) and segments:
        return loaded
    if isinstance(segments, Mapping) and segments.get("dimensions"):
        return loaded
    raise ValueError(f"segment config must contain a non-empty `segments` definition: {config_path}")


def _normalize_segment_values(raw: Any) -> list[str]:
    if raw is None:
        return ["*"]
    if isinstance(raw, str):
        value = raw.strip()
        return ["*"] if value in {"", "*"} else [value]
    if isinstance(raw, (list, tuple, set)):
        cleaned = [str(item).strip() for item in raw if str(item).strip()]
        if not cleaned or "*" in cleaned:
            return ["*"]
        return cleaned
    value = str(raw).strip()
    return ["*"] if value in {"", "*"} else [value]


def _normalize_segment_entry(entry: Mapping[str, Any], index: int) -> list[dict[str, Any]]:
    segment_id = str(entry.get("segment_id") or entry.get("id") or f"segment_{index + 1}")
    segment_label = str(entry.get("segment_label") or entry.get("label") or segment_id)
    match = entry.get("match")
    if match is None:
        match = {}
    if not isinstance(match, Mapping):
        raise ValueError(f"segment `{segment_id}` has a non-mapping `match` block")

    dimension_values: dict[str, list[str]] = {}
    for dimension in SEGMENT_DIMENSIONS:
        source_value = match.get(dimension, entry.get(dimension))
        dimension_values[dimension] = _normalize_segment_values(source_value)

    rows: list[dict[str, Any]] = []
    for child_age, provider_type, channel in product(
        dimension_values["child_age"],
        dimension_values["provider_type"],
        dimension_values["channel"],
    ):
        rows.append(
            {
                "segment_id": segment_id,
                "segment_label": segment_label,
                "segment_order": int(index),
                "child_age": child_age,
                "provider_type": provider_type,
                "channel": channel,
            }
        )
    return rows


def _expand_dimension_segments(segment_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    dimensions = segment_config.get("dimensions", list(SEGMENT_DIMENSIONS))
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("segment config dimensions must be a non-empty list")

    normalized_values: dict[str, list[str]] = {}
    for dimension in SEGMENT_DIMENSIONS:
        if dimension not in dimensions:
            normalized_values[dimension] = ["*"]
            continue
        values = _normalize_segment_values(segment_config.get(dimension))
        if values == ["*"]:
            raise ValueError(f"segment config dimension `{dimension}` must declare explicit values")
        normalized_values[dimension] = values

    rows: list[dict[str, Any]] = []
    for index, (child_age, provider_type, channel) in enumerate(
        product(
            normalized_values["child_age"],
            normalized_values["provider_type"],
            normalized_values["channel"],
        )
    ):
        segment_id = "_".join([child_age, provider_type, channel]).replace("-", "_")
        rows.append(
            {
                "segment_id": segment_id,
                "segment_label": segment_id,
                "segment_order": index,
                "child_age": child_age,
                "provider_type": provider_type,
                "channel": channel,
            }
        )
    return rows


def _resolve_segment_entries(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    segments = config.get("segments")
    if isinstance(segments, list) and segments:
        rows: list[dict[str, Any]] = []
        for index, raw_entry in enumerate(segments):
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"segment index {index} is not a mapping")
            rows.extend(_normalize_segment_entry(raw_entry, index))
        return rows
    if isinstance(segments, Mapping):
        return _expand_dimension_segments(segments)
    raise ValueError("segment config must contain a non-empty `segments` definition")


def build_segment_definitions(config: Mapping[str, Any]) -> pd.DataFrame:
    rows = _resolve_segment_entries(config)
    definitions = pd.DataFrame(rows)
    if definitions.empty:
        return pd.DataFrame(columns=SEGMENT_DEFINITION_COLUMNS)
    definitions = definitions.drop_duplicates(
        subset=["segment_id", "child_age", "provider_type", "channel"], keep="first"
    )
    return definitions[SEGMENT_DEFINITION_COLUMNS].sort_values(
        ["segment_order", "segment_id", "child_age", "provider_type", "channel"], kind="stable"
    ).reset_index(drop=True)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid = numeric_values.notna() & numeric_weights.gt(0)
    if valid.any():
        return float(np.average(numeric_values.loc[valid], weights=numeric_weights.loc[valid]))
    if numeric_values.notna().any():
        return float(numeric_values.loc[numeric_values.notna()].mean())
    return float("nan")


def _required_dimensions(definitions: pd.DataFrame) -> list[str]:
    required: list[str] = []
    for dimension in SEGMENT_DIMENSIONS:
        if dimension in definitions.columns and definitions[dimension].astype(str).ne("*").any():
            required.append(dimension)
    return required


def build_ndcp_segment_price_panel(
    ndcp_frame: pd.DataFrame,
    segment_definitions: pd.DataFrame,
    geography_keys: tuple[str, ...] = ("state_fips", "year"),
    price_col: str = "annual_price",
    weight_col: str = "sample_weight",
    imputed_flag_col: str = "imputed_flag",
) -> pd.DataFrame:
    required_columns = list(geography_keys) + [price_col]
    missing_required = [column for column in required_columns if column not in ndcp_frame.columns]
    if missing_required:
        missing = ", ".join(missing_required)
        raise KeyError(f"NDCP frame is missing required columns: {missing}")

    missing_def_columns = [
        column for column in SEGMENT_DEFINITION_COLUMNS if column not in segment_definitions.columns
    ]
    if missing_def_columns:
        missing = ", ".join(missing_def_columns)
        raise KeyError(f"segment definitions are missing required columns: {missing}")

    working = ndcp_frame.copy()
    needed_dimensions = _required_dimensions(segment_definitions)
    missing_dimensions = [column for column in needed_dimensions if column not in working.columns]
    unresolved_dimensions: list[str] = []
    for column in missing_dimensions:
        values = sorted(segment_definitions[column].dropna().astype(str).unique().tolist())
        if len(values) == 1:
            working[column] = values[0]
        else:
            unresolved_dimensions.append(column)
    if unresolved_dimensions:
        missing = ", ".join(unresolved_dimensions)
        raise KeyError(f"NDCP frame is missing required segment dimensions: {missing}")

    if weight_col in working.columns:
        working["_segment_weight"] = pd.to_numeric(working[weight_col], errors="coerce").fillna(0.0)
    else:
        working["_segment_weight"] = 1.0
    if imputed_flag_col in working.columns:
        working["_segment_imputed_flag"] = pd.to_numeric(working[imputed_flag_col], errors="coerce")
    else:
        working["_segment_imputed_flag"] = np.nan

    expanded_parts: list[pd.DataFrame] = []
    for row in segment_definitions.itertuples(index=False):
        subset = working
        for dimension in SEGMENT_DIMENSIONS:
            segment_value = getattr(row, dimension)
            if str(segment_value) == "*" or dimension not in subset.columns:
                continue
            subset = subset.loc[subset[dimension].astype(str).eq(str(segment_value))]
        if subset.empty:
            continue
        tagged = subset.copy()
        tagged["segment_id"] = row.segment_id
        tagged["segment_label"] = row.segment_label
        tagged["segment_order"] = row.segment_order
        tagged["segment_child_age"] = row.child_age
        tagged["segment_provider_type"] = row.provider_type
        tagged["segment_channel"] = row.channel
        expanded_parts.append(tagged)

    output_columns = list(geography_keys) + [
        "segment_id",
        "segment_label",
        "segment_order",
        "segment_child_age",
        "segment_provider_type",
        "segment_channel",
        "segment_annual_price",
        "segment_weight_sum",
        "segment_observation_count",
        "segment_ndcp_imputed_share",
        "segment_ndcp_imputed_any",
        "segment_ndcp_imputed_all",
    ]
    if not expanded_parts:
        return pd.DataFrame(columns=output_columns)

    expanded = pd.concat(expanded_parts, ignore_index=True)
    group_columns = list(geography_keys) + [
        "segment_id",
        "segment_label",
        "segment_order",
        "segment_child_age",
        "segment_provider_type",
        "segment_channel",
    ]
    grouped = (
        expanded.groupby(group_columns, as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "segment_annual_price": _weighted_average(
                        group[price_col], group["_segment_weight"]
                    ),
                    "segment_weight_sum": float(
                        pd.to_numeric(group["_segment_weight"], errors="coerce")
                        .fillna(0.0)
                        .clip(lower=0.0)
                        .sum()
                    ),
                    "segment_observation_count": int(len(group)),
                    "segment_ndcp_imputed_share": _weighted_average(
                        group["_segment_imputed_flag"], group["_segment_weight"]
                    ),
                    "segment_ndcp_imputed_any": bool(
                        pd.to_numeric(group["_segment_imputed_flag"], errors="coerce")
                        .fillna(0.0)
                        .gt(0)
                        .any()
                    ),
                    "segment_ndcp_imputed_all": bool(
                        pd.to_numeric(group["_segment_imputed_flag"], errors="coerce").notna().any()
                        and pd.to_numeric(group["_segment_imputed_flag"], errors="coerce")
                        .fillna(0.0)
                        .gt(0)
                        .all()
                    ),
                }
            )
        )
        .reset_index(drop=True)
    )
    return grouped[output_columns].sort_values(
        list(geography_keys) + ["segment_order", "segment_id"], kind="stable"
    ).reset_index(drop=True)


def aggregate_segment_panel_to_pooled(
    segment_panel: pd.DataFrame,
    geography_keys: tuple[str, ...] = ("state_fips", "year"),
) -> pd.DataFrame:
    required = list(geography_keys) + ["segment_annual_price", "segment_weight_sum"]
    missing = [column for column in required if column not in segment_panel.columns]
    if missing:
        raise KeyError(f"segment panel is missing required columns: {', '.join(missing)}")

    grouped = (
        segment_panel.groupby(list(geography_keys), as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "pooled_annual_price": _weighted_average(
                        group["segment_annual_price"], group["segment_weight_sum"]
                    ),
                    "pooled_weight_sum": float(
                        pd.to_numeric(group["segment_weight_sum"], errors="coerce")
                        .fillna(0.0)
                        .clip(lower=0.0)
                        .sum()
                    ),
                    "pooled_segment_count": int(group["segment_id"].nunique())
                    if "segment_id" in group.columns
                    else int(len(group)),
                    "pooled_ndcp_imputed_share": _weighted_average(
                        group.get("segment_ndcp_imputed_share", pd.Series(dtype=float)),
                        group["segment_weight_sum"],
                    ),
                }
            )
        )
        .reset_index(drop=True)
    )
    return grouped.sort_values(list(geography_keys), kind="stable").reset_index(drop=True)


def build_segment_to_pooled_mapping(
    segment_panel: pd.DataFrame,
    pooled_benchmark: pd.DataFrame | None = None,
    geography_keys: tuple[str, ...] = ("state_fips", "year"),
) -> pd.DataFrame:
    aggregated = aggregate_segment_panel_to_pooled(
        segment_panel, geography_keys=geography_keys
    ).rename(
        columns={
            "pooled_annual_price": "pooled_annual_price_from_segments",
            "pooled_weight_sum": "pooled_weight_sum_from_segments",
            "pooled_ndcp_imputed_share": "pooled_ndcp_imputed_share_from_segments",
        }
    )
    if pooled_benchmark is None or pooled_benchmark.empty:
        aggregated["pooled_annual_price_direct"] = np.nan
        aggregated["pooled_price_gap"] = np.nan
        return aggregated

    merged = aggregated.merge(pooled_benchmark, on=list(geography_keys), how="left")
    merged["pooled_price_gap"] = (
        pd.to_numeric(merged["pooled_annual_price_from_segments"], errors="coerce")
        - pd.to_numeric(merged.get("pooled_annual_price"), errors="coerce")
    )
    return merged.rename(
        columns={
            "pooled_annual_price": "pooled_annual_price_direct",
            "pooled_weight_sum": "pooled_weight_sum_direct",
            "pooled_ndcp_imputed_share": "pooled_ndcp_imputed_share_direct",
        }
    ).sort_values(list(geography_keys), kind="stable").reset_index(drop=True)


def build_pooled_ndcp_price_benchmark(
    ndcp_frame: pd.DataFrame,
    geography_keys: tuple[str, ...] = ("state_fips", "year"),
    price_col: str = "annual_price",
    weight_col: str = "sample_weight",
    imputed_flag_col: str = "imputed_flag",
) -> pd.DataFrame:
    required = list(geography_keys) + [price_col]
    missing = [column for column in required if column not in ndcp_frame.columns]
    if missing:
        raise KeyError(f"NDCP frame is missing required columns: {', '.join(missing)}")

    working = ndcp_frame.copy()
    if weight_col in working.columns:
        working["_pooled_weight"] = pd.to_numeric(working[weight_col], errors="coerce").fillna(0.0)
    else:
        working["_pooled_weight"] = 1.0
    if imputed_flag_col in working.columns:
        working["_pooled_imputed_flag"] = pd.to_numeric(working[imputed_flag_col], errors="coerce")
    else:
        working["_pooled_imputed_flag"] = np.nan

    grouped = (
        working.groupby(list(geography_keys), as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "pooled_annual_price": _weighted_average(
                        group[price_col], group["_pooled_weight"]
                    ),
                    "pooled_weight_sum": float(
                        pd.to_numeric(group["_pooled_weight"], errors="coerce")
                        .fillna(0.0)
                        .clip(lower=0.0)
                        .sum()
                    ),
                    "pooled_ndcp_imputed_share": _weighted_average(
                        group["_pooled_imputed_flag"], group["_pooled_weight"]
                    ),
                }
            )
        )
        .reset_index(drop=True)
    )
    return grouped.sort_values(list(geography_keys), kind="stable").reset_index(drop=True)
