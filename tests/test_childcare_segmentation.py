from __future__ import annotations

import numpy as np
import pandas as pd

from unpaidwork import cli
from unpaidwork.childcare.segmentation import (
    aggregate_segment_panel_to_pooled,
    build_ndcp_segment_price_panel,
    build_pooled_ndcp_price_benchmark,
    build_segment_definitions,
    build_segment_to_pooled_mapping,
    load_segment_config,
)
from unpaidwork.ingest.provenance import sidecar_path
from unpaidwork.storage import read_parquet, write_parquet


def _sample_ndcp_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "01",
                "county_fips": "01001",
                "year": 2021,
                "child_age": "infant",
                "provider_type": "center",
                "channel": "private",
                "annual_price": 12000.0,
                "sample_weight": 1.0,
                "imputed_flag": 0,
            },
            {
                "state_fips": "01",
                "county_fips": "01001",
                "year": 2021,
                "child_age": "toddler",
                "provider_type": "home",
                "channel": "private",
                "annual_price": 9000.0,
                "sample_weight": 3.0,
                "imputed_flag": 1,
            },
            {
                "state_fips": "01",
                "county_fips": "01003",
                "year": 2022,
                "child_age": "infant",
                "provider_type": "center",
                "channel": "private",
                "annual_price": 13000.0,
                "sample_weight": 2.0,
                "imputed_flag": 0,
            },
            {
                "state_fips": "02",
                "county_fips": "02001",
                "year": 2021,
                "child_age": "preschool",
                "provider_type": "center",
                "channel": "private",
                "annual_price": 8000.0,
                "sample_weight": 4.0,
                "imputed_flag": 0,
            },
        ]
    )


def test_load_segment_config_and_build_definitions(tmp_path):
    config_path = tmp_path / "segments.yaml"
    config_path.write_text(
        "\n".join(
            [
                "segments:",
                "  - segment_id: infant_center_private",
                "    child_age: infant",
                "    provider_type: center",
                "    channel: private",
                "  - segment_id: private_mix",
                "    match:",
                "      child_age: [infant, toddler]",
                "      provider_type: [center, home]",
                "      channel: private",
            ]
        ),
        encoding="utf-8",
    )

    config = load_segment_config(config_path)
    definitions = build_segment_definitions(config)

    assert len(definitions) == 5
    expected_columns = {
        "segment_id",
        "segment_label",
        "segment_order",
        "child_age",
        "provider_type",
        "channel",
    }
    assert expected_columns <= set(definitions.columns)
    assert set(definitions["segment_id"]) == {"infant_center_private", "private_mix"}
    assert len(definitions.loc[definitions["segment_id"] == "private_mix"]) == 4


def test_build_segment_definitions_from_dimension_style_config(tmp_path):
    config_path = tmp_path / "segmented_solver.yaml"
    config_path.write_text(
        "\n".join(
            [
                "segments:",
                "  dimensions: [child_age, provider_type, channel]",
                "  child_age: [infant, toddler]",
                "  provider_type: [center, home]",
                "  channel: [private, public]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_segment_config(config_path)
    definitions = build_segment_definitions(config)

    assert len(definitions) == 8
    assert set(definitions["child_age"]) == {"infant", "toddler"}
    assert set(definitions["provider_type"]) == {"center", "home"}
    assert set(definitions["channel"]) == {"private", "public"}
    assert "infant_center_private" in set(definitions["segment_id"])


def test_build_ndcp_segment_price_panel_schema_and_metadata():
    ndcp = _sample_ndcp_frame()
    definitions = build_segment_definitions(
        {
            "segments": [
                {
                    "segment_id": "infant_center_private",
                    "child_age": "infant",
                    "provider_type": "center",
                    "channel": "private",
                },
                {
                    "segment_id": "toddler_home_private",
                    "child_age": "toddler",
                    "provider_type": "home",
                    "channel": "private",
                },
            ]
        }
    )

    panel = build_ndcp_segment_price_panel(ndcp, definitions)

    expected_columns = {
        "state_fips",
        "year",
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
    }
    assert expected_columns <= set(panel.columns)
    assert panel["segment_id"].isin({"infant_center_private", "toddler_home_private"}).all()

    infant_row = panel.loc[
        (panel["state_fips"] == "01")
        & (panel["year"] == 2021)
        & (panel["segment_id"] == "infant_center_private")
    ].iloc[0]
    assert infant_row["segment_annual_price"] == 12000.0
    assert infant_row["segment_weight_sum"] == 1.0
    assert infant_row["segment_ndcp_imputed_share"] == 0.0
    assert bool(infant_row["segment_ndcp_imputed_any"]) is False

    toddler_row = panel.loc[
        (panel["state_fips"] == "01")
        & (panel["year"] == 2021)
        & (panel["segment_id"] == "toddler_home_private")
    ].iloc[0]
    assert toddler_row["segment_annual_price"] == 9000.0
    assert toddler_row["segment_weight_sum"] == 3.0
    assert toddler_row["segment_ndcp_imputed_share"] == 1.0
    assert bool(toddler_row["segment_ndcp_imputed_any"]) is True
    assert bool(toddler_row["segment_ndcp_imputed_all"]) is True


def test_one_segment_panel_matches_weighted_pooled_benchmark():
    ndcp = _sample_ndcp_frame()
    definitions = build_segment_definitions(
        {
            "segments": [
                {
                    "segment_id": "all_private_care",
                    "match": {"child_age": "*", "provider_type": "*", "channel": "private"},
                }
            ]
        }
    )

    segment_panel = build_ndcp_segment_price_panel(ndcp, definitions)
    pooled_from_segments = aggregate_segment_panel_to_pooled(segment_panel)
    pooled_direct = build_pooled_ndcp_price_benchmark(ndcp)
    merged = pooled_from_segments.merge(
        pooled_direct,
        on=["state_fips", "year"],
        how="inner",
        suffixes=("_segment", "_direct"),
    )

    assert not merged.empty
    assert np.allclose(
        merged["pooled_annual_price_segment"],
        merged["pooled_annual_price_direct"],
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(
        merged["pooled_weight_sum_segment"],
        merged["pooled_weight_sum_direct"],
        rtol=0.0,
        atol=1e-12,
    )


def test_missing_channel_dimension_defaults_to_single_supported_value():
    ndcp = _sample_ndcp_frame().drop(columns=["channel"])
    definitions = build_segment_definitions(
        {
            "segments": [
                {
                    "segment_id": "all_private_care",
                    "match": {"child_age": "*", "provider_type": "*", "channel": "private"},
                }
            ]
        }
    )

    segment_panel = build_ndcp_segment_price_panel(ndcp, definitions)
    pooled_direct = build_pooled_ndcp_price_benchmark(ndcp)
    mapping = build_segment_to_pooled_mapping(segment_panel, pooled_benchmark=pooled_direct)

    assert not segment_panel.empty
    assert mapping["pooled_price_gap"].abs().max() < 1e-12


def test_build_childcare_segments_writes_outputs(project_paths):
    ndcp_path = project_paths.interim / "ndcp" / "ndcp.parquet"
    write_parquet(_sample_ndcp_frame().drop(columns=["channel"]), ndcp_path)

    config_dir = project_paths.root / "configs" / "extensions"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "segmented_solver.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: segmented_solver",
                "mode: segmented_baseline",
                "segments:",
                "  - segment_id: infant_center_private",
                "    segment_label: infant / center / private",
                "    child_age: infant",
                "    provider_type: center",
                "    channel: private",
                "  - segment_id: toddler_home_private",
                "    segment_label: toddler / home / private",
                "    child_age: toddler",
                "    provider_type: home",
                "    channel: private",
                "output_namespace:",
                "  namespace: data/interim/childcare/segmented_solver",
                "  files:",
                "    segment_definitions: segment_definitions.parquet",
                "    segment_price_panel: ndcp_segment_prices.parquet",
                "    segment_mappings: segmented_to_pooled_mapping.parquet",
            ]
        ),
        encoding="utf-8",
    )

    cli.build_childcare_segments(
        project_paths,
        sample=True,
        refresh=False,
        dry_run=False,
        config_name_or_path="segmented_solver",
    )

    output_root = project_paths.root / "data" / "interim" / "childcare" / "segmented_solver"
    definitions_path = output_root / "segment_definitions.parquet"
    panel_path = output_root / "ndcp_segment_prices.parquet"
    mapping_path = output_root / "segmented_to_pooled_mapping.parquet"
    assert definitions_path.exists()
    assert panel_path.exists()
    assert mapping_path.exists()
    assert sidecar_path(definitions_path).exists()
    assert sidecar_path(panel_path).exists()
    assert sidecar_path(mapping_path).exists()
    assert len(read_parquet(definitions_path)) == 2
