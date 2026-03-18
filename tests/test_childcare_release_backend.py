from __future__ import annotations

from pathlib import Path

import pandas as pd

from unpaidwork.childcare.release_backend import (
    build_childcare_release_backend_outputs,
    build_childcare_release_ccdf_proxy_gap_tables,
    build_childcare_release_headline_summary,
    build_childcare_release_manifest,
    build_childcare_release_manual_requirements,
    build_childcare_release_methods_summary,
    build_childcare_release_segmented_comparison,
    build_childcare_release_schema_inventory,
    build_childcare_release_source_readiness,
    build_childcare_release_support_tables,
)


def _pooled_headline_summary() -> dict[str, object]:
    return {
        "selected_headline_sample": "mvp",
        "mode": "real",
        "n_obs": 12,
        "n_states": 4,
        "year_min": 2021,
        "year_max": 2023,
    }


def _ccdf_admin_state_year() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_fips": ["01", "01", "02"],
            "year": [2021, 2022, 2023],
            "ccdf_support_flag": [
                "ccdf_explicit_split_observed",
                "ccdf_inferred_public_admin_from_children_served_minus_subsidized_private",
                "ccdf_split_proxy_from_payment_method_shares_large_gap_downgraded_to_children_served_proxy",
            ],
            "ccdf_admin_support_status": [
                "explicit_split_support",
                "inferred_split_support",
                "observed_long_payment_method_share_proxy_large_gap_downgraded_to_children_served_proxy",
            ],
            "ccdf_children_served": [100.0, 120.0, 80.0],
            "ccdf_payment_method_total_children": [pd.NA, pd.NA, 176.0],
            "ccdf_payment_method_gap_vs_children_served": [pd.NA, pd.NA, 96.0],
            "ccdf_payment_method_ratio_vs_children_served": [pd.NA, pd.NA, 2.2],
            "ccdf_grants_contracts_share": [pd.NA, pd.NA, 0.25],
            "ccdf_certificates_share": [pd.NA, pd.NA, 0.70],
            "ccdf_cash_share": [pd.NA, pd.NA, 0.05],
        }
    )


def _ccdf_policy_controls_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "control_name": ["copayment_required"],
            "control_kind": ["binary"],
            "observed_state_year_rows": [3],
            "missing_state_year_rows": [1],
            "state_year_coverage_rate": [0.75],
            "coverage_support_status": ["observed"],
            "promoted_state_year_rows": [2],
        }
    )


def _ccdf_policy_promoted_controls_state_year() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_fips": ["01", "02"],
            "year": [2022, 2023],
            "ccdf_control_copayment_required": [1, 1],
            "ccdf_policy_control_support_status": ["observed", "observed"],
        }
    )


def _segmented_state_fallback_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_fips": ["01", "02"],
            "year": [2022, 2023],
            "support_quality_tier": ["explicit_split_support", "proxy_or_fallback_support"],
            "ccdf_support_flag": ["observed", "proxy"],
            "ccdf_admin_support_status": ["explicit_split_support", "proxy_or_fallback_support"],
            "q0_support_flag": ["observed", "proxy"],
            "public_program_support_status": ["observed", "proxy"],
            "promoted_control_observed": [1, 0],
            "proxy_ccdf_row_count": [0, 1],
            "any_segment_allocation_fallback": [False, True],
            "any_private_allocation_fallback": [False, True],
        }
    )


def _segmented_channel_scenarios() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_fips": ["01", "01", "02"],
            "year": [2022, 2022, 2023],
            "solver_channel": ["private_unsubsidized", "public_admin", "private_subsidized"],
            "alpha": [0.5, 0.5, 0.7],
            "market_quantity_proxy": [100.0, 20.0, 80.0],
            "unpaid_quantity_proxy": [10.0, 2.0, 8.0],
            "p_baseline": [100.0, 50.0, 90.0],
            "p_shadow_marginal": [110.0, 50.0, 99.0],
            "p_alpha": [105.0, 50.0, 94.5],
        }
    )


def _licensing_rules_raw_audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_fips": ["01", "02"],
            "rule_text": ["age cap", "staff ratio"],
        }
    )


def _licensing_rules_harmonized() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_fips": ["01", "02", "02"],
            "rule_support_status": ["observed", "proxy", "observed"],
            "rule_family": ["age", "ratio", "ratio"],
        }
    )


def _licensing_stringency_index() -> pd.DataFrame:
    return pd.DataFrame({"state_fips": ["01"], "year": [2023], "stringency_index": [0.9]})


def _licensing_iv_summary() -> dict[str, object]:
    return {"status": "ready", "n_obs": 12, "first_stage_f_stat": 18.4}


def _licensing_iv_results() -> pd.DataFrame:
    return pd.DataFrame({"term": ["licensing_index"], "estimate": [0.3], "std_error": [0.1]})


def _licensing_first_stage_diagnostics() -> pd.DataFrame:
    return pd.DataFrame({"diagnostic": ["f_stat"], "value": [18.4]})


def _licensing_treatment_timing() -> pd.DataFrame:
    return pd.DataFrame({"state_fips": ["01"], "adoption_year": [2022]})


def _licensing_leave_one_state_out() -> pd.DataFrame:
    return pd.DataFrame({"state_fips": ["01"], "estimate": [0.28]})


def _release_contract() -> dict[str, object]:
    return {
        "release_name": "childcare_backend",
        "artifact_status": "canonical",
        "artifact_grain": "bundle",
        "primary_key_columns": ["artifact_name"],
        "join_key_columns": ["artifact_name"],
        "frontend_priority": "high",
    }


def _release_artifact_contracts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "artifact_name": ["release_contract", "release_bundle_index"],
            "artifact_grain": ["bundle", "bundle"],
            "artifact_status": ["canonical", "canonical"],
            "frontend_priority": ["high", "high"],
        }
    )


def _release_column_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "artifact_name": ["release_bundle_index", "release_bundle_index"],
            "column_name": ["output_path", "producer_command"],
            "semantic_type": ["path", "command"],
            "unit": ["path", "text"],
            "nullable": [False, False],
            "description": ["published output path", "command used to build artifact"],
        }
    )


def _release_bundle_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "output_path": [
                "outputs/reports/childcare_backend_release_bundle_index.json",
                "outputs/tables/childcare_backend_release_bundle_index.csv",
            ],
            "provenance_path": [
                "outputs/reports/childcare_backend_release_bundle_index.json.provenance.json",
                "outputs/tables/childcare_backend_release_bundle_index.csv.provenance.json",
            ],
            "producer_command": [
                "python -m unpaidwork.cli build-childcare-release-backend --real --refresh",
                "python -m unpaidwork.cli build-childcare-release-backend --real --refresh",
            ],
            "checksum": ["abc123", "def456"],
            "release_tier": ["release", "release"],
        }
    )


def test_build_childcare_release_source_readiness_tracks_required_and_optional_sources(tmp_path: Path) -> None:
    required_path = tmp_path / "required.parquet"
    required_path.write_text("ok", encoding="utf-8")

    readiness = build_childcare_release_source_readiness(
        required_sources=[
            {
                "name": "required_source",
                "path": required_path.name,
                "manual_download": False,
                "real_release_required": True,
                "description": "required test source",
            }
        ],
        optional_sources=[
            {
                "name": "optional_manual_source",
                "path": "missing_optional.csv",
                "manual_download": True,
                "real_release_required": True,
                "description": "optional missing source",
            }
        ],
        sample_mode=True,
        repo_root=tmp_path,
    )

    assert list(readiness.columns) == [
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
    required_row = readiness.loc[readiness["source_name"] == "required_source"].iloc[0]
    optional_row = readiness.loc[readiness["source_name"] == "optional_manual_source"].iloc[0]
    assert bool(required_row["exists"]) is True
    assert required_row["readiness_status"] == "ready"
    assert optional_row["readiness_status"] == "not_required_in_sample"
    assert bool(optional_row["manual_download"]) is True


def test_build_childcare_release_manual_requirements_summarizes_missing_sources(tmp_path: Path) -> None:
    readiness = pd.DataFrame(
        [
            {
                "source_name": "ccdf_policy_raw_directory",
                "source_tier": "optional",
                "source_path": str(tmp_path / "data/raw/ccdf/policies"),
                "path_kind": "directory",
                "exists": False,
                "manual_download": True,
                "real_release_required": True,
                "readiness_status": "missing_optional",
                "current_mode": "real",
                "description": "policy source",
            },
            {
                "source_name": "licensing_iv_results",
                "source_tier": "required",
                "source_path": str(tmp_path / "data/interim/licensing_iv_results.parquet"),
                "path_kind": "parquet",
                "exists": False,
                "manual_download": False,
                "real_release_required": True,
                "readiness_status": "missing_required",
                "current_mode": "real",
                "description": "iv output",
            },
        ]
    )

    manual_requirements = build_childcare_release_manual_requirements(readiness, sample_mode=False)

    assert manual_requirements["summary"]["missing_required_count"] == 1
    assert manual_requirements["summary"]["real_release_blocker_count"] == 2
    assert manual_requirements["summary"]["status"] == "missing_required_sources"
    assert manual_requirements["table"]["source_name"].tolist() == [
        "ccdf_policy_raw_directory",
        "licensing_iv_results",
    ]
    assert "Childcare backend manual requirements" in manual_requirements["markdown"]
    assert manual_requirements["json"] == manual_requirements["summary"]


def test_build_childcare_release_headline_summary_tracks_release_metrics() -> None:
    source_readiness = build_childcare_release_source_readiness(
        required_sources=[
            {
                "name": "ccdf_admin_state_year",
                "path": "/tmp/ccdf_admin_state_year.parquet",
                "manual_download": False,
                "real_release_required": True,
                "description": "required release input",
            }
        ],
        optional_sources=[
            {
                "name": "ccdf_policy_raw_directory",
                "path": "/tmp/ccdf_policy_raw_directory",
                "manual_download": True,
                "real_release_required": True,
                "description": "manual policy input",
            }
        ],
        sample_mode=True,
    )
    manual_requirements = build_childcare_release_manual_requirements(source_readiness, sample_mode=True)
    summary = build_childcare_release_headline_summary(
        pooled_headline_summary=_pooled_headline_summary(),
        ccdf_admin_state_year=_ccdf_admin_state_year(),
        ccdf_policy_controls_coverage=_ccdf_policy_controls_coverage(),
        ccdf_policy_promoted_controls_state_year=_ccdf_policy_promoted_controls_state_year(),
        segmented_state_year_summary=_segmented_state_fallback_summary(),
        licensing_rules_raw_audit=_licensing_rules_raw_audit(),
        licensing_iv_summary=_licensing_iv_summary(),
        licensing_rules_harmonized=_licensing_rules_harmonized(),
        licensing_stringency_index=_licensing_stringency_index(),
        release_source_readiness=source_readiness,
        release_manual_requirements=manual_requirements,
    )

    assert summary["json"]["release_name"] == "childcare_backend"
    assert summary["json"]["release_scope"] == "backend_only"
    assert summary["json"]["ccdf_admin_state_year_rows"] == 3
    assert summary["json"]["ccdf_admin_explicit_rows"] == 1
    assert summary["json"]["ccdf_admin_inferred_rows"] == 1
    assert summary["json"]["ccdf_admin_proxy_rows"] == 1
    assert summary["json"]["ccdf_proxy_gap_rows"] == 1
    assert summary["json"]["ccdf_proxy_gap_downgraded_rows"] == 1
    assert summary["json"]["ccdf_proxy_gap_mean_ratio_vs_children_served"] == 2.2
    assert summary["json"]["ccdf_proxy_gap_max_abs_vs_children_served"] == 96.0
    assert summary["json"]["licensing_raw_audit_rows"] == 2
    assert summary["json"]["licensing_rule_rows"] == 3
    assert summary["json"]["manual_action_count"] == 1
    assert summary["json"]["real_release_blocker_count"] == 2
    assert summary["json"]["source_readiness_status"] == "real_release_blocked"
    assert summary["json"]["release_ready"] is False
    assert summary["json"]["real_release_ready"] is False
    assert summary["json"]["ccdf_admin_retained_proxy_rows"] == 0
    assert summary["json"]["ccdf_admin_downgraded_proxy_rows"] == 1
    assert summary["json"]["ccdf_headline_decomposition_eligible_rows"] == 2
    assert summary["json"]["ccdf_headline_decomposition_excluded_rows"] == 1
    assert summary["json"]["ccdf_headline_decomposition_safe"] is False
    assert summary["json"]["licensing_iv_recommended_use"] == "diagnostics_only"
    assert list(summary["table"].columns) == list(summary["json"].keys())


def test_build_childcare_release_support_tables_returns_expected_tables() -> None:
    support_tables = build_childcare_release_support_tables(
        ccdf_admin_state_year=_ccdf_admin_state_year(),
        segmented_support_quality_summary=_segmented_state_fallback_summary(),
        licensing_harmonized_rules=_licensing_rules_harmonized(),
        ccdf_policy_controls_coverage=_ccdf_policy_controls_coverage(),
        ccdf_policy_promoted_controls_state_year=_ccdf_policy_promoted_controls_state_year(),
    )

    assert set(support_tables) == {"support_quality_summary", "ccdf_support_mix_summary", "policy_coverage_summary"}
    support_quality = support_tables["support_quality_summary"]
    support_mix = support_tables["ccdf_support_mix_summary"]
    policy_coverage = support_tables["policy_coverage_summary"]
    assert set(support_quality["component"]) == {"ccdf_admin", "segmented", "licensing"}
    assert set(support_mix["support_bucket"]) == {"downgraded_proxy", "explicit", "inferred"}
    assert set(support_mix["publication_treatment"]) == {"headline_eligible", "diagnostic_only"}
    assert bool(
        support_mix.loc[
            support_mix["support_bucket"].eq("downgraded_proxy"),
            "headline_decomposition_eligible",
        ].iloc[0]
    ) is False
    assert "explicit_split_support" in set(support_quality["support_tier"])
    assert "proxy_or_fallback_support" in set(support_quality["support_tier"])
    assert set(policy_coverage["component"]) == {"ccdf_policy_controls", "ccdf_promoted_controls", "licensing_rules"}
    assert "copayment_required" in set(policy_coverage["item_name"])
    assert policy_coverage["coverage_rate"].between(0.0, 1.0).all()


def test_build_childcare_release_ccdf_proxy_gap_tables_summarizes_proxy_rows() -> None:
    tables = build_childcare_release_ccdf_proxy_gap_tables(_ccdf_admin_state_year())

    assert set(tables) == {"proxy_gap_summary", "proxy_gap_state_years"}
    summary = tables["proxy_gap_summary"].iloc[0]
    top_rows = tables["proxy_gap_state_years"]
    assert summary["proxy_support_rows"] == 1
    assert summary["proxy_downgraded_children_served_rows"] == 1
    assert summary["proxy_large_gap_rows"] == 0
    assert summary["rows_with_payment_method_counts"] == 1
    assert summary["mean_payment_method_ratio_vs_children_served"] == 2.2
    assert summary["max_abs_payment_method_gap_vs_children_served"] == 96.0
    assert top_rows.iloc[0]["state_fips"] == "02"
    assert top_rows.iloc[0]["ccdf_payment_method_gap_vs_children_served"] == 96.0
    assert top_rows.iloc[0]["proxy_reliability_tier"] == "large_gap"
    assert top_rows.iloc[0]["proxy_split_treatment"] == "children_served_proxy_downgraded"


def test_build_childcare_release_segmented_comparison_adds_deltas_and_support_metadata() -> None:
    comparison = build_childcare_release_segmented_comparison(
        segmented_channel_scenarios=_segmented_channel_scenarios(),
        segmented_state_fallback_summary=_segmented_state_fallback_summary(),
    )

    assert list(comparison["solver_channel"]) == [
        "private_unsubsidized",
        "public_admin",
        "private_subsidized",
    ]
    private_row = comparison.loc[comparison["solver_channel"] == "private_unsubsidized"].iloc[0]
    public_row = comparison.loc[comparison["solver_channel"] == "public_admin"].iloc[0]
    assert private_row["p_shadow_delta_vs_baseline"] == 10.0
    assert private_row["p_alpha_delta_vs_baseline"] == 5.0
    assert private_row["p_shadow_pct_change_vs_baseline"] == 0.1
    assert private_row["comparison_role"] == "headline_alpha"
    assert bool(public_row["public_admin_price_invariant"]) is True
    assert "support_quality_tier" in comparison.columns
    assert "proxy_ccdf_row_count" in comparison.columns


def test_build_childcare_release_methods_summary_records_backend_limits() -> None:
    source_readiness = build_childcare_release_source_readiness(
        required_sources=[
            {
                "name": "ccdf_admin_state_year",
                "path": "/tmp/ccdf_admin_state_year.parquet",
                "manual_download": False,
                "real_release_required": True,
                "description": "required release input",
            }
        ],
        optional_sources=[
            {
                "name": "ccdf_policy_raw_directory",
                "path": "/tmp/ccdf_policy_raw_directory",
                "manual_download": True,
                "real_release_required": True,
                "description": "manual policy input",
            }
        ],
        sample_mode=True,
    )
    manual_requirements = build_childcare_release_manual_requirements(source_readiness, sample_mode=True)
    methods = build_childcare_release_methods_summary(
        pooled_headline_summary=_pooled_headline_summary(),
        ccdf_admin_state_year=_ccdf_admin_state_year(),
        ccdf_policy_controls_coverage=_ccdf_policy_controls_coverage(),
        segmented_state_fallback_summary=_segmented_state_fallback_summary(),
        licensing_rules_raw_audit=_licensing_rules_raw_audit(),
        licensing_iv_summary=_licensing_iv_summary(),
        licensing_rules_harmonized=_licensing_rules_harmonized(),
        release_manual_requirements=manual_requirements,
    )

    assert methods["json"]["backend_only"] is True
    assert methods["json"]["licensing_raw_audit_rows"] == 2
    assert methods["json"]["manual_action_count"] == 1
    assert methods["json"]["real_release_blocker_count"] == 2
    assert methods["json"]["proxy_support_rows"] == 1
    assert methods["json"]["retained_proxy_rows"] == 0
    assert methods["json"]["downgraded_proxy_rows"] == 1
    assert methods["json"]["headline_decomposition_excluded_rows"] == 1
    assert methods["json"]["licensing_iv_recommended_use"] == "diagnostics_only"
    assert methods["json"]["mean_payment_method_ratio_vs_children_served"] == 2.2
    assert any("Licensing raw audit rows are retained" in assumption for assumption in methods["json"]["assumptions"])
    assert any("Payment-method proxy-backed CCDF rows are retained" in assumption for assumption in methods["json"]["assumptions"])
    assert any("downgraded to a children-served fallback" in assumption for assumption in methods["json"]["assumptions"])
    assert any("manual-source blocker" in limitation for limitation in methods["json"]["limitations"])
    assert any("Headline-grade CCDF decomposition claims should exclude retained-proxy and downgraded-proxy rows" in limitation for limitation in methods["json"]["limitations"])
    assert "Childcare backend methods and limitations" in methods["markdown"]


def test_build_childcare_release_manifest_indexes_artifacts_by_type() -> None:
    manifest = build_childcare_release_manifest(
        {
            "ccdf_support_summary": pd.DataFrame({"support_tier": ["explicit_split_support"]}),
            "release_headline_summary": {"data": pd.DataFrame({"a": [1]}), "role": "release", "description": "headline"},
            "release_headline_summary_json": {"json": {"release_name": "childcare_backend"}},
            "methods_markdown": {"markdown": "backend methods"},
            "segmented_comparison": pd.DataFrame({"state_fips": ["01"], "year": [2022]}),
            "licensing_iv_results": pd.DataFrame({"estimate": [0.3]}),
            "release_contract": {"json": _release_contract(), "role": "release", "description": "release contract"},
            "release_artifact_contracts": {
                "data": _release_artifact_contracts(),
                "role": "release",
                "description": "artifact contracts table",
            },
            "release_column_dictionary": {
                "data": _release_column_dictionary(),
                "role": "release",
                "description": "column dictionary table",
            },
            "release_bundle_index": {
                "data": _release_bundle_index(),
                "role": "release",
                "description": "bundle index table",
            },
            "release_bundle_index_json": {
                "json": {"bundle_index_path": "outputs/reports/childcare_backend_release_bundle_index.json"},
                "role": "release",
                "description": "bundle index json",
            },
        }
    )

    assert set(manifest["artifact_group"]) == {"ccdf", "licensing", "pooled", "release", "segmented"}
    headline = manifest.loc[manifest["artifact_name"] == "release_headline_summary"].iloc[0]
    assert int(headline["row_count"]) == 1
    assert int(headline["column_count"]) == 1
    assert bool(headline["is_empty"]) is False
    methods = manifest.loc[manifest["artifact_name"] == "methods_markdown"].iloc[0]
    assert int(methods["text_length"]) > 0
    assert int(methods["key_count"]) == 0
    assert any(name == "release_contract" for name in manifest["artifact_name"])
    assert any(name == "release_bundle_index" for name in manifest["artifact_name"])
    bundle_index_row = manifest.loc[manifest["artifact_name"] == "release_bundle_index"].iloc[0]
    assert int(bundle_index_row["row_count"]) == 2
    assert bool(bundle_index_row["is_empty"]) is False


def test_build_childcare_release_backend_outputs_bundles_release_artifacts() -> None:
    source_readiness = build_childcare_release_source_readiness(
        required_sources=[
            {
                "name": "ccdf_admin_state_year",
                "path": "/tmp/ccdf_admin_state_year.parquet",
                "manual_download": False,
                "real_release_required": True,
                "description": "required release input",
            }
        ],
        optional_sources=[
            {
                "name": "ccdf_policy_raw_directory",
                "path": "/tmp/ccdf_policy_raw_directory",
                "manual_download": True,
                "real_release_required": True,
                "description": "manual policy input",
            }
        ],
        sample_mode=True,
    )
    manual_requirements = build_childcare_release_manual_requirements(source_readiness, sample_mode=True)
    outputs = build_childcare_release_backend_outputs(
        pooled_headline_summary=_pooled_headline_summary(),
        ccdf_admin_state_year=_ccdf_admin_state_year(),
        ccdf_policy_controls_coverage=_ccdf_policy_controls_coverage(),
        ccdf_policy_promoted_controls_state_year=_ccdf_policy_promoted_controls_state_year(),
        segmented_state_year_summary=_segmented_state_fallback_summary(),
        segmented_state_fallback_summary=_segmented_state_fallback_summary(),
        segmented_channel_scenarios=_segmented_channel_scenarios(),
        licensing_rules_raw_audit=_licensing_rules_raw_audit(),
        licensing_harmonized_rules=_licensing_rules_harmonized(),
        licensing_stringency_index=_licensing_stringency_index(),
        licensing_iv_summary=_licensing_iv_summary(),
        licensing_iv_results=_licensing_iv_results(),
        licensing_first_stage_diagnostics=_licensing_first_stage_diagnostics(),
        licensing_treatment_timing=_licensing_treatment_timing(),
        licensing_leave_one_state_out=_licensing_leave_one_state_out(),
        release_source_readiness=source_readiness,
        release_manual_requirements=manual_requirements,
    )

    assert set(outputs) == {
        "release_headline_summary",
        "release_support_tables",
        "release_ccdf_proxy_gap_tables",
        "release_segmented_comparison",
        "release_methods_summary",
        "release_schema_inventory",
        "release_contract_inventory",
        "release_manifest",
        "release_source_readiness",
        "release_manual_requirements",
    }
    assert outputs["release_headline_summary"]["json"]["release_ready"] is False
    assert set(outputs["release_support_tables"]) == {"support_quality_summary", "ccdf_support_mix_summary", "policy_coverage_summary"}
    assert set(outputs["release_ccdf_proxy_gap_tables"]) == {"proxy_gap_summary", "proxy_gap_state_years"}
    assert not outputs["release_segmented_comparison"].empty
    assert outputs["release_methods_summary"]["json"]["backend_only"] is True
    schema_inventory = outputs["release_schema_inventory"]
    assert set(schema_inventory) == {"artifact_summary", "column_schema", "json"}
    assert not schema_inventory["artifact_summary"].empty
    assert not schema_inventory["column_schema"].empty
    assert "artifact_summary" in schema_inventory["json"]
    assert "column_schema" in schema_inventory["json"]
    source_readiness_output = outputs["release_source_readiness"]
    manual_requirements_output = outputs["release_manual_requirements"]
    assert isinstance(source_readiness_output, dict)
    assert isinstance(manual_requirements_output, dict)
    assert set(source_readiness_output) >= {"summary", "json", "table"}
    assert set(manual_requirements_output) >= {"summary", "json", "table", "markdown"}
    assert {"source_name", "source_tier", "exists", "readiness_status"} <= set(source_readiness_output["table"].columns)
    assert {"source_name", "action_needed", "priority", "readiness_status"} <= set(manual_requirements_output["table"].columns)
    assert source_readiness_output["summary"]["missing_required_count"] == 0
    assert source_readiness_output["summary"]["missing_count"] == 2
    assert source_readiness_output["summary"]["real_release_blocker_count"] == 2
    assert source_readiness_output["summary"]["status"] == "real_release_blocked"
    assert "rows" in source_readiness_output["json"]
    assert manual_requirements_output["summary"]["manual_action_count"] == 1
    assert manual_requirements_output["summary"]["status"] == "real_release_blocked"
    assert "rows" in manual_requirements_output["json"]
    manifest = outputs["release_manifest"]
    assert "licensing_rules_raw_audit" in set(manifest["artifact_name"])
    assert "release_headline_summary_json" in set(manifest["artifact_name"])
    assert "release_source_readiness_table" in set(manifest["artifact_name"])
    assert "release_manual_requirements_table" in set(manifest["artifact_name"])
    assert "ccdf_proxy_gap_summary" in set(manifest["artifact_name"])


def test_build_childcare_release_schema_inventory_covers_contract_and_bundle_index_artifacts() -> None:
    inventory = build_childcare_release_schema_inventory(
        {
            "release_contract": {"json": _release_contract(), "role": "release", "description": "release contract"},
            "release_artifact_contracts": {
                "data": _release_artifact_contracts(),
                "role": "release",
                "description": "artifact contracts table",
            },
            "release_column_dictionary": {
                "data": _release_column_dictionary(),
                "role": "release",
                "description": "column dictionary table",
            },
            "release_bundle_index": {
                "data": _release_bundle_index(),
                "role": "release",
                "description": "bundle index table",
            },
            "release_bundle_index_json": {
                "json": {"bundle_index_path": "outputs/reports/childcare_backend_release_bundle_index.json"},
                "role": "release",
                "description": "bundle index json",
            },
        }
    )

    artifact_summary = inventory["artifact_summary"]
    column_schema = inventory["column_schema"]
    assert any(name == "release_contract" for name in artifact_summary["artifact_name"])
    assert any(name == "release_bundle_index" for name in artifact_summary["artifact_name"])
    assert "mapping" in set(artifact_summary["artifact_kind"])
    assert "dataframe" in set(artifact_summary["artifact_kind"])
    assert "output_path" in set(column_schema["column_name"])
    assert "provenance_path" in set(column_schema["column_name"])
    assert "producer_command" in set(column_schema["column_name"])
    assert "checksum" in set(column_schema["column_name"])
    assert "release_tier" in set(column_schema["column_name"])
