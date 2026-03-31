from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import types

import pandas as pd
import pytest

from unpriced.assumptions import childcare_model_assumptions
from unpriced import cli
from unpriced.childcare import ccdf as childcare_ccdf
from unpriced.errors import UnpaidWorkError
from unpriced.ingest.provenance import write_provenance_sidecar
from unpriced.ingest.ndcp import ingest as ingest_ndcp
from unpriced.storage import read_json, read_parquet, write_json, write_parquet


def _write_mode_provenance(project_paths, dataset_path: Path, sample_mode: bool) -> None:
    write_provenance_sidecar(
        dataset_path,
        source_files=[],
        parameters={"sample_mode": sample_mode},
        repo_root=project_paths.root,
    )


def _write_registry_mode(project_paths, dataset_path: Path, sample_mode: bool, last_fetched: str) -> None:
    registry_path = project_paths.registry
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        registry = read_parquet(registry_path)
    else:
        registry = pd.DataFrame()
    row = pd.DataFrame(
        [
            {
                "source_name": dataset_path.stem,
                "license": "test",
                "retrieval_method": "test",
                "checksum": "test",
                "last_fetched": last_fetched,
                "citation": "test",
                "raw_path": str(project_paths.raw / "test" / f"{dataset_path.stem}.dat"),
                "normalized_path": str(dataset_path.resolve()),
                "sample_mode": sample_mode,
            }
        ]
    )
    write_parquet(pd.concat([registry, row], ignore_index=True), registry_path)


def _write_licensing_iv_config(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "licensing_iv.yaml").write_text(
        "\n".join(
            [
                "name: licensing_iv",
                "mode: supply_identification",
                "output_namespace:",
                "  namespace: data/interim/childcare/licensing_iv",
                "  files:",
                "    harmonized_rules: licensing_rules_harmonized.parquet",
                "    rule_audit: licensing_rules_raw_audit.parquet",
                "    stringency_index: licensing_stringency_index.parquet",
                "    harmonization_summary: licensing_harmonization_summary.parquet",
                "    event_study_results: licensing_event_study_results.parquet",
                "    iv_results: licensing_iv_results.parquet",
                "    iv_usability_summary: licensing_iv_usability_summary.parquet",
                "    first_stage_diagnostics: licensing_first_stage_diagnostics.parquet",
                "    treatment_timing: licensing_treatment_timing.parquet",
                "    leave_one_state_out: licensing_leave_one_state_out.parquet",
                "    elasticity_panel: supply_iv_elasticities.parquet",
                "    iv_summary_json: licensing_iv_summary.json",
            ]
        ),
        encoding="utf-8",
    )


def _write_release_backend_config(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "release_backend.yaml").write_text(
        "\n".join(
            [
                "name: release_backend",
                "mode: backend_release",
                "sources:",
                "  required:",
                "    - name: ccdf_admin_state_year",
                "      path: data/interim/ccdf/ccdf_admin_state_year.parquet",
                "      manual_download: false",
                "      real_release_required: true",
                "      description: required release input",
                "    - name: segmented_state_fallback_summary",
                "      path: data/interim/childcare/segmented_reports/segmented_state_fallback_summary.parquet",
                "      manual_download: false",
                "      real_release_required: true",
                "      description: segmented diagnostics input",
                "  optional:",
                "    - name: ccdf_policy_raw_directory",
                "      path: data/raw/ccdf/policies",
                "      manual_download: true",
                "      real_release_required: true",
                "      description: manual policy workbook directory",
                "    - name: licensing_rules_long",
                "      path: data/raw/licensing/licensing_rules_long.csv",
                "      manual_download: true",
                "      real_release_required: true",
                "      description: optional richer licensing rule input",
                "outputs:",
                "  reports_namespace: outputs/reports",
                "  tables_namespace: outputs/tables",
                "  files:",
                "    headline_summary_json: childcare_backend_release_headline_summary.json",
                "    methods_summary_json: childcare_backend_release_methods_summary.json",
                "    methods_markdown: childcare_backend_release_methods.md",
                "    rebuild_markdown: childcare_backend_release_rebuild.md",
                "    manifest_json: childcare_backend_release_manifest.json",
                "    source_readiness_json: childcare_backend_release_source_readiness.json",
                "    manual_requirements_json: childcare_backend_release_manual_requirements.json",
                "    manual_requirements_markdown: childcare_backend_release_manual_requirements.md",
                "    schema_inventory_json: childcare_backend_release_schema_inventory.json",
                "    headline_summary_csv: childcare_backend_release_headline_summary.csv",
                "    support_quality_csv: childcare_backend_support_quality_summary.csv",
                "    ccdf_support_mix_csv: childcare_backend_ccdf_support_mix_summary.csv",
                "    policy_coverage_csv: childcare_backend_policy_coverage_summary.csv",
                "    ccdf_proxy_gap_summary_csv: childcare_backend_ccdf_proxy_gap_summary.csv",
                "    ccdf_proxy_gap_state_years_csv: childcare_backend_ccdf_proxy_gap_state_years.csv",
                "    segmented_comparison_csv: childcare_backend_segmented_comparison.csv",
                "    licensing_iv_results_csv: childcare_backend_licensing_iv_results.csv",
                "    licensing_iv_usability_summary_csv: childcare_backend_licensing_iv_usability_summary.csv",
                "    licensing_first_stage_diagnostics_csv: childcare_backend_licensing_first_stage_diagnostics.csv",
                "    licensing_treatment_timing_csv: childcare_backend_licensing_treatment_timing.csv",
                "    licensing_leave_one_state_out_csv: childcare_backend_licensing_leave_one_state_out.csv",
                "    manifest_csv: childcare_backend_release_manifest.csv",
                "    source_readiness_csv: childcare_backend_release_source_readiness.csv",
                "    manual_requirements_csv: childcare_backend_release_manual_requirements.csv",
                "    schema_artifact_csv: childcare_backend_release_schema_artifact_summary.csv",
                "    schema_columns_csv: childcare_backend_release_schema_column_summary.csv",
                "    release_contract_json: childcare_backend_release_contract.json",
                "    release_artifact_contracts_csv: childcare_backend_release_artifact_contracts.csv",
                "    release_column_dictionary_csv: childcare_backend_release_column_dictionary.csv",
                "    frontend_handoff_summary_json: childcare_backend_release_frontend_handoff_summary.json",
                "    bundle_index_json: childcare_backend_release_bundle_index.json",
                "    bundle_index_csv: childcare_backend_release_bundle_index.csv",
            ]
        ),
        encoding="utf-8",
    )


def test_real_pull_dry_run_is_ok_from_cli():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unpriced.cli",
            "pull-core",
            "--real",
            "--dry-run",
            "--year",
            "2024",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    assert result.returncode == 0
    assert "planned qcew" in result.stdout or "planned qcew" in result.stderr


def test_real_pull_ccdf_dry_run_is_ok_from_cli():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unpriced.cli",
            "pull-ccdf",
            "--real",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    assert result.returncode == 0


def test_pull_ccdf_cli_dispatches_ingest_with_expected_args(project_paths, monkeypatch):
    observed: dict[str, object] = {}

    def _fake_ccdf_ingest(paths, sample=True, refresh=False, dry_run=False, year=None):
        observed["paths"] = paths
        observed["sample"] = sample
        observed["refresh"] = refresh
        observed["dry_run"] = dry_run
        observed["year"] = year
        return None

    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    monkeypatch.setattr(cli.ccdf, "ingest", _fake_ccdf_ingest)

    cli.main(["pull-ccdf", "--real", "--refresh", "--dry-run", "--year", "2024"])

    assert observed["paths"] == project_paths
    assert observed["sample"] is False
    assert observed["refresh"] is True
    assert observed["dry_run"] is True
    assert observed["year"] == 2024


def test_pull_ccdf_cli_sample_writes_extended_outputs(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)

    cli.main(["pull-ccdf", "--sample", "--refresh"])

    output_dir = project_paths.interim / "ccdf"
    parquet_paths = sorted(output_dir.glob("*.parquet"))
    assert output_dir / "ccdf.parquet" in parquet_paths
    assert len(parquet_paths) >= 4


def test_ensure_childcare_segmented_report_artifacts_refresh_rebuilds_existing_outputs(project_paths, monkeypatch):
    report_dir = project_paths.root / "data" / "interim" / "childcare" / "segmented_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "segmented_channel_response_summary.parquet",
        "segmented_state_fallback_summary.parquet",
        "childcare_segmented_headline_summary.json",
    ):
        (report_dir / name).write_text("existing", encoding="utf-8")

    observed: dict[str, object] = {}

    def _fake_build(*args, **kwargs):
        observed["called"] = True
        observed["refresh"] = kwargs["refresh"]

    monkeypatch.setattr(cli, "build_childcare_segmented_report", _fake_build)

    cli._ensure_childcare_segmented_report_artifacts(
        project_paths,
        sample=False,
        refresh=True,
        dry_run=False,
        year=2024,
    )

    assert observed == {"called": True, "refresh": True}


def test_ensure_licensing_harmonization_artifacts_refresh_rebuilds_existing_outputs(project_paths, monkeypatch):
    config_dir = project_paths.root / "configs" / "extensions"
    _write_licensing_iv_config(config_dir)
    namespace = project_paths.root / "data" / "interim" / "childcare" / "licensing_iv"
    namespace.mkdir(parents=True, exist_ok=True)
    for name in (
        "licensing_rules_raw_audit.parquet",
        "licensing_rules_harmonized.parquet",
        "licensing_stringency_index.parquet",
        "licensing_harmonization_summary.parquet",
    ):
        write_parquet(pd.DataFrame([{"a": 1}]), namespace / name)

    observed: dict[str, object] = {}

    def _fake_build(*args, **kwargs):
        observed["called"] = True
        observed["refresh"] = kwargs["refresh"]

    monkeypatch.setattr(cli, "build_licensing_harmonization", _fake_build)

    cli._ensure_licensing_harmonization_artifacts(
        project_paths,
        sample=False,
        refresh=True,
        dry_run=False,
        year=2024,
        config_name_or_path="licensing_iv",
    )

    assert observed == {"called": True, "refresh": True}


def test_ensure_licensing_iv_artifacts_refresh_rebuilds_existing_outputs_and_requires_usability_summary(
    project_paths,
    monkeypatch,
):
    config_dir = project_paths.root / "configs" / "extensions"
    _write_licensing_iv_config(config_dir)
    namespace = project_paths.root / "data" / "interim" / "childcare" / "licensing_iv"
    namespace.mkdir(parents=True, exist_ok=True)
    write_parquet(pd.DataFrame([{"a": 1}]), namespace / "licensing_event_study_results.parquet")
    write_parquet(pd.DataFrame([{"a": 1}]), namespace / "licensing_iv_results.parquet")
    write_parquet(pd.DataFrame([{"a": 1}]), namespace / "licensing_first_stage_diagnostics.parquet")
    write_parquet(pd.DataFrame([{"a": 1}]), namespace / "licensing_treatment_timing.parquet")
    write_parquet(pd.DataFrame([{"a": 1}]), namespace / "licensing_leave_one_state_out.parquet")
    write_json({"status": "ok"}, namespace / "licensing_iv_summary.json")

    observed: dict[str, object] = {}

    def _fake_build(*args, **kwargs):
        observed["called"] = True
        observed["refresh"] = kwargs["refresh"]

    monkeypatch.setattr(cli, "build_licensing_iv", _fake_build)

    cli._ensure_licensing_iv_artifacts(
        project_paths,
        sample=False,
        refresh=False,
        dry_run=False,
        year=2024,
        config_name_or_path="licensing_iv",
    )
    assert observed == {"called": True, "refresh": False}

    observed.clear()
    write_parquet(pd.DataFrame([{"a": 1}]), namespace / "licensing_iv_usability_summary.parquet")
    cli._ensure_licensing_iv_artifacts(
        project_paths,
        sample=False,
        refresh=True,
        dry_run=False,
        year=2024,
        config_name_or_path="licensing_iv",
    )
    assert observed == {"called": True, "refresh": True}


def test_build_ccdf_state_year_writes_outputs_and_provenance(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    config_dir = project_paths.root / "configs" / "extensions"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "ccdf_state_year.yaml").write_text(
        "\n".join(
            [
                "name: ccdf_state_year",
                "mode: state_year_mapping",
                "output_namespace:",
                "  namespace: data/interim/ccdf",
                "  files:",
                "    admin_state_year: ccdf_admin_state_year.parquet",
                "    policy_features_state_year: ccdf_policy_features_state_year.parquet",
                "    policy_controls_state_year: ccdf_policy_controls_state_year.parquet",
                "    policy_controls_coverage: ccdf_policy_controls_coverage.parquet",
                "    policy_promoted_controls_state_year: ccdf_policy_promoted_controls_state_year.parquet",
                "    policy_feature_audit: ccdf_policy_feature_audit.parquet",
                "policy_controls_promotion:",
                "  min_state_year_coverage: 0.75",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_controls_coverage",
        lambda policy_long: pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2023,
                    "ccdf_policy_control_count": 1,
                    "ccdf_policy_control_support_status": "observed_policy_controls",
                    "ccdf_control_copayment_required": "yes",
                }
            ]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_promoted_controls_state_year",
        lambda policy_long, min_state_year_coverage=0.75: pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2023,
                    "ccdf_policy_control_count": 1,
                    "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                    "ccdf_policy_promoted_control_rule": "state_year_coverage_gte_threshold",
                    "ccdf_policy_promoted_min_state_year_coverage": float(min_state_year_coverage),
                    "ccdf_control_copayment_required": "yes",
                }
            ]
        ),
        raising=False,
    )

    cli.main(["build-ccdf-state-year", "--sample", "--refresh"])

    output_dir = project_paths.interim / "ccdf"
    admin_path = output_dir / "ccdf_admin_state_year.parquet"
    policy_features_path = output_dir / "ccdf_policy_features_state_year.parquet"
    policy_controls_path = output_dir / "ccdf_policy_controls_state_year.parquet"
    policy_controls_coverage_path = output_dir / "ccdf_policy_controls_coverage.parquet"
    policy_promoted_controls_path = output_dir / "ccdf_policy_promoted_controls_state_year.parquet"
    audit_path = output_dir / "ccdf_policy_feature_audit.parquet"

    admin = read_parquet(admin_path)
    policy_features = read_parquet(policy_features_path)
    policy_controls = read_parquet(policy_controls_path)
    policy_controls_coverage = read_parquet(policy_controls_coverage_path)
    policy_promoted_controls = read_parquet(policy_promoted_controls_path)
    audit = read_parquet(audit_path)

    assert not admin.empty
    assert not policy_features.empty
    assert not policy_controls.empty
    assert not policy_controls_coverage.empty
    assert not policy_promoted_controls.empty
    assert not audit.empty
    assert read_json(Path(f"{admin_path}.provenance.json"))["config"]["name"] == "ccdf_state_year"
    assert read_json(Path(f"{policy_controls_coverage_path}.provenance.json"))["config"]["name"] == "ccdf_state_year"
    assert read_json(Path(f"{policy_promoted_controls_path}.provenance.json"))["config"]["name"] == "ccdf_state_year"


def test_build_ccdf_state_year_refresh_reingests_existing_long_inputs(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    config_dir = project_paths.root / "configs" / "extensions"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "ccdf_state_year.yaml").write_text(
        "\n".join(
            [
                "name: ccdf_state_year",
                "mode: state_year_mapping",
                "output_namespace:",
                "  namespace: data/interim/ccdf",
                "  files:",
                "    admin_state_year: ccdf_admin_state_year.parquet",
                "    policy_features_state_year: ccdf_policy_features_state_year.parquet",
                "    policy_controls_state_year: ccdf_policy_controls_state_year.parquet",
                "    policy_controls_coverage: ccdf_policy_controls_coverage.parquet",
                "    policy_promoted_controls_state_year: ccdf_policy_promoted_controls_state_year.parquet",
                "    policy_feature_audit: ccdf_policy_feature_audit.parquet",
                "policy_controls_promotion:",
                "  min_state_year_coverage: 0.75",
            ]
        ),
        encoding="utf-8",
    )

    ccdf_dir = project_paths.interim / "ccdf"
    ccdf_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(
        pd.DataFrame(
            [
                {
                    "source_component": "admin",
                    "raw_relpath": "existing-admin.csv",
                    "filename": "existing-admin.csv",
                    "file_format": "csv",
                    "landing_page": "https://example.com",
                    "source_sheet": "__default__",
                    "row_number": 1,
                    "column_name": "children_served_average_monthly",
                    "value_text": "10",
                    "value_numeric": 10.0,
                    "table_year": 2023,
                    "parse_status": "parsed",
                }
            ]
        ),
        ccdf_dir / "ccdf_admin_long.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "source_component": "policies",
                    "raw_relpath": "existing-policy.csv",
                    "filename": "existing-policy.csv",
                    "file_format": "csv",
                    "landing_page": "https://example.com",
                    "source_sheet": "__default__",
                    "row_number": 1,
                    "column_name": "CopayCollect",
                    "value_text": "yes",
                    "value_numeric": pd.NA,
                    "table_year": 2023,
                    "parse_status": "parsed",
                }
            ]
        ),
        ccdf_dir / "ccdf_policy_long.parquet",
    )

    observed: dict[str, object] = {"calls": 0}

    def _fake_ccdf_ingest(paths, sample=True, refresh=False, dry_run=False, year=None):
        observed["calls"] = int(observed["calls"]) + 1
        observed["refresh"] = refresh
        observed["sample"] = sample
        observed["dry_run"] = dry_run
        return types.SimpleNamespace(
            dry_run=dry_run,
            source_name="ccdf",
            normalized_path=ccdf_dir / "ccdf.parquet",
            detail="ok",
            skipped=False,
        )

    monkeypatch.setattr(cli.ccdf, "ingest", _fake_ccdf_ingest)
    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_admin_state_year",
        lambda admin_long: pd.DataFrame([{"state_fips": "06", "year": 2023, "ccdf_children_served": 10.0}]),
    )
    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_features_state_year",
        lambda policy_long: pd.DataFrame([{"state_fips": "06", "year": 2023, "ccdf_policy_copayment_required": "yes"}]),
    )
    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_controls_state_year",
        lambda policy_long: pd.DataFrame([{"state_fips": "06", "year": 2023, "ccdf_control_copayment_required": "yes"}]),
    )
    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_controls_coverage",
        lambda policy_long: pd.DataFrame([{"feature_name": "ccdf_control_copayment_required", "state_year_coverage": 1.0}]),
        raising=False,
    )
    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_promoted_controls_state_year",
        lambda policy_long, min_state_year_coverage=0.75: pd.DataFrame(
            [{"state_fips": "06", "year": 2023, "ccdf_control_copayment_required": "yes"}]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_feature_audit",
        lambda policy_long: pd.DataFrame([{"state_fips": "06", "year": 2023, "feature_name": "copayment_required"}]),
    )

    cli.main(["build-ccdf-state-year", "--sample", "--refresh"])

    assert observed["calls"] == 1
    assert observed["refresh"] is True
    assert observed["sample"] is True


def test_write_release_backend_config_includes_contract_and_bundle_index_files(project_paths) -> None:
    config_dir = project_paths.root / "configs" / "extensions"
    _write_release_backend_config(config_dir)
    text = (config_dir / "release_backend.yaml").read_text(encoding="utf-8")

    assert "childcare_backend_release_contract.json" in text
    assert "childcare_backend_release_artifact_contracts.csv" in text
    assert "childcare_backend_release_column_dictionary.csv" in text
    assert "childcare_backend_release_bundle_index.json" in text
    assert "childcare_backend_release_bundle_index.csv" in text


def test_build_licensing_harmonization_writes_outputs_and_provenance(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    config_dir = project_paths.root / "configs" / "extensions"
    _write_licensing_iv_config(config_dir)

    cli.main(["build-licensing-harmonization", "--sample", "--refresh"])

    output_dir = project_paths.root / "data" / "interim" / "childcare" / "licensing_iv"
    raw_audit_path = output_dir / "licensing_rules_raw_audit.parquet"
    harmonized_path = output_dir / "licensing_rules_harmonized.parquet"
    stringency_path = output_dir / "licensing_stringency_index.parquet"
    summary_path = output_dir / "licensing_harmonization_summary.parquet"

    raw_audit = read_parquet(raw_audit_path)
    harmonized = read_parquet(harmonized_path)
    stringency = read_parquet(stringency_path)
    summary = read_parquet(summary_path)

    assert not raw_audit.empty
    assert not harmonized.empty
    assert not stringency.empty
    assert not summary.empty
    assert read_json(Path(f"{harmonized_path}.provenance.json"))["parameters"]["mode"] == "licensing_harmonization"
    assert read_json(Path(f"{stringency_path}.provenance.json"))["config"]["name"] == "licensing_iv"


def test_build_licensing_harmonization_accepts_rule_level_only_input(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    config_dir = project_paths.root / "configs" / "extensions"
    _write_licensing_iv_config(config_dir)

    raw_dir = project_paths.raw / "licensing"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "state_fips": "01",
                "year": 2022,
                "provider_type": "center",
                "age_group": "infant",
                "rule_family": "max_children_per_staff",
                "rule_name": "center_infant_ratio",
                "rule_value": 4,
                "source_note": "rule-level only",
            }
        ]
    ).to_csv(raw_dir / "licensing_rules_long.csv", index=False)

    cli.main(["build-licensing-harmonization", "--real", "--refresh"])

    output_dir = project_paths.root / "data" / "interim" / "childcare" / "licensing_iv"
    harmonized_path = output_dir / "licensing_rules_harmonized.parquet"
    stringency_path = output_dir / "licensing_stringency_index.parquet"
    provenance = read_json(Path(f"{harmonized_path}.provenance.json"))

    harmonized = read_parquet(harmonized_path)
    stringency = read_parquet(stringency_path)

    assert not harmonized.empty
    assert not stringency.empty
    assert provenance["parameters"]["selected_input_contract"] == "rule_level"
    assert "licensing_rules_long.parquet" in " ".join(provenance["source_files"])


def test_build_licensing_harmonization_accepts_icpsr_rule_level_input(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    config_dir = project_paths.root / "configs" / "extensions"
    _write_licensing_iv_config(config_dir)

    raw_2017_dir = project_paths.raw / "licensing" / "icpsr_2017" / "ICPSR_37700" / "DS0001"
    raw_2020_dir = project_paths.raw / "licensing" / "icpsr_2020" / "ICPSR_38539" / "DS0001"
    raw_2017_dir.mkdir(parents=True, exist_ok=True)
    raw_2020_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "STATE": "CA",
                "C_CHR_ALLSTAFF": 1,
                "F_CHR_ALLSTAFF": 1,
                "LF_CHR_ALLSTAFF": 0,
                "C_FINGERPRT_ALLSTAFF": 1,
                "F_FINGERPRT_ALLSTAFF": 1,
                "LF_FINGERPRT_ALLSTAFF": 0,
                "C_CAN_ALLSTAFF": 1,
                "F_CAN_ALLSTAFF": 1,
                "LF_CAN_ALLSTAFF": 0,
                "C_SX_ALLSTAFF": 1,
                "F_SX_ALLSTAFF": 1,
                "LF_SX_ALLSTAFF": 0,
            }
        ]
    ).to_csv(raw_2017_dir / "37700-0001-Data.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "STATE": "CA",
                "PQ_T_MINAGE": 18,
                "D_R_INFTOD": 1,
                "PS_MX_RATIO_YN": 1,
                "PS_CNTRSIZE": 0,
                "PS_SIZERATIO": 1,
                "PS_SIZEGRPSZ": 1,
                "PS_LGGROUP": 1,
                "PS_GSEXCEED": 0,
                "SPND_CAPACITY": 1,
                "SPND_RATIO": 1,
                "SPND_GROUPSIZE": 0,
            }
        ]
    ).to_csv(raw_2020_dir / "38539-0001-Data.tsv", sep="\t", index=False)

    cli.main(["build-licensing-harmonization", "--real", "--refresh"])

    output_dir = project_paths.root / "data" / "interim" / "childcare" / "licensing_iv"
    harmonized_path = output_dir / "licensing_rules_harmonized.parquet"
    provenance = read_json(Path(f"{harmonized_path}.provenance.json"))
    harmonized = read_parquet(harmonized_path)

    assert not harmonized.empty
    assert provenance["parameters"]["selected_input_contract"] == "rule_level"
    joined_sources = " ".join(provenance["source_files"])
    assert "licensing_rules_long.parquet" in joined_sources
    assert provenance["parameters"]["selected_input_path"].endswith("licensing_rules_long.parquet")


def test_build_licensing_iv_writes_outputs_and_provenance(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    config_dir = project_paths.root / "configs" / "extensions"
    _write_licensing_iv_config(config_dir)

    cli.main(["build-licensing-iv", "--sample", "--refresh"])

    output_dir = project_paths.root / "data" / "interim" / "childcare" / "licensing_iv"
    event_study_path = output_dir / "licensing_event_study_results.parquet"
    iv_results_path = output_dir / "licensing_iv_results.parquet"
    first_stage_path = output_dir / "licensing_first_stage_diagnostics.parquet"
    timing_path = output_dir / "licensing_treatment_timing.parquet"
    leave_one_out_path = output_dir / "licensing_leave_one_state_out.parquet"
    elasticity_panel_path = output_dir / "supply_iv_elasticities.parquet"
    summary_path = output_dir / "licensing_iv_summary.json"

    event_study = read_parquet(event_study_path)
    iv_results = read_parquet(iv_results_path)
    first_stage = read_parquet(first_stage_path)
    timing = read_parquet(timing_path)
    leave_one_out = read_parquet(leave_one_out_path)
    elasticity_panel = read_parquet(elasticity_panel_path)
    summary = read_json(summary_path)

    assert not event_study.empty
    assert len(iv_results) == 2
    assert len(first_stage) == 1
    assert not timing.empty
    assert not leave_one_out.empty
    assert not elasticity_panel.empty
    assert summary["iv_result_row_count"] == len(iv_results)
    assert read_json(Path(f"{summary_path}.provenance.json"))["parameters"]["mode"] == "licensing_iv_backend"
    assert read_json(Path(f"{iv_results_path}.provenance.json"))["config"]["name"] == "licensing_iv"


def test_build_licensing_iv_accepts_rule_level_only_harmonized_inputs(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    config_dir = project_paths.root / "configs" / "extensions"
    _write_licensing_iv_config(config_dir)

    licensing_dir = project_paths.root / "data" / "interim" / "childcare" / "licensing_iv"
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "year": 2022,
                    "provider_type": "center",
                    "age_group": "infant",
                    "rule_family": "max_children_per_staff",
                    "rule_id": "center__infant__max_children_per_staff",
                    "licensing_rule_missing_original": False,
                    "licensing_rule_support_status": "observed_rule",
                },
                {
                    "state_fips": "01",
                    "year": 2023,
                    "provider_type": "center",
                    "age_group": "infant",
                    "rule_family": "max_children_per_staff",
                    "rule_id": "center__infant__max_children_per_staff",
                    "licensing_rule_missing_original": False,
                    "licensing_rule_support_status": "observed_rule",
                },
                {
                    "state_fips": "02",
                    "year": 2022,
                    "provider_type": "center",
                    "age_group": "infant",
                    "rule_family": "max_children_per_staff",
                    "rule_id": "center__infant__max_children_per_staff",
                    "licensing_rule_missing_original": False,
                    "licensing_rule_support_status": "observed_rule",
                },
                {
                    "state_fips": "02",
                    "year": 2023,
                    "provider_type": "center",
                    "age_group": "infant",
                    "rule_family": "max_children_per_staff",
                    "rule_id": "center__infant__max_children_per_staff",
                    "licensing_rule_missing_original": False,
                    "licensing_rule_support_status": "observed_rule",
                },
            ]
        ),
        licensing_dir / "licensing_rules_harmonized.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {"state_fips": "01", "year": 2022, "stringency_equal_weight_index": 1.0},
                {"state_fips": "01", "year": 2023, "stringency_equal_weight_index": 1.2},
                {"state_fips": "02", "year": 2022, "stringency_equal_weight_index": 1.0},
                {"state_fips": "02", "year": 2023, "stringency_equal_weight_index": 1.0},
            ]
        ),
        licensing_dir / "licensing_stringency_index.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {"state_fips": "01", "year": 2022, "harmonized_rule_count": 1},
                {"state_fips": "01", "year": 2023, "harmonized_rule_count": 1},
                {"state_fips": "02", "year": 2022, "harmonized_rule_count": 1},
                {"state_fips": "02", "year": 2023, "harmonized_rule_count": 1},
            ]
        ),
        licensing_dir / "licensing_harmonization_summary.parquet",
    )

    cli.main(["build-licensing-iv", "--sample", "--refresh"])

    output_dir = project_paths.root / "data" / "interim" / "childcare" / "licensing_iv"
    summary_path = output_dir / "licensing_iv_summary.json"
    elasticity_panel_path = output_dir / "supply_iv_elasticities.parquet"
    summary_provenance = read_json(Path(f"{summary_path}.provenance.json"))

    assert summary_path.exists()
    assert elasticity_panel_path.exists()
    assert summary_provenance["parameters"]["elasticity_panel_source"] in {
        "licensing_shock_panel",
        "stringency_index_fallback",
    }


def test_build_childcare_release_backend_writes_outputs_and_provenance(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    config_dir = project_paths.root / "configs" / "extensions"
    _write_licensing_iv_config(config_dir)
    _write_release_backend_config(config_dir)

    ccdf_dir = project_paths.root / "data" / "interim" / "ccdf"
    segmented_reports_dir = project_paths.root / "data" / "interim" / "childcare" / "segmented_reports"
    segmented_scenarios_dir = project_paths.root / "data" / "interim" / "childcare" / "segmented_scenarios"
    licensing_dir = project_paths.root / "data" / "interim" / "childcare" / "licensing_iv"

    write_parquet(
        pd.DataFrame(
            [
                {"state_fips": "01", "year": 2022, "ccdf_admin_support_status": "explicit_split_support"},
                {"state_fips": "02", "year": 2023, "ccdf_admin_support_status": "proxy_or_fallback_support"},
            ]
        ),
        ccdf_dir / "ccdf_admin_state_year.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "control_name": "copayment_required",
                    "control_kind": "binary",
                    "observed_state_year_rows": 2,
                    "missing_state_year_rows": 0,
                    "state_year_coverage_rate": 1.0,
                    "coverage_support_status": "observed",
                    "promoted_state_year_rows": 2,
                }
            ]
        ),
        ccdf_dir / "ccdf_policy_controls_coverage.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "year": 2022,
                    "ccdf_control_copayment_required": 1,
                    "ccdf_policy_control_support_status": "observed",
                }
            ]
        ),
        ccdf_dir / "ccdf_policy_promoted_controls_state_year.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "year": 2022,
                    "support_quality_tier": "explicit_split_support",
                    "ccdf_support_flag": "observed",
                    "ccdf_admin_support_status": "explicit_split_support",
                    "q0_support_flag": "observed",
                    "public_program_support_status": "observed",
                    "promoted_control_observed": 1,
                    "proxy_ccdf_row_count": 0,
                    "any_segment_allocation_fallback": False,
                    "any_private_allocation_fallback": False,
                }
            ]
        ),
        segmented_reports_dir / "segmented_channel_response_summary.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "year": 2022,
                    "support_quality_tier": "explicit_split_support",
                    "ccdf_support_flag": "observed",
                    "ccdf_admin_support_status": "explicit_split_support",
                    "q0_support_flag": "observed",
                    "public_program_support_status": "observed",
                    "promoted_control_observed": 1,
                    "proxy_ccdf_row_count": 0,
                    "any_segment_allocation_fallback": False,
                    "any_private_allocation_fallback": False,
                }
            ]
        ),
        segmented_reports_dir / "segmented_state_fallback_summary.parquet",
    )
    write_json(
        {"state_year_count": 1, "support_quality_tiers": ["explicit_split_support"]},
        segmented_reports_dir / "childcare_segmented_headline_summary.json",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "year": 2022,
                    "solver_channel": "private_unsubsidized",
                    "alpha": 0.5,
                    "market_quantity_proxy": 100.0,
                    "unpaid_quantity_proxy": 10.0,
                    "p_baseline": 100.0,
                    "p_shadow_marginal": 110.0,
                    "p_alpha": 105.0,
                }
            ]
        ),
        segmented_scenarios_dir / "segmented_channel_scenarios.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "source_row_number": 1,
                    "raw_column_name": "center_infant_ratio",
                    "value_text": "4",
                    "value_numeric": 4.0,
                    "value_kind": "numeric",
                    "source_structure_status": "observed_raw_column",
                }
            ]
        ),
        licensing_dir / "licensing_rules_raw_audit.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "year": 2022,
                    "rule_id": "center__infant__max_children_per_staff",
                    "rule_family": "max_children_per_staff",
                    "licensing_rule_support_status": "observed_rule",
                }
            ]
        ),
        licensing_dir / "licensing_rules_harmonized.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "year": 2022,
                    "stringency_equal_weight_index": 0.5,
                    "stringency_pca_like_index": 0.5,
                }
            ]
        ),
        licensing_dir / "licensing_stringency_index.parquet",
    )
    write_parquet(
        pd.DataFrame([{"outcome": "log_provider_density", "event_time": 0, "estimate": 0.2, "status": "ok"}]),
        licensing_dir / "licensing_event_study_results.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {"outcome": "provider_density", "estimate": 0.3, "status": "ok"},
                {"outcome": "employer_establishment_density", "estimate": 0.1, "status": "ok"},
            ]
        ),
        licensing_dir / "licensing_iv_results.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "outcome": "provider_density",
                    "recommended_use_tier": "headline",
                    "usable_for_headline": True,
                    "first_stage_strength_tier": "strong",
                }
            ]
        ),
        licensing_dir / "licensing_iv_usability_summary.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "first_stage_strength_flag": "strong",
                    "first_stage_strength_tier": "strong",
                    "recommended_use_tier": "headline",
                    "usable_for_headline": True,
                    "first_stage_f_stat": 12.0,
                }
            ]
        ),
        licensing_dir / "licensing_first_stage_diagnostics.parquet",
    )
    write_parquet(
        pd.DataFrame([{"state_fips": "01", "treatment_start_year": 2022, "ever_treated": True}]),
        licensing_dir / "licensing_treatment_timing.parquet",
    )
    write_parquet(
        pd.DataFrame([{"omitted_state_fips": "baseline", "status": "ok"}]),
        licensing_dir / "licensing_leave_one_state_out.parquet",
    )
    write_json(
        {"status": "ok", "iv_result_row_count": 2, "first_stage_row_count": 1},
        licensing_dir / "licensing_iv_summary.json",
    )
    write_json(
        {
            "selected_headline_sample": "mvp",
            "mode": "sample",
            "n_obs": 10,
            "n_states": 2,
            "year_min": 2022,
            "year_max": 2023,
        },
        project_paths.outputs_reports / "childcare_headline_summary.json",
    )

    cli.main(["build-childcare-release-backend", "--sample"])

    headline_json_path = project_paths.outputs_reports / "childcare_backend_release_headline_summary.json"
    methods_json_path = project_paths.outputs_reports / "childcare_backend_release_methods_summary.json"
    methods_md_path = project_paths.outputs_reports / "childcare_backend_release_methods.md"
    rebuild_md_path = project_paths.outputs_reports / "childcare_backend_release_rebuild.md"
    manifest_json_path = project_paths.outputs_reports / "childcare_backend_release_manifest.json"
    source_readiness_json_path = project_paths.outputs_reports / "childcare_backend_release_source_readiness.json"
    manual_requirements_json_path = project_paths.outputs_reports / "childcare_backend_release_manual_requirements.json"
    manual_requirements_md_path = project_paths.outputs_reports / "childcare_backend_release_manual_requirements.md"
    schema_inventory_json_path = project_paths.outputs_reports / "childcare_backend_release_schema_inventory.json"
    support_quality_csv_path = project_paths.outputs_tables / "childcare_backend_support_quality_summary.csv"
    ccdf_support_mix_csv_path = project_paths.outputs_tables / "childcare_backend_ccdf_support_mix_summary.csv"
    policy_coverage_csv_path = project_paths.outputs_tables / "childcare_backend_policy_coverage_summary.csv"
    ccdf_proxy_gap_summary_csv_path = project_paths.outputs_tables / "childcare_backend_ccdf_proxy_gap_summary.csv"
    ccdf_proxy_gap_state_years_csv_path = project_paths.outputs_tables / "childcare_backend_ccdf_proxy_gap_state_years.csv"
    segmented_comparison_csv_path = project_paths.outputs_tables / "childcare_backend_segmented_comparison.csv"
    iv_results_csv_path = project_paths.outputs_tables / "childcare_backend_licensing_iv_results.csv"
    iv_usability_csv_path = project_paths.outputs_tables / "childcare_backend_licensing_iv_usability_summary.csv"
    manifest_csv_path = project_paths.outputs_tables / "childcare_backend_release_manifest.csv"
    source_readiness_csv_path = project_paths.outputs_tables / "childcare_backend_release_source_readiness.csv"
    manual_requirements_csv_path = project_paths.outputs_tables / "childcare_backend_release_manual_requirements.csv"
    schema_artifact_csv_path = project_paths.outputs_tables / "childcare_backend_release_schema_artifact_summary.csv"
    schema_columns_csv_path = project_paths.outputs_tables / "childcare_backend_release_schema_column_summary.csv"
    frontend_handoff_json_path = project_paths.outputs_reports / "childcare_backend_release_frontend_handoff_summary.json"
    bundle_index_json_path = project_paths.outputs_reports / "childcare_backend_release_bundle_index.json"
    bundle_index_csv_path = project_paths.outputs_tables / "childcare_backend_release_bundle_index.csv"

    headline = read_json(headline_json_path)
    methods = read_json(methods_json_path)
    manifest = read_json(manifest_json_path)
    source_readiness = read_json(source_readiness_json_path)
    manual_requirements = read_json(manual_requirements_json_path)
    schema_inventory = read_json(schema_inventory_json_path)
    frontend_handoff = read_json(frontend_handoff_json_path)
    bundle_index = read_json(bundle_index_json_path)

    assert headline["release_name"] == "childcare_backend"
    assert methods["backend_only"] is True
    assert methods_md_path.exists()
    assert rebuild_md_path.exists()
    assert "rows" in manifest
    assert source_readiness_json_path.exists()
    assert manual_requirements_json_path.exists()
    assert manual_requirements_md_path.exists()
    assert schema_inventory_json_path.exists()
    assert source_readiness["summary"]["missing_required_count"] == 0
    assert source_readiness["summary"]["missing_count"] >= 1
    assert source_readiness["summary"]["status"] == "real_release_blocked"
    assert len(source_readiness["rows"]) >= 4
    assert manual_requirements["summary"]["manual_action_count"] >= 1
    assert manual_requirements["summary"]["status"] == "real_release_blocked"
    assert len(manual_requirements["rows"]) >= 1
    assert "artifact_summary" in schema_inventory
    assert "column_schema" in schema_inventory
    assert support_quality_csv_path.exists()
    assert ccdf_support_mix_csv_path.exists()
    assert policy_coverage_csv_path.exists()
    assert ccdf_proxy_gap_summary_csv_path.exists()
    assert ccdf_proxy_gap_state_years_csv_path.exists()
    assert segmented_comparison_csv_path.exists()
    assert iv_results_csv_path.exists()
    assert iv_usability_csv_path.exists()
    assert manifest_csv_path.exists()
    assert source_readiness_csv_path.exists()
    assert manual_requirements_csv_path.exists()
    assert schema_artifact_csv_path.exists()
    assert schema_columns_csv_path.exists()
    assert frontend_handoff_json_path.exists()
    assert bundle_index_json_path.exists()
    assert bundle_index_csv_path.exists()
    assert frontend_handoff["canonical_backend"] is True
    assert frontend_handoff["segmented_outputs_additive_only"] is True
    assert frontend_handoff["publication_rules"][0]["topic"] == "ccdf_headline_decomposition"
    assert frontend_handoff["publication_rules"][0]["diagnostic_only_support_buckets"] == [
        "retained_proxy",
        "downgraded_proxy",
    ]
    assert len(bundle_index["artifacts"]) >= 10
    assert read_json(Path(f"{headline_json_path}.provenance.json"))["config"]["name"] == "release_backend"
    assert read_json(Path(f"{manifest_csv_path}.provenance.json"))["parameters"]["mode"] == "backend_release"


def test_build_childcare_segments_writes_outputs_and_provenance(project_paths):
    ingest_ndcp(project_paths, sample=True)
    config_dir = project_paths.root / "configs" / "extensions"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "segmented_solver.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: segmented_solver",
                "mode: segmented_baseline",
                "segments:",
                "  dimensions: [child_age, provider_type, channel]",
                "  child_age: [infant, toddler, preschool]",
                "  provider_type: [center, home]",
                "  channel: [private]",
                "build:",
                "  compatibility:",
                "    enabled: true",
                "    tolerance: 1.0e-9",
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

    cli.build_childcare_segments(project_paths, sample=True, config_name_or_path="segmented_solver")

    output_dir = project_paths.root / "data" / "interim" / "childcare" / "segmented_solver"
    definitions_path = output_dir / "segment_definitions.parquet"
    prices_path = output_dir / "ndcp_segment_prices.parquet"
    mapping_path = output_dir / "segmented_to_pooled_mapping.parquet"

    definitions = read_parquet(definitions_path)
    prices = read_parquet(prices_path)
    mapping = read_parquet(mapping_path)

    assert definitions["segment_id"].nunique() == 6
    assert not prices.empty
    assert not mapping.empty
    assert float(mapping["pooled_price_gap"].abs().max()) <= 1.0e-9
    assert read_json(Path(f"{prices_path}.provenance.json"))["config"]["name"] == "segmented_solver"


def test_build_childcare_utilization_writes_outputs_and_provenance(project_paths, monkeypatch):
    config_dir = project_paths.root / "configs" / "extensions"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "segmented_solver.yaml").write_text(
        "\n".join(
            [
                "name: segmented_solver",
                "mode: segmented_baseline",
                "segments:",
                "  - segment_id: infant_center_private",
                "    child_age: infant",
                "    provider_type: center",
                "    channel: private",
                "  - segment_id: preschool_home_private",
                "    child_age: preschool",
                "    provider_type: home",
                "    channel: private",
                "build:",
                "  compatibility:",
                "    enabled: true",
                "    tolerance: 1.0e-9",
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
    (config_dir / "utilization_stack.yaml").write_text(
        "\n".join(
            [
                "name: utilization_stack",
                "mode: observed_utilization",
                "year_window:",
                "  start: 2021",
                "  end: 2023",
                "output_namespace:",
                "  namespace: data/interim/childcare/utilization_stack",
                "  files:",
                "    public_program_slots: public_program_slots_state_year.parquet",
                "    survey_paid_use_targets: survey_paid_use_targets.parquet",
                "    quantity_by_segment: q0_segmented.parquet",
                "    reconciliation_diagnostics: utilization_reconciliation_diagnostics.parquet",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "ccdf_state_year.yaml").write_text(
        "\n".join(
            [
                "name: ccdf_state_year",
                "mode: state_year_mapping",
                "output_namespace:",
                "  namespace: data/interim/ccdf",
                "  files:",
                "    admin_state_year: ccdf_admin_state_year.parquet",
                "    policy_features_state_year: ccdf_policy_features_state_year.parquet",
                "    policy_controls_state_year: ccdf_policy_controls_state_year.parquet",
                "    policy_controls_coverage: ccdf_policy_controls_coverage.parquet",
                "    policy_promoted_controls_state_year: ccdf_policy_promoted_controls_state_year.parquet",
                "    policy_feature_audit: ccdf_policy_feature_audit.parquet",
                "policy_controls_promotion:",
                "  min_state_year_coverage: 0.75",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_controls_coverage",
        lambda policy_long: pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2021,
                    "ccdf_policy_control_count": 1,
                    "ccdf_policy_control_support_status": "observed_policy_controls",
                    "ccdf_control_copayment_required": "yes",
                }
            ]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_promoted_controls_state_year",
        lambda policy_long, min_state_year_coverage=0.75: pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2021,
                    "ccdf_policy_control_count": 1,
                    "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                    "ccdf_policy_promoted_control_rule": "state_year_coverage_gte_threshold",
                    "ccdf_policy_promoted_min_state_year_coverage": float(min_state_year_coverage),
                    "ccdf_control_copayment_required": "yes",
                }
            ]
        ),
        raising=False,
    )

    cli.build_childcare_utilization(project_paths, sample=True, config_name_or_path="utilization_stack")

    output_dir = project_paths.root / "data" / "interim" / "childcare" / "utilization_stack"
    public_path = output_dir / "public_program_slots_state_year.parquet"
    survey_path = output_dir / "survey_paid_use_targets.parquet"
    quantity_path = output_dir / "q0_segmented.parquet"
    diagnostics_path = output_dir / "utilization_reconciliation_diagnostics.parquet"
    promoted_controls_path = project_paths.interim / "ccdf" / "ccdf_policy_promoted_controls_state_year.parquet"

    public_programs = read_parquet(public_path)
    survey_targets = read_parquet(survey_path)
    quantities = read_parquet(quantity_path)
    diagnostics = read_parquet(diagnostics_path)
    promoted_controls = read_parquet(promoted_controls_path)

    assert not public_programs.empty
    assert not survey_targets.empty
    assert not quantities.empty
    assert not diagnostics.empty
    assert not promoted_controls.empty
    assert "public_head_start" in set(quantities["segment_id"])
    assert "ccdf_policy_control_count" in diagnostics.columns
    assert "ccdf_control_copayment_required" in diagnostics.columns
    assert "ccdf_policy_feature_count" in diagnostics.columns
    assert read_json(Path(f"{quantity_path}.provenance.json"))["config"]["name"] == "utilization_stack"
    assert read_json(Path(f"{promoted_controls_path}.provenance.json"))["config"]["name"] == "ccdf_state_year"


def test_build_childcare_solver_inputs_writes_outputs_and_provenance(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    _write_childcare_additive_extension_configs(project_paths)
    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_controls_coverage",
        lambda policy_long: pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2021,
                    "ccdf_policy_control_count": 1,
                    "ccdf_policy_control_support_status": "observed_policy_controls",
                    "ccdf_control_copayment_required": "yes",
                }
            ]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_promoted_controls_state_year",
        lambda policy_long, min_state_year_coverage=0.75: pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2021,
                    "ccdf_policy_control_count": 1,
                    "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                    "ccdf_policy_promoted_control_rule": "state_year_coverage_gte_threshold",
                    "ccdf_policy_promoted_min_state_year_coverage": float(min_state_year_coverage),
                    "ccdf_control_copayment_required": "yes",
                }
            ]
        ),
        raising=False,
    )

    cli.build_childcare_utilization(project_paths, sample=True, config_name_or_path="utilization_stack")
    observed = _install_fake_childcare_solver_inputs_module(monkeypatch)

    cli.main(["build-childcare-solver-inputs", "--sample"])

    output_dir = project_paths.root / "data" / "interim" / "childcare" / "solver_inputs"
    channel_path = output_dir / "solver_channel_quantities.parquet"
    baseline_path = output_dir / "solver_baseline_state_year.parquet"
    elasticity_path = output_dir / "solver_elasticity_mapping.parquet"
    controls_path = output_dir / "solver_policy_controls_state_year.parquet"

    channel = read_parquet(channel_path)
    baseline = read_parquet(baseline_path)
    elasticity = read_parquet(elasticity_path)
    controls = read_parquet(controls_path)

    assert not channel.empty
    assert not baseline.empty
    assert not elasticity.empty
    assert not controls.empty
    assert set(channel["solver_channel"]) == {"private_unsubsidized", "private_subsidized", "public_admin"}
    assert channel.loc[channel["solver_channel"] == "public_admin", "price_responsive"].eq(False).all()
    assert baseline["total_paid_slots"].eq(
        baseline["private_unsubsidized"] + baseline["private_subsidized"] + baseline["public_admin"]
    ).all()
    assert elasticity["active_in_price_solver"].tolist() == [True, True, False]
    assert read_json(Path(f"{channel_path}.provenance.json"))["config"]["name"] == "solver_inputs"
    assert observed["kwargs"]["q0_segmented"].empty is False
    assert observed["kwargs"]["ndcp_segment_prices"].empty is False
    assert observed["kwargs"]["ccdf_policy_controls_state_year"].empty is False


def test_build_childcare_report_tables_writes_outputs_and_provenance(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    _write_childcare_additive_extension_configs(project_paths)
    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_controls_coverage",
        lambda policy_long: pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2021,
                    "ccdf_policy_control_count": 1,
                    "ccdf_policy_control_support_status": "observed_policy_controls",
                    "ccdf_control_copayment_required": "yes",
                }
            ]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        childcare_ccdf,
        "build_ccdf_policy_promoted_controls_state_year",
        lambda policy_long, min_state_year_coverage=0.75: pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2021,
                    "ccdf_policy_control_count": 1,
                    "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                    "ccdf_policy_promoted_control_rule": "state_year_coverage_gte_threshold",
                    "ccdf_policy_promoted_min_state_year_coverage": float(min_state_year_coverage),
                    "ccdf_control_copayment_required": "yes",
                }
            ]
        ),
        raising=False,
    )

    cli.build_childcare_utilization(project_paths, sample=True, config_name_or_path="utilization_stack")
    observed = _install_fake_childcare_report_tables_module(monkeypatch)

    cli.main(["build-childcare-report-tables", "--sample"])

    output_dir = project_paths.root / "data" / "interim" / "childcare" / "report_tables"
    channel_path = output_dir / "state_year_channel_summary.parquet"
    policy_path = output_dir / "state_year_policy_quantity_summary.parquet"
    support_path = output_dir / "state_year_support_summary.parquet"

    channel = read_parquet(channel_path)
    policy = read_parquet(policy_path)
    support = read_parquet(support_path)

    assert not channel.empty
    assert not policy.empty
    assert not support.empty
    totals = channel.groupby(["state_fips", "year"])["channel_share_of_total_paid_care"].sum()
    assert totals.round(10).eq(1.0).all()
    assert "ccdf_policy_control_support_status" in policy.columns
    assert "ccdf_support_flag" in support.columns
    assert "ccdf_admin_support_status" in support.columns
    assert read_json(Path(f"{support_path}.provenance.json"))["config"]["name"] == "report_tables"
    assert observed["kwargs"]["q0_segmented"].empty is False
    assert observed["kwargs"]["utilization_diagnostics"].empty is False
    assert observed["kwargs"]["ccdf_policy_controls_state_year"].empty is False


def test_simulate_childcare_segmented_writes_outputs_and_provenance(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    _write_childcare_simulation_inputs(project_paths)
    _write_childcare_segmented_scenario_inputs(project_paths)
    _write_childcare_additive_extension_configs(project_paths)
    pooled_path = project_paths.processed / "childcare_marketization_scenarios.parquet"
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2021,
                    "demand_sample_name": "pooled_sentinel",
                }
            ]
        ),
        pooled_path,
    )
    observed = _install_fake_childcare_segmented_scenarios_module(monkeypatch)

    cli.main(["simulate-childcare-segmented", "--sample"])

    output_dir = project_paths.root / "data" / "interim" / "childcare" / "segmented_scenarios"
    channel_inputs_path = output_dir / "segmented_channel_inputs.parquet"
    channel_scenarios_path = output_dir / "segmented_channel_scenarios.parquet"
    summary_path = output_dir / "segmented_state_year_summary.parquet"
    diagnostics_path = output_dir / "segmented_state_year_diagnostics.parquet"

    channel_inputs = read_parquet(channel_inputs_path)
    channel_scenarios = read_parquet(channel_scenarios_path)
    summary = read_parquet(summary_path)
    diagnostics = read_parquet(diagnostics_path)

    assert not channel_inputs.empty
    assert not channel_scenarios.empty
    assert not summary.empty
    assert not diagnostics.empty
    assert read_json(Path(f"{channel_inputs_path}.provenance.json"))["config"]["name"] == "segmented_scenarios"
    assert read_json(Path(f"{channel_scenarios_path}.provenance.json"))["config"]["name"] == "segmented_scenarios"
    assert observed["kwargs"]["demand_summary"]["specification_profile"] == "household_parsimonious"
    assert observed["kwargs"]["state_frame"].empty is False
    pooled_after = read_parquet(pooled_path)
    assert pooled_after.equals(
        pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2021,
                    "demand_sample_name": "pooled_sentinel",
                }
            ]
        )
    )


def test_build_childcare_segmented_report_writes_outputs_and_preserves_pooled_artifacts(
    project_paths, monkeypatch
):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    _write_childcare_additive_extension_configs(project_paths)

    pooled_scenario_path = project_paths.processed / "childcare_marketization_scenarios.parquet"
    pooled_report_path = project_paths.outputs_reports / "childcare_mvp_report.md"
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2021,
                    "demand_sample_name": "pooled_sentinel",
                }
            ]
        ),
        pooled_scenario_path,
    )
    pooled_report_path.write_text("pooled sentinel", encoding="utf-8")

    observed: dict[str, bool] = {}

    def fake_segmented_scenarios(
        paths,
        sample=True,
        refresh=False,
        dry_run=False,
        year=None,
        config_name_or_path="segmented_scenarios",
    ):
        observed["segmented_scenarios_called"] = True
        segmented_dir = paths.root / "data" / "interim" / "childcare" / "segmented_scenarios"
        segmented_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(
            pd.DataFrame(
                [
                    {
                        "state_fips": "06",
                        "year": 2021,
                        "solver_channel": "private_unsubsidized",
                        "alpha": 0.50,
                        "state_price_index": 100.0,
                        "market_quantity_proxy": 60.0,
                        "unpaid_quantity_proxy": 20.0,
                        "p_baseline": 100.0,
                        "p_shadow_marginal": 108.0,
                        "p_alpha": 102.0,
                        "price_responsive": True,
                        "state_price_observation_status": "observed_ndcp_support",
                        "state_price_nowcast": False,
                        "state_price_support_window": "in_support",
                        "state_ndcp_imputed_share": 0.1,
                        "is_sensitivity_year": False,
                        "demand_sample_name": "observed_core",
                        "demand_specification_profile": "household_parsimonious",
                        "ccdf_policy_control_count": 1,
                        "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                        "ccdf_control_copayment_required": "yes",
                    },
                    {
                        "state_fips": "06",
                        "year": 2021,
                        "solver_channel": "private_subsidized",
                        "alpha": 0.50,
                        "state_price_index": 100.0,
                        "market_quantity_proxy": 30.0,
                        "unpaid_quantity_proxy": 10.0,
                        "p_baseline": 100.0,
                        "p_shadow_marginal": 108.0,
                        "p_alpha": 101.0,
                        "price_responsive": True,
                        "state_price_observation_status": "observed_ndcp_support",
                        "state_price_nowcast": False,
                        "state_price_support_window": "in_support",
                        "state_ndcp_imputed_share": 0.1,
                        "is_sensitivity_year": False,
                        "demand_sample_name": "observed_core",
                        "demand_specification_profile": "household_parsimonious",
                        "ccdf_policy_control_count": 1,
                        "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                        "ccdf_control_copayment_required": "yes",
                    },
                    {
                        "state_fips": "06",
                        "year": 2021,
                        "solver_channel": "public_admin",
                        "alpha": 0.50,
                        "state_price_index": 100.0,
                        "market_quantity_proxy": 20.0,
                        "unpaid_quantity_proxy": 0.0,
                        "p_baseline": 100.0,
                        "p_shadow_marginal": 100.0,
                        "p_alpha": 100.0,
                        "price_responsive": False,
                        "state_price_observation_status": "observed_ndcp_support",
                        "state_price_nowcast": False,
                        "state_price_support_window": "in_support",
                        "state_ndcp_imputed_share": 0.1,
                        "is_sensitivity_year": False,
                        "demand_sample_name": "observed_core",
                        "demand_specification_profile": "household_parsimonious",
                        "ccdf_policy_control_count": 1,
                        "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                        "ccdf_control_copayment_required": "yes",
                    },
                ]
            ),
            segmented_dir / "segmented_channel_scenarios.parquet",
        )
        write_parquet(
            pd.DataFrame(
                [
                    {
                        "state_fips": "06",
                        "year": 2021,
                        "alpha": 0.50,
                        "market_quantity_total": 110.0,
                        "unpaid_quantity_total": 30.0,
                        "private_market_quantity_total": 90.0,
                        "public_market_quantity_total": 20.0,
                        "private_unpaid_quantity_total": 30.0,
                        "public_unpaid_quantity_total": 0.0,
                        "quantity_weighted_p_baseline": 100.0,
                        "quantity_weighted_p_shadow_marginal": 104.0,
                        "quantity_weighted_p_alpha": 101.0,
                    }
                ]
            ),
            segmented_dir / "segmented_state_year_summary.parquet",
        )
        write_parquet(
            pd.DataFrame(
                [
                    {
                        "state_fips": "06",
                        "year": 2021,
                        "scenario_rows": 3,
                        "private_rows": 2,
                        "public_rows": 1,
                        "public_admin_invariant_prices": True,
                    }
                ]
            ),
            segmented_dir / "segmented_state_year_diagnostics.parquet",
        )

    def fake_report_tables(
        paths,
        sample=True,
        refresh=False,
        dry_run=False,
        year=None,
        config_name_or_path="report_tables",
    ):
        observed["report_tables_called"] = True
        report_dir = paths.root / "data" / "interim" / "childcare" / "report_tables"
        report_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(
            pd.DataFrame(
                [
                    {
                        "state_fips": "06",
                        "year": 2021,
                        "quantity_slots_total": 110.0,
                        "total_paid_slots_target": 110.0,
                        "reconciled_paid_slots": 110.0,
                            "accounting_gap_from_target_max_abs": 0.0,
                            "any_segment_allocation_fallback": True,
                            "ccdf_support_flag": "explicit",
                            "ccdf_admin_support_status": "explicit_admin_support",
                            "q0_support_flag": "explicit_q0_support",
                        "public_program_support_status": "supported",
                        "explicit_ccdf_row_count": 1,
                        "inferred_ccdf_row_count": 0,
                        "proxy_ccdf_row_count": 0,
                        "missing_ccdf_row_count": 0,
                        "any_private_allocation_fallback": False,
                        "any_negative_quantity": False,
                        "ccdf_policy_control_count": 1,
                        "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                        "ccdf_control_copayment_required": "yes",
                    }
                ]
            ),
            report_dir / "state_year_support_summary.parquet",
        )

    monkeypatch.setattr(cli, "build_childcare_segmented_scenarios", fake_segmented_scenarios)
    monkeypatch.setattr(cli, "build_childcare_report_tables", fake_report_tables)

    cli.main(["build-childcare-segmented-report", "--sample"])

    output_dir = project_paths.root / "data" / "interim" / "childcare" / "segmented_reports"
    segmented_dir = project_paths.root / "data" / "interim" / "childcare" / "segmented_scenarios"
    response_path = output_dir / "segmented_channel_response_summary.parquet"
    fallback_path = output_dir / "segmented_state_fallback_summary.parquet"
    headline_path = output_dir / "childcare_segmented_headline_summary.json"
    report_path = output_dir / "childcare_segmented_report.md"

    response = read_parquet(response_path)
    fallback = read_parquet(fallback_path)
    headline = read_json(headline_path)
    report = report_path.read_text(encoding="utf-8")

    assert observed["segmented_scenarios_called"] is True
    assert observed["report_tables_called"] is True
    assert not response.empty
    assert not fallback.empty
    assert headline["headline_alpha"] == 0.5
    assert headline["state_year_count"] == 1
    assert headline["public_admin_invariant_prices"] is True
    assert headline["fallback_state_year_count"] == 1
    assert headline["fallback_state_count"] == 1
    assert "# segmented_childcare_report" in report
    assert "headline_alpha: 0.500000" in report
    response_provenance = read_json(Path(f"{response_path}.provenance.json"))
    assert response_provenance["config"]["name"] == "segmented_reports"
    assert str(segmented_dir / "segmented_state_year_diagnostics.parquet") in response_provenance["source_files"]
    assert read_json(Path(f"{fallback_path}.provenance.json"))["config"]["name"] == "segmented_reports"
    assert read_parquet(pooled_scenario_path).equals(
        pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2021,
                    "demand_sample_name": "pooled_sentinel",
                }
            ]
        )
    )
    assert pooled_report_path.read_text(encoding="utf-8") == "pooled sentinel"


def test_report_childcare_segmented_publishes_outputs_and_preserves_pooled_report(project_paths, monkeypatch):
    monkeypatch.setattr(cli, "load_project_paths", lambda _root: project_paths)
    _write_childcare_additive_extension_configs(project_paths)

    pooled_report_path = project_paths.outputs_reports / "childcare_mvp_report.md"
    pooled_report_path.write_text("pooled sentinel", encoding="utf-8")
    ccdf_dir = project_paths.root / "data" / "interim" / "ccdf"
    ccdf_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(
        pd.DataFrame(
            [
                {
                    "source_component": "admin",
                    "raw_relpath": "data/raw/ccdf/admin/fy2023_table_1.xlsx",
                    "filename": "fy2023_table_1.xlsx",
                    "source_sheet": "Table 1",
                    "file_format": "xlsx",
                    "parse_status": "parsed",
                    "output_table": "ccdf_admin_long",
                    "row_count": 4,
                    "parsed_row_count": 4,
                    "table_year": 2023,
                    "parse_detail": "",
                }
            ]
        ),
        ccdf_dir / "ccdf_parse_inventory.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "filename": "fy2023_table_1.xlsx",
                    "source_sheet": "Table 1",
                    "column_name": "state",
                },
                {
                    "filename": "fy2023_table_1.xlsx",
                    "source_sheet": "Table 1",
                    "column_name": "children_served_average_monthly",
                },
            ]
        ),
        ccdf_dir / "ccdf_admin_long.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "06",
                    "year": 2023,
                    "ccdf_expenditures_support_status": "missing_metric",
                }
            ]
        ),
        ccdf_dir / "ccdf_admin_state_year.parquet",
    )

    observed: dict[str, bool] = {}

    def fake_segmented_report(
        paths,
        sample=True,
        refresh=False,
        dry_run=False,
        year=None,
        config_name_or_path="segmented_reports",
    ):
        observed["segmented_report_called"] = True
        segmented_dir = paths.root / "data" / "interim" / "childcare" / "segmented_reports"
        segmented_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(
            pd.DataFrame(
                [
                    {
                        "state_fips": "06",
                        "year": 2021,
                        "solver_channel": "private_unsubsidized",
                        "alpha": 0.50,
                        "market_quantity_proxy": 60.0,
                        "unpaid_quantity_proxy": 20.0,
                        "price_responsive": True,
                        "p_baseline": 100.0,
                        "p_shadow_marginal": 108.0,
                        "p_alpha": 102.0,
                        "p_shadow_delta_vs_baseline": 8.0,
                        "p_shadow_pct_change_vs_baseline": 0.08,
                        "p_alpha_delta_vs_baseline": 2.0,
                        "p_alpha_pct_change_vs_baseline": 0.02,
                        "public_admin_price_invariant": False,
                    },
                    {
                        "state_fips": "06",
                        "year": 2021,
                        "solver_channel": "public_admin",
                        "alpha": 0.50,
                        "market_quantity_proxy": 20.0,
                        "unpaid_quantity_proxy": 0.0,
                        "price_responsive": False,
                        "p_baseline": 100.0,
                        "p_shadow_marginal": 100.0,
                        "p_alpha": 100.0,
                        "p_shadow_delta_vs_baseline": 0.0,
                        "p_shadow_pct_change_vs_baseline": 0.0,
                        "p_alpha_delta_vs_baseline": 0.0,
                        "p_alpha_pct_change_vs_baseline": 0.0,
                        "public_admin_price_invariant": True,
                    },
                ]
            ),
            segmented_dir / "segmented_channel_response_summary.parquet",
        )
        write_parquet(
            pd.DataFrame(
                [
                    {
                        "state_fips": "06",
                        "year": 2021,
                        "headline_alpha": 0.50,
                        "has_support_row": True,
                        "any_segment_allocation_fallback": False,
                        "any_private_allocation_fallback": False,
                        "ccdf_support_flag": "explicit",
                        "ccdf_admin_support_status": "explicit_admin_support",
                        "q0_support_flag": "explicit_q0_support",
                        "public_program_support_status": "supported",
                        "promoted_control_observed": True,
                        "proxy_ccdf_row_count": 0,
                    }
                ]
            ),
            segmented_dir / "segmented_state_fallback_summary.parquet",
        )
        write_json(
            {
                "headline_alpha": 0.5,
                "state_year_count": 1,
                "fallback_state_year_count": 0,
                "fallback_state_count": 0,
                "promoted_control_observed_state_year_count": 1,
                "proxy_ccdf_state_year_count": 0,
                "public_admin_invariant_prices": True,
                "channel_price_response": {
                    "private_unsubsidized": {
                        "state_year_count": 1,
                        "price_responsive": True,
                        "mean_p_baseline": 100.0,
                        "mean_p_alpha": 102.0,
                        "mean_p_shadow_marginal": 108.0,
                        "mean_p_alpha_pct_change_vs_baseline": 0.02,
                    },
                    "private_subsidized": {
                        "state_year_count": 0,
                        "price_responsive": True,
                        "mean_p_baseline": 0.0,
                        "mean_p_alpha": 0.0,
                        "mean_p_shadow_marginal": 0.0,
                        "mean_p_alpha_pct_change_vs_baseline": 0.0,
                    },
                    "public_admin": {
                        "state_year_count": 1,
                        "price_responsive": False,
                        "mean_p_baseline": 100.0,
                        "mean_p_alpha": 100.0,
                        "mean_p_shadow_marginal": 100.0,
                        "mean_p_alpha_pct_change_vs_baseline": 0.0,
                    },
                },
            },
            segmented_dir / "childcare_segmented_headline_summary.json",
        )

    monkeypatch.setattr(cli, "build_childcare_segmented_report", fake_segmented_report)

    cli.main(["report-childcare-segmented", "--sample"])

    headline_path = project_paths.outputs_reports / "childcare_segmented_headline_summary.json"
    support_quality_json_path = project_paths.outputs_reports / "childcare_segmented_support_quality_summary.json"
    support_priority_json_path = project_paths.outputs_reports / "childcare_segmented_support_priority_summary.json"
    support_issue_json_path = project_paths.outputs_reports / "childcare_segmented_support_issue_breakdown.json"
    parser_focus_json_path = project_paths.outputs_reports / "childcare_segmented_parser_focus_summary.json"
    parser_action_plan_json_path = project_paths.outputs_reports / "childcare_segmented_parser_action_plan_summary.json"
    readout_path = project_paths.outputs_reports / "childcare_segmented_headline_readout.md"
    report_path = project_paths.outputs_reports / "childcare_segmented_report.md"
    channel_csv_path = project_paths.outputs_tables / "childcare_segmented_channel_response_summary.csv"
    fallback_csv_path = project_paths.outputs_tables / "childcare_segmented_state_fallback_summary.csv"
    support_quality_csv_path = project_paths.outputs_tables / "childcare_segmented_support_quality_summary.csv"
    support_priority_states_path = project_paths.outputs_tables / "childcare_segmented_support_priority_states.csv"
    support_issue_csv_path = project_paths.outputs_tables / "childcare_segmented_support_issue_breakdown.csv"
    parser_focus_csv_path = project_paths.outputs_tables / "childcare_segmented_parser_focus_areas.csv"
    admin_targets_csv_path = project_paths.outputs_tables / "childcare_segmented_admin_sheet_targets.csv"
    parser_action_plan_csv_path = project_paths.outputs_tables / "childcare_segmented_parser_action_plan.csv"

    assert observed["segmented_report_called"] is True
    assert read_json(headline_path)["headline_alpha"] == 0.5
    support_quality_json = read_json(support_quality_json_path)
    assert support_quality_json["state_year_count_total"] == 1
    assert support_quality_json["tiers"][0]["support_quality_tier"] == "explicit_split_support"
    support_priority_json = read_json(support_priority_json_path)
    assert support_priority_json["weak_support_state_year_count"] == 0
    assert support_priority_json["priority_state_count"] == 0
    support_issue_json = read_json(support_issue_json_path)
    assert support_issue_json["rows"][0]["ccdf_support_flag"] == "explicit"
    parser_focus_json = read_json(parser_focus_json_path)
    assert parser_focus_json["rows"][0]["priority_tier"] in {"high", "medium", "low"}
    parser_action_plan_json = read_json(parser_action_plan_json_path)
    assert parser_action_plan_json["row_count"] > 0
    assert parser_action_plan_json["rows"]
    assert "# Childcare Segmented Headline Readout" in readout_path.read_text(encoding="utf-8")
    assert "# childcare_segmented_report" in report_path.read_text(encoding="utf-8")
    assert "## Support quality" in readout_path.read_text(encoding="utf-8")
    assert "## Priority states" in readout_path.read_text(encoding="utf-8")
    assert "## Support issue mix" in readout_path.read_text(encoding="utf-8")
    assert "## Parser focus areas" in readout_path.read_text(encoding="utf-8")
    assert pd.read_csv(channel_csv_path)["solver_channel"].tolist() == ["private_unsubsidized", "public_admin"]
    assert pd.read_csv(fallback_csv_path)["state_fips"].astype(str).tolist() == ["6"]
    assert pd.read_csv(support_quality_csv_path)["support_quality_tier"].tolist() == ["explicit_split_support"]
    assert pd.read_csv(support_priority_states_path)["primary_support_issue"].tolist() == ["stable_supported"]
    assert pd.read_csv(support_issue_csv_path)["ccdf_support_flag"].tolist() == ["explicit"]
    assert not pd.read_csv(parser_focus_csv_path).empty
    assert pd.read_csv(admin_targets_csv_path)["filename"].tolist() == ["fy2023_table_1.xlsx"]
    parser_action_plan_csv = pd.read_csv(parser_action_plan_csv_path)
    assert not parser_action_plan_csv.empty
    assert "subsidized_private_split_columns" in parser_action_plan_csv["focus_area"].tolist()
    headline_provenance = read_json(Path(f"{headline_path}.provenance.json"))
    assert headline_provenance["config"]["name"] == "segmented_publication"
    assert str(project_paths.root / "data" / "interim" / "childcare" / "segmented_reports" / "childcare_segmented_headline_summary.json") in headline_provenance["source_files"]
    assert read_json(Path(f"{support_quality_json_path}.provenance.json"))["config"]["name"] == "segmented_publication"
    assert read_json(Path(f"{support_priority_json_path}.provenance.json"))["config"]["name"] == "segmented_publication"
    assert read_json(Path(f"{support_issue_json_path}.provenance.json"))["config"]["name"] == "segmented_publication"
    assert read_json(Path(f"{parser_focus_json_path}.provenance.json"))["config"]["name"] == "segmented_publication"
    assert read_json(Path(f"{parser_action_plan_json_path}.provenance.json"))["config"]["name"] == "segmented_publication"
    assert read_json(Path(f"{parser_action_plan_csv_path}.provenance.json"))["config"]["name"] == "segmented_publication"
    assert pooled_report_path.read_text(encoding="utf-8") == "pooled sentinel"


def _write_childcare_simulation_inputs(project_paths) -> None:
    state = pd.DataFrame(
        [
            {
                "state_fips": "01",
                "year": 2009,
                "subgroup": "all",
                "state_price_index": 10000.0,
                "unpaid_childcare_hours": 18.0,
                "atus_weight_sum": 100.0,
                "outside_option_wage": 20.0,
                "parent_employment_rate": 0.72,
                "single_parent_share": 0.24,
                "median_income": 55000.0,
                "unemployment_rate": 0.05,
                "market_quantity_proxy": 1000.0,
                "unpaid_quantity_proxy": 200.0,
                "benchmark_replacement_cost": 8000.0,
                "state_direct_care_price_index": 4200.0,
                "state_direct_care_price_index_raw": 4300.0,
                "state_non_direct_care_price_index": 5800.0,
                "state_direct_care_labor_share": 0.42,
                "state_direct_care_price_clip_binding_share": 0.0,
                "state_direct_care_price_clip_binding": False,
                "state_effective_children_per_worker": 5.0,
                "state_implied_direct_care_hourly_wage": 8.08,
                "eligible_broad_complete": True,
                "eligible_observed_core": False,
                "eligible_observed_core_low_impute": False,
                "state_controls_source": "acs_observed",
                "state_unemployment_source": "laus_observed",
                "state_price_support_window": "out_of_support",
                "state_price_observation_status": "pre_ndcp_support_gap",
                "state_price_nowcast": False,
                "state_ndcp_imputed_share": 0.35,
                "state_qcew_wage_observed_share": 0.98,
                "state_qcew_employment_observed_share": 0.98,
                "state_qcew_labor_observed_share": 0.98,
                "observed_core_exclusion_reason": "outside_year_window",
                "observed_core_low_impute_exclusion_reason": "outside_year_window",
                "is_sensitivity_year": False,
            },
            {
                "state_fips": "01",
                "year": 2014,
                "subgroup": "all",
                "state_price_index": 11000.0,
                "unpaid_childcare_hours": 17.0,
                "atus_weight_sum": 120.0,
                "outside_option_wage": 21.0,
                "parent_employment_rate": 0.74,
                "single_parent_share": 0.22,
                "median_income": 57000.0,
                "unemployment_rate": 0.045,
                "market_quantity_proxy": 1100.0,
                "unpaid_quantity_proxy": 180.0,
                "benchmark_replacement_cost": 8200.0,
                "state_direct_care_price_index": 4800.0,
                "state_direct_care_price_index_raw": 4900.0,
                "state_non_direct_care_price_index": 6200.0,
                "state_direct_care_labor_share": 4800.0 / 11000.0,
                "state_direct_care_price_clip_binding_share": 0.0,
                "state_direct_care_price_clip_binding": False,
                "state_effective_children_per_worker": 5.5,
                "state_implied_direct_care_hourly_wage": 8.73,
                "eligible_broad_complete": True,
                "eligible_observed_core": True,
                "eligible_observed_core_low_impute": True,
                "state_controls_source": "acs_observed",
                "state_unemployment_source": "laus_observed",
                "state_price_support_window": "in_support",
                "state_price_observation_status": "observed_ndcp_support",
                "state_price_nowcast": False,
                "state_ndcp_imputed_share": 0.12,
                "state_qcew_wage_observed_share": 0.99,
                "state_qcew_employment_observed_share": 0.99,
                "state_qcew_labor_observed_share": 0.99,
                "observed_core_exclusion_reason": "",
                "observed_core_low_impute_exclusion_reason": "",
                "is_sensitivity_year": False,
            },
            {
                "state_fips": "02",
                "year": 2015,
                "subgroup": "all",
                "state_price_index": 12000.0,
                "unpaid_childcare_hours": 16.0,
                "atus_weight_sum": 130.0,
                "outside_option_wage": 22.0,
                "parent_employment_rate": 0.75,
                "single_parent_share": 0.21,
                "median_income": 59000.0,
                "unemployment_rate": 0.04,
                "market_quantity_proxy": 1200.0,
                "unpaid_quantity_proxy": 150.0,
                "benchmark_replacement_cost": 8400.0,
                "state_direct_care_price_index": 5200.0,
                "state_direct_care_price_index_raw": 13000.0,
                "state_non_direct_care_price_index": 6800.0,
                "state_direct_care_labor_share": 5200.0 / 12000.0,
                "state_direct_care_price_clip_binding_share": 1.0,
                "state_direct_care_price_clip_binding": True,
                "state_effective_children_per_worker": 5.0,
                "state_implied_direct_care_hourly_wage": 10.42,
                "eligible_broad_complete": True,
                "eligible_observed_core": True,
                "eligible_observed_core_low_impute": False,
                "state_controls_source": "acs_observed",
                "state_unemployment_source": "laus_observed",
                "state_price_support_window": "in_support",
                "state_price_observation_status": "observed_ndcp_support",
                "state_price_nowcast": False,
                "state_ndcp_imputed_share": 0.31,
                "state_qcew_wage_observed_share": 0.97,
                "state_qcew_employment_observed_share": 0.97,
                "state_qcew_labor_observed_share": 0.97,
                "observed_core_exclusion_reason": "",
                "observed_core_low_impute_exclusion_reason": "imputation_share_above_threshold",
                "is_sensitivity_year": False,
            },
            {
                "state_fips": "02",
                "year": 2023,
                "subgroup": "all",
                "state_price_index": 12500.0,
                "unpaid_childcare_hours": 15.5,
                "atus_weight_sum": 140.0,
                "outside_option_wage": 22.5,
                "parent_employment_rate": 0.76,
                "single_parent_share": 0.2,
                "median_income": 60000.0,
                "unemployment_rate": 0.038,
                "market_quantity_proxy": 1210.0,
                "unpaid_quantity_proxy": 140.0,
                "benchmark_replacement_cost": 8500.0,
                "state_direct_care_price_index": 5300.0,
                "state_direct_care_price_index_raw": 5400.0,
                "state_non_direct_care_price_index": 7200.0,
                "state_direct_care_labor_share": 5300.0 / 12500.0,
                "state_direct_care_price_clip_binding_share": 0.0,
                "state_direct_care_price_clip_binding": False,
                "state_effective_children_per_worker": 5.0,
                "state_implied_direct_care_hourly_wage": 10.61,
                "eligible_broad_complete": True,
                "eligible_observed_core": False,
                "eligible_observed_core_low_impute": False,
                "state_controls_source": "acs_observed",
                "state_unemployment_source": "laus_observed",
                "state_price_support_window": "out_of_support",
                "state_price_observation_status": "post_ndcp_nowcast",
                "state_price_nowcast": True,
                "state_ndcp_imputed_share": 0.28,
                "state_qcew_wage_observed_share": 0.96,
                "state_qcew_employment_observed_share": 0.96,
                "state_qcew_labor_observed_share": 0.96,
                "observed_core_exclusion_reason": "outside_year_window",
                "observed_core_low_impute_exclusion_reason": "outside_year_window",
                "is_sensitivity_year": False,
            },
        ]
    )
    county = pd.DataFrame(
        [
            {
                "state_fips": "01",
                "year": 2014,
                "provider_density": 1.0,
                "annual_price": 10000.0,
                "under5_population": 1000.0,
                "direct_care_price_index": 4800.0,
                "non_direct_care_price_index": 5200.0,
                "benchmark_childcare_wage": 8.4,
            },
            {
                "state_fips": "02",
                "year": 2015,
                "provider_density": 1.1,
                "annual_price": 10200.0,
                "under5_population": 1200.0,
                "direct_care_price_index": 5200.0,
                "non_direct_care_price_index": 5000.0,
                "benchmark_childcare_wage": 8.8,
            },
        ]
    )
    acs = pd.DataFrame(
        [
            {"county_fips": "01001", "state_fips": "01", "year": 2014, "under5_population": 1300.0},
            {"county_fips": "02001", "state_fips": "02", "year": 2015, "under5_population": 1500.0},
        ]
    )
    write_parquet(state, project_paths.processed / "childcare_state_year_panel.parquet")
    write_parquet(county, project_paths.processed / "childcare_county_year_price_panel.parquet")
    write_parquet(acs, project_paths.interim / "acs" / "acs.parquet")
    write_json(
        {
            "comparison_specification_profile": "household_parsimonious",
            "samples": {
                "broad_complete": {
                    "n_obs": 228,
                    "n_states": 22,
                    "n_years": 15,
                    "price_coefficient": -0.001,
                    "elasticity_at_mean": -0.14,
                    "economically_admissible": True,
                    "reduced_form_price_coefficient": -0.001,
                    "reduced_form_sign_consistent": True,
                    "first_stage_f": 8.0,
                    "headline_eligible": False,
                },
                "observed_core": {
                    "n_obs": 139,
                    "n_states": 21,
                    "n_years": 8,
                    "price_coefficient": -0.001,
                    "elasticity_at_mean": -0.11,
                    "economically_admissible": True,
                    "reduced_form_price_coefficient": -0.001,
                    "reduced_form_sign_consistent": True,
                    "first_stage_f": 12.0,
                    "headline_eligible": True,
                },
                "observed_core_low_impute": {
                    "n_obs": 55,
                    "n_states": 13,
                    "n_years": 8,
                    "price_coefficient": -0.001,
                    "elasticity_at_mean": -0.10,
                    "economically_admissible": True,
                    "reduced_form_price_coefficient": -0.001,
                    "reduced_form_sign_consistent": True,
                    "first_stage_f": 9.0,
                    "headline_eligible": False,
                },
            }
        },
        project_paths.outputs_reports / "childcare_demand_sample_comparison.json",
    )
    for sample_name, elasticity in (
        ("broad_complete", -0.15),
        ("observed_core", 0.30),
        ("observed_core_low_impute", 0.80),
    ):
        write_json(
            {
                "mode": sample_name,
                "specification_profile": "full_controls",
                "price_coefficient": -0.001,
                "elasticity_at_mean": elasticity,
                "first_stage_r2": 0.8,
                "first_stage_f": 12.0 if elasticity <= 0 else 8.0,
                "reduced_form_price_coefficient": -0.001 if elasticity <= 0 else 0.001,
                "reduced_form_sign_consistent": elasticity <= 0,
                "n_obs": 10,
                "n_states": 2,
                "n_years": 2,
                "year_min": 2014,
                "year_max": 2015,
                "economically_admissible": elasticity <= 0,
            },
            project_paths.outputs_reports / f"childcare_demand_iv_{sample_name}.json",
        )
    write_json(
        {
            "mode": "broad_complete",
            "specification_profile": "household_parsimonious",
            "price_coefficient": -0.001,
            "elasticity_at_mean": -0.14,
            "first_stage_r2": 0.79,
            "first_stage_f": 8.0,
            "reduced_form_price_coefficient": -0.001,
            "reduced_form_sign_consistent": True,
            "n_obs": 12,
            "n_states": 2,
            "n_years": 3,
            "year_min": 2009,
            "year_max": 2023,
            "economically_admissible": True,
        },
        project_paths.outputs_reports / "childcare_demand_iv_broad_complete_household_parsimonious.json",
    )
    write_json(
        {
            "mode": "observed_core",
            "specification_profile": "household_parsimonious",
            "price_coefficient": -0.001,
            "elasticity_at_mean": -0.11,
            "first_stage_r2": 0.78,
            "first_stage_f": 12.0,
            "reduced_form_price_coefficient": -0.001,
            "reduced_form_sign_consistent": True,
            "n_obs": 10,
            "n_states": 2,
            "n_years": 2,
            "year_min": 2014,
            "year_max": 2015,
            "economically_admissible": True,
        },
        project_paths.outputs_reports / "childcare_demand_iv_observed_core_household_parsimonious.json",
    )
    write_json(
        {
            "mode": "observed_core_low_impute",
            "specification_profile": "household_parsimonious",
            "price_coefficient": -0.001,
            "elasticity_at_mean": -0.10,
            "first_stage_r2": 0.77,
            "first_stage_f": 9.0,
            "reduced_form_price_coefficient": -0.001,
            "reduced_form_sign_consistent": True,
            "n_obs": 8,
            "n_states": 2,
            "n_years": 2,
            "year_min": 2014,
            "year_max": 2015,
            "economically_admissible": True,
        },
        project_paths.outputs_reports / "childcare_demand_iv_observed_core_low_impute_household_parsimonious.json",
    )
    write_json(
        {
            "mode": "observed_core",
            "specification_profile": "instrument_only",
            "price_coefficient": -0.001,
            "elasticity_at_mean": -0.09,
            "first_stage_r2": 0.75,
            "first_stage_f": 12.0,
            "reduced_form_price_coefficient": -0.001,
            "reduced_form_sign_consistent": True,
            "n_obs": 10,
            "n_states": 2,
            "n_years": 2,
            "year_min": 2014,
            "year_max": 2015,
            "economically_admissible": True,
        },
        project_paths.outputs_reports / "childcare_demand_iv_observed_core_instrument_only.json",
    )
    write_json(
        {
            "profiles": {
                "household_parsimonious": {},
                "instrument_only": {},
            }
        },
        project_paths.outputs_reports / "childcare_demand_specification_sweep.json",
    )


def _write_childcare_segmented_scenario_inputs(project_paths) -> None:
    solver_dir = project_paths.root / "data" / "interim" / "childcare" / "solver_inputs"
    solver_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "year": 2014,
                    "solver_private_unsubsidized_slots": 60.0,
                    "solver_private_subsidized_slots": 40.0,
                    "solver_public_admin_slots": 20.0,
                    "solver_total_private_slots": 100.0,
                    "solver_total_paid_slots": 120.0,
                    "solver_exogenous_public_admin_slots": 20.0,
                },
                {
                    "state_fips": "01",
                    "year": 2009,
                    "solver_private_unsubsidized_slots": 55.0,
                    "solver_private_subsidized_slots": 45.0,
                    "solver_public_admin_slots": 15.0,
                    "solver_total_private_slots": 100.0,
                    "solver_total_paid_slots": 115.0,
                    "solver_exogenous_public_admin_slots": 15.0,
                },
                {
                    "state_fips": "02",
                    "year": 2015,
                    "solver_private_unsubsidized_slots": 70.0,
                    "solver_private_subsidized_slots": 35.0,
                    "solver_public_admin_slots": 25.0,
                    "solver_total_private_slots": 105.0,
                    "solver_total_paid_slots": 130.0,
                    "solver_exogenous_public_admin_slots": 25.0,
                },
                {
                    "state_fips": "02",
                    "year": 2023,
                    "solver_private_unsubsidized_slots": 68.0,
                    "solver_private_subsidized_slots": 42.0,
                    "solver_public_admin_slots": 28.0,
                    "solver_total_private_slots": 110.0,
                    "solver_total_paid_slots": 138.0,
                    "solver_exogenous_public_admin_slots": 28.0,
                },
            ]
        ),
        solver_dir / "solver_baseline_state_year.parquet",
    )
    write_parquet(
        pd.DataFrame(
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
        ),
        solver_dir / "solver_elasticity_mapping.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "year": 2014,
                    "ccdf_policy_control_count": 1,
                    "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                    "ccdf_policy_promoted_controls_selected": "ccdf_control_copayment_required",
                    "ccdf_policy_promoted_control_rule": "state_year_coverage_gte_threshold",
                    "ccdf_policy_promoted_min_state_year_coverage": 0.75,
                    "ccdf_control_copayment_required": "yes",
                }
            ]
        ),
        solver_dir / "solver_policy_controls_state_year.parquet",
    )
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "year": 2014,
                    "solver_channel": "private_unsubsidized",
                    "quantity_slots": 60.0,
                    "price_responsive": True,
                },
                {
                    "state_fips": "01",
                    "year": 2014,
                    "solver_channel": "private_subsidized",
                    "quantity_slots": 40.0,
                    "price_responsive": True,
                },
                {
                    "state_fips": "01",
                    "year": 2014,
                    "solver_channel": "public_admin",
                    "quantity_slots": 20.0,
                    "price_responsive": False,
                },
            ]
        ),
        solver_dir / "solver_channel_quantities.parquet",
    )


def _write_childcare_additive_extension_configs(project_paths) -> None:
    config_dir = project_paths.root / "configs" / "extensions"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "segmented_solver.yaml").write_text(
        "\n".join(
            [
                "name: segmented_solver",
                "mode: segmented_baseline",
                "segments:",
                "  - segment_id: infant_center_private",
                "    child_age: infant",
                "    provider_type: center",
                "    channel: private",
                "  - segment_id: preschool_home_private",
                "    child_age: preschool",
                "    provider_type: home",
                "    channel: private",
                "build:",
                "  compatibility:",
                "    enabled: true",
                "    tolerance: 1.0e-9",
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
    (config_dir / "utilization_stack.yaml").write_text(
        "\n".join(
            [
                "name: utilization_stack",
                "sector: childcare",
                "mode: observed_utilization",
                "description: Replace proxy paid-care quantity with measured utilization components and reconciliation targets.",
                "geography:",
                "  canonical_core: state_year",
                "  support_level: state_year",
                "year_window:",
                "  start: 2021",
                "  end: 2023",
                "output_namespace:",
                "  namespace: data/interim/childcare/utilization_stack",
                "  files:",
                "    public_program_slots: public_program_slots_state_year.parquet",
                "    survey_paid_use_targets: survey_paid_use_targets.parquet",
                "    quantity_by_segment: q0_segmented.parquet",
                "    reconciliation_diagnostics: utilization_reconciliation_diagnostics.parquet",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "ccdf_state_year.yaml").write_text(
        "\n".join(
            [
                "name: ccdf_state_year",
                "sector: childcare",
                "mode: state_year_mapping",
                "description: Map normalized CCDF admin and policy long-form tables into additive state-year features.",
                "geography:",
                "  canonical_core: state_year",
                "  support_level: state_year",
                "sources:",
                "  required:",
                "    - name: ccdf_admin_long",
                "      path: data/interim/ccdf/ccdf_admin_long.parquet",
                "    - name: ccdf_policy_long",
                "      path: data/interim/ccdf/ccdf_policy_long.parquet",
                "output_namespace:",
                "  namespace: data/interim/ccdf",
                "  files:",
                "    admin_state_year: ccdf_admin_state_year.parquet",
                "    policy_features_state_year: ccdf_policy_features_state_year.parquet",
                "    policy_controls_state_year: ccdf_policy_controls_state_year.parquet",
                "    policy_controls_coverage: ccdf_policy_controls_coverage.parquet",
                "    policy_promoted_controls_state_year: ccdf_policy_promoted_controls_state_year.parquet",
                "    policy_feature_audit: ccdf_policy_feature_audit.parquet",
                "policy_controls_promotion:",
                "  min_state_year_coverage: 0.75",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "solver_inputs.yaml").write_text(
        "\n".join(
            [
                "name: solver_inputs",
                "sector: childcare",
                "mode: additive_solver_inputs",
                "description: Build additive solver-ready childcare channel inputs from segmented utilization and promoted CCDF controls.",
                "geography:",
                "  canonical_core: state_year",
                "  support_level: state_year",
                "year_window:",
                "  start: 2008",
                "  end: 2023",
                "  sensitivity_year: 2020",
                "sources:",
                "  required:",
                "    - name: q0_segmented",
                "      path: data/interim/childcare/utilization_stack/q0_segmented.parquet",
                "    - name: ndcp_segment_prices",
                "      path: data/interim/childcare/segmented_solver/ndcp_segment_prices.parquet",
                "  optional:",
                "    - name: utilization_reconciliation_diagnostics",
                "      path: data/interim/childcare/utilization_stack/utilization_reconciliation_diagnostics.parquet",
                "    - name: ccdf_policy_promoted_controls_state_year",
                "      path: data/interim/ccdf/ccdf_policy_promoted_controls_state_year.parquet",
                "    - name: ccdf_policy_controls_state_year",
                "      path: data/interim/ccdf/ccdf_policy_controls_state_year.parquet",
                "build:",
                "  channels:",
                "    private_unsubsidized:",
                "      quantity_component: private_unsubsidized",
                "      price_responsive: true",
                "    private_subsidized:",
                "      quantity_component: private_subsidized",
                "      price_responsive: true",
                "    public_admin:",
                "      quantity_component: public_admin",
                "      price_responsive: false",
                "  elasticity_family: pooled_childcare_demand",
                "output_namespace:",
                "  namespace: data/interim/childcare/solver_inputs",
                "  files:",
                "    solver_channel_quantities: solver_channel_quantities.parquet",
                "    solver_baseline_state_year: solver_baseline_state_year.parquet",
                "    solver_elasticity_mapping: solver_elasticity_mapping.parquet",
                "    solver_policy_controls_state_year: solver_policy_controls_state_year.parquet",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "report_tables.yaml").write_text(
        "\n".join(
            [
                "name: report_tables",
                "sector: childcare",
                "mode: additive_report_tables",
                "description: Build additive childcare report tables from segmented utilization and promoted CCDF controls.",
                "geography:",
                "  canonical_core: state_year",
                "  support_level: state_year",
                "year_window:",
                "  start: 2008",
                "  end: 2023",
                "  sensitivity_year: 2020",
                "sources:",
                "  required:",
                "    - name: q0_segmented",
                "      path: data/interim/childcare/utilization_stack/q0_segmented.parquet",
                "    - name: utilization_reconciliation_diagnostics",
                "      path: data/interim/childcare/utilization_stack/utilization_reconciliation_diagnostics.parquet",
                "  optional:",
                "    - name: ccdf_policy_promoted_controls_state_year",
                "      path: data/interim/ccdf/ccdf_policy_promoted_controls_state_year.parquet",
                "    - name: ccdf_policy_controls_state_year",
                "      path: data/interim/ccdf/ccdf_policy_controls_state_year.parquet",
                "build:",
                "  summary_tables:",
                "    - state_year_channel_summary",
                "    - state_year_policy_quantity_summary",
                "    - state_year_support_summary",
                "output_namespace:",
                "  namespace: data/interim/childcare/report_tables",
                "  files:",
                "    state_year_channel_summary: state_year_channel_summary.parquet",
                "    state_year_policy_quantity_summary: state_year_policy_quantity_summary.parquet",
                "    state_year_support_summary: state_year_support_summary.parquet",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "segmented_reports.yaml").write_text(
        "\n".join(
            [
                "name: segmented_reports",
                "sector: childcare",
                "mode: segmented_reports",
                "description: Build additive segmented childcare report outputs from segmented scenarios and support diagnostics.",
                "geography:",
                "  canonical_core: state_year",
                "  support_level: state_year",
                "year_window:",
                "  start: 2008",
                "  end: 2023",
                "  sensitivity_year: 2020",
                "sources:",
                "  required:",
                "    - name: segmented_channel_scenarios",
                "      path: data/interim/childcare/segmented_scenarios/segmented_channel_scenarios.parquet",
                "    - name: segmented_state_year_summary",
                "      path: data/interim/childcare/segmented_scenarios/segmented_state_year_summary.parquet",
                "    - name: segmented_state_year_diagnostics",
                "      path: data/interim/childcare/segmented_scenarios/segmented_state_year_diagnostics.parquet",
                "  optional:",
                "    - name: state_year_support_summary",
                "      path: data/interim/childcare/report_tables/state_year_support_summary.parquet",
                "build:",
                "  summary_surfaces:",
                "    - segmented_channel_response_summary",
                "    - segmented_state_fallback_summary",
                "output_namespace:",
                "  namespace: data/interim/childcare/segmented_reports",
                "  files:",
                "    segmented_channel_response_summary: segmented_channel_response_summary.parquet",
                "    segmented_state_fallback_summary: segmented_state_fallback_summary.parquet",
                "    childcare_segmented_headline_summary: childcare_segmented_headline_summary.json",
                "    childcare_segmented_report: childcare_segmented_report.md",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "segmented_publication.yaml").write_text(
        "\n".join(
            [
                "name: segmented_publication",
                "sector: childcare",
                "mode: segmented_publication",
                "description: Publish additive segmented childcare report outputs into outputs/reports and outputs/tables.",
                "geography:",
                "  canonical_core: state_year",
                "  support_level: state_year",
                "year_window:",
                "  start: 2008",
                "  end: 2023",
                "  sensitivity_year: 2020",
                "sources:",
                "  required:",
                "    - name: segmented_channel_response_summary",
                "      path: data/interim/childcare/segmented_reports/segmented_channel_response_summary.parquet",
                "    - name: segmented_state_fallback_summary",
                "      path: data/interim/childcare/segmented_reports/segmented_state_fallback_summary.parquet",
                "    - name: childcare_segmented_headline_summary",
                "      path: data/interim/childcare/segmented_reports/childcare_segmented_headline_summary.json",
                "outputs:",
                "  reports_namespace: outputs/reports",
                "  tables_namespace: outputs/tables",
                "  files:",
                "    headline_summary_json: childcare_segmented_headline_summary.json",
                "    support_quality_summary_json: childcare_segmented_support_quality_summary.json",
                "    support_priority_summary_json: childcare_segmented_support_priority_summary.json",
                "    support_issue_breakdown_json: childcare_segmented_support_issue_breakdown.json",
                "    parser_focus_summary_json: childcare_segmented_parser_focus_summary.json",
                "    parser_action_plan_summary_json: childcare_segmented_parser_action_plan_summary.json",
                "    headline_readout_markdown: childcare_segmented_headline_readout.md",
                "    full_report_markdown: childcare_segmented_report.md",
                "    channel_response_csv: childcare_segmented_channel_response_summary.csv",
                "    fallback_csv: childcare_segmented_state_fallback_summary.csv",
                "    support_quality_csv: childcare_segmented_support_quality_summary.csv",
                "    support_priority_states_csv: childcare_segmented_support_priority_states.csv",
                "    support_issue_breakdown_csv: childcare_segmented_support_issue_breakdown.csv",
                "    parser_focus_areas_csv: childcare_segmented_parser_focus_areas.csv",
                "    admin_sheet_targets_csv: childcare_segmented_admin_sheet_targets.csv",
                "    parser_action_plan_csv: childcare_segmented_parser_action_plan.csv",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "segmented_scenarios.yaml").write_text(
        "\n".join(
            [
                "name: segmented_scenarios",
                "sector: childcare",
                "mode: segmented_scenarios",
                "description: Build additive segmented childcare scenario outputs from solver-ready channel inputs and selected demand fits.",
                "geography:",
                "  canonical_core: state_year",
                "  support_level: state_year",
                "year_window:",
                "  start: 2008",
                "  end: 2023",
                "  sensitivity_year: 2020",
                "sources:",
                "  required:",
                "    - name: childcare_state_year_panel",
                "      path: data/processed/childcare_state_year_panel.parquet",
                "    - name: childcare_county_year_price_panel",
                "      path: data/processed/childcare_county_year_price_panel.parquet",
                "    - name: childcare_demand_sample_comparison",
                "      path: outputs/reports/childcare_demand_sample_comparison.json",
                "  optional:",
                "    - name: solver_baseline_state_year",
                "      path: data/interim/childcare/solver_inputs/solver_baseline_state_year.parquet",
                "    - name: solver_elasticity_mapping",
                "      path: data/interim/childcare/solver_inputs/solver_elasticity_mapping.parquet",
                "    - name: solver_policy_controls_state_year",
                "      path: data/interim/childcare/solver_inputs/solver_policy_controls_state_year.parquet",
                "build:",
                "  selection:",
                "    sample_choice: headline_sample",
                "    demand_comparison: childcare_demand_sample_comparison.json",
                "  scenario_grid:",
                "    alphas: project_alpha_grid",
                "output_namespace:",
                "  namespace: data/interim/childcare/segmented_scenarios",
                "  files:",
                "    segmented_channel_inputs: segmented_channel_inputs.parquet",
                "    segmented_channel_scenarios: segmented_channel_scenarios.parquet",
                "    segmented_state_year_summary: segmented_state_year_summary.parquet",
                "    segmented_state_year_diagnostics: segmented_state_year_diagnostics.parquet",
            ]
        ),
        encoding="utf-8",
    )


def _install_fake_childcare_solver_inputs_module(monkeypatch):
    module = types.ModuleType("unpriced.childcare.solver_inputs")
    observed: dict[str, object] = {}

    def build_childcare_solver_inputs(**kwargs):
        observed["kwargs"] = kwargs
        q0 = kwargs["q0_segmented"].copy()
        for column, default in (
            ("q0_support_flag", pd.NA),
            ("ccdf_support_flag", pd.NA),
            ("ccdf_admin_support_status", pd.NA),
            ("public_program_support_status", pd.NA),
            ("segment_allocation_fallback", False),
        ):
            if column not in q0.columns:
                q0[column] = default
        grouped = (
            q0.groupby(["state_fips", "year", "quantity_component"], as_index=False)
            .agg(
                quantity_slots=("quantity_slots", "sum"),
                q0_support_flag=("q0_support_flag", "first"),
                ccdf_support_flag=("ccdf_support_flag", "first"),
                ccdf_admin_support_status=("ccdf_admin_support_status", "first"),
                public_program_support_status=("public_program_support_status", "first"),
                segment_allocation_fallback=("segment_allocation_fallback", "max"),
            )
            .rename(columns={"quantity_component": "solver_channel"})
        )
        grouped["source_quantity_component"] = grouped["solver_channel"]
        grouped["price_responsive"] = grouped["solver_channel"].ne("public_admin")
        grouped = grouped[
            [
                "state_fips",
                "year",
                "solver_channel",
                "source_quantity_component",
                "quantity_slots",
                "price_responsive",
                "q0_support_flag",
                "ccdf_support_flag",
                "ccdf_admin_support_status",
                "public_program_support_status",
                "segment_allocation_fallback",
            ]
        ]
        all_channels = pd.DataFrame({"solver_channel": ["private_unsubsidized", "private_subsidized", "public_admin"]})
        state_years = grouped[["state_fips", "year"]].drop_duplicates()
        grouped = (
            state_years.assign(_key=1)
            .merge(all_channels.assign(_key=1), on="_key", how="outer")
            .drop(columns="_key")
            .merge(grouped, on=["state_fips", "year", "solver_channel"], how="left")
        )
        grouped["source_quantity_component"] = grouped["source_quantity_component"].fillna(grouped["solver_channel"])
        grouped["quantity_slots"] = pd.to_numeric(grouped["quantity_slots"], errors="coerce").fillna(0.0)
        grouped["price_responsive"] = grouped["price_responsive"].fillna(grouped["solver_channel"].ne("public_admin"))
        grouped["q0_support_flag"] = grouped["q0_support_flag"].fillna("missing")
        grouped["ccdf_support_flag"] = grouped["ccdf_support_flag"].fillna("missing")
        grouped["ccdf_admin_support_status"] = grouped["ccdf_admin_support_status"].fillna("missing")
        grouped["public_program_support_status"] = grouped["public_program_support_status"].fillna("missing")
        grouped["segment_allocation_fallback"] = grouped["segment_allocation_fallback"].fillna(False)
        grouped = grouped.sort_values(["state_fips", "year", "solver_channel"], kind="stable").reset_index(drop=True)

        baseline = grouped.pivot_table(
            index=["state_fips", "year"],
            columns="solver_channel",
            values="quantity_slots",
            aggfunc="sum",
            fill_value=0.0,
        ).reset_index()
        baseline.columns = [str(column) for column in baseline.columns]
        for column in ("private_unsubsidized", "private_subsidized", "public_admin"):
            if column not in baseline.columns:
                baseline[column] = 0.0
        baseline["total_paid_slots"] = baseline[
            ["private_unsubsidized", "private_subsidized", "public_admin"]
        ].sum(axis=1)
        baseline["exogenous_public_slots"] = baseline["public_admin"]
        baseline["price_responsive_private_slots"] = baseline[
            ["private_unsubsidized", "private_subsidized"]
        ].sum(axis=1)
        baseline = baseline[
            [
                "state_fips",
                "year",
                "private_unsubsidized",
                "private_subsidized",
                "public_admin",
                "total_paid_slots",
                "exogenous_public_slots",
                "price_responsive_private_slots",
            ]
        ]

        elasticity_mapping = pd.DataFrame(
            [
                {
                    "solver_channel": "private_unsubsidized",
                    "elasticity_family": "pooled_childcare_demand",
                    "active_in_price_solver": True,
                    "inheritance_rule": "inherits pooled childcare demand elasticity",
                },
                {
                    "solver_channel": "private_subsidized",
                    "elasticity_family": "pooled_childcare_demand",
                    "active_in_price_solver": True,
                    "inheritance_rule": "inherits pooled childcare demand elasticity",
                },
                {
                    "solver_channel": "public_admin",
                    "elasticity_family": "pooled_childcare_demand",
                    "active_in_price_solver": False,
                    "inheritance_rule": "non-price-responsive public admin channel",
                },
            ]
        )

        controls = kwargs.get("ccdf_policy_controls_state_year")
        if controls is None or controls.empty:
            controls_out = pd.DataFrame(
                columns=[
                    "state_fips",
                    "year",
                    "ccdf_policy_control_count",
                    "ccdf_policy_control_support_status",
                    "ccdf_control_copayment_required",
                ]
            )
        else:
            controls_out = controls.copy()
            if "ccdf_policy_control_count" not in controls_out.columns:
                controls_out["ccdf_policy_control_count"] = 1
            if "ccdf_policy_control_support_status" not in controls_out.columns:
                controls_out["ccdf_policy_control_support_status"] = "observed_policy_promoted_controls"

        return {
            "solver_channel_quantities": grouped.reset_index(drop=True),
            "solver_baseline_state_year": baseline.reset_index(drop=True),
            "solver_elasticity_mapping": elasticity_mapping,
            "solver_policy_controls_state_year": controls_out.reset_index(drop=True),
        }

    module.build_childcare_solver_inputs = build_childcare_solver_inputs
    monkeypatch.setitem(sys.modules, "unpriced.childcare.solver_inputs", module)
    return observed


def _install_fake_childcare_report_tables_module(monkeypatch):
    module = types.ModuleType("unpriced.childcare.report_tables")
    observed: dict[str, object] = {}

    def build_childcare_report_tables(**kwargs):
        observed["kwargs"] = kwargs
        q0 = kwargs["q0_segmented"].copy()
        for column, default in (
            ("q0_support_flag", pd.NA),
            ("ccdf_support_flag", pd.NA),
            ("ccdf_admin_support_status", pd.NA),
            ("public_program_support_status", pd.NA),
            ("segment_allocation_fallback", False),
        ):
            if column not in q0.columns:
                q0[column] = default
        channel_summary = (
            q0.groupby(["state_fips", "year", "quantity_component"], as_index=False)
            .agg(
                quantity_slots=("quantity_slots", "sum"),
                q0_support_flag=("q0_support_flag", "first"),
                ccdf_support_flag=("ccdf_support_flag", "first"),
                ccdf_admin_support_status=("ccdf_admin_support_status", "first"),
                public_program_support_status=("public_program_support_status", "first"),
                segment_allocation_fallback=("segment_allocation_fallback", "max"),
            )
            .rename(columns={"quantity_component": "solver_channel"})
        )
        totals = channel_summary.groupby(["state_fips", "year"], as_index=False).agg(
            total_paid_slots=("quantity_slots", "sum")
        )
        channel_summary = channel_summary.merge(totals, on=["state_fips", "year"], how="left")
        channel_summary["channel_share_of_total_paid_care"] = (
            channel_summary["quantity_slots"] / channel_summary["total_paid_slots"].replace({0.0: pd.NA})
        ).fillna(0.0)
        channel_summary["channel_group"] = channel_summary["solver_channel"].map(
            lambda value: "public" if value == "public_admin" else "private"
        )

        policy_quantity_summary = totals.merge(
            channel_summary.pivot_table(
                index=["state_fips", "year"],
                columns="solver_channel",
                values="quantity_slots",
                aggfunc="sum",
                fill_value=0.0,
            ).reset_index(),
            on=["state_fips", "year"],
            how="left",
        )
        policy_quantity_summary.columns = [str(column) for column in policy_quantity_summary.columns]
        for column in ("private_unsubsidized", "private_subsidized", "public_admin"):
            if column not in policy_quantity_summary.columns:
                policy_quantity_summary[column] = 0.0
        promoted = kwargs.get("ccdf_policy_controls_state_year")
        if promoted is not None and not promoted.empty:
            policy_quantity_summary = policy_quantity_summary.merge(
                promoted,
                on=["state_fips", "year"],
                how="left",
                suffixes=("", "_control"),
            )
        else:
            policy_quantity_summary["ccdf_policy_control_count"] = 0
            policy_quantity_summary["ccdf_policy_control_support_status"] = "missing"
            policy_quantity_summary["ccdf_control_copayment_required"] = pd.NA

        support_summary = (
            channel_summary.groupby(["state_fips", "year"], as_index=False)
            .agg(
                total_paid_slots=("total_paid_slots", "max"),
                q0_support_flag=("q0_support_flag", "first"),
                ccdf_support_flag=("ccdf_support_flag", "first"),
                ccdf_admin_support_status=("ccdf_admin_support_status", "first"),
                public_program_support_status=("public_program_support_status", "first"),
                segment_allocation_fallback=("segment_allocation_fallback", "max"),
            )
        )
        diagnostics = kwargs.get("utilization_diagnostics")
        if diagnostics is not None and not diagnostics.empty:
            diagnostics = diagnostics[
                [
                    column
                    for column in diagnostics.columns
                    if column
                    in {
                        "state_fips",
                        "year",
                        "component_sum_gap",
                        "any_private_allocation_fallback",
                        "any_negative_quantity",
                        "ccdf_policy_control_count",
                        "ccdf_policy_control_support_status",
                        "ccdf_control_copayment_required",
                    }
                ]
            ].drop_duplicates(["state_fips", "year"])
            support_summary = support_summary.merge(diagnostics, on=["state_fips", "year"], how="left")
        else:
            support_summary["component_sum_gap"] = 0.0
            support_summary["any_private_allocation_fallback"] = False
            support_summary["any_negative_quantity"] = False
        support_summary["quantity_accounting_gap"] = support_summary["component_sum_gap"].fillna(0.0)
        return {
            "state_year_channel_summary": channel_summary.reset_index(drop=True),
            "state_year_policy_quantity_summary": policy_quantity_summary.reset_index(drop=True),
            "state_year_support_summary": support_summary.reset_index(drop=True),
        }

    module.build_childcare_report_tables = build_childcare_report_tables
    monkeypatch.setitem(sys.modules, "unpriced.childcare.report_tables", module)
    return observed


def _install_fake_childcare_segmented_scenarios_module(monkeypatch):
    module = types.ModuleType("unpriced.childcare.segmented_scenarios")
    observed: dict[str, object] = {}

    def build_childcare_segmented_scenarios(**kwargs):
        observed["kwargs"] = kwargs
        state = kwargs["state_frame"].copy()
        if "eligible_observed_core" in state.columns:
            state = state.loc[state["eligible_observed_core"].fillna(False).astype(bool)].copy()
        if state.empty:
            state = kwargs["state_frame"].head(1).copy()
        channel_rows = []
        for row in state.to_dict(orient="records"):
            for channel in ("private_unsubsidized", "private_subsidized", "public_admin"):
                channel_rows.append(
                    {
                        "state_fips": row["state_fips"],
                        "year": row["year"],
                        "solver_channel": channel,
                        "state_price_index": float(row.get("state_price_index", 100.0)),
                        "state_market_quantity_proxy": float(row.get("market_quantity_proxy", 1000.0)),
                        "state_unpaid_quantity_proxy": float(row.get("unpaid_quantity_proxy", 100.0)),
                        "solver_channel_market_quantity": 60.0 if channel == "private_unsubsidized" else (
                            40.0 if channel == "private_subsidized" else 20.0
                        ),
                        "solver_channel_unpaid_quantity": 50.0 if channel != "public_admin" else 0.0,
                        "price_responsive": channel != "public_admin",
                        "elasticity_family": (
                            "pooled_childcare_demand" if channel != "public_admin" else "exogenous_non_price"
                        ),
                        "solver_total_paid_slots": 120.0,
                        "solver_total_private_slots": 100.0,
                        "solver_exogenous_public_admin_slots": 20.0,
                        "ccdf_policy_control_count": 1,
                        "ccdf_policy_control_support_status": "observed_policy_promoted_controls",
                        "public_admin_support_status": "head_start_plus_ccdf_observed",
                        "state_price_observation_status": row.get("state_price_observation_status", "observed_ndcp_support"),
                        "state_price_nowcast": bool(row.get("state_price_nowcast", False)),
                        "demand_sample_name": "observed_core",
                        "demand_specification_profile": "household_parsimonious",
                    }
                )
        channel_inputs = pd.DataFrame(channel_rows)
        channel_scenarios = channel_inputs.copy()
        channel_scenarios["alpha"] = 0.5
        channel_scenarios["p_baseline"] = channel_scenarios["state_price_index"]
        channel_scenarios["p_shadow_marginal"] = channel_scenarios["state_price_index"]
        channel_scenarios["p_alpha"] = channel_scenarios["state_price_index"]
        summary = pd.DataFrame(
            [
                {
                    "state_fips": row["state_fips"],
                    "year": row["year"],
                    "alpha": 0.5,
                    "channel_rows": 3,
                    "channel_quantity_total": 120.0,
                    "weighted_p_alpha": float(row.get("state_price_index", 100.0)),
                }
                for row in state.to_dict(orient="records")
            ]
        )
        diagnostics = pd.DataFrame(
            [
                {
                    "state_fips": row["state_fips"],
                    "year": row["year"],
                    "scenario_rows": 3,
                    "private_rows": 2,
                    "public_rows": 1,
                    "private_unpaid_quantity": 100.0,
                    "public_unpaid_quantity": 0.0,
                    "price_responsive_row_share": 2.0 / 3.0,
                    "p_alpha_median": float(row.get("state_price_index", 100.0)),
                    "p_baseline_median": float(row.get("state_price_index", 100.0)),
                }
                for row in state.to_dict(orient="records")
            ]
        )
        return {
            "segmented_channel_inputs": channel_inputs.reset_index(drop=True),
            "segmented_channel_scenarios": channel_scenarios.reset_index(drop=True),
            "segmented_state_year_summary": summary.reset_index(drop=True),
            "segmented_state_year_diagnostics": diagnostics.reset_index(drop=True),
        }

    module.build_childcare_segmented_scenarios = build_childcare_segmented_scenarios
    monkeypatch.setitem(sys.modules, "unpriced.childcare.segmented_scenarios", module)
    return observed


def test_simulate_childcare_writes_all_sample_outputs(project_paths):
    _write_childcare_simulation_inputs(project_paths)

    cli.simulate_childcare(project_paths)

    canonical = read_parquet(project_paths.processed / "childcare_marketization_scenarios.parquet")
    combined = read_parquet(project_paths.processed / "childcare_marketization_scenarios_all_samples.parquet")
    diagnostics = read_json(project_paths.outputs_reports / "childcare_scenario_diagnostics.json")
    comparison = read_json(project_paths.outputs_reports / "childcare_scenario_sample_comparison.json")
    spec_comparison = read_json(project_paths.outputs_reports / "childcare_scenario_specification_comparison.json")
    decomposition = read_json(project_paths.outputs_reports / "childcare_price_decomposition.json")
    supply = read_json(project_paths.outputs_reports / "childcare_supply_elasticity.json")
    piecewise = read_json(project_paths.outputs_reports / "childcare_piecewise_supply_demo.json")

    assert canonical["demand_sample_name"].eq("observed_core").all()
    assert set(combined["demand_sample_name"].unique()) == {
        "broad_complete",
        "observed_core",
        "observed_core_low_impute",
    }
    assert diagnostics["demand_sample_name"] == "observed_core"
    assert diagnostics["demand_specification_profile"] == "household_parsimonious"
    assert diagnostics["scenario_price_nowcast_rows"] == 0
    assert diagnostics["demand_fit_quarantined"] is False
    assert diagnostics["supply_estimation_method"] == supply["estimation_method"]
    assert comparison["selected_headline_sample"] == "observed_core"
    assert comparison["comparison_specification_profile"] == "household_parsimonious"
    assert comparison["samples"]["broad_complete"]["scenario_price_nowcast_rows"] == 4
    assert comparison["samples"]["broad_complete"]["demand_sample_selection_reason"] == "comparison_only"
    assert comparison["samples"]["observed_core"]["demand_sample_selection_reason"] == "observed_core_passes_minimum_support"
    assert comparison["samples"]["broad_complete"]["demand_specification_profile"] == "household_parsimonious"
    assert set(spec_comparison["profiles"]) == {"household_parsimonious", "instrument_only"}
    assert spec_comparison["comparison_specification_profile"] == "household_parsimonious"
    assert "state_price_observation_status" in canonical.columns
    assert "state_price_nowcast" in canonical.columns
    assert "p_alpha_direct_care" in canonical.columns
    assert "wage_alpha_implied" in canonical.columns
    assert "demand_elasticity_signed" in canonical.columns
    assert "solver_demand_elasticity_magnitude" in canonical.columns
    assert canonical["p_alpha_direct_care"].gt(0).all()
    assert decomposition["selected_headline_sample"] == "observed_core"
    assert decomposition["canonical"]["baseline_direct_care_clip_binding_row_share"] > 0
    assert supply["supply_elasticity"] > 0
    assert "0.50" in decomposition["canonical"]["alphas"]
    assert piecewise["demo_sample_name"] == "observed_core"
    assert piecewise["piecewise_method"] == "state_year_piecewise_isoelastic"
    assert (project_paths.processed / "childcare_piecewise_supply_demo.parquet").exists()


def test_simulate_childcare_writes_price_decomposition_sensitivity(project_paths):
    _write_childcare_simulation_inputs(project_paths)

    cli.simulate_childcare(project_paths)

    sensitivity = read_json(
        project_paths.outputs_reports / "childcare_price_decomposition_sensitivity.json"
    )
    assert sensitivity["n_cases"] == 9
    assert len(sensitivity["cases"]) == 9

    # Check that the base/base case matches the canonical decomposition.
    base_case = [
        c for c in sensitivity["cases"]
        if c["staffing_case"] == "base" and c["fringe_case"] == "base"
    ]
    assert len(base_case) == 1
    canonical = read_json(
        project_paths.outputs_reports / "childcare_price_decomposition.json"
    )
    # Base/base baseline price should match canonical baseline price.
    assert abs(
        base_case[0]["baseline_price_p50"] - canonical["canonical"]["baseline_price_p50"]
    ) < 0.01

    # Low staffing (more workers per child) should yield higher direct-care prices
    # than high staffing (fewer workers per child), holding fringe constant.
    low_staff_base_fringe = [
        c for c in sensitivity["cases"]
        if c["staffing_case"] == "low" and c["fringe_case"] == "base"
    ][0]
    high_staff_base_fringe = [
        c for c in sensitivity["cases"]
        if c["staffing_case"] == "high" and c["fringe_case"] == "base"
    ][0]
    assert (
        low_staff_base_fringe["baseline_direct_care_price_p50"]
        >= high_staff_base_fringe["baseline_direct_care_price_p50"]
    )

    # Higher fringe should yield higher direct-care prices, holding staffing constant.
    base_staff_low_fringe = [
        c for c in sensitivity["cases"]
        if c["staffing_case"] == "base" and c["fringe_case"] == "low"
    ][0]
    base_staff_high_fringe = [
        c for c in sensitivity["cases"]
        if c["staffing_case"] == "base" and c["fringe_case"] == "high"
    ][0]
    assert (
        base_staff_high_fringe["baseline_direct_care_price_p50"]
        >= base_staff_low_fringe["baseline_direct_care_price_p50"]
    )


def test_recompute_decomposition_preserves_gross_prices(project_paths):
    """Sensitivity sweep must not change gross prices."""
    frame = pd.DataFrame(
        [
            {
                "state_fips": "01",
                "year": 2015,
                "alpha": 0.50,
                "p_baseline": 11000.0,
                "p_alpha": 11500.0,
                "effective_children_per_worker": 5.5,
                "direct_care_fringe_multiplier": 1.20,
                "direct_care_labor_share": 0.44,
            },
        ]
    )

    result = cli._recompute_decomposition_under_assumptions(
        frame,
        0.80,
        1.35,
        childcare_model_assumptions(project_paths),
    )

    # Gross baseline and alpha prices are untouched.
    assert result["baseline_price_p50"] == 11000.0
    assert result["alphas"]["0.50"]["price_p50"] == 11500.0
    # Direct-care price must not exceed gross.
    assert result["baseline_direct_care_price_p50"] <= 11000.0
    assert result["alphas"]["0.50"]["direct_care_price_p50"] <= 11500.0


def test_simulate_childcare_dual_shift_writes_additive_outputs_without_mutating_canonical(project_paths):
    _write_childcare_simulation_inputs(project_paths)

    cli.simulate_childcare(project_paths)
    canonical_before = read_parquet(project_paths.processed / "childcare_marketization_scenarios.parquet")
    diagnostics_before = read_json(project_paths.outputs_reports / "childcare_scenario_diagnostics.json")

    cli.simulate_childcare_dual_shift(project_paths)

    canonical_after = read_parquet(project_paths.processed / "childcare_marketization_scenarios.parquet")
    diagnostics_after = read_json(project_paths.outputs_reports / "childcare_scenario_diagnostics.json")
    dual_shift = read_parquet(project_paths.processed / "childcare_dual_shift_marketization_scenarios.parquet")
    dual_shift_summary = read_json(project_paths.outputs_reports / "childcare_dual_shift_summary.json")
    dual_shift_table = pd.read_csv(project_paths.outputs_tables / "childcare_dual_shift_headline_alpha.csv")

    pd.testing.assert_frame_equal(canonical_before, canonical_after)
    assert diagnostics_before == diagnostics_after
    assert set(dual_shift["solver_family"].unique()) == {"short_run_fixed_supply", "medium_run_dual_shift"}
    assert dual_shift["headline_alpha_flag"].any()
    assert dual_shift["p_alpha_delta_vs_baseline"].notna().all()
    assert dual_shift["p_alpha_pct_change_vs_baseline"].notna().all()
    assert dual_shift_summary["headline_alpha"] == 0.5
    assert len(dual_shift_table) == 35
    assert (project_paths.outputs_figures / "childcare_dual_shift_frontier.svg").exists()


def test_simulate_childcare_fails_without_defensible_observed_core(project_paths):
    _write_childcare_simulation_inputs(project_paths)
    write_json(
        {
            "samples": {
                "broad_complete": {
                    "n_obs": 228,
                    "n_states": 22,
                    "n_years": 15,
                },
                "observed_core": {
                    "n_obs": 20,
                    "n_states": 5,
                    "n_years": 3,
                },
                "observed_core_low_impute": {
                    "n_obs": 10,
                    "n_states": 4,
                    "n_years": 2,
                },
            }
        },
        project_paths.outputs_reports / "childcare_demand_sample_comparison.json",
    )

    with pytest.raises(UnpaidWorkError, match="only exploratory samples are available"):
        cli.simulate_childcare(project_paths)


def test_simulate_childcare_quarantines_inadmissible_comparison_fit(project_paths):
    _write_childcare_simulation_inputs(project_paths)
    write_json(
        {
            "mode": "broad_complete",
            "specification_profile": "household_parsimonious",
            "price_coefficient": 0.001,
            "elasticity_at_mean": 0.14,
            "first_stage_r2": 0.79,
            "n_obs": 12,
            "n_states": 2,
            "n_years": 3,
            "year_min": 2009,
            "year_max": 2023,
            "economically_admissible": False,
        },
        project_paths.outputs_reports / "childcare_demand_iv_broad_complete_household_parsimonious.json",
    )

    cli.simulate_childcare(project_paths)

    comparison = read_json(project_paths.outputs_reports / "childcare_scenario_sample_comparison.json")
    combined = read_parquet(project_paths.processed / "childcare_marketization_scenarios_all_samples.parquet")

    assert comparison["samples"]["broad_complete"]["demand_fit_quarantined"] is True
    assert "positive price response" in comparison["samples"]["broad_complete"]["demand_fit_quarantine_reason"]
    assert set(combined["demand_sample_name"].unique()) == {"observed_core", "observed_core_low_impute"}


def test_report_writes_figure_assets(project_paths):
    _write_childcare_simulation_inputs(project_paths)
    cli.simulate_childcare(project_paths)
    write_json(
        {"holdout_year": 2022, "rmse_test": 123.4},
        project_paths.outputs_reports / "childcare_price_surface.json",
    )
    write_json(
        {
            "state_year_rows": 3,
            "county_year_rows": 2,
            "births_cdc_wonder_observed": 2,
            "state_controls_acs_observed": 2,
            "county_controls_acs_direct": 1,
            "county_wage_observed": 1,
            "county_employment_observed": 1,
            "county_laus_observed": 2,
        },
        project_paths.outputs_reports / "childcare_pipeline_diagnostics.json",
    )
    shock_panel_path = project_paths.interim / "licensing" / "licensing_supply_shocks.parquet"
    write_parquet(
        pd.DataFrame(
            [
                {
                    "state_fips": "01",
                    "year": 2017,
                    "center_labor_intensity_index": 0.20,
                    "center_labor_intensity_shock": 0.0,
                },
                {
                    "state_fips": "01",
                    "year": 2019,
                    "center_labor_intensity_index": 0.24,
                    "center_labor_intensity_shock": 0.04,
                },
            ]
        ),
        shock_panel_path,
    )
    write_json(
        {
            "status": "ok",
            "pilot_scope": "single_state_pilot",
            "sample_mode": False,
            "n_obs": 364,
            "n_counties": 91,
            "n_states": 1,
            "year_min": 2017,
            "year_max": 2022,
            "shock_state_count": 1,
            "treated_state_fips": ["01"],
            "first_stage_strength_flag": "weak_or_unknown",
            "shock_panel_path": str(shock_panel_path),
            "first_stage_price": {"beta": 1.868},
            "reduced_form_provider_density": {"beta": 29.796},
            "reduced_form_employer_establishment_density": {"beta": 0.0},
            "iv_supply_elasticity_provider_density": 15.951,
            "local_iv_supply_elasticity_provider_density": 15.951,
            "secondary_supply_estimate": {
                "name": "local_iv_supply_elasticity_provider_density",
                "value": 15.951,
                "scope": "treated_state_local_wald",
            },
        },
        project_paths.outputs_reports / "childcare_supply_iv.json",
    )

    cli.report(project_paths)

    assert (project_paths.outputs_reports / "childcare_mvp_report.md").exists()
    assert (project_paths.outputs_reports / "childcare_headline_summary.json").exists()
    assert (project_paths.outputs_reports / "childcare_headline_readout.md").exists()
    assert (project_paths.outputs_reports / "childcare_satellite_account.json").exists()
    assert (project_paths.outputs_reports / "childcare_satellite_account.md").exists()
    assert (project_paths.outputs_tables / "childcare_satellite_account_annual.csv").exists()
    assert (project_paths.outputs_reports / "model_assumption_audit.json").exists()
    assert (project_paths.outputs_figures / "childcare_marketization_diagram.svg").exists()
    assert (project_paths.outputs_figures / "childcare_sample_ladder.svg").exists()
    assert (project_paths.outputs_figures / "childcare_alpha_intervals.svg").exists()
    assert (project_paths.outputs_figures / "childcare_price_decomposition_by_alpha.svg").exists()
    assert (project_paths.outputs_figures / "childcare_alpha_examples.svg").exists()
    assert (project_paths.outputs_figures / "childcare_solver_implied_curves.svg").exists()
    assert (project_paths.outputs_figures / "childcare_support_boundary.svg").exists()
    assert (project_paths.outputs_figures / "childcare_scenario_specification_comparison.svg").exists()
    assert (project_paths.outputs_figures / "childcare_piecewise_supply_demo.svg").exists()
    assert (project_paths.outputs_figures / "childcare_supply_iv_pilot.svg").exists()
    assert (project_paths.outputs_figures / "childcare_local_iv_marketization_demo.svg").exists()
    manifest = (project_paths.outputs_figures / "figure_manifest.md").read_text(encoding="utf-8")
    report = (project_paths.outputs_reports / "childcare_mvp_report.md").read_text(encoding="utf-8")
    readout = (project_paths.outputs_reports / "childcare_headline_readout.md").read_text(encoding="utf-8")
    satellite = read_json(project_paths.outputs_reports / "childcare_satellite_account.json")
    satellite_md = (project_paths.outputs_reports / "childcare_satellite_account.md").read_text(encoding="utf-8")
    assert "Stylized marketization diagram" in manifest
    assert "Price decomposition by alpha" in manifest
    assert "Alpha examples with implied wages" in manifest
    assert "Solver-implied supply and demand curves" in manifest
    assert "Piecewise supply demo" in manifest
    assert "Supply IV pilot" in manifest
    assert "Local IV-informed marketization demo" in manifest
    assert "## Piecewise supply demo" in report
    assert "## National childcare benchmarks" in report
    assert "secondary local IV supply elasticity (provider density)" in report
    assert "Childcare Headline Readout" in readout
    assert satellite["default_methodology"] == "annual_hours_childcare_account"
    assert satellite["preferred_series"] == "direct_care_total_value"
    assert "Top-level preferred_series" in satellite["compatibility_note"]
    assert satellite["latest_year"] == satellite["benchmark_methodologies"]["annual_hours_childcare_account"]["latest_year"]
    assert satellite["latest_year"]["preferred_value"] > 0.0
    assert satellite["latest_year"]["price_support_population_share"] < 1.0
    assert "active_care_bridge_benchmark" in satellite["benchmark_methodologies"]
    assert "annual" in satellite["benchmark_methodologies"]["active_care_bridge_benchmark"]
    assert "annual" in satellite["benchmark_methodologies"]["annual_hours_childcare_account"]
    assert "hourly replacement price x unpaid annual childcare hours" in satellite_md
    assert "Active-care bridge benchmark (scaled to under-5 population)" in satellite_md


def test_mode_only_commands_accept_real_and_sample_flags():
    parser = cli.build_parser()

    for command in ("fit-childcare", "simulate-childcare", "simulate-childcare-dual-shift", "fit-home", "report"):
        sample_args = parser.parse_args([command, "--sample"])
        real_args = parser.parse_args([command, "--real"])

        assert sample_args.command == command
        assert sample_args.sample is True
        assert sample_args.real is False
        assert real_args.command == command
        assert real_args.real is True
        assert real_args.sample is False


def test_fit_childcare_real_rejects_sample_built_processed_panels(project_paths):
    county_path = project_paths.processed / "childcare_county_year_price_panel.parquet"
    state_path = project_paths.processed / "childcare_state_year_panel.parquet"
    write_parquet(pd.DataFrame([{"state_fips": "01"}]), county_path)
    write_parquet(pd.DataFrame([{"state_fips": "01"}]), state_path)
    _write_mode_provenance(project_paths, county_path, sample_mode=True)
    _write_mode_provenance(project_paths, state_path, sample_mode=True)

    with pytest.raises(UnpaidWorkError, match="processed childcare county-year panel was built with sample data"):
        cli.fit_childcare(project_paths, sample=False)


def test_build_childcare_real_refreshes_sample_built_interim_core_sources(project_paths, monkeypatch):
    core_sources = {
        "atus": pd.DataFrame(
            [
                {
                    "active_childcare_hours": 1.0,
                    "active_household_childcare_hours": 1.0,
                    "active_nonhousehold_childcare_hours": 0.0,
                    "supervisory_childcare_hours": 0.5,
                }
            ]
        ),
        "acs": pd.DataFrame(
            [
                {
                    "year": 2021,
                    "state_fips": "06",
                    "county_fips": "06037",
                    "under5_population": 10,
                    "under5_male_population": 5,
                    "under5_female_population": 5,
                }
            ]
        ),
        "oews": pd.DataFrame(
            [
                {
                    "oews_childcare_worker_wage": 1.0,
                    "oews_preschool_teacher_wage": 2.0,
                    "oews_outside_option_wage": 3.0,
                }
            ]
        ),
        "ndcp": pd.DataFrame([{"year": 2021}]),
        "qcew": pd.DataFrame([{"year": 2021}]),
        "cdc_wonder": pd.DataFrame([{"year": 2021}]),
        "head_start": pd.DataFrame([{"year": 2021}]),
        "nces_ccd": pd.DataFrame([{"year": 2021}]),
        "cbp": pd.DataFrame([{"year": 2021}]),
        "nes": pd.DataFrame([{"year": 2021}]),
        "laus": pd.DataFrame([{"year": 2021}]),
    }
    for name, frame in core_sources.items():
        path = project_paths.interim / name / f"{name}.parquet"
        write_parquet(frame, path)
    _write_registry_mode(
        project_paths,
        project_paths.interim / "atus" / "atus.parquet",
        sample_mode=True,
        last_fetched="2026-03-30T17:53:23+00:00",
    )

    observed: dict[str, object] = {}

    def _fake_pull_core(paths, sample, refresh=False, dry_run=False, year=None, ingestors=None):
        observed["sample"] = sample
        observed["refresh"] = refresh
        observed["year"] = year
        observed["ingestors"] = ingestors

    monkeypatch.setattr(cli, "pull_core", _fake_pull_core)
    monkeypatch.setattr(
        cli,
        "build_childcare_panels",
        lambda paths: (
            pd.DataFrame([{"county_fips": "06037", "state_fips": "06", "year": 2021}]),
            pd.DataFrame([{"state_fips": "06", "year": 2021}]),
        ),
    )
    monkeypatch.setattr(cli, "apply_replacement_cost", lambda frame, *_args, **_kwargs: frame)
    monkeypatch.setattr(
        cli,
        "diagnose_childcare_pipeline",
        lambda county, state, paths: {
            "state_controls_acs_observed": 1,
            "state_year_rows": 1,
            "births_cdc_wonder_observed": 1,
        },
    )
    monkeypatch.setattr(
        cli.acs,
        "ingest_year_range",
        lambda *args, **kwargs: types.SimpleNamespace(detail="ok"),
    )
    monkeypatch.setattr(
        cli.qcew,
        "ingest_year_range",
        lambda *args, **kwargs: types.SimpleNamespace(detail="ok"),
    )
    monkeypatch.setattr(
        cli.laus,
        "ingest_year_range",
        lambda *args, **kwargs: types.SimpleNamespace(detail="ok"),
    )

    cli.build_childcare(project_paths, sample=False)

    assert observed["sample"] is False
    assert observed["refresh"] is True
    assert observed["year"] is None
    assert observed["ingestors"] is not None


def test_build_childcare_real_drops_sample_laus_when_refresh_fails(project_paths, monkeypatch):
    core_sources = {
        "atus": pd.DataFrame(
            [
                {
                    "active_childcare_hours": 1.0,
                    "active_household_childcare_hours": 1.0,
                    "active_nonhousehold_childcare_hours": 0.0,
                    "supervisory_childcare_hours": 0.5,
                }
            ]
        ),
        "acs": pd.DataFrame(
            [
                {
                    "year": 2021,
                    "state_fips": "06",
                    "county_fips": "06037",
                    "under5_population": 10,
                    "under5_male_population": 5,
                    "under5_female_population": 5,
                }
            ]
        ),
        "oews": pd.DataFrame(
            [
                {
                    "oews_childcare_worker_wage": 1.0,
                    "oews_preschool_teacher_wage": 2.0,
                    "oews_outside_option_wage": 3.0,
                }
            ]
        ),
        "ndcp": pd.DataFrame([{"year": 2021}]),
        "qcew": pd.DataFrame([{"year": 2021}]),
        "cdc_wonder": pd.DataFrame([{"year": 2021}]),
        "head_start": pd.DataFrame([{"year": 2021}]),
        "nces_ccd": pd.DataFrame([{"year": 2021}]),
        "cbp": pd.DataFrame([{"year": 2021}]),
        "nes": pd.DataFrame([{"year": 2021}]),
        "laus": pd.DataFrame([{"year": 2021, "geography": "state", "state_fips": "06"}]),
    }
    for name, frame in core_sources.items():
        path = project_paths.interim / name / f"{name}.parquet"
        write_parquet(frame, path)
    _write_registry_mode(
        project_paths,
        project_paths.interim / "atus" / "atus.parquet",
        sample_mode=True,
        last_fetched="2026-03-30T17:53:23+00:00",
    )
    _write_registry_mode(
        project_paths,
        project_paths.interim / "laus" / "laus.parquet",
        sample_mode=True,
        last_fetched="2026-03-30T17:53:23+00:00",
    )

    monkeypatch.setattr(cli, "pull_core", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "build_childcare_panels",
        lambda paths: (
            pd.DataFrame([{"county_fips": "06037", "state_fips": "06", "year": 2021}]),
            pd.DataFrame([{"state_fips": "06", "year": 2021}]),
        ),
    )
    monkeypatch.setattr(cli, "apply_replacement_cost", lambda frame, *_args, **_kwargs: frame)
    monkeypatch.setattr(
        cli,
        "diagnose_childcare_pipeline",
        lambda county, state, paths: {
            "state_controls_acs_observed": 1,
            "state_year_rows": 1,
            "births_cdc_wonder_observed": 1,
        },
    )
    monkeypatch.setattr(
        cli.acs,
        "ingest_year_range",
        lambda *args, **kwargs: types.SimpleNamespace(detail="ok"),
    )
    monkeypatch.setattr(
        cli.qcew,
        "ingest_year_range",
        lambda *args, **kwargs: types.SimpleNamespace(detail="ok"),
    )

    def _raise_laus(*args, **kwargs):
        raise cli.SourceAccessError("blocked")

    monkeypatch.setattr(cli.laus, "ingest_year_range", _raise_laus)

    laus_path = project_paths.interim / "laus" / "laus.parquet"
    assert laus_path.exists()

    cli.build_childcare(project_paths, sample=False)

    assert not laus_path.exists()


def test_simulate_childcare_real_rejects_sample_built_demand_summary(project_paths):
    _write_childcare_simulation_inputs(project_paths)

    state_path = project_paths.processed / "childcare_state_year_panel.parquet"
    county_path = project_paths.processed / "childcare_county_year_price_panel.parquet"
    comparison_path = project_paths.outputs_reports / "childcare_demand_sample_comparison.json"
    broad_path = project_paths.outputs_reports / "childcare_demand_iv_broad_complete_household_parsimonious.json"
    observed_path = project_paths.outputs_reports / "childcare_demand_iv_observed_core_household_parsimonious.json"
    low_impute_path = project_paths.outputs_reports / "childcare_demand_iv_observed_core_low_impute_household_parsimonious.json"
    instrument_only_path = project_paths.outputs_reports / "childcare_demand_iv_observed_core_instrument_only.json"

    for path in (state_path, county_path, comparison_path, broad_path, low_impute_path, instrument_only_path):
        _write_mode_provenance(project_paths, path, sample_mode=False)
    _write_mode_provenance(project_paths, observed_path, sample_mode=True)

    with pytest.raises(UnpaidWorkError, match="childcare demand summary for observed_core was built with sample data"):
        cli.simulate_childcare(project_paths, sample=False)


def test_simulate_childcare_real_persists_failing_diagnostics_before_gate(project_paths, monkeypatch):
    _write_childcare_simulation_inputs(project_paths)

    state_path = project_paths.processed / "childcare_state_year_panel.parquet"
    county_path = project_paths.processed / "childcare_county_year_price_panel.parquet"
    comparison_path = project_paths.outputs_reports / "childcare_demand_sample_comparison.json"
    broad_path = project_paths.outputs_reports / "childcare_demand_iv_broad_complete_household_parsimonious.json"
    observed_path = project_paths.outputs_reports / "childcare_demand_iv_observed_core_household_parsimonious.json"
    low_impute_path = project_paths.outputs_reports / "childcare_demand_iv_observed_core_low_impute_household_parsimonious.json"
    instrument_only_path = project_paths.outputs_reports / "childcare_demand_iv_observed_core_instrument_only.json"

    for path in (state_path, county_path, comparison_path, broad_path, observed_path, low_impute_path, instrument_only_path):
        _write_mode_provenance(project_paths, path, sample_mode=False)

    def _fake_simulate_sample(paths, state, county, alphas, demand_summary, sample_name, sample, selection_reason="comparison_only", specification_profile=None):
        scenarios = pd.DataFrame(
            [
                {
                    "demand_sample_name": sample_name,
                    "alpha": 0.5,
                    "p_baseline": 100.0,
                    "p_alpha": 110.0,
                }
            ]
        )
        diagnostics = {
            "current_mode": "real",
            "demand_sample_name": sample_name,
            "demand_specification_profile": specification_profile or "household_parsimonious",
            "demand_instrument": "outside_option_wage",
            "scenario_rows": 1,
            "scenario_states": 1,
            "skipped_state_rows": 0,
            "bootstrap_acceptance_rate": 0.78 if sample_name == "observed_core" else 0.95,
            "bootstrap_draws_requested": 100,
            "bootstrap_draws_accepted": 78 if sample_name == "observed_core" else 95,
            "bootstrap_draws_rejected": 22 if sample_name == "observed_core" else 5,
        }
        return scenarios, diagnostics

    monkeypatch.setattr(cli, "_simulate_childcare_sample", _fake_simulate_sample)
    monkeypatch.setattr(cli, "_selected_sample_specification_profiles", lambda paths, selected_sample: [])

    with pytest.raises(UnpaidWorkError, match="below the headline threshold"):
        cli.simulate_childcare(project_paths, sample=False)

    diagnostics = read_json(project_paths.outputs_reports / "childcare_scenario_diagnostics.json")
    assert diagnostics["current_mode"] == "real"
    assert diagnostics["demand_sample_name"] == "observed_core"
    assert diagnostics["bootstrap_acceptance_rate"] == pytest.approx(0.78)
    assert diagnostics["headline_gate_passed"] is False


def test_report_real_rejects_sample_built_scenarios(project_paths):
    scenarios_path = project_paths.processed / "childcare_marketization_scenarios.parquet"
    state_path = project_paths.processed / "childcare_state_year_panel.parquet"
    county_path = project_paths.processed / "childcare_county_year_price_panel.parquet"
    write_parquet(pd.DataFrame([{"state_fips": "01", "year": 2022, "alpha": 0.5, "p_alpha": 100.0}]), scenarios_path)
    write_parquet(pd.DataFrame([{"state_fips": "01"}]), state_path)
    write_parquet(pd.DataFrame([{"state_fips": "01"}]), county_path)
    _write_mode_provenance(project_paths, scenarios_path, sample_mode=True)
    _write_mode_provenance(project_paths, state_path, sample_mode=False)
    _write_mode_provenance(project_paths, county_path, sample_mode=False)

    with pytest.raises(UnpaidWorkError, match="childcare marketization scenarios was built with sample data"):
        cli.report(project_paths, sample=False)
