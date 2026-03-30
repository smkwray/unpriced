from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from unpriced.assumptions import childcare_model_assumptions, solver_assumptions, write_assumption_audit
from unpriced.config import (
    ensure_project_dirs,
    load_extension_config,
    load_project_paths,
    load_yaml,
    resolve_extension_config_path,
)
from unpriced.errors import SourceAccessError, UnpaidWorkError
from unpriced.features.childcare_panel import (
    build_childcare_panels,
    diagnose_childcare_pipeline,
)
from unpriced.features.home_maintenance_panel import build_home_maintenance_panel
from unpriced.ingest import (
    acs,
    ahs,
    atus,
    cbp,
    ccdf,
    cdc_wonder,
    ce,
    head_start,
    laus,
    licensing,
    nces_ccd,
    ndcp,
    nes,
    noaa,
    oews,
    qcew,
    sipp,
)
from unpriced.ingest.provenance import read_provenance_sidecar, write_provenance_sidecar
from unpriced.ingest.acs import ACS_FIRST_AVAILABLE_YEAR
from unpriced.ingest.qcew import QCEW_FIRST_AVAILABLE_YEAR
from unpriced.logging import get_logger
from unpriced.models.demand_iv import (
    CANONICAL_SAMPLE_MODES,
    CANONICAL_COMPARISON_SPECIFICATION_PROFILE,
    CANONICAL_OBSERVED_SPECIFICATION_PROFILE,
    DEFAULT_SPECIFICATION_PROFILE,
    build_childcare_demand_sample_comparison,
    build_childcare_imputation_sweep,
    build_childcare_labor_support_sweep,
    build_childcare_specification_sweep,
    fit_childcare_demand_iv,
    select_headline_sample,
)
from unpriced.models.price_surface import fit_price_surface
from unpriced.models.replacement_cost import apply_replacement_cost
from unpriced.models.scenario_solver import (
    MARGINAL_ALPHA,
    SolverMetadata,
    bootstrap_childcare_intervals,
    bootstrap_childcare_dual_shift_headline_table,
    dual_shift_zero_price_frontier,
    prepare_childcare_scenario_inputs,
    resolve_solver_demand_elasticity,
    solve_alpha_grid,
    solve_alpha_grid_dual_shift,
    solve_alpha_grid_piecewise_supply,
    solve_price,
    solve_price_dual_shift,
    summarize_childcare_scenario_diagnostics,
)
from unpriced.models.supply_curve import (
    summarize_piecewise_supply_curve,
    summarize_supply_elasticity,
)
from unpriced.models.supply_iv import build_supply_iv_panel, fit_supply_iv_exposure_design
from unpriced.models.switching import fit_home_switching
from unpriced.registry import ensure_registry
from unpriced.reports.export import (
    build_childcare_headline_summary,
    build_childcare_satellite_account,
    build_markdown_report,
)
from unpriced.reports.figures import (
    write_childcare_dual_shift_figure,
    write_childcare_figure_manifest,
)
from unpriced.reports.tables import summarize_scenarios
from unpriced.storage import read_json, read_parquet, write_json, write_parquet

LOGGER = get_logger()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _core_ingestors():
    return [
        atus.ingest,
        ndcp.ingest,
        ahs.ingest,
        qcew.ingest,
        cbp.ingest,
        nes.ingest,
        acs.ingest_with_options,
        cdc_wonder.ingest,
        laus.ingest,
        ce.ingest,
        head_start.ingest,
        nces_ccd.ingest,
        oews.ingest,
        sipp.ingest,
    ]


def _childcare_core_ingestors():
    return [
        atus.ingest,
        ndcp.ingest,
        qcew.ingest,
        cbp.ingest,
        nes.ingest,
        acs.ingest_with_options,
        cdc_wonder.ingest,
        head_start.ingest,
        nces_ccd.ingest,
        oews.ingest,
    ]


def _state_panel_has_sample_ladder(state: pd.DataFrame) -> bool:
    required = {
        "state_controls_source",
        "state_unemployment_source",
        "state_price_support_window",
        "state_price_observation_status",
        "state_price_nowcast",
        "state_ndcp_imputed_share",
        "state_qcew_wage_observed_share",
        "state_qcew_employment_observed_share",
        "state_qcew_labor_observed_share",
        "eligible_broad_complete",
        "eligible_observed_core",
        "eligible_observed_core_low_impute",
        "observed_core_exclusion_reason",
        "observed_core_low_impute_exclusion_reason",
    }
    return required.issubset(state.columns)


def _artifact_mode(path: Path) -> str | None:
    try:
        metadata = read_provenance_sidecar(path)
    except Exception:
        return None
    params = metadata.get("parameters", {})
    if not isinstance(params, dict):
        return None
    sample_mode = params.get("sample_mode", params.get("sample"))
    if isinstance(sample_mode, bool):
        return "sample" if sample_mode else "real"
    return None


def _registry_artifact_mode(paths, path: Path) -> str | None:
    registry_path = getattr(paths, "registry", None)
    if registry_path is None or not Path(registry_path).exists():
        return None
    try:
        registry = read_parquet(Path(registry_path))
    except Exception:
        return None
    if registry.empty or "normalized_path" not in registry.columns or "sample_mode" not in registry.columns:
        return None

    target = str(path.resolve())

    def _matches_target(value: object) -> bool:
        if not isinstance(value, str) or not value:
            return False
        try:
            return str(Path(value).resolve()) == target
        except Exception:
            return False

    matches = registry.loc[registry["normalized_path"].map(_matches_target)].copy()
    if matches.empty:
        return None
    if "last_fetched" in matches.columns:
        matches = matches.sort_values("last_fetched", kind="stable")
    sample_mode = matches.iloc[-1]["sample_mode"]
    if pd.isna(sample_mode):
        return None
    return "sample" if bool(sample_mode) else "real"


def _current_artifact_mode(paths, path: Path) -> str | None:
    mode = _artifact_mode(path)
    if mode is not None:
        return mode
    return _registry_artifact_mode(paths, path)


def _ensure_real_mode_artifact(path: Path, prereq_command: str, artifact_name: str) -> None:
    if not path.exists():
        raise UnpaidWorkError(
            f"{artifact_name} is missing for --real mode. Build with `unpriced {prereq_command}` before rerunning."
        )
    mode = _artifact_mode(path)
    if mode == "sample":
        raise UnpaidWorkError(
            f"{artifact_name} was built with sample data. Rebuild it with `unpriced {prereq_command}` before running in --real mode."
        )
    if mode is None:
        raise UnpaidWorkError(
            f"{artifact_name} does not include mode provenance. Rebuild with `unpriced {prereq_command}` before running in --real mode."
        )


def _ensure_real_mode_artifacts(paths: list[tuple[Path, str]], prereq_command: str) -> None:
    for path, artifact_name in paths:
        _ensure_real_mode_artifact(path, prereq_command, artifact_name)


def _write_mode_artifact(
    path: Path,
    source_files: list[Path],
    sample: bool,
    repo_root: Path,
    extra_parameters: dict[str, object] | None = None,
) -> None:
    provenance_kwargs = {
        "source_files": [artifact for artifact in source_files if artifact.exists()],
        "parameters": {
            "sample_mode": sample,
            **(extra_parameters or {}),
        },
        "repo_root": repo_root,
    }
    write_provenance_sidecar(path, **provenance_kwargs)


def _parquet_has_columns(path: Path, required_columns: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        columns = set(read_parquet(path).columns)
    except Exception:
        return False
    return required_columns.issubset(columns)


def _refresh_childcare_panels_from_interim(paths, sample: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    county, state = build_childcare_panels(paths)
    state = apply_replacement_cost(state, "unpaid_childcare_hours", "benchmark_childcare_wage")
    county_path = paths.processed / "childcare_county_year_price_panel.parquet"
    state_path = paths.processed / "childcare_state_year_panel.parquet"
    static_hours_path = paths.processed / "childcare_state_year_static_hours.parquet"
    write_parquet(state, state_path)
    diagnose_childcare_pipeline(county, state, paths)
    source_files = [
        paths.interim / name / f"{name}.parquet"
        for name in ("atus", "ndcp", "qcew", "acs", "cdc_wonder", "head_start", "nces_ccd", "oews", "cbp", "nes", "laus")
    ]
    _write_mode_artifact(county_path, source_files, sample=sample, repo_root=paths.root, extra_parameters={"command": "build-childcare"})
    _write_mode_artifact(state_path, source_files, sample=sample, repo_root=paths.root, extra_parameters={"command": "build-childcare"})
    if static_hours_path.exists():
        _write_mode_artifact(static_hours_path, source_files, sample=sample, repo_root=paths.root, extra_parameters={"command": "build-childcare"})
    return county, state


def bootstrap(paths) -> None:
    ensure_project_dirs(paths)
    ensure_registry(paths)
    LOGGER.info("bootstrap complete")


def pull_core(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    ingestors=None,
) -> None:
    for ingestor in (ingestors or _core_ingestors()):
        result = ingestor(paths, sample=sample, refresh=refresh, dry_run=dry_run, year=year)
        LOGGER.info(
            "%s %s -> %s (%s)",
            "planned" if result.dry_run else "ingested",
            result.source_name,
            result.normalized_path,
            result.detail or ("skipped" if result.skipped else "ok"),
        )


def build_childcare(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> None:
    county_path = paths.processed / "childcare_county_year_price_panel.parquet"
    state_path = paths.processed / "childcare_state_year_panel.parquet"
    static_hours_path = paths.processed / "childcare_state_year_static_hours.parquet"
    core_sources = [
        paths.interim / name / f"{name}.parquet"
        for name in ("atus", "ndcp", "qcew", "acs", "cdc_wonder", "head_start", "nces_ccd", "oews", "cbp", "nes", "laus")
    ]
    sample_built_core = (
        not sample and any(path.exists() and _current_artifact_mode(paths, path) == "sample" for path in core_sources)
    )
    stale_schema = not _parquet_has_columns(
        paths.interim / "atus" / "atus.parquet",
        {
            "active_childcare_hours",
            "active_household_childcare_hours",
            "active_nonhousehold_childcare_hours",
            "supervisory_childcare_hours",
        },
    ) or not _parquet_has_columns(
        paths.interim / "acs" / "acs.parquet",
        {"under5_population", "under5_male_population", "under5_female_population"},
    ) or not _parquet_has_columns(
        paths.interim / "oews" / "oews.parquet",
        {"oews_childcare_worker_wage", "oews_preschool_teacher_wage", "oews_outside_option_wage"},
    )
    force_core_refresh = refresh or stale_schema or sample_built_core
    if any(not path.exists() for path in core_sources) or stale_schema or sample_built_core:
        if sample_built_core:
            LOGGER.info("detected sample-built childcare core inputs during --real build; forcing real refresh")
        pull_core(
            paths,
            sample=sample,
            refresh=force_core_refresh,
            dry_run=dry_run,
            year=year,
            ingestors=_childcare_core_ingestors(),
        )
    if dry_run:
        LOGGER.info("planned childcare build")
        return
    # Ensure ACS, QCEW, and LAUS cover the NDCP year range.
    if not sample:
        ndcp_path = paths.interim / "ndcp" / "ndcp.parquet"
        laus_path = paths.interim / "laus" / "laus.parquet"
        if ndcp_path.exists():
            ndcp_frame = read_parquet(ndcp_path)
            ndcp_years = sorted(ndcp_frame["year"].dropna().astype(int).unique())
            if ndcp_years:
                ndcp_start = int(ndcp_years[0])
                ndcp_end = int(ndcp_years[-1])
                acs_result = acs.ingest_year_range(
                    paths,
                    max(ndcp_start, ACS_FIRST_AVAILABLE_YEAR),
                    ndcp_end,
                    refresh=force_core_refresh,
                )
                LOGGER.info("ACS year-range check: %s", acs_result.detail or "ok")
                qcew_result = qcew.ingest_year_range(
                    paths,
                    max(ndcp_start, QCEW_FIRST_AVAILABLE_YEAR),
                    ndcp_end,
                    refresh=force_core_refresh,
                )
                LOGGER.info("QCEW year-range check: %s", qcew_result.detail or "ok")
                try:
                    laus_result = laus.ingest_year_range(paths, ndcp_start, ndcp_end, refresh=force_core_refresh)
                    LOGGER.info("LAUS year-range check: %s", laus_result.detail or "ok")
                except SourceAccessError as exc:
                    laus_mode = _current_artifact_mode(paths, laus_path)
                    if laus_mode == "sample" and laus_path.exists():
                        laus_path.unlink()
                    LOGGER.warning(
                        "LAUS year-range check failed (%s); continuing without refreshed LAUS controls",
                        exc,
                    )
    county, state = build_childcare_panels(paths)
    state = apply_replacement_cost(state, "unpaid_childcare_hours", "benchmark_childcare_wage")
    write_parquet(state, state_path)
    diag = diagnose_childcare_pipeline(county, state, paths)
    _write_mode_artifact(county_path, core_sources, sample=sample, repo_root=paths.root, extra_parameters={"command": "build-childcare"})
    _write_mode_artifact(state_path, core_sources, sample=sample, repo_root=paths.root, extra_parameters={"command": "build-childcare"})
    if static_hours_path.exists():
        _write_mode_artifact(static_hours_path, core_sources, sample=sample, repo_root=paths.root, extra_parameters={"command": "build-childcare"})
    pipeline_path = paths.outputs_reports / "childcare_pipeline_diagnostics.json"
    if pipeline_path.exists():
        _write_mode_artifact(pipeline_path, [county_path, state_path], sample=sample, repo_root=paths.root, extra_parameters={"command": "build-childcare"})
    LOGGER.info(
        "built childcare panels: county=%s state=%s (observed_controls=%s/%s, births_observed=%s/%s)",
        len(county),
        len(state),
        diag["state_controls_acs_observed"],
        diag["state_year_rows"],
        diag["births_cdc_wonder_observed"],
        diag["state_year_rows"],
    )


def _output_namespace_path(paths, namespace: str | Path) -> Path:
    target = Path(namespace)
    return target if target.is_absolute() else paths.root / target


def _read_optional_parquet(path: Path) -> pd.DataFrame | None:
    return read_parquet(path) if path.exists() else None


def _read_optional_json(path: Path) -> dict[str, object] | None:
    return read_json(path) if path.exists() else None


def _write_csv_output(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_json_records(frame: pd.DataFrame, path: Path, record_key: str = "rows") -> Path:
    return write_json({record_key: frame.to_dict(orient="records")}, path)


def _load_pooled_backend_summary(paths, sample: bool, refresh: bool = False) -> dict[str, object]:
    headline_path = paths.outputs_reports / "childcare_headline_summary.json"
    demand_path = paths.outputs_reports / "childcare_demand_iv.json"
    state_path = paths.processed / "childcare_state_year_panel.parquet"
    if headline_path.exists():
        return read_json(headline_path)
    if demand_path.exists():
        return read_json(demand_path)
    if not state_path.exists():
        build_childcare(paths, sample=sample, refresh=refresh, dry_run=False, year=None)
    state = read_parquet(state_path)
    if state.empty:
        return {
            "selected_headline_sample": "missing",
            "mode": "sample" if sample else "real",
            "n_obs": 0,
            "n_states": 0,
            "year_min": None,
            "year_max": None,
        }
    years = pd.to_numeric(state["year"], errors="coerce").dropna().astype(int)
    return {
        "selected_headline_sample": "state_panel_fallback",
        "mode": "sample" if sample else "real",
        "n_obs": int(len(state)),
        "n_states": int(state["state_fips"].astype(str).nunique()) if "state_fips" in state.columns else 0,
        "year_min": int(years.min()) if not years.empty else None,
        "year_max": int(years.max()) if not years.empty else None,
    }


def build_childcare_segments(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "segmented_solver",
) -> None:
    config_path = resolve_extension_config_path(paths.root, config_name_or_path)
    config = load_extension_config(paths.root, config_name_or_path)
    if not config:
        raise UnpaidWorkError(f"extension config is empty or missing: {config_path}")

    ndcp_path = paths.interim / "ndcp" / "ndcp.parquet"
    if not ndcp_path.exists():
        result = ndcp.ingest(paths, sample=sample, refresh=refresh, dry_run=dry_run, year=year)
        LOGGER.info(
            "%s %s -> %s (%s)",
            "planned" if result.dry_run else "ingested",
            result.source_name,
            result.normalized_path,
            result.detail or ("skipped" if result.skipped else "ok"),
        )
    if dry_run:
        LOGGER.info("planned segmented childcare build using %s", config_path)
        return
    if not ndcp_path.exists():
        raise UnpaidWorkError(f"missing normalized NDCP input: {ndcp_path}")

    from unpriced.childcare.segmentation import (
        build_ndcp_segment_price_panel,
        build_pooled_ndcp_price_benchmark,
        build_segment_definitions,
        build_segment_to_pooled_mapping,
    )

    output_config = config.get("output_namespace", {})
    output_namespace = _output_namespace_path(
        paths,
        str(output_config.get("namespace", "data/interim/childcare/segmented_solver")),
    )
    file_names = output_config.get("files", {})
    segment_definitions_path = output_namespace / file_names.get(
        "segment_definitions", "segment_definitions.parquet"
    )
    segment_price_panel_path = output_namespace / file_names.get(
        "segment_price_panel", "ndcp_segment_prices.parquet"
    )
    segment_mapping_path = output_namespace / file_names.get(
        "segment_mappings", "segmented_to_pooled_mapping.parquet"
    )

    ndcp_frame = read_parquet(ndcp_path)
    segment_definitions = build_segment_definitions(config)
    segment_prices = build_ndcp_segment_price_panel(ndcp_frame, segment_definitions)
    pooled_benchmark = build_pooled_ndcp_price_benchmark(ndcp_frame)
    segment_mapping = build_segment_to_pooled_mapping(segment_prices, pooled_benchmark=pooled_benchmark)

    compatibility = config.get("build", {}).get("compatibility", {})
    if compatibility.get("enabled", False) and not segment_mapping.empty:
        tolerance = float(compatibility.get("tolerance", 1.0e-6))
        max_gap = float(pd.to_numeric(segment_mapping.get("pooled_price_gap"), errors="coerce").abs().max())
        if pd.notna(max_gap) and max_gap > tolerance:
            raise UnpaidWorkError(
                f"segmented childcare compatibility check failed: max pooled price gap {max_gap:.6g} exceeds tolerance {tolerance:.6g}"
            )

    write_parquet(segment_definitions, segment_definitions_path)
    write_parquet(segment_prices, segment_price_panel_path)
    write_parquet(segment_mapping, segment_mapping_path)

    provenance_kwargs = {
        "source_files": [ndcp_path, config_path],
        "parameters": {
            "sample_mode": sample,
            "requested_year": year,
            "extension_name": config.get("name", "segmented_solver"),
            "mode": config.get("mode", "segmented_baseline"),
        },
        "config": config,
        "repo_root": paths.root,
    }
    write_provenance_sidecar(segment_definitions_path, **provenance_kwargs)
    write_provenance_sidecar(segment_price_panel_path, **provenance_kwargs)
    write_provenance_sidecar(segment_mapping_path, **provenance_kwargs)
    LOGGER.info(
        "built segmented childcare NDCP baseline: definitions=%s prices=%s mappings=%s",
        len(segment_definitions),
        len(segment_prices),
        len(segment_mapping),
    )


def build_childcare_utilization(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "utilization_stack",
    segmented_config_name_or_path: str = "segmented_solver",
) -> None:
    config_path = resolve_extension_config_path(paths.root, config_name_or_path)
    config = load_extension_config(paths.root, config_name_or_path)
    if not config:
        raise UnpaidWorkError(f"extension config is empty or missing: {config_path}")

    source_requirements = {
        "acs": paths.interim / "acs" / "acs.parquet",
        "sipp": paths.interim / "sipp" / "sipp.parquet",
        "head_start": paths.interim / "head_start" / "head_start.parquet",
    }
    ingestors = {
        "acs": acs.ingest_with_options,
        "sipp": sipp.ingest,
        "head_start": head_start.ingest,
    }
    for source_name, source_path in source_requirements.items():
        if source_path.exists():
            continue
        result = ingestors[source_name](paths, sample=sample, refresh=refresh, dry_run=dry_run, year=year)
        LOGGER.info(
            "%s %s -> %s (%s)",
            "planned" if result.dry_run else "ingested",
            result.source_name,
            result.normalized_path,
            result.detail or ("skipped" if result.skipped else "ok"),
        )

    segmented_output_path = (
        _output_namespace_path(paths, "data/interim/childcare/segmented_solver")
        / "ndcp_segment_prices.parquet"
    )
    ccdf_admin_state_year_path = paths.interim / "ccdf" / "ccdf_admin_state_year.parquet"
    ccdf_policy_features_state_year_path = paths.interim / "ccdf" / "ccdf_policy_features_state_year.parquet"
    ccdf_policy_controls_state_year_path = paths.interim / "ccdf" / "ccdf_policy_controls_state_year.parquet"
    ccdf_policy_promoted_controls_state_year_path = (
        paths.interim / "ccdf" / "ccdf_policy_promoted_controls_state_year.parquet"
    )
    ccdf_long_inputs_exist = (
        (paths.interim / "ccdf" / "ccdf_admin_long.parquet").exists()
        and (paths.interim / "ccdf" / "ccdf_policy_long.parquet").exists()
    )
    if not segmented_output_path.exists():
        build_childcare_segments(
            paths,
            sample=sample,
            refresh=refresh,
            dry_run=dry_run,
            year=year,
            config_name_or_path=segmented_config_name_or_path,
        )
    if sample or ccdf_long_inputs_exist or paths.raw.joinpath("ccdf").exists():
        try:
            build_ccdf_state_year(
                paths,
                sample=sample,
                refresh=refresh,
                dry_run=dry_run,
            )
        except UnpaidWorkError:
            if sample:
                raise
            LOGGER.info("skipping optional CCDF state-year mapper because mapped inputs are unavailable")

    if dry_run:
        LOGGER.info("planned childcare utilization build using %s", config_path)
        return

    for source_name, source_path in source_requirements.items():
        if not source_path.exists():
            raise UnpaidWorkError(f"missing normalized {source_name} input: {source_path}")
    if not segmented_output_path.exists():
        raise UnpaidWorkError(f"missing segmented NDCP input: {segmented_output_path}")

    from unpriced.childcare.utilization import build_childcare_utilization_outputs

    output_config = config.get("output_namespace", {})
    output_namespace = _output_namespace_path(
        paths,
        str(output_config.get("namespace", "data/interim/childcare/utilization_stack")),
    )
    file_names = output_config.get("files", {})
    public_program_slots_path = output_namespace / file_names.get(
        "public_program_slots", "public_program_slots_state_year.parquet"
    )
    survey_targets_path = output_namespace / file_names.get(
        "survey_paid_use_targets", "survey_paid_use_targets.parquet"
    )
    quantity_by_segment_path = output_namespace / file_names.get(
        "quantity_by_segment", "q0_segmented.parquet"
    )
    diagnostics_path = output_namespace / file_names.get(
        "reconciliation_diagnostics", "utilization_reconciliation_diagnostics.parquet"
    )

    outputs = build_childcare_utilization_outputs(
        acs_frame=read_parquet(source_requirements["acs"]),
        sipp_frame=read_parquet(source_requirements["sipp"]),
        head_start_frame=read_parquet(source_requirements["head_start"]),
        ccdf_state_year=read_parquet(ccdf_admin_state_year_path) if ccdf_admin_state_year_path.exists() else None,
        ccdf_policy_features_state_year=read_parquet(ccdf_policy_features_state_year_path)
        if ccdf_policy_features_state_year_path.exists()
        else None,
        ccdf_policy_controls_state_year=(
            read_parquet(ccdf_policy_promoted_controls_state_year_path)
            if ccdf_policy_promoted_controls_state_year_path.exists()
            else (
                read_parquet(ccdf_policy_controls_state_year_path)
                if ccdf_policy_controls_state_year_path.exists()
                else None
            )
        ),
        segment_price_panel=read_parquet(segmented_output_path),
        config=config,
    )
    write_parquet(outputs["public_program_slots"], public_program_slots_path)
    write_parquet(outputs["survey_paid_use_targets"], survey_targets_path)
    write_parquet(outputs["quantity_by_segment"], quantity_by_segment_path)
    write_parquet(outputs["reconciliation_diagnostics"], diagnostics_path)

    provenance_kwargs = {
        "source_files": [
            source_requirements["acs"],
            source_requirements["sipp"],
            source_requirements["head_start"],
            segmented_output_path,
            config_path,
        ]
        + ([ccdf_admin_state_year_path] if ccdf_admin_state_year_path.exists() else [])
        + (
            [ccdf_policy_promoted_controls_state_year_path]
            if ccdf_policy_promoted_controls_state_year_path.exists()
            else ([ccdf_policy_controls_state_year_path] if ccdf_policy_controls_state_year_path.exists() else [])
        )
        + ([ccdf_policy_features_state_year_path] if ccdf_policy_features_state_year_path.exists() else []),
        "parameters": {
            "sample_mode": sample,
            "requested_year": year,
            "extension_name": config.get("name", "utilization_stack"),
            "mode": config.get("mode", "observed_utilization"),
        },
        "config": config,
        "repo_root": paths.root,
    }
    write_provenance_sidecar(public_program_slots_path, **provenance_kwargs)
    write_provenance_sidecar(survey_targets_path, **provenance_kwargs)
    write_provenance_sidecar(quantity_by_segment_path, **provenance_kwargs)
    write_provenance_sidecar(diagnostics_path, **provenance_kwargs)
    LOGGER.info(
        "built childcare utilization stack: public=%s survey=%s q0=%s diagnostics=%s",
        len(outputs["public_program_slots"]),
        len(outputs["survey_paid_use_targets"]),
        len(outputs["quantity_by_segment"]),
        len(outputs["reconciliation_diagnostics"]),
    )


def _ccdf_policy_controls_input_path(paths) -> Path | None:
    promoted = paths.interim / "ccdf" / "ccdf_policy_promoted_controls_state_year.parquet"
    controls = paths.interim / "ccdf" / "ccdf_policy_controls_state_year.parquet"
    if promoted.exists():
        return promoted
    if controls.exists():
        return controls
    return None


def _ensure_childcare_utilization_artifacts(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> None:
    quantity_path = paths.root / "data" / "interim" / "childcare" / "utilization_stack" / "q0_segmented.parquet"
    diagnostics_path = (
        paths.root
        / "data"
        / "interim"
        / "childcare"
        / "utilization_stack"
        / "utilization_reconciliation_diagnostics.parquet"
    )
    if not refresh and quantity_path.exists() and diagnostics_path.exists():
        return
    build_childcare_utilization(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
        config_name_or_path="utilization_stack",
        segmented_config_name_or_path="segmented_solver",
    )


def _ensure_childcare_segmented_prices_artifact(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> None:
    segmented_price_panel_path = (
        paths.root / "data" / "interim" / "childcare" / "segmented_solver" / "ndcp_segment_prices.parquet"
    )
    if not refresh and segmented_price_panel_path.exists():
        return
    build_childcare_segments(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
        config_name_or_path="segmented_solver",
    )


def build_childcare_solver_inputs(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "solver_inputs",
) -> None:
    config_path = resolve_extension_config_path(paths.root, config_name_or_path)
    config = load_extension_config(paths.root, config_name_or_path)
    if not config:
        raise UnpaidWorkError(f"extension config is empty or missing: {config_path}")

    if dry_run:
        LOGGER.info("planned childcare solver inputs build using %s", config_path)
        return

    _ensure_childcare_utilization_artifacts(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
    )
    _ensure_childcare_segmented_prices_artifact(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
    )

    q0_segmented_path = (
        paths.root / "data" / "interim" / "childcare" / "utilization_stack" / "q0_segmented.parquet"
    )
    diagnostics_path = (
        paths.root
        / "data"
        / "interim"
        / "childcare"
        / "utilization_stack"
        / "utilization_reconciliation_diagnostics.parquet"
    )
    segmented_price_panel_path = (
        paths.root / "data" / "interim" / "childcare" / "segmented_solver" / "ndcp_segment_prices.parquet"
    )
    if not q0_segmented_path.exists() or not diagnostics_path.exists() or not segmented_price_panel_path.exists():
        raise UnpaidWorkError(
            "missing upstream childcare additive inputs: "
            f"{q0_segmented_path}, {diagnostics_path}, and/or {segmented_price_panel_path}"
        )

    controls_path = _ccdf_policy_controls_input_path(paths)

    from importlib import import_module

    childcare_solver_inputs = import_module("unpriced.childcare.solver_inputs")
    build_childcare_solver_inputs_impl = childcare_solver_inputs.build_childcare_solver_inputs

    output_config = config.get("output_namespace", {})
    output_namespace = _output_namespace_path(
        paths,
        str(output_config.get("namespace", "data/interim/childcare/solver_inputs")),
    )
    file_names = output_config.get("files", {})
    channel_quantities_path = output_namespace / file_names.get(
        "solver_channel_quantities", "solver_channel_quantities.parquet"
    )
    baseline_state_year_path = output_namespace / file_names.get(
        "solver_baseline_state_year", "solver_baseline_state_year.parquet"
    )
    elasticity_mapping_path = output_namespace / file_names.get(
        "solver_elasticity_mapping", "solver_elasticity_mapping.parquet"
    )
    policy_controls_path = output_namespace / file_names.get(
        "solver_policy_controls_state_year", "solver_policy_controls_state_year.parquet"
    )

    outputs = build_childcare_solver_inputs_impl(
        q0_segmented=read_parquet(q0_segmented_path),
        promoted_controls=read_parquet(controls_path) if controls_path is not None else None,
        ccdf_policy_controls_state_year=read_parquet(controls_path) if controls_path is not None else None,
        ndcp_segment_prices=read_parquet(segmented_price_panel_path),
    )

    write_parquet(outputs["solver_channel_quantities"], channel_quantities_path)
    write_parquet(outputs["solver_baseline_state_year"], baseline_state_year_path)
    write_parquet(outputs["solver_elasticity_mapping"], elasticity_mapping_path)
    write_parquet(outputs["solver_policy_controls_state_year"], policy_controls_path)

    provenance_kwargs = {
        "source_files": [
            q0_segmented_path,
            diagnostics_path,
            segmented_price_panel_path,
            config_path,
        ]
        + ([controls_path] if controls_path is not None else []),
        "parameters": {
            "sample_mode": sample,
            "requested_year": year,
            "extension_name": config.get("name", "solver_inputs"),
            "mode": config.get("mode", "additive_solver_inputs"),
        },
        "config": config,
        "repo_root": paths.root,
    }
    write_provenance_sidecar(channel_quantities_path, **provenance_kwargs)
    write_provenance_sidecar(baseline_state_year_path, **provenance_kwargs)
    write_provenance_sidecar(elasticity_mapping_path, **provenance_kwargs)
    write_provenance_sidecar(policy_controls_path, **provenance_kwargs)
    LOGGER.info(
        "built childcare solver inputs: quantities=%s baseline=%s elasticity=%s controls=%s",
        len(outputs["solver_channel_quantities"]),
        len(outputs["solver_baseline_state_year"]),
        len(outputs["solver_elasticity_mapping"]),
        len(outputs["solver_policy_controls_state_year"]),
    )


def build_childcare_report_tables(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "report_tables",
) -> None:
    config_path = resolve_extension_config_path(paths.root, config_name_or_path)
    config = load_extension_config(paths.root, config_name_or_path)
    if not config:
        raise UnpaidWorkError(f"extension config is empty or missing: {config_path}")

    if dry_run:
        LOGGER.info("planned childcare report tables build using %s", config_path)
        return

    _ensure_childcare_utilization_artifacts(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
    )

    q0_segmented_path = (
        paths.root / "data" / "interim" / "childcare" / "utilization_stack" / "q0_segmented.parquet"
    )
    diagnostics_path = (
        paths.root
        / "data"
        / "interim"
        / "childcare"
        / "utilization_stack"
        / "utilization_reconciliation_diagnostics.parquet"
    )
    if not q0_segmented_path.exists() or not diagnostics_path.exists():
        raise UnpaidWorkError(
            "missing upstream childcare additive inputs: "
            f"{q0_segmented_path} and/or {diagnostics_path}"
        )

    controls_path = _ccdf_policy_controls_input_path(paths)

    from importlib import import_module

    childcare_report_tables = import_module("unpriced.childcare.report_tables")
    build_childcare_report_tables_impl = childcare_report_tables.build_childcare_report_tables

    output_config = config.get("output_namespace", {})
    output_namespace = _output_namespace_path(
        paths,
        str(output_config.get("namespace", "data/interim/childcare/report_tables")),
    )
    file_names = output_config.get("files", {})
    channel_summary_path = output_namespace / file_names.get(
        "state_year_channel_summary", "state_year_channel_summary.parquet"
    )
    policy_quantity_summary_path = output_namespace / file_names.get(
        "state_year_policy_quantity_summary", "state_year_policy_quantity_summary.parquet"
    )
    support_summary_path = output_namespace / file_names.get(
        "state_year_support_summary", "state_year_support_summary.parquet"
    )

    outputs = build_childcare_report_tables_impl(
        q0_segmented=read_parquet(q0_segmented_path),
        utilization_diagnostics=read_parquet(diagnostics_path),
        promoted_controls=read_parquet(controls_path) if controls_path is not None else None,
        ccdf_policy_controls_state_year=read_parquet(controls_path) if controls_path is not None else None,
    )

    write_parquet(outputs["state_year_channel_summary"], channel_summary_path)
    write_parquet(outputs["state_year_policy_quantity_summary"], policy_quantity_summary_path)
    write_parquet(outputs["state_year_support_summary"], support_summary_path)

    provenance_kwargs = {
        "source_files": [q0_segmented_path, diagnostics_path, config_path]
        + ([controls_path] if controls_path is not None else []),
        "parameters": {
            "sample_mode": sample,
            "requested_year": year,
            "extension_name": config.get("name", "report_tables"),
            "mode": config.get("mode", "additive_report_tables"),
        },
        "config": config,
        "repo_root": paths.root,
    }
    write_provenance_sidecar(channel_summary_path, **provenance_kwargs)
    write_provenance_sidecar(policy_quantity_summary_path, **provenance_kwargs)
    write_provenance_sidecar(support_summary_path, **provenance_kwargs)
    LOGGER.info(
        "built childcare report tables: channel=%s policy=%s support=%s",
        len(outputs["state_year_channel_summary"]),
        len(outputs["state_year_policy_quantity_summary"]),
        len(outputs["state_year_support_summary"]),
    )


def build_childcare_segmented_scenarios(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "segmented_scenarios",
) -> None:
    config_path = resolve_extension_config_path(paths.root, config_name_or_path)
    config = load_extension_config(paths.root, config_name_or_path)
    if not config:
        raise UnpaidWorkError(f"extension config is empty or missing: {config_path}")

    state_path = paths.processed / "childcare_state_year_panel.parquet"
    county_path = paths.processed / "childcare_county_year_price_panel.parquet"
    comparison_path = paths.outputs_reports / "childcare_demand_sample_comparison.json"
    solver_namespace = paths.root / "data" / "interim" / "childcare" / "solver_inputs"
    baseline_path = solver_namespace / "solver_baseline_state_year.parquet"
    elasticity_path = solver_namespace / "solver_elasticity_mapping.parquet"
    controls_path = solver_namespace / "solver_policy_controls_state_year.parquet"

    if dry_run:
        LOGGER.info("planned childcare segmented scenarios build using %s", config_path)
        return

    if not state_path.exists() or not county_path.exists() or not comparison_path.exists():
        fit_childcare(paths, sample=sample)
    if not baseline_path.exists() or not elasticity_path.exists() or not controls_path.exists():
        build_childcare_solver_inputs(
            paths,
            sample=sample,
            refresh=refresh,
            dry_run=dry_run,
            year=year,
        )

    if not state_path.exists() or not county_path.exists() or not comparison_path.exists():
        raise UnpaidWorkError(
            "missing pooled childcare artifacts required for segmented scenarios: "
            f"{state_path}, {county_path}, and/or {comparison_path}"
        )
    if not baseline_path.exists() or not elasticity_path.exists():
        raise UnpaidWorkError(
            "missing segmented solver inputs required for segmented scenarios: "
            f"{baseline_path} and/or {elasticity_path}"
        )

    state = read_parquet(state_path)
    county = read_parquet(county_path)
    if not _state_panel_has_sample_ladder(state):
        fit_childcare(paths, sample=sample)
        state = read_parquet(state_path)
        county = read_parquet(county_path)
    comparison = read_json(comparison_path)
    samples = comparison.get("samples", {})
    selected_sample, selection_reason = select_headline_sample(samples)
    if selected_sample is None:
        raise UnpaidWorkError(
            "no defensible observed-core childcare sample passed the minimum support rule; only exploratory samples are available"
        )
    canonical_profile = _canonical_specification_profile_for_sample(selected_sample)
    demand_summary_path = _demand_summary_path(paths, selected_sample, specification_profile=canonical_profile)
    if not demand_summary_path.exists():
        fit_childcare(paths, sample=sample)
    if not demand_summary_path.exists():
        raise UnpaidWorkError(f"missing demand summary for selected sample: {demand_summary_path}")

    eligible_column = f"eligible_{selected_sample}"
    if eligible_column in state.columns:
        state = state.loc[state[eligible_column].fillna(False).astype(bool)].copy()

    from importlib import import_module

    childcare_segmented_scenarios = import_module("unpriced.childcare.segmented_scenarios")
    build_childcare_segmented_scenarios_impl = childcare_segmented_scenarios.build_childcare_segmented_scenarios

    supply_summary = summarize_supply_elasticity(county)
    project_config = load_yaml(paths.root / "configs" / "project.yaml")
    alphas = [float(alpha) for alpha in project_config["alpha_grid"]]
    outputs = build_childcare_segmented_scenarios_impl(
        state_frame=state,
        solver_baseline_state_year=read_parquet(baseline_path),
        solver_elasticity_mapping=read_parquet(elasticity_path),
        solver_policy_controls_state_year=read_parquet(controls_path) if controls_path.exists() else None,
        alphas=alphas,
        demand_summary=read_json(demand_summary_path),
        supply_summary=supply_summary,
    )

    output_config = config.get("output_namespace", {})
    output_namespace = _output_namespace_path(
        paths,
        str(output_config.get("namespace", "data/interim/childcare/segmented_scenarios")),
    )
    file_names = output_config.get("files", {})
    channel_inputs_path = output_namespace / file_names.get(
        "segmented_channel_inputs", "segmented_channel_inputs.parquet"
    )
    channel_scenarios_path = output_namespace / file_names.get(
        "segmented_channel_scenarios", "segmented_channel_scenarios.parquet"
    )
    summary_path = output_namespace / file_names.get(
        "segmented_state_year_summary", "segmented_state_year_summary.parquet"
    )
    diagnostics_path = output_namespace / file_names.get(
        "segmented_state_year_diagnostics", "segmented_state_year_diagnostics.parquet"
    )

    write_parquet(outputs["segmented_channel_inputs"], channel_inputs_path)
    write_parquet(outputs["segmented_channel_scenarios"], channel_scenarios_path)
    write_parquet(outputs["segmented_state_year_summary"], summary_path)
    write_parquet(outputs["segmented_state_year_diagnostics"], diagnostics_path)

    provenance_kwargs = {
        "source_files": [
            state_path,
            county_path,
            comparison_path,
            demand_summary_path,
            baseline_path,
            elasticity_path,
            config_path,
        ]
        + ([controls_path] if controls_path.exists() else []),
        "parameters": {
            "sample_mode": sample,
            "requested_year": year,
            "selected_sample": selected_sample,
            "selection_reason": selection_reason,
            "extension_name": config.get("name", "segmented_scenarios"),
            "mode": config.get("mode", "segmented_scenarios"),
        },
        "config": config,
        "repo_root": paths.root,
    }
    write_provenance_sidecar(channel_inputs_path, **provenance_kwargs)
    write_provenance_sidecar(channel_scenarios_path, **provenance_kwargs)
    write_provenance_sidecar(summary_path, **provenance_kwargs)
    write_provenance_sidecar(diagnostics_path, **provenance_kwargs)
    LOGGER.info(
        "built childcare segmented scenarios: inputs=%s scenarios=%s summary=%s diagnostics=%s",
        len(outputs["segmented_channel_inputs"]),
        len(outputs["segmented_channel_scenarios"]),
        len(outputs["segmented_state_year_summary"]),
        len(outputs["segmented_state_year_diagnostics"]),
    )


def build_ccdf_state_year(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "ccdf_state_year",
) -> None:
    del year
    config_path = resolve_extension_config_path(paths.root, config_name_or_path)
    config = load_extension_config(paths.root, config_name_or_path)
    if not config:
        raise UnpaidWorkError(f"extension config is empty or missing: {config_path}")

    ccdf_dir = paths.interim / "ccdf"
    admin_long_path = ccdf_dir / "ccdf_admin_long.parquet"
    policy_long_path = ccdf_dir / "ccdf_policy_long.parquet"
    if refresh or not admin_long_path.exists() or not policy_long_path.exists():
        result = ccdf.ingest(paths, sample=sample, refresh=refresh, dry_run=dry_run)
        LOGGER.info(
            "%s %s -> %s (%s)",
            "planned" if result.dry_run else "ingested",
            result.source_name,
            result.normalized_path,
            result.detail or ("skipped" if result.skipped else "ok"),
        )

    if dry_run:
        LOGGER.info("planned CCDF state-year mapping build using %s", config_path)
        return

    if not admin_long_path.exists() or not policy_long_path.exists():
        raise UnpaidWorkError(
            f"missing normalized CCDF long inputs: {admin_long_path} and/or {policy_long_path}"
        )

    from unpriced.childcare import ccdf as childcare_ccdf

    build_ccdf_admin_state_year = childcare_ccdf.build_ccdf_admin_state_year
    build_ccdf_policy_features_state_year = childcare_ccdf.build_ccdf_policy_features_state_year
    build_ccdf_policy_controls_state_year = childcare_ccdf.build_ccdf_policy_controls_state_year
    build_ccdf_policy_feature_audit = childcare_ccdf.build_ccdf_policy_feature_audit
    policy_feature_rows_from_policy_long = getattr(childcare_ccdf, "_policy_feature_rows_from_policy_long", None)
    policy_features_state_year_from_feature_rows = getattr(
        childcare_ccdf,
        "_policy_features_state_year_from_feature_rows",
        None,
    )
    policy_controls_state_year_from_feature_rows = getattr(
        childcare_ccdf,
        "_policy_controls_state_year_from_feature_rows",
        None,
    )
    policy_controls_coverage_from_feature_rows = getattr(
        childcare_ccdf,
        "_policy_controls_coverage_from_feature_rows",
        None,
    )
    policy_promoted_controls_state_year_from_feature_rows = getattr(
        childcare_ccdf,
        "_policy_promoted_controls_state_year_from_feature_rows",
        None,
    )
    policy_feature_audit_from_feature_rows = getattr(
        childcare_ccdf,
        "_policy_feature_audit_from_feature_rows",
        None,
    )
    build_ccdf_policy_controls_coverage = getattr(
        childcare_ccdf,
        "build_ccdf_policy_controls_coverage",
        None,
    )
    build_ccdf_policy_promoted_controls_state_year = getattr(
        childcare_ccdf,
        "build_ccdf_policy_promoted_controls_state_year",
        None,
    )
    if build_ccdf_policy_controls_coverage is None:
        raise UnpaidWorkError(
            "missing CCDF policy controls coverage builder: "
            "unpriced.childcare.ccdf.build_ccdf_policy_controls_coverage"
        )
    if build_ccdf_policy_promoted_controls_state_year is None:
        raise UnpaidWorkError(
            "missing CCDF policy promoted controls builder: "
            "unpriced.childcare.ccdf.build_ccdf_policy_promoted_controls_state_year"
        )

    output_config = config.get("output_namespace", {})
    output_namespace = _output_namespace_path(
        paths,
        str(output_config.get("namespace", "data/interim/ccdf")),
    )
    file_names = output_config.get("files", {})
    admin_state_year_path = output_namespace / file_names.get(
        "admin_state_year", "ccdf_admin_state_year.parquet"
    )
    policy_features_path = output_namespace / file_names.get(
        "policy_features_state_year", "ccdf_policy_features_state_year.parquet"
    )
    policy_controls_path = output_namespace / file_names.get(
        "policy_controls_state_year", "ccdf_policy_controls_state_year.parquet"
    )
    policy_controls_coverage_path = output_namespace / file_names.get(
        "policy_controls_coverage", "ccdf_policy_controls_coverage.parquet"
    )
    policy_promoted_controls_path = output_namespace / file_names.get(
        "policy_promoted_controls_state_year",
        "ccdf_policy_promoted_controls_state_year.parquet",
    )
    policy_audit_path = output_namespace / file_names.get(
        "policy_feature_audit", "ccdf_policy_feature_audit.parquet"
    )
    promotion_config = config.get("policy_controls_promotion", {})
    min_state_year_coverage = float(promotion_config.get("min_state_year_coverage", 0.75))

    admin_long = read_parquet(admin_long_path)
    policy_long = read_parquet(policy_long_path)
    admin_state_year = build_ccdf_admin_state_year(admin_long)
    if all(
        helper is not None
        for helper in (
            policy_feature_rows_from_policy_long,
            policy_features_state_year_from_feature_rows,
            policy_controls_state_year_from_feature_rows,
            policy_controls_coverage_from_feature_rows,
            policy_promoted_controls_state_year_from_feature_rows,
            policy_feature_audit_from_feature_rows,
        )
    ):
        policy_feature_rows = policy_feature_rows_from_policy_long(policy_long)
        policy_features = policy_features_state_year_from_feature_rows(policy_feature_rows)
        policy_controls = policy_controls_state_year_from_feature_rows(policy_feature_rows)
        policy_controls_coverage = policy_controls_coverage_from_feature_rows(policy_feature_rows)
        policy_promoted_controls = policy_promoted_controls_state_year_from_feature_rows(
            policy_feature_rows,
            min_state_year_coverage=min_state_year_coverage,
        )
        policy_audit = policy_feature_audit_from_feature_rows(policy_feature_rows)
    else:
        policy_features = build_ccdf_policy_features_state_year(policy_long)
        policy_controls = build_ccdf_policy_controls_state_year(policy_long)
        policy_controls_coverage = build_ccdf_policy_controls_coverage(policy_long)
        policy_promoted_controls = build_ccdf_policy_promoted_controls_state_year(
            policy_long,
            min_state_year_coverage=min_state_year_coverage,
        )
        policy_audit = build_ccdf_policy_feature_audit(policy_long)

    write_parquet(admin_state_year, admin_state_year_path)
    write_parquet(policy_features, policy_features_path)
    write_parquet(policy_controls, policy_controls_path)
    write_parquet(policy_controls_coverage, policy_controls_coverage_path)
    write_parquet(policy_promoted_controls, policy_promoted_controls_path)
    write_parquet(policy_audit, policy_audit_path)

    provenance_kwargs = {
        "source_files": [admin_long_path, policy_long_path, config_path],
        "parameters": {
            "sample_mode": sample,
            "extension_name": config.get("name", "ccdf_state_year"),
            "mode": config.get("mode", "state_year_mapping"),
        },
        "config": config,
        "repo_root": paths.root,
    }
    write_provenance_sidecar(admin_state_year_path, **provenance_kwargs)
    write_provenance_sidecar(policy_features_path, **provenance_kwargs)
    write_provenance_sidecar(policy_controls_path, **provenance_kwargs)
    write_provenance_sidecar(policy_controls_coverage_path, **provenance_kwargs)
    write_provenance_sidecar(policy_promoted_controls_path, **provenance_kwargs)
    write_provenance_sidecar(policy_audit_path, **provenance_kwargs)
    LOGGER.info(
        "built CCDF state-year outputs: admin=%s policy=%s controls=%s coverage=%s promoted=%s audit=%s",
        len(admin_state_year),
        len(policy_features),
        len(policy_controls),
        len(policy_controls_coverage),
        len(policy_promoted_controls),
        len(policy_audit),
    )


def _ensure_childcare_segmented_scenarios_artifacts(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> None:
    segmented_namespace = paths.root / "data" / "interim" / "childcare" / "segmented_scenarios"
    channel_scenarios_path = segmented_namespace / "segmented_channel_scenarios.parquet"
    summary_path = segmented_namespace / "segmented_state_year_summary.parquet"
    diagnostics_path = segmented_namespace / "segmented_state_year_diagnostics.parquet"
    if not refresh and channel_scenarios_path.exists() and summary_path.exists() and diagnostics_path.exists():
        return
    build_childcare_segmented_scenarios(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
        config_name_or_path="segmented_scenarios",
    )


def _ensure_childcare_report_tables_artifacts(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> None:
    report_namespace = paths.root / "data" / "interim" / "childcare" / "report_tables"
    support_summary_path = report_namespace / "state_year_support_summary.parquet"
    if not refresh and support_summary_path.exists():
        return
    build_childcare_report_tables(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
        config_name_or_path="report_tables",
    )


def _ensure_childcare_segmented_report_artifacts(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> None:
    report_namespace = paths.root / "data" / "interim" / "childcare" / "segmented_reports"
    response_summary_path = report_namespace / "segmented_channel_response_summary.parquet"
    fallback_summary_path = report_namespace / "segmented_state_fallback_summary.parquet"
    headline_summary_path = report_namespace / "childcare_segmented_headline_summary.json"
    if not refresh and response_summary_path.exists() and fallback_summary_path.exists() and headline_summary_path.exists():
        return
    build_childcare_segmented_report(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
        config_name_or_path="segmented_reports",
    )


def build_childcare_segmented_report(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "segmented_reports",
) -> None:
    config_path = resolve_extension_config_path(paths.root, config_name_or_path)
    config = load_extension_config(paths.root, config_name_or_path)
    if not config:
        raise UnpaidWorkError(f"extension config is empty or missing: {config_path}")

    segmented_namespace = paths.root / "data" / "interim" / "childcare" / "segmented_scenarios"
    report_namespace = paths.root / "data" / "interim" / "childcare" / "report_tables"
    channel_scenarios_path = segmented_namespace / "segmented_channel_scenarios.parquet"
    summary_path = segmented_namespace / "segmented_state_year_summary.parquet"
    diagnostics_path = segmented_namespace / "segmented_state_year_diagnostics.parquet"
    support_summary_path = report_namespace / "state_year_support_summary.parquet"

    if dry_run:
        LOGGER.info("planned childcare segmented report build using %s", config_path)
        return

    _ensure_childcare_segmented_scenarios_artifacts(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
    )
    _ensure_childcare_report_tables_artifacts(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
    )

    if not channel_scenarios_path.exists() or not summary_path.exists() or not diagnostics_path.exists():
        raise UnpaidWorkError(
            "missing segmented childcare scenario artifacts required for report build: "
            f"{channel_scenarios_path}, {summary_path}, and/or {diagnostics_path}"
        )
    if not support_summary_path.exists():
        raise UnpaidWorkError(
            f"missing segmented childcare support summary required for report build: {support_summary_path}"
        )

    from importlib import import_module

    childcare_segmented_reports = import_module("unpriced.childcare.segmented_reports")
    build_childcare_segmented_reports_impl = childcare_segmented_reports.build_childcare_segmented_reports

    output_config = config.get("output_namespace", {})
    output_namespace = _output_namespace_path(
        paths,
        str(output_config.get("namespace", "data/interim/childcare/segmented_reports")),
    )
    file_names = output_config.get("files", {})
    response_summary_path = output_namespace / file_names.get(
        "segmented_channel_response_summary", "segmented_channel_response_summary.parquet"
    )
    fallback_summary_path = output_namespace / file_names.get(
        "segmented_state_fallback_summary", "segmented_state_fallback_summary.parquet"
    )
    headline_summary_path = output_namespace / file_names.get(
        "childcare_segmented_headline_summary", "childcare_segmented_headline_summary.json"
    )
    report_path = output_namespace / file_names.get(
        "childcare_segmented_report", "childcare_segmented_report.md"
    )

    outputs = build_childcare_segmented_reports_impl(
        segmented_channel_scenarios=read_parquet(channel_scenarios_path),
        segmented_state_year_summary=read_parquet(summary_path),
        segmented_state_year_diagnostics=read_parquet(diagnostics_path),
        state_year_support_summary=read_parquet(support_summary_path),
    )

    write_parquet(outputs["segmented_channel_response_summary"], response_summary_path)
    write_parquet(outputs["segmented_state_fallback_summary"], fallback_summary_path)
    write_json(outputs["segmented_headline_summary"], headline_summary_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(outputs["segmented_report_markdown"], encoding="utf-8")

    provenance_kwargs = {
        "source_files": [channel_scenarios_path, summary_path, diagnostics_path, support_summary_path, config_path],
        "parameters": {
            "sample_mode": sample,
            "requested_year": year,
            "extension_name": config.get("name", "segmented_reports"),
            "mode": config.get("mode", "segmented_reports"),
        },
        "config": config,
        "repo_root": paths.root,
    }
    write_provenance_sidecar(response_summary_path, **provenance_kwargs)
    write_provenance_sidecar(fallback_summary_path, **provenance_kwargs)
    LOGGER.info(
        "built childcare segmented report: response=%s fallback=%s headline=%s",
        len(outputs["segmented_channel_response_summary"]),
        len(outputs["segmented_state_fallback_summary"]),
        headline_summary_path,
    )


def _build_childcare_segmented_parser_action_plan(
    parser_focus_areas: pd.DataFrame,
    admin_sheet_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    focus_columns = [
        "focus_area",
        "priority_tier",
        "current_signal_present",
        "matched_column_count",
        "matched_column_examples",
        "recommended_keywords",
        "reason",
    ]
    target_columns = [
        "filename",
        "source_sheet",
        "table_year",
        "parse_status",
        "parsed_row_count",
        "matched_focus_areas",
        "missing_priority_focus_areas",
        "parser_target_recommendation",
    ]

    focus = parser_focus_areas.copy() if parser_focus_areas is not None else pd.DataFrame(columns=focus_columns)
    target = admin_sheet_targets.copy() if admin_sheet_targets is not None else pd.DataFrame(columns=target_columns)

    for column in focus_columns:
        if column not in focus.columns:
            focus[column] = pd.NA
    for column in target_columns:
        if column not in target.columns:
            target[column] = pd.NA

    focus = focus[focus_columns].copy()
    target = target[target_columns].copy()
    priority_order = {"high": 0, "medium": 1, "low": 2}
    focus["priority_rank"] = focus["priority_tier"].astype(str).str.lower().map(priority_order).fillna(99).astype(int)

    if target.empty and focus.empty:
        plan_columns = focus_columns + target_columns + ["priority_rank", "missing_priority_focus_area"]
        return pd.DataFrame(columns=plan_columns), {
            "row_count": 0,
            "focus_area_count": 0,
            "target_row_count": 0,
            "target_file_count": 0,
            "parsed_target_count": 0,
            "priority_tier_counts": {},
            "rows": [],
        }

    target = target.copy()
    target["missing_priority_focus_areas"] = target["missing_priority_focus_areas"].fillna("").astype(str)
    target["missing_priority_focus_area"] = target["missing_priority_focus_areas"].str.split(r"\s*;\s*")
    target = target.explode("missing_priority_focus_area", ignore_index=True)
    target["missing_priority_focus_area"] = (
        target["missing_priority_focus_area"].astype("string").str.strip().replace("", pd.NA)
    )
    target = target.dropna(subset=["missing_priority_focus_area"]).copy()
    target["focus_area"] = target["missing_priority_focus_area"].astype(str)

    if target.empty:
        plan = focus.copy()
        for column in target_columns:
            plan[column] = pd.NA
        plan["missing_priority_focus_area"] = plan["focus_area"]
    else:
        plan = target.merge(focus, on="focus_area", how="left", suffixes=("", "_focus"))
        plan["missing_priority_focus_area"] = plan["focus_area"]

    for column in ["current_signal_present"]:
        if column in plan.columns:
            plan[column] = plan[column].fillna(False).astype(bool)
    for column in ["matched_column_count", "parsed_row_count", "table_year"]:
        if column in plan.columns:
            plan[column] = pd.to_numeric(plan[column], errors="coerce")

    plan["target_rank"] = plan["parse_status"].astype(str).str.lower().ne("parsed").astype(int)
    plan = plan.sort_values(
        ["priority_rank", "target_rank", "filename", "source_sheet", "focus_area"],
        ascending=[True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    plan = plan[
        [
            "priority_rank",
            "focus_area",
            "priority_tier",
            "current_signal_present",
            "matched_column_count",
            "matched_column_examples",
            "recommended_keywords",
            "reason",
            "filename",
            "source_sheet",
            "table_year",
            "parse_status",
            "parsed_row_count",
            "matched_focus_areas",
            "missing_priority_focus_areas",
            "parser_target_recommendation",
            "missing_priority_focus_area",
            "target_rank",
        ]
    ].copy()

    priority_counts = {}
    if "priority_tier" in focus.columns and not focus.empty:
        counts = focus["priority_tier"].astype(str).value_counts(dropna=False).to_dict()
        priority_counts = {str(key): int(value) for key, value in counts.items()}

    summary = {
        "row_count": int(len(plan)),
        "focus_area_count": int(focus["focus_area"].nunique()) if "focus_area" in focus.columns else 0,
        "target_row_count": int(len(target)),
        "target_file_count": int(target["filename"].astype(str).nunique()) if "filename" in target.columns else 0,
        "parsed_target_count": int(target["parse_status"].astype(str).str.lower().eq("parsed").sum())
        if "parse_status" in target.columns
        else 0,
        "priority_tier_counts": priority_counts,
        "rows": plan.to_dict(orient="records"),
    }
    return plan, summary


def report_childcare_segmented(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "segmented_publication",
) -> None:
    config_path = resolve_extension_config_path(paths.root, config_name_or_path)
    config = load_extension_config(paths.root, config_name_or_path)
    if not config:
        raise UnpaidWorkError(f"extension config is empty or missing: {config_path}")

    report_namespace = paths.root / "data" / "interim" / "childcare" / "segmented_reports"
    ccdf_namespace = paths.root / "data" / "interim" / "ccdf"
    response_summary_path = report_namespace / "segmented_channel_response_summary.parquet"
    fallback_summary_path = report_namespace / "segmented_state_fallback_summary.parquet"
    headline_summary_path = report_namespace / "childcare_segmented_headline_summary.json"
    ccdf_parse_inventory_path = ccdf_namespace / "ccdf_parse_inventory.parquet"
    ccdf_admin_long_path = ccdf_namespace / "ccdf_admin_long.parquet"
    ccdf_admin_state_year_path = ccdf_namespace / "ccdf_admin_state_year.parquet"

    if dry_run:
        LOGGER.info("planned childcare segmented publication using %s", config_path)
        return

    _ensure_childcare_segmented_report_artifacts(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
    )

    if not response_summary_path.exists() or not fallback_summary_path.exists() or not headline_summary_path.exists():
        raise UnpaidWorkError(
            "missing segmented childcare report artifacts required for publication: "
            f"{response_summary_path}, {fallback_summary_path}, and/or {headline_summary_path}"
        )

    from importlib import import_module

    childcare_segmented_publication = import_module("unpriced.childcare.segmented_publication")
    build_childcare_segmented_publication_outputs_impl = (
        childcare_segmented_publication.build_childcare_segmented_publication_outputs
    )

    outputs = build_childcare_segmented_publication_outputs_impl(
        segmented_headline_summary=read_json(headline_summary_path),
        segmented_channel_response_summary=read_parquet(response_summary_path),
        segmented_state_fallback_summary=read_parquet(fallback_summary_path),
        ccdf_parse_inventory=read_parquet(ccdf_parse_inventory_path) if ccdf_parse_inventory_path.exists() else None,
        ccdf_admin_long=read_parquet(ccdf_admin_long_path) if ccdf_admin_long_path.exists() else None,
        ccdf_admin_state_year=read_parquet(ccdf_admin_state_year_path) if ccdf_admin_state_year_path.exists() else None,
    )

    output_config = config.get("outputs", {})
    reports_namespace = _output_namespace_path(
        paths,
        str(output_config.get("reports_namespace", "outputs/reports")),
    )
    tables_namespace = _output_namespace_path(
        paths,
        str(output_config.get("tables_namespace", "outputs/tables")),
    )
    file_names = output_config.get("files", {})
    publication_headline_summary_path = reports_namespace / file_names.get(
        "headline_summary_json", "childcare_segmented_headline_summary.json"
    )
    publication_readout_path = reports_namespace / file_names.get(
        "headline_readout_markdown", "childcare_segmented_headline_readout.md"
    )
    publication_report_path = reports_namespace / file_names.get(
        "full_report_markdown", "childcare_segmented_report.md"
    )
    support_quality_summary_path = reports_namespace / file_names.get(
        "support_quality_summary_json", "childcare_segmented_support_quality_summary.json"
    )
    support_priority_summary_path = reports_namespace / file_names.get(
        "support_priority_summary_json", "childcare_segmented_support_priority_summary.json"
    )
    support_issue_breakdown_path = reports_namespace / file_names.get(
        "support_issue_breakdown_json", "childcare_segmented_support_issue_breakdown.json"
    )
    parser_focus_summary_path = reports_namespace / file_names.get(
        "parser_focus_summary_json", "childcare_segmented_parser_focus_summary.json"
    )
    parser_action_plan_summary_path = reports_namespace / file_names.get(
        "parser_action_plan_summary_json", "childcare_segmented_parser_action_plan_summary.json"
    )
    publication_channel_table_path = tables_namespace / file_names.get(
        "channel_response_csv", "childcare_segmented_channel_response_summary.csv"
    )
    publication_fallback_table_path = tables_namespace / file_names.get(
        "fallback_csv", "childcare_segmented_state_fallback_summary.csv"
    )
    support_quality_table_path = tables_namespace / file_names.get(
        "support_quality_csv", "childcare_segmented_support_quality_summary.csv"
    )
    support_priority_states_path = tables_namespace / file_names.get(
        "support_priority_states_csv", "childcare_segmented_support_priority_states.csv"
    )
    support_issue_breakdown_table_path = tables_namespace / file_names.get(
        "support_issue_breakdown_csv", "childcare_segmented_support_issue_breakdown.csv"
    )
    parser_focus_areas_path = tables_namespace / file_names.get(
        "parser_focus_areas_csv", "childcare_segmented_parser_focus_areas.csv"
    )
    admin_sheet_targets_path = tables_namespace / file_names.get(
        "admin_sheet_targets_csv", "childcare_segmented_admin_sheet_targets.csv"
    )
    parser_action_plan_path = tables_namespace / file_names.get(
        "parser_action_plan_csv", "childcare_segmented_parser_action_plan.csv"
    )

    write_json(outputs["segmented_publication_headline_summary"], publication_headline_summary_path)
    write_json(
        {
            "state_year_count_total": int(outputs["segmented_publication_support_quality_summary"]["state_year_count"].sum())
            if not outputs["segmented_publication_support_quality_summary"].empty
            else 0,
            "tiers": outputs["segmented_publication_support_quality_summary"].to_dict(orient="records"),
        },
        support_quality_summary_path,
    )
    write_json(outputs["segmented_publication_priority_summary"], support_priority_summary_path)
    write_json(
        {"rows": outputs["segmented_publication_issue_breakdown"].to_dict(orient="records")},
        support_issue_breakdown_path,
    )
    write_json(
        {"rows": outputs["segmented_publication_parser_focus_areas"].to_dict(orient="records")},
        parser_focus_summary_path,
    )
    parser_action_plan_table = outputs.get("segmented_publication_parser_action_plan")
    parser_action_plan_summary = outputs.get("segmented_publication_parser_action_plan_summary")
    if parser_action_plan_table is None or parser_action_plan_summary is None:
        fallback_action_plan_table, fallback_action_plan_summary = _build_childcare_segmented_parser_action_plan(
            outputs["segmented_publication_parser_focus_areas"],
            outputs["segmented_publication_admin_sheet_targets"],
        )
        if parser_action_plan_table is None:
            parser_action_plan_table = fallback_action_plan_table
        if parser_action_plan_summary is None:
            parser_action_plan_summary = fallback_action_plan_summary
    write_json(parser_action_plan_summary, parser_action_plan_summary_path)
    publication_readout_path.parent.mkdir(parents=True, exist_ok=True)
    publication_readout_path.write_text(outputs["segmented_publication_readout_markdown"], encoding="utf-8")
    publication_report_path.parent.mkdir(parents=True, exist_ok=True)
    publication_report_path.write_text(outputs["segmented_publication_report_markdown"], encoding="utf-8")
    publication_channel_table_path.parent.mkdir(parents=True, exist_ok=True)
    outputs["segmented_publication_channel_table"].to_csv(publication_channel_table_path, index=False)
    publication_fallback_table_path.parent.mkdir(parents=True, exist_ok=True)
    outputs["segmented_publication_fallback_table"].to_csv(publication_fallback_table_path, index=False)
    support_quality_table_path.parent.mkdir(parents=True, exist_ok=True)
    outputs["segmented_publication_support_quality_summary"].to_csv(support_quality_table_path, index=False)
    support_priority_states_path.parent.mkdir(parents=True, exist_ok=True)
    outputs["segmented_publication_priority_states"].to_csv(support_priority_states_path, index=False)
    support_issue_breakdown_table_path.parent.mkdir(parents=True, exist_ok=True)
    outputs["segmented_publication_issue_breakdown"].to_csv(support_issue_breakdown_table_path, index=False)
    parser_focus_areas_path.parent.mkdir(parents=True, exist_ok=True)
    outputs["segmented_publication_parser_focus_areas"].to_csv(parser_focus_areas_path, index=False)
    admin_sheet_targets_path.parent.mkdir(parents=True, exist_ok=True)
    outputs["segmented_publication_admin_sheet_targets"].to_csv(admin_sheet_targets_path, index=False)
    parser_action_plan_path.parent.mkdir(parents=True, exist_ok=True)
    parser_action_plan_table.to_csv(parser_action_plan_path, index=False)

    source_files = [response_summary_path, fallback_summary_path, headline_summary_path]
    for optional_path in (ccdf_parse_inventory_path, ccdf_admin_long_path, ccdf_admin_state_year_path):
        if optional_path.exists():
            source_files.append(optional_path)
    source_files.append(config_path)
    provenance_kwargs = {
        "source_files": source_files,
        "parameters": {
            "sample_mode": sample,
            "requested_year": year,
            "extension_name": config.get("name", "segmented_publication"),
            "mode": config.get("mode", "segmented_publication"),
        },
        "config": config,
        "repo_root": paths.root,
    }
    write_provenance_sidecar(publication_headline_summary_path, **provenance_kwargs)
    write_provenance_sidecar(support_quality_summary_path, **provenance_kwargs)
    write_provenance_sidecar(support_priority_summary_path, **provenance_kwargs)
    write_provenance_sidecar(support_issue_breakdown_path, **provenance_kwargs)
    write_provenance_sidecar(parser_focus_summary_path, **provenance_kwargs)
    write_provenance_sidecar(publication_channel_table_path, **provenance_kwargs)
    write_provenance_sidecar(publication_fallback_table_path, **provenance_kwargs)
    write_provenance_sidecar(support_quality_table_path, **provenance_kwargs)
    write_provenance_sidecar(support_priority_states_path, **provenance_kwargs)
    write_provenance_sidecar(support_issue_breakdown_table_path, **provenance_kwargs)
    write_provenance_sidecar(parser_focus_areas_path, **provenance_kwargs)
    write_provenance_sidecar(admin_sheet_targets_path, **provenance_kwargs)
    write_provenance_sidecar(parser_action_plan_summary_path, **provenance_kwargs)
    write_provenance_sidecar(parser_action_plan_path, **provenance_kwargs)
    LOGGER.info(
        "published childcare segmented outputs: report=%s channel_rows=%s fallback_rows=%s support_tiers=%s priority_states=%s issue_rows=%s parser_focus_rows=%s admin_targets=%s parser_action_rows=%s",
        publication_report_path,
        len(outputs["segmented_publication_channel_table"]),
        len(outputs["segmented_publication_fallback_table"]),
        len(outputs["segmented_publication_support_quality_summary"]),
        len(outputs["segmented_publication_priority_states"]),
        len(outputs["segmented_publication_issue_breakdown"]),
        len(outputs["segmented_publication_parser_focus_areas"]),
        len(outputs["segmented_publication_admin_sheet_targets"]),
        len(parser_action_plan_table),
    )


def _ensure_licensing_harmonization_artifacts(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "licensing_iv",
) -> None:
    config = load_extension_config(paths.root, config_name_or_path)
    output_config = config.get("output_namespace", {}) if config else {}
    file_names = output_config.get("files", {}) if output_config else {}
    namespace = _output_namespace_path(
        paths,
        str(output_config.get("namespace", "data/interim/childcare/licensing_iv")),
    )
    required_paths = [
        namespace / file_names.get("rule_audit", "licensing_rules_raw_audit.parquet"),
        namespace / file_names.get("harmonized_rules", "licensing_rules_harmonized.parquet"),
        namespace / file_names.get("stringency_index", "licensing_stringency_index.parquet"),
        namespace / file_names.get("harmonization_summary", "licensing_harmonization_summary.parquet"),
    ]
    if not refresh and all(path.exists() for path in required_paths):
        return
    build_licensing_harmonization(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
        config_name_or_path=config_name_or_path,
    )


def _ensure_licensing_iv_artifacts(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "licensing_iv",
) -> None:
    config = load_extension_config(paths.root, config_name_or_path)
    output_config = config.get("output_namespace", {}) if config else {}
    file_names = output_config.get("files", {}) if output_config else {}
    namespace = _output_namespace_path(
        paths,
        str(output_config.get("namespace", "data/interim/childcare/licensing_iv")),
    )
    required_paths = [
        namespace / file_names.get("event_study_results", "licensing_event_study_results.parquet"),
        namespace / file_names.get("iv_results", "licensing_iv_results.parquet"),
        namespace / file_names.get("iv_usability_summary", "licensing_iv_usability_summary.parquet"),
        namespace / file_names.get("first_stage_diagnostics", "licensing_first_stage_diagnostics.parquet"),
        namespace / file_names.get("treatment_timing", "licensing_treatment_timing.parquet"),
        namespace / file_names.get("leave_one_state_out", "licensing_leave_one_state_out.parquet"),
        namespace / file_names.get("iv_summary_json", "licensing_iv_summary.json"),
    ]
    if not refresh and all(path.exists() for path in required_paths):
        return
    build_licensing_iv(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
        config_name_or_path=config_name_or_path,
    )


def build_licensing_harmonization(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "licensing_iv",
) -> None:
    config_path = resolve_extension_config_path(paths.root, config_name_or_path)
    config = load_extension_config(paths.root, config_name_or_path)
    if not config:
        raise UnpaidWorkError(f"extension config is empty or missing: {config_path}")

    licensing_output = paths.interim / "licensing" / ("licensing.parquet" if sample else "licensing_supply_shocks.parquet")
    licensing_rules_long_path = paths.interim / "licensing" / "licensing_rules_long.parquet"
    licensing_raw_dir = paths.raw / "licensing"
    raw_rule_level_present = (
        (licensing_raw_dir / "licensing_rules_long.csv").exists()
        or (licensing_raw_dir / "icpsr_2017" / "ICPSR_37700" / "DS0001" / "37700-0001-Data.tsv").exists()
        or (licensing_raw_dir / "icpsr_2020" / "ICPSR_38539" / "DS0001" / "38539-0001-Data.tsv").exists()
    )
    should_refresh_licensing_ingest = refresh or (
        raw_rule_level_present and not licensing_rules_long_path.exists()
    ) or (
        not licensing_output.exists() and not licensing_rules_long_path.exists()
    )
    if should_refresh_licensing_ingest:
        result = licensing.ingest(paths, sample=sample, refresh=refresh, dry_run=dry_run, year=year)
        LOGGER.info(
            "%s %s -> %s (%s)",
            "planned" if result.dry_run else "ingested",
            result.source_name,
            result.normalized_path,
            result.detail or ("skipped" if result.skipped else "ok"),
        )
        licensing_output = result.normalized_path
        licensing_rules_long_path = paths.interim / "licensing" / "licensing_rules_long.parquet"

    if dry_run:
        LOGGER.info("planned licensing harmonization build using %s", config_path)
        return

    selected_input_path = licensing_rules_long_path if licensing_rules_long_path.exists() else licensing_output
    if not selected_input_path.exists():
        raise UnpaidWorkError(
            f"missing normalized licensing inputs: {licensing_output} and/or {licensing_rules_long_path}"
        )

    from unpriced.childcare.licensing_harmonization import build_licensing_backend_outputs

    output_config = config.get("output_namespace", {})
    output_namespace = _output_namespace_path(
        paths,
        str(output_config.get("namespace", "data/interim/childcare/licensing_iv")),
    )
    file_names = output_config.get("files", {})
    raw_audit_path = output_namespace / file_names.get("rule_audit", "licensing_rules_raw_audit.parquet")
    harmonized_rules_path = output_namespace / file_names.get("harmonized_rules", "licensing_rules_harmonized.parquet")
    stringency_index_path = output_namespace / file_names.get("stringency_index", "licensing_stringency_index.parquet")
    harmonization_summary_path = output_namespace / file_names.get(
        "harmonization_summary", "licensing_harmonization_summary.parquet"
    )

    outputs = build_licensing_backend_outputs(read_parquet(selected_input_path))
    write_parquet(outputs["licensing_rules_raw_audit"], raw_audit_path)
    write_parquet(outputs["licensing_rules_harmonized"], harmonized_rules_path)
    write_parquet(outputs["licensing_stringency_index"], stringency_index_path)
    write_parquet(outputs["licensing_harmonization_summary"], harmonization_summary_path)

    provenance_kwargs = {
        "source_files": [path for path in (licensing_output, licensing_rules_long_path, config_path) if path.exists()],
        "parameters": {
            "sample_mode": sample,
            "requested_year": year,
            "extension_name": config.get("name", "licensing_iv"),
            "mode": "licensing_harmonization",
            "selected_input_path": str(selected_input_path),
            "selected_input_contract": "rule_level" if selected_input_path == licensing_rules_long_path else "wide_shock",
        },
        "config": config,
        "repo_root": paths.root,
    }
    for path in (
        raw_audit_path,
        harmonized_rules_path,
        stringency_index_path,
        harmonization_summary_path,
    ):
        write_provenance_sidecar(path, **provenance_kwargs)
    LOGGER.info(
        "built licensing harmonization outputs: raw=%s harmonized=%s stringency=%s summary=%s",
        len(outputs["licensing_rules_raw_audit"]),
        len(outputs["licensing_rules_harmonized"]),
        len(outputs["licensing_stringency_index"]),
        len(outputs["licensing_harmonization_summary"]),
    )


def build_licensing_iv(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "licensing_iv",
) -> None:
    config_path = resolve_extension_config_path(paths.root, config_name_or_path)
    config = load_extension_config(paths.root, config_name_or_path)
    if not config:
        raise UnpaidWorkError(f"extension config is empty or missing: {config_path}")

    county_path = paths.processed / "childcare_county_year_price_panel.parquet"
    state_path = paths.processed / "childcare_state_year_panel.parquet"
    if not county_path.exists() or not state_path.exists():
        build_childcare(paths, sample=sample, refresh=refresh, dry_run=dry_run, year=year)
    _ensure_licensing_harmonization_artifacts(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=dry_run,
        year=year,
        config_name_or_path=config_name_or_path,
    )

    if dry_run:
        LOGGER.info("planned licensing IV backend build using %s", config_path)
        return

    if not county_path.exists() or not state_path.exists():
        raise UnpaidWorkError(f"missing childcare county/state panels: {county_path} and/or {state_path}")

    output_config = config.get("output_namespace", {})
    output_namespace = _output_namespace_path(
        paths,
        str(output_config.get("namespace", "data/interim/childcare/licensing_iv")),
    )
    file_names = output_config.get("files", {})
    harmonized_rules_path = output_namespace / file_names.get("harmonized_rules", "licensing_rules_harmonized.parquet")
    stringency_index_path = output_namespace / file_names.get("stringency_index", "licensing_stringency_index.parquet")
    event_study_results_path = output_namespace / file_names.get(
        "event_study_results", "licensing_event_study_results.parquet"
    )
    iv_results_path = output_namespace / file_names.get("iv_results", "licensing_iv_results.parquet")
    iv_usability_summary_path = output_namespace / file_names.get(
        "iv_usability_summary", "licensing_iv_usability_summary.parquet"
    )
    first_stage_path = output_namespace / file_names.get(
        "first_stage_diagnostics", "licensing_first_stage_diagnostics.parquet"
    )
    treatment_timing_path = output_namespace / file_names.get(
        "treatment_timing", "licensing_treatment_timing.parquet"
    )
    leave_one_state_out_path = output_namespace / file_names.get(
        "leave_one_state_out", "licensing_leave_one_state_out.parquet"
    )
    elasticity_panel_path = output_namespace / file_names.get("elasticity_panel", "supply_iv_elasticities.parquet")
    iv_summary_path = output_namespace / file_names.get("iv_summary_json", "licensing_iv_summary.json")

    if not harmonized_rules_path.exists() or not stringency_index_path.exists():
        raise UnpaidWorkError(
            f"missing harmonized licensing inputs: {harmonized_rules_path} and/or {stringency_index_path}"
        )

    from unpriced.childcare.licensing_iv_backend import build_licensing_iv_backend_outputs

    licensing_output = paths.interim / "licensing" / ("licensing.parquet" if sample else "licensing_supply_shocks.parquet")
    if not licensing_output.exists():
        result = licensing.ingest(paths, sample=sample, refresh=refresh, dry_run=False, year=year)
        licensing_output = result.normalized_path

    county = read_parquet(county_path)
    state = read_parquet(state_path)
    harmonized_rules = read_parquet(harmonized_rules_path)
    stringency_index = read_parquet(stringency_index_path)
    outputs = build_licensing_iv_backend_outputs(
        harmonized_rules,
        stringency_index,
        county,
        state,
    )
    elasticity_panel_source = pd.DataFrame()
    elasticity_panel_source_name = "missing"
    if licensing_output.exists():
        raw_elasticity_source = read_parquet(licensing_output)
        if {"state_fips", "year"} <= set(raw_elasticity_source.columns) and (
            "center_labor_intensity_index" in raw_elasticity_source.columns
            or {
                "center_infant_ratio",
                "center_toddler_ratio",
                "center_infant_group_size",
                "center_toddler_group_size",
            }
            & set(raw_elasticity_source.columns)
        ):
            elasticity_panel_source = raw_elasticity_source
            elasticity_panel_source_name = "licensing_shock_panel"
    if elasticity_panel_source.empty:
        elasticity_panel_source = stringency_index.rename(
            columns={"stringency_equal_weight_index": "center_labor_intensity_index"}
        )
        if not elasticity_panel_source.empty:
            elasticity_panel_source_name = "stringency_index_fallback"
    elasticity_panel = (
        build_supply_iv_panel(county, elasticity_panel_source)
        if not elasticity_panel_source.empty
        else pd.DataFrame()
    )

    write_parquet(outputs["event_study_results"], event_study_results_path)
    write_parquet(outputs["iv_results"], iv_results_path)
    write_parquet(outputs["iv_usability_summary"], iv_usability_summary_path)
    write_parquet(outputs["first_stage_diagnostics"], first_stage_path)
    write_parquet(outputs["treatment_timing"], treatment_timing_path)
    write_parquet(outputs["leave_one_state_out"], leave_one_state_out_path)
    write_parquet(elasticity_panel, elasticity_panel_path)
    write_json(outputs["summary"], iv_summary_path)

    provenance_kwargs = {
        "source_files": [
            path
            for path in (
                county_path,
                state_path,
                harmonized_rules_path,
                stringency_index_path,
                licensing_output,
                config_path,
            )
            if path.exists()
        ],
        "parameters": {
            "sample_mode": sample,
            "requested_year": year,
            "extension_name": config.get("name", "licensing_iv"),
            "mode": "licensing_iv_backend",
            "elasticity_panel_source": elasticity_panel_source_name,
        },
        "config": config,
        "repo_root": paths.root,
    }
    for path in (
        event_study_results_path,
        iv_results_path,
        iv_usability_summary_path,
        first_stage_path,
        treatment_timing_path,
        leave_one_state_out_path,
        elasticity_panel_path,
        iv_summary_path,
    ):
        write_provenance_sidecar(path, **provenance_kwargs)
    LOGGER.info(
        "built licensing IV backend outputs: event=%s iv=%s usability=%s first_stage=%s timing=%s loo=%s elasticity=%s",
        len(outputs["event_study_results"]),
        len(outputs["iv_results"]),
        len(outputs["iv_usability_summary"]),
        len(outputs["first_stage_diagnostics"]),
        len(outputs["treatment_timing"]),
        len(outputs["leave_one_state_out"]),
        len(elasticity_panel),
    )


def build_childcare_release_backend(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
    config_name_or_path: str = "release_backend",
) -> None:
    config_path = resolve_extension_config_path(paths.root, config_name_or_path)
    config = load_extension_config(paths.root, config_name_or_path)
    if not config:
        raise UnpaidWorkError(f"extension config is empty or missing: {config_path}")

    if dry_run:
        LOGGER.info("planned childcare backend release build using %s", config_path)
        return

    ccdf_namespace = paths.root / "data" / "interim" / "ccdf"
    required_ccdf_paths = (
        ccdf_namespace / "ccdf_admin_state_year.parquet",
        ccdf_namespace / "ccdf_policy_controls_coverage.parquet",
        ccdf_namespace / "ccdf_policy_promoted_controls_state_year.parquet",
    )
    if refresh or not all(path.exists() for path in required_ccdf_paths):
        build_ccdf_state_year(
            paths,
            sample=sample,
            refresh=refresh,
            dry_run=False,
            year=year,
            config_name_or_path="ccdf_state_year",
        )
    _ensure_childcare_segmented_report_artifacts(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=False,
        year=year,
    )
    _ensure_licensing_iv_artifacts(
        paths,
        sample=sample,
        refresh=refresh,
        dry_run=False,
        year=year,
        config_name_or_path="licensing_iv",
    )

    from unpriced.childcare.release_backend import (
        build_childcare_release_backend_outputs,
        build_childcare_release_bundle_index,
        build_childcare_frontend_handoff_summary,
        build_childcare_release_contract,
        build_childcare_release_manifest,
        build_childcare_release_manual_requirements,
        build_childcare_release_schema_inventory,
        build_childcare_release_source_readiness,
    )

    segmented_report_namespace = paths.root / "data" / "interim" / "childcare" / "segmented_reports"
    segmented_scenario_namespace = paths.root / "data" / "interim" / "childcare" / "segmented_scenarios"
    licensing_namespace = paths.root / "data" / "interim" / "childcare" / "licensing_iv"

    pooled_summary = _load_pooled_backend_summary(paths, sample=sample, refresh=refresh)
    source_config = config.get("sources", {})
    source_readiness = build_childcare_release_source_readiness(
        required_sources=list(source_config.get("required", []) or []),
        optional_sources=list(source_config.get("optional", []) or []),
        sample_mode=sample,
        repo_root=paths.root,
    )
    manual_requirements = build_childcare_release_manual_requirements(
        source_readiness,
        sample_mode=sample,
    )
    release_outputs = build_childcare_release_backend_outputs(
        pooled_headline_summary=pooled_summary,
        ccdf_admin_state_year=_read_optional_parquet(ccdf_namespace / "ccdf_admin_state_year.parquet"),
        ccdf_policy_controls_coverage=_read_optional_parquet(ccdf_namespace / "ccdf_policy_controls_coverage.parquet"),
        ccdf_policy_promoted_controls_state_year=_read_optional_parquet(
            ccdf_namespace / "ccdf_policy_promoted_controls_state_year.parquet"
        ),
        segmented_state_year_summary=_read_optional_parquet(
            segmented_scenario_namespace / "segmented_state_year_summary.parquet"
        ),
        segmented_state_fallback_summary=_read_optional_parquet(
            segmented_report_namespace / "segmented_state_fallback_summary.parquet"
        ),
        segmented_channel_scenarios=_read_optional_parquet(
            segmented_scenario_namespace / "segmented_channel_scenarios.parquet"
        ),
        licensing_rules_raw_audit=_read_optional_parquet(licensing_namespace / "licensing_rules_raw_audit.parquet"),
        licensing_harmonized_rules=_read_optional_parquet(
            licensing_namespace / "licensing_rules_harmonized.parquet"
        ),
        licensing_stringency_index=_read_optional_parquet(
            licensing_namespace / "licensing_stringency_index.parquet"
        ),
        licensing_iv_summary=_read_optional_json(licensing_namespace / "licensing_iv_summary.json"),
        licensing_iv_results=_read_optional_parquet(licensing_namespace / "licensing_iv_results.parquet"),
        licensing_iv_usability_summary=_read_optional_parquet(
            licensing_namespace / "licensing_iv_usability_summary.parquet"
        ),
        licensing_first_stage_diagnostics=_read_optional_parquet(
            licensing_namespace / "licensing_first_stage_diagnostics.parquet"
        ),
        licensing_treatment_timing=_read_optional_parquet(
            licensing_namespace / "licensing_treatment_timing.parquet"
        ),
        licensing_leave_one_state_out=_read_optional_parquet(
            licensing_namespace / "licensing_leave_one_state_out.parquet"
        ),
        release_source_readiness=source_readiness,
        release_manual_requirements=manual_requirements,
    )

    output_config = config.get("outputs", {})
    reports_namespace = _output_namespace_path(
        paths,
        str(output_config.get("reports_namespace", "outputs/reports")),
    )
    tables_namespace = _output_namespace_path(
        paths,
        str(output_config.get("tables_namespace", "outputs/tables")),
    )
    file_names = output_config.get("files", {})
    headline_summary_json_path = reports_namespace / file_names.get(
        "headline_summary_json", "childcare_backend_release_headline_summary.json"
    )
    methods_summary_json_path = reports_namespace / file_names.get(
        "methods_summary_json", "childcare_backend_release_methods_summary.json"
    )
    methods_markdown_path = reports_namespace / file_names.get(
        "methods_markdown", "childcare_backend_release_methods.md"
    )
    rebuild_markdown_path = reports_namespace / file_names.get(
        "rebuild_markdown", "childcare_backend_release_rebuild.md"
    )
    manifest_json_path = reports_namespace / file_names.get(
        "manifest_json", "childcare_backend_release_manifest.json"
    )
    source_readiness_json_path = reports_namespace / file_names.get(
        "source_readiness_json", "childcare_backend_release_source_readiness.json"
    )
    manual_requirements_json_path = reports_namespace / file_names.get(
        "manual_requirements_json", "childcare_backend_release_manual_requirements.json"
    )
    manual_requirements_markdown_path = reports_namespace / file_names.get(
        "manual_requirements_markdown", "childcare_backend_release_manual_requirements.md"
    )
    schema_inventory_json_path = reports_namespace / file_names.get(
        "schema_inventory_json", "childcare_backend_release_schema_inventory.json"
    )
    contract_json_path = reports_namespace / file_names.get(
        "contract_json",
        file_names.get("release_contract_json", "childcare_backend_release_contract.json"),
    )
    bundle_index_json_path = reports_namespace / file_names.get(
        "bundle_index_json", "childcare_backend_release_bundle_index.json"
    )
    headline_summary_csv_path = tables_namespace / file_names.get(
        "headline_summary_csv", "childcare_backend_release_headline_summary.csv"
    )
    support_quality_csv_path = tables_namespace / file_names.get(
        "support_quality_csv", "childcare_backend_support_quality_summary.csv"
    )
    ccdf_support_mix_csv_path = tables_namespace / file_names.get(
        "ccdf_support_mix_csv", "childcare_backend_ccdf_support_mix_summary.csv"
    )
    policy_coverage_csv_path = tables_namespace / file_names.get(
        "policy_coverage_csv", "childcare_backend_policy_coverage_summary.csv"
    )
    ccdf_proxy_gap_summary_csv_path = tables_namespace / file_names.get(
        "ccdf_proxy_gap_summary_csv", "childcare_backend_ccdf_proxy_gap_summary.csv"
    )
    ccdf_proxy_gap_state_years_csv_path = tables_namespace / file_names.get(
        "ccdf_proxy_gap_state_years_csv", "childcare_backend_ccdf_proxy_gap_state_years.csv"
    )
    segmented_comparison_csv_path = tables_namespace / file_names.get(
        "segmented_comparison_csv", "childcare_backend_segmented_comparison.csv"
    )
    licensing_iv_results_csv_path = tables_namespace / file_names.get(
        "licensing_iv_results_csv", "childcare_backend_licensing_iv_results.csv"
    )
    licensing_iv_usability_csv_path = tables_namespace / file_names.get(
        "licensing_iv_usability_summary_csv",
        "childcare_backend_licensing_iv_usability_summary.csv",
    )
    first_stage_csv_path = tables_namespace / file_names.get(
        "licensing_first_stage_diagnostics_csv",
        "childcare_backend_licensing_first_stage_diagnostics.csv",
    )
    treatment_timing_csv_path = tables_namespace / file_names.get(
        "licensing_treatment_timing_csv", "childcare_backend_licensing_treatment_timing.csv"
    )
    leave_one_state_out_csv_path = tables_namespace / file_names.get(
        "licensing_leave_one_state_out_csv", "childcare_backend_licensing_leave_one_state_out.csv"
    )
    manifest_csv_path = tables_namespace / file_names.get(
        "manifest_csv", "childcare_backend_release_manifest.csv"
    )
    source_readiness_csv_path = tables_namespace / file_names.get(
        "source_readiness_csv", "childcare_backend_release_source_readiness.csv"
    )
    manual_requirements_csv_path = tables_namespace / file_names.get(
        "manual_requirements_csv", "childcare_backend_release_manual_requirements.csv"
    )
    schema_artifact_csv_path = tables_namespace / file_names.get(
        "schema_artifact_csv", "childcare_backend_release_schema_artifact_summary.csv"
    )
    schema_columns_csv_path = tables_namespace / file_names.get(
        "schema_columns_csv", "childcare_backend_release_schema_column_summary.csv"
    )
    contract_artifacts_csv_path = tables_namespace / file_names.get(
        "contract_artifacts_csv",
        file_names.get("release_artifact_contracts_csv", "childcare_backend_release_artifact_contracts.csv"),
    )
    contract_columns_csv_path = tables_namespace / file_names.get(
        "contract_columns_csv",
        file_names.get("release_column_dictionary_csv", "childcare_backend_release_column_dictionary.csv"),
    )
    frontend_handoff_summary_json_path = reports_namespace / file_names.get(
        "frontend_handoff_summary_json",
        "childcare_backend_release_frontend_handoff_summary.json",
    )
    bundle_index_csv_path = tables_namespace / file_names.get(
        "bundle_index_csv", "childcare_backend_release_bundle_index.csv"
    )

    headline_summary = release_outputs["release_headline_summary"]
    support_tables = release_outputs["release_support_tables"]
    proxy_gap_tables = release_outputs["release_ccdf_proxy_gap_tables"]
    methods_summary = release_outputs["release_methods_summary"]
    segmented_comparison = release_outputs["release_segmented_comparison"]
    source_readiness_output = release_outputs["release_source_readiness"]
    manual_requirements_output = release_outputs["release_manual_requirements"]
    licensing_iv_results = _read_optional_parquet(licensing_namespace / "licensing_iv_results.parquet")
    if licensing_iv_results is None:
        licensing_iv_results = pd.DataFrame()
    licensing_iv_usability_summary = _read_optional_parquet(
        licensing_namespace / "licensing_iv_usability_summary.parquet"
    )
    if licensing_iv_usability_summary is None:
        licensing_iv_usability_summary = pd.DataFrame()
    first_stage_diagnostics = _read_optional_parquet(
        licensing_namespace / "licensing_first_stage_diagnostics.parquet"
    )
    if first_stage_diagnostics is None:
        first_stage_diagnostics = pd.DataFrame()
    treatment_timing = _read_optional_parquet(licensing_namespace / "licensing_treatment_timing.parquet")
    if treatment_timing is None:
        treatment_timing = pd.DataFrame()
    leave_one_state_out = _read_optional_parquet(licensing_namespace / "licensing_leave_one_state_out.parquet")
    if leave_one_state_out is None:
        leave_one_state_out = pd.DataFrame()

    write_json(headline_summary["json"], headline_summary_json_path)
    _write_csv_output(headline_summary["table"], headline_summary_csv_path)
    write_json(methods_summary["json"], methods_summary_json_path)
    methods_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    methods_markdown_path.write_text(methods_summary["markdown"], encoding="utf-8")
    rebuild_markdown = "\n".join(
        [
            "# Childcare backend rebuild",
            "",
            "Run the backend release with these commands:",
            "",
            f"1. `python -m unpriced.cli pull-ccdf {'--sample' if sample else '--real'}{' --refresh' if refresh else ''}`",
            f"2. `python -m unpriced.cli build-ccdf-state-year {'--sample' if sample else '--real'}{' --refresh' if refresh else ''}`",
            f"3. `python -m unpriced.cli build-childcare-segmented-report {'--sample' if sample else '--real'}{' --refresh' if refresh else ''}`",
            f"4. `python -m unpriced.cli build-licensing-harmonization {'--sample' if sample else '--real'}{' --refresh' if refresh else ''}`",
            f"5. `python -m unpriced.cli build-licensing-iv {'--sample' if sample else '--real'}{' --refresh' if refresh else ''}`",
            f"6. `python -m unpriced.cli build-childcare-release-backend {'--sample' if sample else '--real'}{' --refresh' if refresh else ''}`",
            "",
            "The pooled childcare path remains canonical. Segmented outputs are additive-only.",
        ]
    )
    rebuild_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    rebuild_markdown_path.write_text(rebuild_markdown, encoding="utf-8")
    write_json(source_readiness_output["json"], source_readiness_json_path)
    write_json(manual_requirements_output["json"], manual_requirements_json_path)
    manual_requirements_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    manual_requirements_markdown_path.write_text(manual_requirements_output["markdown"], encoding="utf-8")
    _write_csv_output(support_tables["support_quality_summary"], support_quality_csv_path)
    _write_csv_output(support_tables["ccdf_support_mix_summary"], ccdf_support_mix_csv_path)
    _write_csv_output(support_tables["policy_coverage_summary"], policy_coverage_csv_path)
    _write_csv_output(proxy_gap_tables["proxy_gap_summary"], ccdf_proxy_gap_summary_csv_path)
    _write_csv_output(proxy_gap_tables["proxy_gap_state_years"], ccdf_proxy_gap_state_years_csv_path)
    _write_csv_output(segmented_comparison, segmented_comparison_csv_path)
    _write_csv_output(licensing_iv_results, licensing_iv_results_csv_path)
    _write_csv_output(licensing_iv_usability_summary, licensing_iv_usability_csv_path)
    _write_csv_output(first_stage_diagnostics, first_stage_csv_path)
    _write_csv_output(treatment_timing, treatment_timing_csv_path)
    _write_csv_output(leave_one_state_out, leave_one_state_out_csv_path)
    _write_csv_output(source_readiness_output["table"], source_readiness_csv_path)
    _write_csv_output(manual_requirements_output["table"], manual_requirements_csv_path)

    source_files = [
        ccdf_namespace / "ccdf_admin_state_year.parquet",
        ccdf_namespace / "ccdf_policy_controls_coverage.parquet",
        ccdf_namespace / "ccdf_policy_promoted_controls_state_year.parquet",
        segmented_report_namespace / "segmented_state_fallback_summary.parquet",
        segmented_scenario_namespace / "segmented_channel_scenarios.parquet",
        licensing_namespace / "licensing_rules_raw_audit.parquet",
        licensing_namespace / "licensing_rules_harmonized.parquet",
        licensing_namespace / "licensing_stringency_index.parquet",
        licensing_namespace / "licensing_iv_results.parquet",
        licensing_namespace / "licensing_iv_usability_summary.parquet",
        licensing_namespace / "licensing_first_stage_diagnostics.parquet",
        licensing_namespace / "licensing_treatment_timing.parquet",
        licensing_namespace / "licensing_leave_one_state_out.parquet",
        licensing_namespace / "licensing_iv_summary.json",
        config_path,
    ]
    provenance_kwargs = {
        "source_files": [path for path in source_files if path.exists()],
        "parameters": {
            "sample_mode": sample,
            "requested_year": year,
            "extension_name": config.get("name", "release_backend"),
            "mode": config.get("mode", "backend_release"),
        },
        "config": config,
        "repo_root": paths.root,
    }
    producer_command = (
        f"python -m unpriced.cli build-childcare-release-backend "
        f"{'--sample' if sample else '--real'}{' --refresh' if refresh else ''}"
    )

    def _published_artifact(
        artifact_name: str,
        artifact_group: str,
        artifact_kind: str,
        artifact_status: str,
        frontend_priority: str,
        publication_tier: str,
        output_path: Path,
    ) -> dict[str, object]:
        return {
            "artifact_name": artifact_name,
            "artifact_group": artifact_group,
            "artifact_kind": artifact_kind,
            "artifact_status": artifact_status,
            "frontend_priority": frontend_priority,
            "publication_tier": publication_tier,
            "producer_command": producer_command,
            "output_path": output_path,
            "provenance_path": Path(f"{output_path}.provenance.json"),
            "sample_available": True,
            "real_available": True,
            "upstream_dependencies": provenance_kwargs["source_files"],
        }

    provisional_published_artifacts = [
        _published_artifact("release_headline_summary", "release", "json", "canonical", "high", "headline", headline_summary_json_path),
        _published_artifact("release_headline_summary_csv", "release", "csv", "canonical", "high", "headline", headline_summary_csv_path),
        _published_artifact("methods_summary", "release", "json", "canonical", "high", "methods", methods_summary_json_path),
        _published_artifact("methods_markdown", "release", "markdown", "canonical", "medium", "methods", methods_markdown_path),
        _published_artifact("release_rebuild_markdown", "release", "markdown", "canonical", "medium", "methods", rebuild_markdown_path),
        _published_artifact("release_source_readiness_json", "release", "json", "diagnostic", "medium", "appendix", source_readiness_json_path),
        _published_artifact("release_manual_requirements_json", "release", "json", "diagnostic", "medium", "appendix", manual_requirements_json_path),
        _published_artifact("release_manual_requirements_markdown", "release", "markdown", "diagnostic", "low", "appendix", manual_requirements_markdown_path),
        _published_artifact("support_quality_summary", "release", "csv", "canonical", "high", "appendix", support_quality_csv_path),
        _published_artifact("ccdf_support_mix_summary", "release", "csv", "canonical", "high", "appendix", ccdf_support_mix_csv_path),
        _published_artifact("policy_coverage_summary", "release", "csv", "canonical", "medium", "appendix", policy_coverage_csv_path),
        _published_artifact("ccdf_proxy_gap_summary", "release", "csv", "diagnostic", "medium", "appendix", ccdf_proxy_gap_summary_csv_path),
        _published_artifact("ccdf_proxy_gap_state_years", "release", "csv", "diagnostic", "low", "appendix", ccdf_proxy_gap_state_years_csv_path),
        _published_artifact("segmented_comparison", "segmented", "csv", "additive", "medium", "appendix", segmented_comparison_csv_path),
        _published_artifact("licensing_iv_results", "licensing", "csv", "canonical", "high", "appendix", licensing_iv_results_csv_path),
        _published_artifact("licensing_iv_usability_summary", "licensing", "csv", "diagnostic", "high", "appendix", licensing_iv_usability_csv_path),
        _published_artifact("licensing_first_stage_diagnostics", "licensing", "csv", "diagnostic", "medium", "appendix", first_stage_csv_path),
        _published_artifact("licensing_treatment_timing", "licensing", "csv", "diagnostic", "medium", "appendix", treatment_timing_csv_path),
        _published_artifact("licensing_leave_one_state_out", "licensing", "csv", "diagnostic", "low", "appendix", leave_one_state_out_csv_path),
    ]
    frontend_handoff_summary = build_childcare_frontend_handoff_summary(
        published_artifacts=provisional_published_artifacts,
        headline_summary=headline_summary["json"],
        methods_summary=methods_summary["json"],
    )
    write_json(frontend_handoff_summary, frontend_handoff_summary_json_path)

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
        "release_source_readiness_table": source_readiness_output["table"],
        "release_source_readiness_summary": source_readiness_output["summary"],
        "release_manual_requirements_table": manual_requirements_output["table"],
        "release_manual_requirements_summary": manual_requirements_output["summary"],
        "release_manual_requirements_markdown": manual_requirements_output["markdown"],
        "release_frontend_handoff_summary": frontend_handoff_summary,
        "licensing_iv_results": licensing_iv_results,
        "licensing_iv_usability_summary": licensing_iv_usability_summary,
        "licensing_first_stage_diagnostics": first_stage_diagnostics,
        "licensing_treatment_timing": treatment_timing,
        "licensing_leave_one_state_out": leave_one_state_out,
    }
    schema_inventory = build_childcare_release_schema_inventory(release_artifacts)
    contract_inventory = build_childcare_release_contract(release_artifacts)
    manifest = build_childcare_release_manifest(
        release_artifacts
        | {
            "release_schema_artifact_summary": schema_inventory["artifact_summary"],
            "release_schema_column_schema": schema_inventory["column_schema"],
            "release_artifact_contracts": contract_inventory["artifact_contracts"],
            "release_column_dictionary": contract_inventory["column_dictionary"],
        }
    )

    write_json({"rows": manifest.to_dict(orient="records")}, manifest_json_path)
    write_json(schema_inventory["json"], schema_inventory_json_path)
    write_json(contract_inventory["json"], contract_json_path)
    _write_csv_output(manifest, manifest_csv_path)
    _write_csv_output(schema_inventory["artifact_summary"], schema_artifact_csv_path)
    _write_csv_output(schema_inventory["column_schema"], schema_columns_csv_path)
    _write_csv_output(contract_inventory["artifact_contracts"], contract_artifacts_csv_path)
    _write_csv_output(contract_inventory["column_dictionary"], contract_columns_csv_path)

    published_artifacts = provisional_published_artifacts + [
        _published_artifact("release_frontend_handoff_summary", "release", "json", "canonical", "high", "contract", frontend_handoff_summary_json_path),
        _published_artifact("release_manifest_json", "release", "json", "diagnostic", "medium", "appendix", manifest_json_path),
        _published_artifact("release_manifest_csv", "release", "csv", "diagnostic", "medium", "appendix", manifest_csv_path),
        _published_artifact("release_schema_inventory_json", "release", "json", "diagnostic", "medium", "appendix", schema_inventory_json_path),
        _published_artifact("release_schema_artifact_summary", "release", "csv", "diagnostic", "medium", "appendix", schema_artifact_csv_path),
        _published_artifact("release_schema_column_summary", "release", "csv", "diagnostic", "medium", "appendix", schema_columns_csv_path),
        _published_artifact("release_contract_json", "release", "json", "diagnostic", "high", "contract", contract_json_path),
        _published_artifact("release_artifact_contracts_csv", "release", "csv", "diagnostic", "high", "contract", contract_artifacts_csv_path),
        _published_artifact("release_column_dictionary_csv", "release", "csv", "diagnostic", "high", "contract", contract_columns_csv_path),
        _published_artifact("release_source_readiness_csv", "release", "csv", "diagnostic", "medium", "appendix", source_readiness_csv_path),
        _published_artifact("release_manual_requirements_csv", "release", "csv", "diagnostic", "medium", "appendix", manual_requirements_csv_path),
    ]
    bundle_index = build_childcare_release_bundle_index(
        published_artifacts=published_artifacts,
        current_mode="sample" if sample else "real",
    )
    write_json(bundle_index["json"], bundle_index_json_path)
    _write_csv_output(bundle_index["table"], bundle_index_csv_path)

    release_artifacts_with_bundle = release_artifacts | {
        "release_bundle_index": bundle_index["table"],
        "release_bundle_index_json": bundle_index["json"],
    }
    schema_inventory = build_childcare_release_schema_inventory(release_artifacts_with_bundle)
    contract_inventory = build_childcare_release_contract(release_artifacts_with_bundle)
    manifest = build_childcare_release_manifest(
        release_artifacts_with_bundle
        | {
            "release_schema_artifact_summary": schema_inventory["artifact_summary"],
            "release_schema_column_schema": schema_inventory["column_schema"],
            "release_artifact_contracts": contract_inventory["artifact_contracts"],
            "release_column_dictionary": contract_inventory["column_dictionary"],
        }
    )
    write_json({"rows": manifest.to_dict(orient="records")}, manifest_json_path)
    write_json(schema_inventory["json"], schema_inventory_json_path)
    write_json(contract_inventory["json"], contract_json_path)
    _write_csv_output(manifest, manifest_csv_path)
    _write_csv_output(schema_inventory["artifact_summary"], schema_artifact_csv_path)
    _write_csv_output(schema_inventory["column_schema"], schema_columns_csv_path)
    _write_csv_output(contract_inventory["artifact_contracts"], contract_artifacts_csv_path)
    _write_csv_output(contract_inventory["column_dictionary"], contract_columns_csv_path)

    for path in (
        headline_summary_json_path,
        methods_summary_json_path,
        methods_markdown_path,
        rebuild_markdown_path,
        manifest_json_path,
        source_readiness_json_path,
        manual_requirements_json_path,
        manual_requirements_markdown_path,
        schema_inventory_json_path,
        contract_json_path,
        frontend_handoff_summary_json_path,
        bundle_index_json_path,
        headline_summary_csv_path,
        support_quality_csv_path,
        ccdf_support_mix_csv_path,
        policy_coverage_csv_path,
        ccdf_proxy_gap_summary_csv_path,
        ccdf_proxy_gap_state_years_csv_path,
        segmented_comparison_csv_path,
        licensing_iv_results_csv_path,
        licensing_iv_usability_csv_path,
        first_stage_csv_path,
        treatment_timing_csv_path,
        leave_one_state_out_csv_path,
        manifest_csv_path,
        source_readiness_csv_path,
        manual_requirements_csv_path,
        schema_artifact_csv_path,
        schema_columns_csv_path,
        contract_artifacts_csv_path,
        contract_columns_csv_path,
    ):
        write_provenance_sidecar(path, **provenance_kwargs)
    bundle_index = build_childcare_release_bundle_index(
        published_artifacts=published_artifacts,
        current_mode="sample" if sample else "real",
    )
    write_json(bundle_index["json"], bundle_index_json_path)
    _write_csv_output(bundle_index["table"], bundle_index_csv_path)
    write_provenance_sidecar(bundle_index_json_path, **provenance_kwargs)
    write_provenance_sidecar(bundle_index_csv_path, **provenance_kwargs)
    LOGGER.info(
        "built childcare backend release outputs: support=%s policy=%s proxy_summary=%s proxy_rows=%s segmented=%s manifest=%s",
        len(support_tables["support_quality_summary"]),
        len(support_tables["policy_coverage_summary"]),
        len(proxy_gap_tables["proxy_gap_summary"]),
        len(proxy_gap_tables["proxy_gap_state_years"]),
        len(segmented_comparison),
        len(manifest),
    )


def fit_childcare(paths, sample: bool = True) -> None:
    county_path = paths.processed / "childcare_county_year_price_panel.parquet"
    state_path = paths.processed / "childcare_state_year_panel.parquet"
    if sample:
        if not county_path.exists() or not state_path.exists():
            build_childcare(paths, sample=True)
    else:
        _ensure_real_mode_artifacts(
            [
                (county_path, "processed childcare county-year panel"),
                (state_path, "processed childcare state-year panel"),
            ],
            "build-childcare --real",
        )
    county = read_parquet(county_path)
    state = read_parquet(state_path)
    if not _state_panel_has_sample_ladder(state):
        if sample:
            county, state = _refresh_childcare_panels_from_interim(paths, sample=True)
        else:
            raise UnpaidWorkError(
                "processed childcare state panel is missing current sample-ladder metadata. "
                "Rebuild with `unpriced build-childcare --real` before rerunning."
            )
    diagnose_childcare_pipeline(county, state, paths)
    fit_price_surface(
        county,
        output_json=paths.outputs_reports / "childcare_price_surface.json",
        output_panel=county_path,
    )
    broad = fit_childcare_demand_iv(
        state,
        output_json=paths.outputs_reports / "childcare_demand_iv_broad_complete.json",
        output_panel=paths.outputs_reports / "childcare_demand_iv_broad_complete.parquet",
        mode="broad_complete",
    )
    broad_canonical = fit_childcare_demand_iv(
        state,
        output_json=paths.outputs_reports / "childcare_demand_iv_broad_complete_household_parsimonious.json",
        output_panel=paths.outputs_reports / "childcare_demand_iv_broad_complete_household_parsimonious.parquet",
        mode="broad_complete",
        specification_profile=CANONICAL_COMPARISON_SPECIFICATION_PROFILE,
    )
    observed = fit_childcare_demand_iv(
        state,
        output_json=paths.outputs_reports / "childcare_demand_iv_observed_core.json",
        output_panel=paths.outputs_reports / "childcare_demand_iv_observed_core.parquet",
        mode="observed_core",
    )
    low_impute = fit_childcare_demand_iv(
        state,
        output_json=paths.outputs_reports / "childcare_demand_iv_observed_core_low_impute.json",
        output_panel=paths.outputs_reports / "childcare_demand_iv_observed_core_low_impute.parquet",
        mode="observed_core_low_impute",
    )
    low_impute_canonical = fit_childcare_demand_iv(
        state,
        output_json=paths.outputs_reports / "childcare_demand_iv_observed_core_low_impute_household_parsimonious.json",
        output_panel=paths.outputs_reports / "childcare_demand_iv_observed_core_low_impute_household_parsimonious.parquet",
        mode="observed_core_low_impute",
        specification_profile=CANONICAL_COMPARISON_SPECIFICATION_PROFILE,
    )
    observed_canonical = fit_childcare_demand_iv(
        state,
        output_json=paths.outputs_reports / "childcare_demand_iv_observed_core_household_parsimonious.json",
        output_panel=paths.outputs_reports / "childcare_demand_iv_observed_core_household_parsimonious.parquet",
        mode="observed_core",
        specification_profile=CANONICAL_OBSERVED_SPECIFICATION_PROFILE,
    )
    fit_childcare_demand_iv(
        state,
        output_json=paths.outputs_reports / "childcare_demand_iv_observed_core_instrument_only.json",
        output_panel=paths.outputs_reports / "childcare_demand_iv_observed_core_instrument_only.parquet",
        mode="observed_core",
        specification_profile="instrument_only",
    )
    fit_childcare_demand_iv(
        state,
        output_json=paths.outputs_reports / "childcare_demand_iv_observed_core_labor_parsimonious.json",
        output_panel=paths.outputs_reports / "childcare_demand_iv_observed_core_labor_parsimonious.parquet",
        mode="observed_core",
        specification_profile="labor_parsimonious",
    )
    comparison = build_childcare_demand_sample_comparison(
        state,
        output_json=paths.outputs_reports / "childcare_demand_sample_comparison.json",
        specification_profile=CANONICAL_COMPARISON_SPECIFICATION_PROFILE,
    )
    build_childcare_imputation_sweep(
        state,
        output_json=paths.outputs_reports / "childcare_demand_imputation_sweep.json",
        specification_profile=CANONICAL_COMPARISON_SPECIFICATION_PROFILE,
    )
    build_childcare_labor_support_sweep(
        state,
        output_json=paths.outputs_reports / "childcare_demand_labor_support_sweep.json",
        specification_profile=CANONICAL_COMPARISON_SPECIFICATION_PROFILE,
    )
    build_childcare_specification_sweep(
        state,
        output_json=paths.outputs_reports / "childcare_demand_specification_sweep.json",
        mode="observed_core",
        current_profile=CANONICAL_OBSERVED_SPECIFICATION_PROFILE,
    )
    write_json(broad, paths.outputs_reports / "childcare_demand_iv.json")
    write_json(observed, paths.outputs_reports / "childcare_demand_iv_strict.json")
    selected_sample = comparison.get("selected_headline_sample")
    if selected_sample == "observed_core":
        write_json(observed_canonical, paths.outputs_reports / "childcare_demand_iv_canonical.json")
    elif selected_sample == "observed_core_low_impute":
        write_json(low_impute_canonical, paths.outputs_reports / "childcare_demand_iv_canonical.json")
    elif selected_sample == "broad_complete":
        write_json(broad_canonical, paths.outputs_reports / "childcare_demand_iv_canonical.json")
    else:
        write_json(observed, paths.outputs_reports / "childcare_demand_iv_canonical.json")
    output_artifacts = [
        paths.outputs_reports / "childcare_price_surface.json",
        paths.outputs_reports / "childcare_demand_iv_broad_complete.json",
        paths.outputs_reports / "childcare_demand_iv_broad_complete_household_parsimonious.json",
        paths.outputs_reports / "childcare_demand_iv_observed_core.json",
        paths.outputs_reports / "childcare_demand_iv_observed_core_low_impute.json",
        paths.outputs_reports / "childcare_demand_iv_observed_core_low_impute_household_parsimonious.json",
        paths.outputs_reports / "childcare_demand_iv_observed_core_household_parsimonious.json",
        paths.outputs_reports / "childcare_demand_iv_observed_core_instrument_only.json",
        paths.outputs_reports / "childcare_demand_iv_observed_core_labor_parsimonious.json",
        paths.outputs_reports / "childcare_demand_sample_comparison.json",
        paths.outputs_reports / "childcare_demand_imputation_sweep.json",
        paths.outputs_reports / "childcare_demand_labor_support_sweep.json",
        paths.outputs_reports / "childcare_demand_specification_sweep.json",
        paths.outputs_reports / "childcare_demand_iv.json",
        paths.outputs_reports / "childcare_demand_iv_strict.json",
        paths.outputs_reports / "childcare_demand_iv_canonical.json",
    ]
    for artifact in output_artifacts:
        if artifact.exists():
            _write_mode_artifact(
                artifact,
                [county_path, state_path],
                sample=sample,
                repo_root=paths.root,
                extra_parameters={"command": "fit-childcare"},
            )
    LOGGER.info(
        "fit childcare models: broad(n=%s) observed_core(n=%s) low_impute(n=%s) selected=%s",
        broad.get("n_obs"),
        observed.get("n_obs"),
        low_impute.get("n_obs"),
        selected_sample,
    )


def _simulate_childcare_sample(
    paths,
    state: pd.DataFrame,
    county: pd.DataFrame,
    alphas: list[float],
    demand_summary: dict[str, object],
    sample_name: str,
    sample: bool,
    selection_reason: str = "comparison_only",
    specification_profile: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | bool | str]]:
    childcare_assumptions = childcare_model_assumptions(paths)
    quarantine_reason: str | None = None
    try:
        demand_elasticity_signed, solver_demand_elasticity = resolve_solver_demand_elasticity(demand_summary)
    except ValueError as exc:
        quarantine_reason = str(exc)
        if selection_reason in {"comparison_only", "specification_sensitivity"}:
            empty = pd.DataFrame()
            diagnostics = summarize_childcare_scenario_diagnostics(
                empty,
                skipped_state_rows=0,
                demand_summary=demand_summary,
                demand_sample_name=sample_name,
                demand_sample_selection_reason=selection_reason,
            )
            diagnostics["current_mode"] = "sample" if sample else "real"
            diagnostics["demand_fit_quarantined"] = True
            diagnostics["demand_fit_quarantine_reason"] = quarantine_reason
            return empty, diagnostics
        raise UnpaidWorkError(quarantine_reason) from exc
    supply_summary = summarize_supply_elasticity(county)
    supply_elasticity = float(supply_summary["supply_elasticity"])
    eligible_column = f"eligible_{sample_name}"
    if eligible_column not in state.columns:
        raise UnpaidWorkError(f"state panel missing eligibility column for sample: {eligible_column}")
    state_selected = state.loc[state[eligible_column].fillna(False).astype(bool)].copy()
    state_valid = prepare_childcare_scenario_inputs(state_selected)
    skipped = len(state_selected) - len(state_valid)
    rows = []
    for row in state_valid.to_dict(orient="records"):
        baseline = float(row["state_price_index"])
        market_q = float(row["market_quantity_proxy"])
        unpaid_q = float(row["unpaid_quantity_proxy"])
        direct_care_share = float(
            pd.to_numeric(pd.Series([row.get("state_direct_care_labor_share")]), errors="coerce").fillna(0.0).iloc[0]
        )
        effective_children_per_worker = float(
            pd.to_numeric(pd.Series([row.get("state_effective_children_per_worker")]), errors="coerce")
            .fillna(float(childcare_assumptions["default_children_per_worker"]))
            .iloc[0]
        )
        try:
            shadow_result = solve_price(
            baseline,
            market_q,
            unpaid_q,
            solver_demand_elasticity,
            supply_elasticity,
            MARGINAL_ALPHA,
            return_metadata=True,
            )
            solved = solve_alpha_grid(
            baseline,
            market_q,
            unpaid_q,
            solver_demand_elasticity,
            supply_elasticity,
            alphas,
            return_metadata=True,
            )
        except RuntimeError as exc:
            if selection_reason in {"comparison_only", "specification_sensitivity"}:
                empty = pd.DataFrame()
                diagnostics = summarize_childcare_scenario_diagnostics(
                    empty,
                    skipped_state_rows=skipped,
                    demand_summary=demand_summary,
                    demand_sample_name=sample_name,
                    demand_sample_selection_reason=selection_reason,
                )
                diagnostics["current_mode"] = "sample" if sample else "real"
                diagnostics["demand_fit_quarantined"] = True
                diagnostics["demand_fit_quarantine_reason"] = str(exc)
                return empty, diagnostics
            raise UnpaidWorkError(str(exc)) from exc
        for result in solved:
            rows.append(
                {
                    "demand_sample_name": sample_name,
                    "demand_specification_profile": demand_summary.get("specification_profile", DEFAULT_SPECIFICATION_PROFILE),
                    "state_fips": row["state_fips"],
                    "year": row["year"],
                    "state_price_observation_status": row.get("state_price_observation_status", "unknown"),
                    "state_price_nowcast": bool(row.get("state_price_nowcast", False)),
                    "p_baseline": baseline,
                    "p_shadow_marginal": shadow_result.price,
                    "alpha": result.alpha,
                    "p_alpha": result.price,
                    "benchmark_replacement_cost": row["benchmark_replacement_cost"],
                    "state_direct_care_price_index": row.get("state_direct_care_price_index"),
                    "state_direct_care_price_index_raw": row.get("state_direct_care_price_index_raw"),
                    "state_non_direct_care_price_index": row.get("state_non_direct_care_price_index"),
                    "direct_care_labor_share": direct_care_share,
                    "direct_care_price_clip_binding": bool(row.get("state_direct_care_price_clip_binding", False)),
                    "direct_care_price_clip_binding_share": row.get("state_direct_care_price_clip_binding_share"),
                    "effective_children_per_worker": effective_children_per_worker,
                    "direct_care_fringe_multiplier": float(childcare_assumptions["direct_care_fringe_multiplier"]),
                    "demand_elasticity": demand_elasticity_signed,
                    "demand_elasticity_signed": demand_elasticity_signed,
                    "solver_demand_elasticity_magnitude": solver_demand_elasticity,
                    "solver_status": result.solver_status,
                    "solver_iterations": result.solver_iterations,
                    "solver_expansion_steps": result.solver_expansion_steps,
                    "solver_bracket_low": result.solver_low,
                    "solver_bracket_high": result.solver_high,
                    "supply_elasticity": supply_elasticity,
                    "supply_estimation_method": supply_summary.get("estimation_method"),
                    "market_quantity_proxy": market_q,
                    "unpaid_quantity_proxy": unpaid_q,
                }
            )
    scenarios, bootstrap_meta = bootstrap_childcare_intervals(
        state_valid,
        county,
        pd.DataFrame(rows),
        demand_mode=sample_name,
        demand_specification_profile=specification_profile,
    )
    if not scenarios.empty:
        direct_share = pd.to_numeric(scenarios.get("direct_care_labor_share"), errors="coerce").clip(lower=0.0, upper=1.0)
        effective_children_per_worker = pd.to_numeric(
            scenarios.get("effective_children_per_worker"), errors="coerce"
        ).fillna(float(childcare_assumptions["default_children_per_worker"]))
        for gross_col, direct_col, residual_col, wage_col in (
            ("p_baseline", "p_baseline_direct_care", "p_baseline_non_direct_care", "wage_baseline_implied"),
            ("p_shadow_marginal", "p_shadow_marginal_direct_care", "p_shadow_marginal_non_direct_care", "wage_shadow_implied"),
            ("p_alpha", "p_alpha_direct_care", "p_alpha_non_direct_care", "wage_alpha_implied"),
        ):
            gross = pd.to_numeric(scenarios[gross_col], errors="coerce")
            direct = gross * direct_share
            scenarios[direct_col] = direct
            scenarios[residual_col] = (gross - direct).clip(lower=0.0)
            scenarios[wage_col] = (
                direct
                * effective_children_per_worker
                / (
                    float(childcare_assumptions["direct_care_hours_per_year"])
                    * float(childcare_assumptions["direct_care_fringe_multiplier"])
                )
            )
    if not scenarios.empty and "demand_sample_name" not in scenarios.columns:
        scenarios["demand_sample_name"] = sample_name
    if not scenarios.empty and "demand_specification_profile" not in scenarios.columns:
        scenarios["demand_specification_profile"] = demand_summary.get("specification_profile", DEFAULT_SPECIFICATION_PROFILE)
    diagnostics = summarize_childcare_scenario_diagnostics(
        scenarios,
        skipped_state_rows=skipped,
        demand_summary=demand_summary,
        demand_sample_name=sample_name,
        demand_sample_selection_reason=selection_reason,
        bootstrap_meta=bootstrap_meta,
    )
    diagnostics["current_mode"] = "sample" if sample else "real"
    diagnostics["demand_fit_quarantined"] = False
    diagnostics["demand_fit_quarantine_reason"] = ""
    diagnostics["supply_elasticity"] = supply_elasticity
    diagnostics["supply_estimation_method"] = supply_summary.get("estimation_method", "unknown")
    diagnostics["supply_fallback_used"] = bool(supply_summary.get("fallback_used", False))
    diagnostics["supply_pooled_loglog_slope"] = float(supply_summary.get("pooled_loglog_slope", float("nan")))
    diagnostics["supply_year_demeaned_loglog_slope"] = float(
        supply_summary.get("year_demeaned_loglog_slope", float("nan"))
    )
    diagnostics["supply_within_state_year_group_count"] = int(
        supply_summary.get("within_state_year_group_count", 0)
    )
    diagnostics["supply_within_state_year_positive_group_count"] = int(
        supply_summary.get("within_state_year_positive_group_count", 0)
    )
    diagnostics["supply_within_state_year_positive_group_share"] = float(
        supply_summary.get("within_state_year_positive_group_share", 0.0)
    )
    diagnostics["supply_within_state_year_weighted_median_positive_slope"] = float(
        supply_summary.get("within_state_year_weighted_median_positive_slope", float("nan"))
    )
    diagnostics["supply_within_state_year_weighted_median_all_slope"] = float(
        supply_summary.get("within_state_year_weighted_median_all_slope", float("nan"))
    )
    diagnostics["supply_within_state_year_weighted_median_gap"] = float(
        supply_summary.get("supply_elasticity_weighted_median_gap", float("nan"))
    )
    return scenarios, diagnostics


def _canonical_specification_profile_for_sample(sample_name: str) -> str | None:
    return CANONICAL_COMPARISON_SPECIFICATION_PROFILE


def _demand_summary_path(paths, sample_name: str, specification_profile: str | None = None) -> Path:
    if specification_profile:
        candidate = paths.outputs_reports / f"childcare_demand_iv_{sample_name}_{specification_profile}.json"
        if candidate.exists():
            return candidate
    return paths.outputs_reports / f"childcare_demand_iv_{sample_name}.json"


def _selected_sample_specification_profiles(paths, selected_sample: str) -> list[str]:
    if selected_sample != "observed_core":
        return []
    sweep_path = paths.outputs_reports / "childcare_demand_specification_sweep.json"
    if not sweep_path.exists():
        return []
    sweep = read_json(sweep_path)
    profiles = sweep.get("profiles", {})
    ordered = [
        CANONICAL_OBSERVED_SPECIFICATION_PROFILE,
        "instrument_only",
        "labor_parsimonious",
        "full_controls",
    ]
    result = []
    for profile in ordered:
        if profile in profiles and profile not in result:
            result.append(profile)
    return result


def _price_decomposition_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"rows": 0, "alphas": {}}
    baseline = frame.iloc[0:0].copy()
    baseline["p_baseline"] = pd.to_numeric(frame["p_baseline"], errors="coerce")
    baseline["p_baseline_direct_care"] = pd.to_numeric(frame.get("p_baseline_direct_care"), errors="coerce")
    baseline["p_baseline_non_direct_care"] = pd.to_numeric(frame.get("p_baseline_non_direct_care"), errors="coerce")
    baseline["wage_baseline_implied"] = pd.to_numeric(frame.get("wage_baseline_implied"), errors="coerce")
    summary: dict[str, object] = {
        "rows": int(len(frame)),
        "states": int(frame["state_fips"].nunique()) if "state_fips" in frame.columns else 0,
        "years": sorted(pd.to_numeric(frame["year"], errors="coerce").dropna().astype(int).unique().tolist())
        if "year" in frame.columns
        else [],
        "baseline_price_p50": float(pd.to_numeric(frame["p_baseline"], errors="coerce").median()),
        "baseline_direct_care_price_p50": float(pd.to_numeric(frame.get("p_baseline_direct_care"), errors="coerce").median()),
        "baseline_non_direct_care_price_p50": float(pd.to_numeric(frame.get("p_baseline_non_direct_care"), errors="coerce").median()),
        "baseline_implied_wage_p50": float(pd.to_numeric(frame.get("wage_baseline_implied"), errors="coerce").median()),
        "baseline_direct_care_labor_share_p50": float(pd.to_numeric(frame.get("direct_care_labor_share"), errors="coerce").median()),
        "baseline_direct_care_clip_binding_share_p50": float(
            pd.to_numeric(frame.get("direct_care_price_clip_binding_share"), errors="coerce").median()
        ),
        "baseline_direct_care_clip_binding_row_share": float(
            pd.to_numeric(frame.get("direct_care_price_clip_binding"), errors="coerce").fillna(0.0).mean()
        ),
        "alphas": {},
    }
    for alpha in sorted(pd.to_numeric(frame["alpha"], errors="coerce").dropna().unique().tolist()):
        subset = frame.loc[pd.to_numeric(frame["alpha"], errors="coerce").round(4).eq(round(float(alpha), 4))].copy()
        summary["alphas"][f"{float(alpha):.2f}"] = {
            "price_p50": float(pd.to_numeric(subset["p_alpha"], errors="coerce").median()),
            "direct_care_price_p50": float(pd.to_numeric(subset.get("p_alpha_direct_care"), errors="coerce").median()),
            "non_direct_care_price_p50": float(pd.to_numeric(subset.get("p_alpha_non_direct_care"), errors="coerce").median()),
            "implied_wage_p50": float(pd.to_numeric(subset.get("wage_alpha_implied"), errors="coerce").median()),
        }
    return summary


def _recompute_decomposition_under_assumptions(
    frame: pd.DataFrame,
    staffing_scale: float,
    fringe_multiplier: float,
    childcare_assumptions: dict[str, object],
) -> dict[str, object]:
    """Recompute direct-care decomposition summaries under alternative assumptions.

    ``staffing_scale`` multiplies effective_children_per_worker (higher = fewer
    workers per child = lower cost).  ``fringe_multiplier`` replaces the
    canonical value.
    """
    if frame.empty:
        return {"rows": 0, "alphas": {}}

    df = frame.copy()
    gross_baseline = pd.to_numeric(df["p_baseline"], errors="coerce")
    original_children = pd.to_numeric(df.get("effective_children_per_worker"), errors="coerce").fillna(
        float(childcare_assumptions["default_children_per_worker"])
    )
    original_fringe = pd.to_numeric(df.get("direct_care_fringe_multiplier"), errors="coerce").fillna(
        float(childcare_assumptions["direct_care_fringe_multiplier"])
    )
    original_direct_share = pd.to_numeric(df.get("direct_care_labor_share"), errors="coerce").clip(0.0, 1.0)
    original_direct_baseline = gross_baseline * original_direct_share

    # Back out the underlying hourly wage using original assumptions.
    underlying_wage = (
        original_direct_baseline
        * original_children
        / (float(childcare_assumptions["direct_care_hours_per_year"]) * original_fringe)
    )

    new_children = original_children * staffing_scale
    new_raw_direct = (
        underlying_wage
        * float(childcare_assumptions["direct_care_hours_per_year"])
        * fringe_multiplier
        / new_children.clip(lower=1e-9)
    )
    new_direct_baseline = pd.concat([gross_baseline, new_raw_direct], axis=1).min(axis=1).clip(lower=0.0)
    new_residual_baseline = (gross_baseline - new_direct_baseline).clip(lower=0.0)
    new_share = new_direct_baseline.div(gross_baseline.replace({0: pd.NA})).clip(0.0, 1.0).fillna(0.0)
    new_wage_baseline = (
        new_direct_baseline
        * new_children
        / (float(childcare_assumptions["direct_care_hours_per_year"]) * fringe_multiplier)
    )

    result: dict[str, object] = {
        "rows": int(len(df)),
        "staffing_scale": staffing_scale,
        "fringe_multiplier": fringe_multiplier,
        "effective_children_per_worker_p50": float(new_children.median()),
        "baseline_price_p50": float(gross_baseline.median()),
        "baseline_direct_care_price_p50": float(new_direct_baseline.median()),
        "baseline_non_direct_care_price_p50": float(new_residual_baseline.median()),
        "baseline_implied_wage_p50": float(new_wage_baseline.median()),
        "alphas": {},
    }

    for alpha_val in sorted(pd.to_numeric(df["alpha"], errors="coerce").dropna().unique().tolist()):
        mask = pd.to_numeric(df["alpha"], errors="coerce").round(4).eq(round(float(alpha_val), 4))
        subset_gross = pd.to_numeric(df.loc[mask, "p_alpha"], errors="coerce")
        subset_direct = subset_gross * new_share.loc[mask]
        subset_residual = (subset_gross - subset_direct).clip(lower=0.0)
        subset_wage = (
            subset_direct
            * new_children.loc[mask]
            / (float(childcare_assumptions["direct_care_hours_per_year"]) * fringe_multiplier)
        )
        result["alphas"][f"{float(alpha_val):.2f}"] = {
            "price_p50": float(subset_gross.median()),
            "direct_care_price_p50": float(subset_direct.median()),
            "non_direct_care_price_p50": float(subset_residual.median()),
            "implied_wage_p50": float(subset_wage.median()),
        }

    return result


def _run_price_decomposition_sensitivity(
    scenarios: pd.DataFrame,
    paths,
) -> dict[str, object]:
    """Run a 3x3 staffing-ratio x fringe-multiplier sensitivity sweep."""
    childcare_assumptions = childcare_model_assumptions(paths)
    staffing_cases = childcare_assumptions["sensitivity_staffing_cases"]
    fringe_cases = childcare_assumptions["sensitivity_fringe_cases"]
    cases: list[dict[str, object]] = []
    for staffing_label, staffing_scale in staffing_cases.items():
        for fringe_label, fringe_value in fringe_cases.items():
            case = _recompute_decomposition_under_assumptions(
                scenarios,
                staffing_scale,
                fringe_value,
                childcare_assumptions,
            )
            case["staffing_case"] = staffing_label
            case["fringe_case"] = fringe_label
            cases.append(case)

    sensitivity: dict[str, object] = {
        "description": "3x3 staffing-ratio x fringe-multiplier sensitivity sweep for the direct-care-equivalent price decomposition",
        "staffing_cases": {k: v for k, v in staffing_cases.items()},
        "fringe_cases": {k: v for k, v in fringe_cases.items()},
        "canonical_fringe_multiplier": float(childcare_assumptions["direct_care_fringe_multiplier"]),
        "canonical_hours_per_year": float(childcare_assumptions["direct_care_hours_per_year"]),
        "n_cases": len(cases),
        "cases": cases,
    }
    write_json(sensitivity, paths.outputs_reports / "childcare_price_decomposition_sensitivity.json")
    return sensitivity


def _run_piecewise_supply_demo(
    state: pd.DataFrame,
    county: pd.DataFrame,
    demand_summary: dict[str, object],
    alphas: list[float],
    paths,
    sample_name: str = "observed_core",
) -> tuple[pd.DataFrame, dict[str, object]]:
    childcare_assumptions = childcare_model_assumptions(paths)
    labor_support_threshold = float(childcare_assumptions["piecewise_supply_labor_support_threshold"])
    eligible_column = f"eligible_{sample_name}"
    subset = state.loc[
        state[eligible_column].fillna(False).astype(bool)
        & pd.to_numeric(state.get("state_qcew_labor_observed_share"), errors="coerce").ge(labor_support_threshold)
    ].copy()
    prepared = prepare_childcare_scenario_inputs(subset)
    if prepared.empty:
        diagnostics = {
            "demo_sample_name": sample_name,
            "labor_support_threshold": labor_support_threshold,
            "demo_rows": 0,
            "demo_states": 0,
            "demo_years": [],
            "piecewise_method": "state_year_piecewise_isoelastic",
            "note": "no eligible high-labor-support rows for the piecewise-supply demo",
        }
        return pd.DataFrame(), diagnostics

    county_demo = county.merge(
        prepared[["state_fips", "year", "state_price_index"]].drop_duplicates(),
        on=["state_fips", "year"],
        how="inner",
    )
    piecewise_summary, piecewise_groups = summarize_piecewise_supply_curve(county_demo, baseline_column="state_price_index")
    assigned = prepared.merge(
        piecewise_groups[
            [
                "state_fips",
                "year",
                "eta_below",
                "eta_above",
                "fallback_below",
                "fallback_above",
                "rows_below",
                "rows_above",
            ]
        ] if not piecewise_groups.empty else pd.DataFrame(columns=["state_fips", "year", "eta_below", "eta_above", "fallback_below", "fallback_above", "rows_below", "rows_above"]),
        on=["state_fips", "year"],
        how="left",
    )
    constant_supply_summary = summarize_supply_elasticity(county_demo)
    demand_elasticity_signed, solver_demand_elasticity = resolve_solver_demand_elasticity(demand_summary)
    rows: list[dict[str, object]] = []
    for row in assigned.to_dict(orient="records"):
        baseline = float(row["state_price_index"])
        market_q = float(row["market_quantity_proxy"])
        unpaid_q = float(row["unpaid_quantity_proxy"])
        eta_below = float(pd.to_numeric(pd.Series([row.get("eta_below")]), errors="coerce").fillna(piecewise_summary["pooled_eta_below"]).iloc[0])
        eta_above = float(pd.to_numeric(pd.Series([row.get("eta_above")]), errors="coerce").fillna(piecewise_summary["pooled_eta_above"]).iloc[0])
        constant_results = solve_alpha_grid(
            baseline,
            market_q,
            unpaid_q,
            solver_demand_elasticity,
            float(constant_supply_summary["supply_elasticity"]),
            alphas,
        )
        piecewise_results = solve_alpha_grid_piecewise_supply(
            baseline,
            market_q,
            unpaid_q,
            solver_demand_elasticity,
            eta_below,
            eta_above,
            alphas,
        )
        piecewise_by_alpha = {round(result.alpha, 4): result.price for result in piecewise_results}
        for result in constant_results:
            rows.append(
                {
                    "state_fips": row["state_fips"],
                    "year": int(row["year"]),
                    "alpha": float(result.alpha),
                    "p_baseline": baseline,
                    "market_quantity_proxy": market_q,
                    "unpaid_quantity_proxy": unpaid_q,
                    "demand_elasticity_signed": demand_elasticity_signed,
                    "solver_demand_elasticity_magnitude": solver_demand_elasticity,
                    "supply_elasticity_constant": float(constant_supply_summary["supply_elasticity"]),
                    "supply_elasticity_below": eta_below,
                    "supply_elasticity_above": eta_above,
                    "fallback_below": bool(row.get("fallback_below", True)),
                    "fallback_above": bool(row.get("fallback_above", True)),
                    "rows_below": int(pd.to_numeric(pd.Series([row.get("rows_below")]), errors="coerce").fillna(0).iloc[0]),
                    "rows_above": int(pd.to_numeric(pd.Series([row.get("rows_above")]), errors="coerce").fillna(0).iloc[0]),
                    "p_alpha_constant_supply": float(result.price),
                    "p_alpha_piecewise_supply": float(piecewise_by_alpha[round(float(result.alpha), 4)]),
                }
            )
    demo = pd.DataFrame(rows)
    if not demo.empty:
        demo["piecewise_minus_constant"] = (
            pd.to_numeric(demo["p_alpha_piecewise_supply"], errors="coerce")
            - pd.to_numeric(demo["p_alpha_constant_supply"], errors="coerce")
        )
    alpha_half = demo.loc[pd.to_numeric(demo["alpha"], errors="coerce").round(4).eq(0.5)].copy()
    alpha_one = demo.loc[pd.to_numeric(demo["alpha"], errors="coerce").round(4).eq(1.0)].copy()
    diagnostics = {
        "demo_sample_name": sample_name,
        "labor_support_threshold": labor_support_threshold,
        "demo_rows": int(len(prepared)),
        "demo_states": int(prepared["state_fips"].nunique()),
        "demo_years": sorted(pd.to_numeric(prepared["year"], errors="coerce").dropna().astype(int).unique().tolist()),
        "piecewise_method": piecewise_summary["piecewise_method"],
        "piecewise_group_count": int(piecewise_summary["group_count"]),
        "piecewise_supported_both_sides_groups": int(piecewise_summary["supported_both_sides_groups"]),
        "piecewise_pooled_eta_below": float(piecewise_summary["pooled_eta_below"]),
        "piecewise_pooled_eta_above": float(piecewise_summary["pooled_eta_above"]),
        "piecewise_fallback_share_any_side": float(piecewise_summary["fallback_share_any_side"]),
        "constant_supply_elasticity": float(constant_supply_summary["supply_elasticity"]),
        "constant_supply_method": constant_supply_summary["estimation_method"],
        "demand_elasticity_signed": demand_elasticity_signed,
        "alpha_50_piecewise_minus_constant_p50": float(pd.to_numeric(alpha_half.get("piecewise_minus_constant"), errors="coerce").median()) if not alpha_half.empty else 0.0,
        "alpha_100_piecewise_minus_constant_p50": float(pd.to_numeric(alpha_one.get("piecewise_minus_constant"), errors="coerce").median()) if not alpha_one.empty else 0.0,
        "median_demo_baseline_price": float(pd.to_numeric(demo["p_baseline"], errors="coerce").median()) if not demo.empty else 0.0,
    }
    return demo, diagnostics


def _summarize_dual_shift_headline_table(
    medium_headline: pd.DataFrame,
    bootstrap_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    table = (
        medium_headline.groupby(["kappa_q", "kappa_c"], as_index=False)
        .agg(
            row_count=("state_fips", "size"),
            median_baseline_price=("p_baseline", "median"),
            median_p_alpha=("p_alpha", "median"),
            median_p_alpha_pct_change=("p_alpha_pct_change_vs_baseline", "median"),
            share_price_increase=("p_alpha_delta_vs_baseline", lambda values: float(pd.Series(values).gt(0).mean())),
            share_price_decrease=("p_alpha_delta_vs_baseline", lambda values: float(pd.Series(values).lt(0).mean())),
        )
        .sort_values(["kappa_q", "kappa_c"], kind="stable")
        .reset_index(drop=True)
    )
    if bootstrap_table is not None and not bootstrap_table.empty:
        keep = [
            "kappa_q",
            "kappa_c",
            "median_p_alpha_lower",
            "median_p_alpha_upper",
            "median_p_alpha_pct_change_lower",
            "median_p_alpha_pct_change_upper",
            "share_price_increase_lower",
            "share_price_increase_upper",
            "share_price_decrease_lower",
            "share_price_decrease_upper",
        ]
        table = table.merge(
            bootstrap_table.loc[:, keep].drop_duplicates(["kappa_q", "kappa_c"]),
            on=["kappa_q", "kappa_c"],
            how="left",
        )
    return table


def _summarize_dual_shift_frontier(
    state_valid: pd.DataFrame,
    kappa_c_grid: list[float],
    supply_elasticity: float,
) -> pd.DataFrame:
    if state_valid.empty:
        return pd.DataFrame(
            columns=[
                "kappa_c",
                "frontier_row_count",
                "kappa_q_zero_price_frontier_p10",
                "kappa_q_zero_price_frontier_p50",
                "kappa_q_zero_price_frontier_p90",
            ]
        )
    rows: list[dict[str, float]] = []
    for record in state_valid.to_dict(orient="records"):
        market_quantity = float(record["market_quantity_proxy"])
        unpaid_quantity = float(record["unpaid_quantity_proxy"])
        for kappa_c in kappa_c_grid:
            rows.append(
                {
                    "state_fips": str(record["state_fips"]),
                    "year": int(record["year"]),
                    "kappa_c": float(kappa_c),
                    "kappa_q_zero_price_frontier": dual_shift_zero_price_frontier(
                        market_quantity=market_quantity,
                        unpaid_quantity=unpaid_quantity,
                        supply_elasticity=supply_elasticity,
                        kappa_c=float(kappa_c),
                    ),
                }
            )
    frontier = pd.DataFrame(rows)
    return (
        frontier.groupby("kappa_c", as_index=False)
        .agg(
            frontier_row_count=("kappa_q_zero_price_frontier", "size"),
            kappa_q_zero_price_frontier_p10=(
                "kappa_q_zero_price_frontier",
                lambda values: float(pd.Series(values).quantile(0.1)),
            ),
            kappa_q_zero_price_frontier_p50=(
                "kappa_q_zero_price_frontier",
                lambda values: float(pd.Series(values).quantile(0.5)),
            ),
            kappa_q_zero_price_frontier_p90=(
                "kappa_q_zero_price_frontier",
                lambda values: float(pd.Series(values).quantile(0.9)),
            ),
        )
        .sort_values(["kappa_c"], kind="stable")
        .reset_index(drop=True)
    )


def simulate_childcare_dual_shift(paths, sample: bool = True) -> None:
    state_path = paths.processed / "childcare_state_year_panel.parquet"
    county_path = paths.processed / "childcare_county_year_price_panel.parquet"
    if sample:
        if not state_path.exists() or not county_path.exists():
            fit_childcare(paths, sample=True)
    else:
        _ensure_real_mode_artifacts(
            [
                (state_path, "processed childcare state-year panel"),
                (county_path, "processed childcare county-year price panel"),
            ],
            "build-childcare --real",
        )
    state = read_parquet(state_path)
    county = read_parquet(county_path)
    if not _state_panel_has_sample_ladder(state):
        if sample:
            county, state = _refresh_childcare_panels_from_interim(paths, sample=True)
        else:
            raise UnpaidWorkError(
                "processed childcare state panel is missing current sample-ladder metadata. "
                "Rebuild with `unpriced build-childcare --real` before rerunning."
            )
    comparison_path = paths.outputs_reports / "childcare_demand_sample_comparison.json"
    if sample:
        if not comparison_path.exists():
            fit_childcare(paths, sample=True)
    else:
        _ensure_real_mode_artifact(
            comparison_path,
            "fit-childcare --real",
            "childcare demand sample comparison",
        )
    comparison = read_json(comparison_path)
    samples = comparison.get("samples", {})
    selected_sample, selection_reason = select_headline_sample(samples)
    if selected_sample is None:
        raise UnpaidWorkError(
            "no defensible observed-core childcare sample passed the minimum support rule; only exploratory samples are available"
        )
    canonical_profile = _canonical_specification_profile_for_sample(selected_sample)
    selected_path = _demand_summary_path(paths, selected_sample, specification_profile=canonical_profile)
    if sample:
        if not selected_path.exists():
            fit_childcare(paths, sample=True)
    else:
        _ensure_real_mode_artifact(
            selected_path,
            "fit-childcare --real",
            "selected childcare demand summary",
        )
    demand_summary = read_json(selected_path)
    demand_elasticity_signed, solver_demand_elasticity = resolve_solver_demand_elasticity(demand_summary)
    supply_summary = summarize_supply_elasticity(county)
    supply_elasticity = float(supply_summary["supply_elasticity"])
    project_config = load_yaml(paths.root / "configs" / "project.yaml")
    alphas = [float(alpha) for alpha in project_config["alpha_grid"]]
    childcare_assumptions = childcare_model_assumptions(paths)
    solver_config = solver_assumptions(paths)
    headline_alpha = float(childcare_assumptions["dual_shift_headline_alpha"])
    kappa_q_grid = [float(value) for value in childcare_assumptions["dual_shift_kappa_q_grid"]]
    kappa_c_grid = [float(value) for value in childcare_assumptions["dual_shift_kappa_c_grid"]]
    eligible_column = f"eligible_{selected_sample}"
    if eligible_column not in state.columns:
        raise UnpaidWorkError(f"state panel missing eligibility column for sample: {eligible_column}")
    state_selected = state.loc[state[eligible_column].fillna(False).astype(bool)].copy()
    state_valid = prepare_childcare_scenario_inputs(state_selected)
    if state_valid.empty:
        raise UnpaidWorkError("no valid state-year rows available for the dual-shift childcare sample")

    rows: list[dict[str, object]] = []
    for record in state_valid.to_dict(orient="records"):
        baseline = float(record["state_price_index"])
        market_q = float(record["market_quantity_proxy"])
        unpaid_q = float(record["unpaid_quantity_proxy"])
        short_run_shadow = solve_price(
            baseline_price=baseline,
            market_quantity=market_q,
            unpaid_quantity=unpaid_q,
            demand_elasticity=solver_demand_elasticity,
            supply_elasticity=supply_elasticity,
            alpha=MARGINAL_ALPHA,
            return_metadata=True,
        )
        if not isinstance(short_run_shadow, SolverMetadata):
            raise RuntimeError("missing short-run solver metadata")
        short_run_results = solve_alpha_grid(
            baseline_price=baseline,
            market_quantity=market_q,
            unpaid_quantity=unpaid_q,
            demand_elasticity=solver_demand_elasticity,
            supply_elasticity=supply_elasticity,
            alphas=alphas,
            return_metadata=True,
        )
        for result in short_run_results:
            rows.append(
                {
                    "solver_family": "short_run_fixed_supply",
                    "state_fips": record["state_fips"],
                    "year": record["year"],
                    "demand_sample_name": selected_sample,
                    "demand_specification_profile": demand_summary.get(
                        "specification_profile",
                        DEFAULT_SPECIFICATION_PROFILE,
                    ),
                    "p_baseline": baseline,
                    "p_shadow_marginal": float(short_run_shadow.price),
                    "alpha": float(result.alpha),
                    "p_alpha": float(result.price),
                    "demand_elasticity_signed": demand_elasticity_signed,
                    "solver_demand_elasticity_magnitude": solver_demand_elasticity,
                    "supply_elasticity": supply_elasticity,
                    "supply_estimation_method": supply_summary.get("estimation_method"),
                    "market_quantity_proxy": market_q,
                    "unpaid_quantity_proxy": unpaid_q,
                    "kappa_q": np.nan,
                    "kappa_c": np.nan,
                    "dual_shift_entry_multiplier": 1.0,
                    "dual_shift_cost_multiplier": 1.0,
                    "headline_alpha_flag": bool(np.isclose(float(result.alpha), headline_alpha)),
                    "solver_status": result.solver_status,
                    "solver_iterations": result.solver_iterations,
                    "solver_expansion_steps": result.solver_expansion_steps,
                    "solver_bracket_low": result.solver_low,
                    "solver_bracket_high": result.solver_high,
                }
            )
        for kappa_q in kappa_q_grid:
            for kappa_c in kappa_c_grid:
                shadow_result = solve_price_dual_shift(
                    baseline_price=baseline,
                    market_quantity=market_q,
                    unpaid_quantity=unpaid_q,
                    demand_elasticity=solver_demand_elasticity,
                    supply_elasticity=supply_elasticity,
                    alpha=MARGINAL_ALPHA,
                    kappa_q=float(kappa_q),
                    kappa_c=float(kappa_c),
                    return_metadata=True,
                )
                if not isinstance(shadow_result, SolverMetadata):
                    raise RuntimeError("missing dual-shift solver metadata")
                dual_shift_results = solve_alpha_grid_dual_shift(
                    baseline_price=baseline,
                    market_quantity=market_q,
                    unpaid_quantity=unpaid_q,
                    demand_elasticity=solver_demand_elasticity,
                    supply_elasticity=supply_elasticity,
                    alphas=alphas,
                    kappa_q=float(kappa_q),
                    kappa_c=float(kappa_c),
                    return_metadata=True,
                )
                frontier = dual_shift_zero_price_frontier(
                    market_quantity=market_q,
                    unpaid_quantity=unpaid_q,
                    supply_elasticity=supply_elasticity,
                    kappa_c=float(kappa_c),
                )
                for result in dual_shift_results:
                    rows.append(
                        {
                            "solver_family": "medium_run_dual_shift",
                            "state_fips": record["state_fips"],
                            "year": record["year"],
                            "demand_sample_name": selected_sample,
                            "demand_specification_profile": demand_summary.get(
                                "specification_profile",
                                DEFAULT_SPECIFICATION_PROFILE,
                            ),
                            "p_baseline": baseline,
                            "p_shadow_marginal": float(shadow_result.price),
                            "alpha": float(result.alpha),
                            "p_alpha": float(result.price),
                            "demand_elasticity_signed": demand_elasticity_signed,
                            "solver_demand_elasticity_magnitude": solver_demand_elasticity,
                            "supply_elasticity": supply_elasticity,
                            "supply_estimation_method": supply_summary.get("estimation_method"),
                            "market_quantity_proxy": market_q,
                            "unpaid_quantity_proxy": unpaid_q,
                            "kappa_q": float(kappa_q),
                            "kappa_c": float(kappa_c),
                            "dual_shift_entry_multiplier": float(np.exp(float(kappa_q) * float(result.alpha))),
                            "dual_shift_cost_multiplier": float(np.exp(float(kappa_c) * float(result.alpha))),
                            "kappa_q_zero_price_frontier": frontier,
                            "headline_alpha_flag": bool(np.isclose(float(result.alpha), headline_alpha)),
                            "solver_status": result.solver_status,
                            "solver_iterations": result.solver_iterations,
                            "solver_expansion_steps": result.solver_expansion_steps,
                            "solver_bracket_low": result.solver_low,
                            "solver_bracket_high": result.solver_high,
                        }
                    )

    scenarios = pd.DataFrame(rows).sort_values(
        ["solver_family", "state_fips", "year", "kappa_q", "kappa_c", "alpha"],
        kind="stable",
    ).reset_index(drop=True)
    scenarios["p_alpha_delta_vs_baseline"] = (
        pd.to_numeric(scenarios["p_alpha"], errors="coerce")
        - pd.to_numeric(scenarios["p_baseline"], errors="coerce")
    )
    scenarios["p_alpha_pct_change_vs_baseline"] = (
        scenarios["p_alpha_delta_vs_baseline"]
        / pd.to_numeric(scenarios["p_baseline"], errors="coerce").replace({0.0: pd.NA})
    ).fillna(0.0)
    raw_path = paths.processed / "childcare_dual_shift_marketization_scenarios.parquet"
    write_parquet(scenarios, raw_path)

    medium_headline = scenarios.loc[
        scenarios["solver_family"].astype(str).eq("medium_run_dual_shift")
        & scenarios["headline_alpha_flag"].fillna(False).astype(bool)
    ].copy()
    short_run_headline = scenarios.loc[
        scenarios["solver_family"].astype(str).eq("short_run_fixed_supply")
        & scenarios["headline_alpha_flag"].fillna(False).astype(bool)
    ].copy()
    bootstrap_table, bootstrap_meta = bootstrap_childcare_dual_shift_headline_table(
        state_frame=state_valid,
        county_frame=county,
        scenarios=medium_headline,
        demand_mode=selected_sample,
        demand_specification_profile=canonical_profile,
        n_boot=int(solver_config["bootstrap_n_boot"]),
        seed=int(solver_config["bootstrap_seed"]),
    )
    headline_table = _summarize_dual_shift_headline_table(medium_headline, bootstrap_table)
    frontier_summary = _summarize_dual_shift_frontier(state_valid, kappa_c_grid, supply_elasticity)
    summary = {
        "current_mode": "sample" if sample else "real",
        "headline_sample": selected_sample,
        "headline_selection_reason": str(selection_reason),
        "demand_instrument": demand_summary.get("instrument"),
        "demand_specification_profile": demand_summary.get("specification_profile", DEFAULT_SPECIFICATION_PROFILE),
        "headline_alpha": headline_alpha,
        "kappa_q_grid": kappa_q_grid,
        "kappa_c_grid": kappa_c_grid,
        "scenario_rows": int(len(scenarios)),
        "medium_run_rows": int(len(scenarios.loc[scenarios["solver_family"].eq("medium_run_dual_shift")])),
        "short_run_rows": int(len(scenarios.loc[scenarios["solver_family"].eq("short_run_fixed_supply")])),
        "median_baseline_price_p50": float(pd.to_numeric(medium_headline["p_baseline"], errors="coerce").median()),
        "short_run_fixed_supply_headline_alpha_price_p50": float(
            pd.to_numeric(short_run_headline["p_alpha"], errors="coerce").median()
        ),
        "short_run_fixed_supply_headline_alpha_pct_change_p50": float(
            pd.to_numeric(short_run_headline["p_alpha_pct_change_vs_baseline"], errors="coerce").median()
        ),
        "bootstrap_draws_requested": bootstrap_meta.get("bootstrap_draws_requested"),
        "bootstrap_draws_accepted": bootstrap_meta.get("bootstrap_draws_accepted"),
        "bootstrap_draws_rejected": bootstrap_meta.get("bootstrap_draws_rejected"),
        "bootstrap_acceptance_rate": bootstrap_meta.get("bootstrap_acceptance_rate"),
        "bootstrap_failed": bootstrap_meta.get("bootstrap_failed"),
        "bootstrap_rejection_reasons": bootstrap_meta.get("bootstrap_rejection_reasons", {}),
        "headline_alpha_table": headline_table.to_dict(orient="records"),
        "frontier_summary": frontier_summary.to_dict(orient="records"),
        "notes": [
            "Short-run fixed-supply benchmark keeps alpha on demand only, so positive alpha raises price by construction.",
            "Medium-run dual-shift results let marketization expand supply through kappa_q and raise costs through kappa_c, so the sign of the price effect becomes ambiguous.",
            "This additive layer is a sensitivity-driven parallel estimand; the canonical pooled headline remains the short-run benchmark.",
        ],
    }
    summary_path = paths.outputs_reports / "childcare_dual_shift_summary.json"
    table_path = paths.outputs_tables / "childcare_dual_shift_headline_alpha.csv"
    write_json(summary, summary_path)
    headline_table.to_csv(table_path, index=False)
    figure_path = write_childcare_dual_shift_figure(
        paths.outputs_figures / "childcare_dual_shift_frontier.svg",
        summary,
        headline_table,
    )
    output_artifacts = [raw_path, summary_path, table_path, figure_path]
    for artifact in output_artifacts:
        if artifact.exists():
            _write_mode_artifact(
                artifact,
                [state_path, county_path, selected_path],
                sample=sample,
                repo_root=paths.root,
                extra_parameters={"command": "simulate-childcare-dual-shift"},
            )
    LOGGER.info("simulated dual-shift childcare marketization scenarios using %s", selected_sample)


def simulate_childcare(paths, sample: bool = True) -> None:
    state_path = paths.processed / "childcare_state_year_panel.parquet"
    county_path = paths.processed / "childcare_county_year_price_panel.parquet"
    if sample:
        if not state_path.exists() or not county_path.exists():
            fit_childcare(paths, sample=True)
    else:
        _ensure_real_mode_artifacts(
            [
                (state_path, "processed childcare state-year panel"),
                (county_path, "processed childcare county-year price panel"),
            ],
            "build-childcare --real",
        )
    state = read_parquet(state_path)
    county = read_parquet(county_path)
    if not _state_panel_has_sample_ladder(state):
        if sample:
            county, state = _refresh_childcare_panels_from_interim(paths, sample=True)
        else:
            raise UnpaidWorkError(
                "processed childcare state panel is missing current sample-ladder metadata. "
                "Rebuild with `unpriced build-childcare --real` before rerunning."
            )
    comparison_path = paths.outputs_reports / "childcare_demand_sample_comparison.json"
    if sample:
        if not comparison_path.exists():
            fit_childcare(paths, sample=True)
    else:
        _ensure_real_mode_artifact(
            comparison_path,
            "fit-childcare --real",
            "childcare demand sample comparison",
        )
    comparison = read_json(comparison_path)
    samples = comparison.get("samples", {})
    selected_sample, selection_reason = select_headline_sample(samples)
    if selected_sample is None:
        raise UnpaidWorkError(
            "no defensible observed-core childcare sample passed the minimum support rule; only exploratory samples are available"
        )
    canonical_profile = _canonical_specification_profile_for_sample(selected_sample)
    project_config = load_yaml(paths.root / "configs" / "project.yaml")
    alphas = [float(alpha) for alpha in project_config["alpha_grid"]]
    all_scenarios: list[pd.DataFrame] = []
    scenario_sample_comparison: dict[str, object] = {
        "selected_headline_sample": selected_sample,
        "selected_headline_reason": str(selection_reason),
        "comparison_specification_profile": canonical_profile,
        "samples": {},
    }
    scenario_specification_comparison: dict[str, object] = {
        "selected_headline_sample": selected_sample,
        "selected_headline_reason": str(selection_reason),
        "comparison_specification_profile": canonical_profile,
        "profiles": {},
    }
    selected_scenarios: pd.DataFrame | None = None
    selected_diagnostics: dict[str, float | int | bool | str] | None = None
    for sample_name in CANONICAL_SAMPLE_MODES:
        sample_profile = canonical_profile
        demand_path = _demand_summary_path(paths, sample_name, specification_profile=sample_profile)
        if not demand_path.exists():
            continue
        if not sample:
            _ensure_real_mode_artifact(
                demand_path,
                "fit-childcare --real",
                f"childcare demand summary for {sample_name}",
            )
        demand_summary = read_json(demand_path)
        scenarios, diagnostics = _simulate_childcare_sample(
            paths,
            state,
            county,
            alphas,
            demand_summary,
            sample_name,
            sample=sample,
            selection_reason=str(selection_reason) if sample_name == selected_sample else "comparison_only",
            specification_profile=sample_profile,
        )
        scenario_sample_comparison["samples"][sample_name] = diagnostics
        if not scenarios.empty:
            all_scenarios.append(scenarios)
        if sample_name == selected_sample:
            selected_scenarios = scenarios
            selected_diagnostics = diagnostics
    specification_frames: list[pd.DataFrame] = []
    if selected_sample is not None:
        for profile in _selected_sample_specification_profiles(paths, selected_sample):
            demand_path = _demand_summary_path(paths, selected_sample, specification_profile=profile)
            if not demand_path.exists():
                continue
            if not sample:
                _ensure_real_mode_artifact(
                    demand_path,
                    "fit-childcare --real",
                    f"childcare demand summary for {selected_sample} ({profile})",
                )
            demand_summary = read_json(demand_path)
            scenarios, diagnostics = _simulate_childcare_sample(
                paths,
                state,
                county,
                alphas,
                demand_summary,
                selected_sample,
                sample=sample,
                selection_reason="canonical_specification"
                if profile == canonical_profile
                else "specification_sensitivity",
                specification_profile=profile,
            )
            scenario_specification_comparison["profiles"][profile] = diagnostics
            if not scenarios.empty:
                specification_frames.append(scenarios)
    if selected_scenarios is None or selected_scenarios.empty or selected_diagnostics is None:
        raise UnpaidWorkError("no valid state-year rows available for the selected childcare headline sample")
    if int(selected_diagnostics.get("skipped_state_rows", 0)) > 0:
        LOGGER.info(
            "skipping %s state-year rows with incomplete scenario inputs for %s",
            selected_diagnostics["skipped_state_rows"],
            selected_sample,
        )
    acceptance_rate = float(selected_diagnostics.get("bootstrap_acceptance_rate", 0.0))
    headline_gate_passed = sample or acceptance_rate >= 0.80
    selected_diagnostics["headline_gate_passed"] = headline_gate_passed
    write_json(selected_diagnostics, paths.outputs_reports / "childcare_scenario_diagnostics.json")
    write_json(
        scenario_sample_comparison,
        paths.outputs_reports / "childcare_scenario_sample_comparison.json",
    )
    if specification_frames or scenario_specification_comparison.get("profiles"):
        write_json(
            scenario_specification_comparison,
            paths.outputs_reports / "childcare_scenario_specification_comparison.json",
        )
    if not headline_gate_passed:
        raise UnpaidWorkError(
            f"childcare bootstrap acceptance rate is below the headline threshold for {selected_sample}: "
            f"{acceptance_rate:.1%}. "
            "Revisit the demand/solver diagnostics before publishing results."
        )
    write_parquet(selected_scenarios, paths.processed / "childcare_marketization_scenarios.parquet")
    summarize_scenarios(selected_scenarios).to_csv(
        paths.outputs_tables / "childcare_marketization_scenarios.csv", index=False
    )
    all_samples_frame = pd.concat(all_scenarios, ignore_index=True) if all_scenarios else selected_scenarios.copy()
    write_parquet(all_samples_frame, paths.processed / "childcare_marketization_scenarios_all_samples.parquet")
    summarize_scenarios(all_samples_frame).to_csv(
        paths.outputs_tables / "childcare_marketization_scenarios_all_samples.csv", index=False
    )
    decomposition = {
        "selected_headline_sample": selected_sample,
        "selected_headline_reason": str(selection_reason),
        "canonical": _price_decomposition_summary(selected_scenarios),
        "samples": {
            sample_name: _price_decomposition_summary(sample_frame)
            for sample_name, sample_frame in all_samples_frame.groupby("demand_sample_name", dropna=False)
        },
    }
    write_json(decomposition, paths.outputs_reports / "childcare_price_decomposition.json")
    write_json(
        summarize_supply_elasticity(county),
        paths.outputs_reports / "childcare_supply_elasticity.json",
    )
    canonical_demand_summary_path = _demand_summary_path(paths, selected_sample, specification_profile=canonical_profile)
    if canonical_demand_summary_path.exists():
        piecewise_demo, piecewise_demo_summary = _run_piecewise_supply_demo(
            state,
            county,
            read_json(canonical_demand_summary_path),
            alphas,
            paths,
            sample_name=selected_sample,
        )
        write_json(piecewise_demo_summary, paths.outputs_reports / "childcare_piecewise_supply_demo.json")
        write_parquet(piecewise_demo, paths.processed / "childcare_piecewise_supply_demo.parquet")
    _run_price_decomposition_sensitivity(selected_scenarios, paths)
    if specification_frames:
        specification_frame = pd.concat(specification_frames, ignore_index=True)
        write_parquet(
            specification_frame,
            paths.processed / "childcare_marketization_scenarios_specifications.parquet",
        )
        summarize_scenarios(specification_frame).to_csv(
            paths.outputs_tables / "childcare_marketization_scenarios_specifications.csv", index=False
        )
    output_artifacts = [
        paths.processed / "childcare_marketization_scenarios.parquet",
        paths.outputs_reports / "childcare_scenario_diagnostics.json",
        paths.processed / "childcare_marketization_scenarios_all_samples.parquet",
        paths.outputs_reports / "childcare_price_decomposition.json",
        paths.outputs_reports / "childcare_price_decomposition_sensitivity.json",
        paths.outputs_reports / "childcare_supply_elasticity.json",
        paths.outputs_reports / "childcare_piecewise_supply_demo.json",
        paths.outputs_reports / "childcare_scenario_sample_comparison.json",
        paths.processed / "childcare_piecewise_supply_demo.parquet",
        paths.processed / "childcare_marketization_scenarios_specifications.parquet",
        paths.outputs_reports / "childcare_scenario_specification_comparison.json",
    ]
    selected_demand_path = _demand_summary_path(paths, selected_sample, specification_profile=canonical_profile)
    for artifact in output_artifacts:
        if artifact.exists():
            _write_mode_artifact(
                artifact,
                [state_path, county_path, selected_demand_path],
                sample=sample,
                repo_root=paths.root,
                extra_parameters={"command": "simulate-childcare"},
            )
    LOGGER.info("simulated childcare marketization scenarios using %s", selected_sample)


def build_home(
    paths,
    sample: bool,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> None:
    ahs_path = paths.interim / "ahs" / "ahs.parquet"
    acs_path = paths.interim / "acs" / "acs.parquet"
    laus_path = paths.interim / "laus" / "laus.parquet"
    noaa_path = paths.interim / "noaa" / "noaa.parquet"
    if refresh or not acs_path.exists():
        acs.ingest_with_options(paths, sample=sample, refresh=refresh, dry_run=dry_run, year=year)
    if refresh or not laus_path.exists():
        laus.ingest(paths, sample=sample, refresh=refresh, dry_run=dry_run, year=year)
    if refresh or not noaa_path.exists():
        noaa.ingest(paths, sample=sample, refresh=refresh, dry_run=dry_run, year=year)
    if refresh or not ahs_path.exists():
        ahs.ingest(paths, sample=sample, refresh=refresh, dry_run=dry_run, year=year)
    if dry_run:
        LOGGER.info("planned home maintenance build")
        return
    panel = build_home_maintenance_panel(paths)
    LOGGER.info("built home maintenance panel: %s rows", len(panel))


def fit_home(paths, sample: bool = True) -> None:
    panel_path = paths.processed / "home_maintenance_panel.parquet"
    if sample:
        if not panel_path.exists():
            build_home(paths, sample=True)
    else:
        _ensure_real_mode_artifact(
            panel_path,
            "build-home --real",
            "processed home maintenance panel",
        )
    panel = read_parquet(panel_path)
    summary = fit_home_switching(panel, paths.outputs_reports / "home_switching.json")
    write_json(summary, paths.outputs_reports / "home_maintenance_summary.json")
    for artifact in (
        paths.outputs_reports / "home_switching.json",
        paths.outputs_reports / "home_maintenance_summary.json",
    ):
        if artifact.exists():
            _write_mode_artifact(
                artifact,
                [panel_path],
                sample=sample,
                repo_root=paths.root,
                extra_parameters={"command": "fit-home"},
            )
    LOGGER.info("fit home maintenance switching model")


def fit_supply_iv(
    paths,
    sample: bool = False,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> None:
    county_path = paths.processed / "childcare_county_year_price_panel.parquet"
    if dry_run:
        LOGGER.info("planned supply IV fit")
        return
    if sample:
        build_childcare(paths, sample=True, refresh=refresh, dry_run=False, year=year)
        licensing.ingest(paths, sample=True, refresh=refresh, dry_run=False, year=year)
        county_path = paths.processed / "childcare_county_year_price_panel.parquet"
        output_json = paths.outputs_reports / "childcare_supply_iv_sample.json"
        output_panel = paths.processed / "childcare_supply_iv_panel_sample.parquet"
        shocks_path = paths.interim / "licensing" / "licensing.parquet"
    else:
        if not county_path.exists():
            build_childcare(paths, sample=True)
        output_json = paths.outputs_reports / "childcare_supply_iv.json"
        output_panel = paths.processed / "childcare_supply_iv_panel.parquet"
        shocks_path = paths.interim / "licensing" / "licensing_supply_shocks.parquet"
    county = read_parquet(county_path)
    if not shocks_path.exists():
        try:
            ingest_result = licensing.ingest(paths, sample=sample, refresh=refresh)
        except SourceAccessError:
            ingest_result = None
        if ingest_result is not None:
            shocks_path = ingest_result.normalized_path
    expected_name = "licensing.parquet" if sample else "licensing_supply_shocks.parquet"
    if not shocks_path.exists() or shocks_path.name != expected_name:
        write_json(
            {
                "status": "missing_licensing_shock_panel",
                "design": "county_fe_state_year_fe_exposure_shock",
                "sample_mode": sample,
                "shock_panel_path": str(
                    paths.interim / "licensing" / ("licensing.parquet" if sample else "licensing_supply_shocks.parquet")
                ),
                "raw_seed_path": (
                    str(paths.raw / "licensing" / "licensing_supply_shocks.csv")
                    if not sample
                    else str(paths.raw / "licensing" / "licensing_sample.json")
                ),
                "note": (
                    "No normalized licensing shock panel found. Create data/raw/licensing/licensing_supply_shocks.csv with state-year center labor-intensity shocks and rerun fit-supply-iv."
                    if not sample
                    else "No sample licensing shock panel found."
                ),
            },
            output_json,
        )
        write_parquet(
            pd.DataFrame(columns=["county_fips", "state_fips", "year", "reg_shock_ct"]),
            output_panel,
        )
        LOGGER.info("skipped supply IV fit: missing licensing shock panel")
        return
    shocks = read_parquet(shocks_path)
    summary, panel = fit_supply_iv_exposure_design(county, shocks)
    summary["shock_panel_path"] = str(shocks_path)
    summary["sample_mode"] = sample
    write_json(summary, output_json)
    write_parquet(panel, output_panel)
    LOGGER.info("fit supply IV demo: status=%s rows=%s", summary.get("status"), summary.get("n_obs", 0))


def report(paths, sample: bool = True) -> None:
    scenarios_path = paths.processed / "childcare_marketization_scenarios.parquet"
    state_path = paths.processed / "childcare_state_year_panel.parquet"
    county_path = paths.processed / "childcare_county_year_price_panel.parquet"
    if sample:
        if not scenarios_path.exists():
            simulate_childcare(paths, sample=True)
    else:
        _ensure_real_mode_artifact(
            scenarios_path,
            "simulate-childcare --real",
            "childcare marketization scenarios",
        )
        _ensure_real_mode_artifacts(
            [
                (state_path, "processed childcare state-year panel"),
                (county_path, "processed childcare county-year price panel"),
            ],
            "build-childcare --real",
        )
    scenarios = read_parquet(scenarios_path)
    state = read_parquet(state_path)
    county = read_parquet(county_path)
    sipp_path = paths.interim / "sipp" / "sipp.parquet"
    ce_path = paths.interim / "ce" / "ce.parquet"
    acs_path = paths.interim / "acs" / "acs.parquet"
    sipp_validation = read_parquet(sipp_path) if sipp_path.exists() else None
    ce_validation = read_parquet(ce_path) if ce_path.exists() else None
    acs_frame = read_parquet(acs_path) if acs_path.exists() else None
    pipeline_diagnostics_path = paths.outputs_reports / "childcare_pipeline_diagnostics.json"
    comparison_path = paths.outputs_reports / "childcare_demand_sample_comparison.json"
    scenario_comparison_path = paths.outputs_reports / "childcare_scenario_sample_comparison.json"
    scenario_specification_comparison_path = paths.outputs_reports / "childcare_scenario_specification_comparison.json"
    imputation_sweep_path = paths.outputs_reports / "childcare_demand_imputation_sweep.json"
    labor_support_sweep_path = paths.outputs_reports / "childcare_demand_labor_support_sweep.json"
    specification_sweep_path = paths.outputs_reports / "childcare_demand_specification_sweep.json"
    piecewise_supply_demo_path = paths.outputs_reports / "childcare_piecewise_supply_demo.json"
    dual_shift_summary_path = paths.outputs_reports / "childcare_dual_shift_summary.json"
    supply_iv_path = paths.outputs_reports / "childcare_supply_iv.json"
    satellite_account_path = paths.outputs_reports / "childcare_satellite_account.json"
    selected_sample = "broad_complete"
    selected_path = paths.outputs_reports / "childcare_demand_iv.json"
    if comparison_path.exists():
        if not sample:
            _ensure_real_mode_artifact(
                comparison_path,
                "fit-childcare --real",
                "childcare demand sample comparison",
            )
        comparison = read_json(comparison_path)
        samples = comparison.get("samples", {})
        selected_sample, _ = select_headline_sample(samples)
        if selected_sample:
            selected_path = _demand_summary_path(
                paths,
                selected_sample,
                specification_profile=_canonical_specification_profile_for_sample(selected_sample),
            )
            if sample and not selected_path.exists():
                fit_childcare(paths, sample=True)
    elif not sample:
        raise UnpaidWorkError(
            "childcare demand sample comparison is missing for --real mode. "
            "Build with `unpriced fit-childcare --real` before rerunning."
        )
    if not sample:
        _ensure_real_mode_artifact(
            selected_path,
            "fit-childcare --real",
            "selected childcare demand summary",
        )
    build_childcare_satellite_account(
        county=county,
        state=state,
        acs=acs_frame,
        childcare_assumptions=childcare_model_assumptions(paths),
        output_json_path=satellite_account_path,
        output_markdown_path=paths.outputs_reports / "childcare_satellite_account.md",
        output_table_path=paths.outputs_tables / "childcare_satellite_account_annual.csv",
    )
    build_markdown_report(
        price_surface_path=paths.outputs_reports / "childcare_price_surface.json",
        demand_iv_path=selected_path,
        scenario_diagnostics_path=paths.outputs_reports / "childcare_scenario_diagnostics.json",
        scenarios=scenarios,
        output_path=paths.outputs_reports / "childcare_mvp_report.md",
        sipp_validation=sipp_validation,
        ce_validation=ce_validation,
        pipeline_diagnostics_path=pipeline_diagnostics_path if pipeline_diagnostics_path.exists() else None,
        demand_iv_strict_path=comparison_path if comparison_path.exists() else None,
        scenario_sample_comparison_path=scenario_comparison_path if scenario_comparison_path.exists() else None,
        scenario_specification_comparison_path=(
            scenario_specification_comparison_path if scenario_specification_comparison_path.exists() else None
        ),
        demand_imputation_sweep_path=imputation_sweep_path if imputation_sweep_path.exists() else None,
        demand_labor_support_sweep_path=labor_support_sweep_path if labor_support_sweep_path.exists() else None,
        demand_specification_sweep_path=specification_sweep_path if specification_sweep_path.exists() else None,
        piecewise_supply_demo_path=piecewise_supply_demo_path if piecewise_supply_demo_path.exists() else None,
        dual_shift_summary_path=dual_shift_summary_path if dual_shift_summary_path.exists() else None,
        supply_iv_path=supply_iv_path if supply_iv_path.exists() else None,
        satellite_account_path=satellite_account_path if satellite_account_path.exists() else None,
    )
    price_decomposition_path = paths.outputs_reports / "childcare_price_decomposition.json"
    price_decomposition_sensitivity_path = paths.outputs_reports / "childcare_price_decomposition_sensitivity.json"
    if price_decomposition_path.exists():
        build_childcare_headline_summary(
            demand_iv_path=selected_path,
            scenario_diagnostics_path=paths.outputs_reports / "childcare_scenario_diagnostics.json",
            price_decomposition_path=price_decomposition_path,
            output_json_path=paths.outputs_reports / "childcare_headline_summary.json",
            output_markdown_path=paths.outputs_reports / "childcare_headline_readout.md",
            price_decomposition_sensitivity_path=(
                price_decomposition_sensitivity_path if price_decomposition_sensitivity_path.exists() else None
            ),
        )
    for artifact in (
        paths.outputs_reports / "childcare_mvp_report.md",
        paths.outputs_reports / "childcare_headline_summary.json",
        paths.outputs_reports / "childcare_headline_readout.md",
        paths.outputs_reports / "childcare_satellite_account.json",
        paths.outputs_reports / "childcare_satellite_account.md",
    ):
        if artifact.exists():
            _write_mode_artifact(
                artifact,
                [scenarios_path, state_path, county_path, selected_path],
                sample=sample,
                repo_root=paths.root,
                extra_parameters={"command": "report"},
            )
    write_assumption_audit(paths)
    write_childcare_figure_manifest(paths, demand_summary_path=selected_path)
    LOGGER.info("wrote report")


def add_ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--year", type=int)


def add_mode_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sample", action="store_true")
    group.add_argument("--real", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unpriced")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("bootstrap")
    for command in ("fit-childcare", "simulate-childcare", "simulate-childcare-dual-shift", "fit-home", "report"):
        add_mode_args(subparsers.add_parser(command))

    supply_iv_parser = subparsers.add_parser("fit-supply-iv")
    add_ingest_args(supply_iv_parser)

    pull_parser = subparsers.add_parser("pull-core")
    add_ingest_args(pull_parser)

    ccdf_parser = subparsers.add_parser("pull-ccdf")
    add_ingest_args(ccdf_parser)

    ccdf_state_year_parser = subparsers.add_parser("build-ccdf-state-year")
    add_ingest_args(ccdf_state_year_parser)
    ccdf_state_year_parser.add_argument("--config", default="ccdf_state_year")

    licensing_parser = subparsers.add_parser("pull-licensing")
    add_ingest_args(licensing_parser)

    licensing_harmonization_parser = subparsers.add_parser("build-licensing-harmonization")
    add_ingest_args(licensing_harmonization_parser)
    licensing_harmonization_parser.add_argument("--config", default="licensing_iv")

    licensing_iv_parser = subparsers.add_parser("build-licensing-iv")
    add_ingest_args(licensing_iv_parser)
    licensing_iv_parser.add_argument("--config", default="licensing_iv")

    childcare_parser = subparsers.add_parser("build-childcare")
    add_ingest_args(childcare_parser)

    segmented_parser = subparsers.add_parser("build-childcare-segments")
    add_ingest_args(segmented_parser)
    segmented_parser.add_argument("--config", default="segmented_solver")

    utilization_parser = subparsers.add_parser("build-childcare-utilization")
    add_ingest_args(utilization_parser)
    utilization_parser.add_argument("--config", default="utilization_stack")
    utilization_parser.add_argument("--segmented-config", default="segmented_solver")

    solver_inputs_parser = subparsers.add_parser("build-childcare-solver-inputs")
    add_ingest_args(solver_inputs_parser)
    solver_inputs_parser.add_argument("--config", default="solver_inputs")

    report_tables_parser = subparsers.add_parser("build-childcare-report-tables")
    add_ingest_args(report_tables_parser)
    report_tables_parser.add_argument("--config", default="report_tables")

    segmented_report_parser = subparsers.add_parser("build-childcare-segmented-report")
    add_ingest_args(segmented_report_parser)
    segmented_report_parser.add_argument("--config", default="segmented_reports")

    segmented_publication_parser = subparsers.add_parser("report-childcare-segmented")
    add_ingest_args(segmented_publication_parser)
    segmented_publication_parser.add_argument("--config", default="segmented_publication")

    segmented_scenarios_parser = subparsers.add_parser("simulate-childcare-segmented")
    add_ingest_args(segmented_scenarios_parser)
    segmented_scenarios_parser.add_argument("--config", default="segmented_scenarios")

    release_backend_parser = subparsers.add_parser("build-childcare-release-backend")
    add_ingest_args(release_backend_parser)
    release_backend_parser.add_argument("--config", default="release_backend")

    home_parser = subparsers.add_parser("build-home")
    add_ingest_args(home_parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = load_project_paths(project_root())
    ensure_project_dirs(paths)

    sample = not getattr(args, "real", False)

    try:
        if args.command == "bootstrap":
            bootstrap(paths)
        elif args.command == "pull-core":
            pull_core(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
            )
        elif args.command == "pull-ccdf":
            ccdf.ingest(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
            )
        elif args.command == "build-ccdf-state-year":
            build_ccdf_state_year(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
                config_name_or_path=args.config,
            )
        elif args.command == "pull-licensing":
            licensing.ingest(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
            )
        elif args.command == "build-licensing-harmonization":
            build_licensing_harmonization(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
                config_name_or_path=args.config,
            )
        elif args.command == "build-licensing-iv":
            build_licensing_iv(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
                config_name_or_path=args.config,
            )
        elif args.command == "build-childcare":
            build_childcare(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
            )
        elif args.command == "build-childcare-segments":
            build_childcare_segments(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
                config_name_or_path=args.config,
            )
        elif args.command == "build-childcare-utilization":
            build_childcare_utilization(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
                config_name_or_path=args.config,
                segmented_config_name_or_path=args.segmented_config,
            )
        elif args.command == "build-childcare-solver-inputs":
            build_childcare_solver_inputs(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
                config_name_or_path=args.config,
            )
        elif args.command == "build-childcare-report-tables":
            build_childcare_report_tables(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
                config_name_or_path=args.config,
            )
        elif args.command == "build-childcare-segmented-report":
            build_childcare_segmented_report(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
                config_name_or_path=args.config,
            )
        elif args.command == "report-childcare-segmented":
            report_childcare_segmented(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
                config_name_or_path=args.config,
            )
        elif args.command == "simulate-childcare-segmented":
            build_childcare_segmented_scenarios(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
                config_name_or_path=args.config,
            )
        elif args.command == "build-childcare-release-backend":
            build_childcare_release_backend(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
                config_name_or_path=args.config,
            )
        elif args.command == "fit-childcare":
            fit_childcare(paths, sample=sample)
        elif args.command == "simulate-childcare":
            simulate_childcare(paths, sample=sample)
        elif args.command == "simulate-childcare-dual-shift":
            simulate_childcare_dual_shift(paths, sample=sample)
        elif args.command == "build-home":
            build_home(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
            )
        elif args.command == "fit-home":
            fit_home(paths, sample=sample)
        elif args.command == "fit-supply-iv":
            fit_supply_iv(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
            )
        elif args.command == "report":
            report(paths, sample=sample)
        else:
            parser.error(f"Unknown command: {args.command}")
    except UnpaidWorkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
