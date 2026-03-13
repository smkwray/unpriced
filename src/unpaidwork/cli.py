from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from unpaidwork.assumptions import childcare_model_assumptions, write_assumption_audit
from unpaidwork.config import ensure_project_dirs, load_project_paths, load_yaml
from unpaidwork.errors import UnpaidWorkError
from unpaidwork.features.childcare_panel import (
    build_childcare_panels,
    diagnose_childcare_pipeline,
)
from unpaidwork.features.home_maintenance_panel import build_home_maintenance_panel
from unpaidwork.ingest import acs, ahs, atus, cbp, cdc_wonder, ce, head_start, laus, licensing, nces_ccd, ndcp, nes, noaa, oews, qcew, sipp
from unpaidwork.ingest.acs import ACS_FIRST_AVAILABLE_YEAR
from unpaidwork.ingest.qcew import QCEW_FIRST_AVAILABLE_YEAR
from unpaidwork.ingest.laus import LAUS_FIRST_AVAILABLE_YEAR
from unpaidwork.logging import get_logger
from unpaidwork.models.demand_iv import (
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
from unpaidwork.models.price_surface import fit_price_surface
from unpaidwork.models.replacement_cost import apply_replacement_cost
from unpaidwork.models.scenario_solver import (
    MARGINAL_ALPHA,
    bootstrap_childcare_intervals,
    prepare_childcare_scenario_inputs,
    resolve_solver_demand_elasticity,
    solve_alpha_grid,
    solve_alpha_grid_piecewise_supply,
    solve_price,
    summarize_childcare_scenario_diagnostics,
)
from unpaidwork.models.supply_curve import (
    calibrate_supply_elasticity,
    summarize_piecewise_supply_curve,
    summarize_supply_elasticity,
)
from unpaidwork.models.supply_iv import fit_supply_iv_exposure_design
from unpaidwork.models.switching import fit_home_switching
from unpaidwork.registry import ensure_registry
from unpaidwork.reports.export import (
    build_childcare_headline_summary,
    build_childcare_satellite_account,
    build_markdown_report,
)
from unpaidwork.reports.figures import write_childcare_figure_manifest
from unpaidwork.reports.tables import summarize_scenarios
from unpaidwork.storage import read_json, read_parquet, write_json, write_parquet

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


def _refresh_childcare_panels_from_interim(paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    county, state = build_childcare_panels(paths)
    state = apply_replacement_cost(state, "unpaid_childcare_hours", "benchmark_childcare_wage")
    write_parquet(state, paths.processed / "childcare_state_year_panel.parquet")
    diagnose_childcare_pipeline(county, state, paths)
    return county, state


def bootstrap(paths) -> None:
    ensure_project_dirs(paths)
    ensure_registry(paths)
    LOGGER.info("bootstrap complete")


def pull_core(paths, sample: bool, refresh: bool = False, dry_run: bool = False, year: int | None = None) -> None:
    for ingestor in _core_ingestors():
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
    missing = [
        paths.interim / name / f"{name}.parquet"
        for name in ("atus", "ndcp", "qcew", "acs", "cdc_wonder", "head_start", "nces_ccd", "oews", "cbp", "nes", "laus")
    ]
    if any(not path.exists() for path in missing):
        pull_core(paths, sample=sample, refresh=refresh, dry_run=dry_run, year=year)
    if dry_run:
        LOGGER.info("planned childcare build")
        return
    # Ensure ACS, QCEW, and LAUS cover the NDCP year range.
    if not sample:
        ndcp_path = paths.interim / "ndcp" / "ndcp.parquet"
        if ndcp_path.exists():
            ndcp_frame = read_parquet(ndcp_path)
            ndcp_years = sorted(ndcp_frame["year"].dropna().astype(int).unique())
            if ndcp_years:
                ndcp_start = int(ndcp_years[0])
                ndcp_end = int(ndcp_years[-1])
                acs_result = acs.ingest_year_range(paths, max(ndcp_start, ACS_FIRST_AVAILABLE_YEAR), ndcp_end, refresh=refresh)
                LOGGER.info("ACS year-range check: %s", acs_result.detail or "ok")
                qcew_result = qcew.ingest_year_range(paths, max(ndcp_start, QCEW_FIRST_AVAILABLE_YEAR), ndcp_end, refresh=refresh)
                LOGGER.info("QCEW year-range check: %s", qcew_result.detail or "ok")
                laus_result = laus.ingest_year_range(paths, ndcp_start, ndcp_end, refresh=refresh)
                LOGGER.info("LAUS year-range check: %s", laus_result.detail or "ok")
    county, state = build_childcare_panels(paths)
    state = apply_replacement_cost(state, "unpaid_childcare_hours", "benchmark_childcare_wage")
    write_parquet(state, paths.processed / "childcare_state_year_panel.parquet")
    diag = diagnose_childcare_pipeline(county, state, paths)
    LOGGER.info(
        "built childcare panels: county=%s state=%s (observed_controls=%s/%s, births_observed=%s/%s)",
        len(county),
        len(state),
        diag["state_controls_acs_observed"],
        diag["state_year_rows"],
        diag["births_cdc_wonder_observed"],
        diag["state_year_rows"],
    )


def fit_childcare(paths) -> None:
    county_path = paths.processed / "childcare_county_year_price_panel.parquet"
    state_path = paths.processed / "childcare_state_year_panel.parquet"
    if not county_path.exists() or not state_path.exists():
        build_childcare(paths, sample=True)
    county = read_parquet(county_path)
    state = read_parquet(state_path)
    if not _state_panel_has_sample_ladder(state):
        county, state = _refresh_childcare_panels_from_interim(paths)
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
        shadow = solve_price(
            baseline,
            market_q,
            unpaid_q,
            solver_demand_elasticity,
            supply_elasticity,
            MARGINAL_ALPHA,
        )
        solved = solve_alpha_grid(
            baseline,
            market_q,
            unpaid_q,
            solver_demand_elasticity,
            supply_elasticity,
            alphas,
        )
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
                    "p_shadow_marginal": shadow,
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
                    "supply_elasticity": supply_elasticity,
                    "supply_estimation_method": supply_summary.get("estimation_method"),
                    "market_quantity_proxy": market_q,
                    "unpaid_quantity_proxy": unpaid_q,
                }
            )
    scenarios = bootstrap_childcare_intervals(
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
    )
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


def simulate_childcare(paths) -> None:
    state_path = paths.processed / "childcare_state_year_panel.parquet"
    county_path = paths.processed / "childcare_county_year_price_panel.parquet"
    if not state_path.exists() or not county_path.exists():
        fit_childcare(paths)
    state = read_parquet(state_path)
    county = read_parquet(county_path)
    if not _state_panel_has_sample_ladder(state):
        county, state = _refresh_childcare_panels_from_interim(paths)
    comparison_path = paths.outputs_reports / "childcare_demand_sample_comparison.json"
    if not comparison_path.exists():
        fit_childcare(paths)
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
        demand_summary = read_json(demand_path)
        scenarios, diagnostics = _simulate_childcare_sample(
            paths,
            state,
            county,
            alphas,
            demand_summary,
            sample_name,
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
            demand_summary = read_json(demand_path)
            scenarios, diagnostics = _simulate_childcare_sample(
                paths,
                state,
                county,
                alphas,
                demand_summary,
                selected_sample,
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
    write_parquet(selected_scenarios, paths.processed / "childcare_marketization_scenarios.parquet")
    write_json(selected_diagnostics, paths.outputs_reports / "childcare_scenario_diagnostics.json")
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
    write_json(
        scenario_sample_comparison,
        paths.outputs_reports / "childcare_scenario_sample_comparison.json",
    )
    if specification_frames:
        specification_frame = pd.concat(specification_frames, ignore_index=True)
        write_parquet(
            specification_frame,
            paths.processed / "childcare_marketization_scenarios_specifications.parquet",
        )
        summarize_scenarios(specification_frame).to_csv(
            paths.outputs_tables / "childcare_marketization_scenarios_specifications.csv", index=False
        )
        write_json(
            scenario_specification_comparison,
            paths.outputs_reports / "childcare_scenario_specification_comparison.json",
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


def fit_home(paths) -> None:
    panel_path = paths.processed / "home_maintenance_panel.parquet"
    if not panel_path.exists():
        build_home(paths, sample=True)
    panel = read_parquet(panel_path)
    summary = fit_home_switching(panel, paths.outputs_reports / "home_switching.json")
    write_json(summary, paths.outputs_reports / "home_maintenance_summary.json")
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
        ingest_result = licensing.ingest(paths, sample=sample, refresh=refresh)
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


def report(paths) -> None:
    scenarios_path = paths.processed / "childcare_marketization_scenarios.parquet"
    state_path = paths.processed / "childcare_state_year_panel.parquet"
    county_path = paths.processed / "childcare_county_year_price_panel.parquet"
    if not scenarios_path.exists():
        simulate_childcare(paths)
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
    supply_iv_path = paths.outputs_reports / "childcare_supply_iv.json"
    satellite_account_path = paths.outputs_reports / "childcare_satellite_account.json"
    selected_sample = "broad_complete"
    selected_path = paths.outputs_reports / "childcare_demand_iv.json"
    if comparison_path.exists():
        comparison = read_json(comparison_path)
        samples = comparison.get("samples", {})
        selected_sample, _ = select_headline_sample(samples)
        if selected_sample:
            selected_path = _demand_summary_path(
                paths,
                selected_sample,
                specification_profile=_canonical_specification_profile_for_sample(selected_sample),
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
    write_assumption_audit(paths)
    write_childcare_figure_manifest(paths, demand_summary_path=selected_path)
    LOGGER.info("wrote report")


def add_ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--year", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unpaidwork")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("bootstrap", "fit-childcare", "simulate-childcare", "fit-home", "report"):
        subparsers.add_parser(command)

    supply_iv_parser = subparsers.add_parser("fit-supply-iv")
    add_ingest_args(supply_iv_parser)

    pull_parser = subparsers.add_parser("pull-core")
    add_ingest_args(pull_parser)

    licensing_parser = subparsers.add_parser("pull-licensing")
    add_ingest_args(licensing_parser)

    childcare_parser = subparsers.add_parser("build-childcare")
    add_ingest_args(childcare_parser)

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
        elif args.command == "pull-licensing":
            licensing.ingest(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
            )
        elif args.command == "build-childcare":
            build_childcare(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
            )
        elif args.command == "fit-childcare":
            fit_childcare(paths)
        elif args.command == "simulate-childcare":
            simulate_childcare(paths)
        elif args.command == "build-home":
            build_home(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
            )
        elif args.command == "fit-home":
            fit_home(paths)
        elif args.command == "fit-supply-iv":
            fit_supply_iv(
                paths,
                sample=sample,
                refresh=args.refresh,
                dry_run=args.dry_run,
                year=args.year,
            )
        elif args.command == "report":
            report(paths)
        else:
            parser.error(f"Unknown command: {args.command}")
    except UnpaidWorkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
