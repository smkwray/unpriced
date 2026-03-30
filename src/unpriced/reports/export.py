from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from unpriced.storage import read_json


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider] + rows)


def _window_summary(
    frame: pd.DataFrame,
    preferred_col: str,
    gross_col: str,
    quantity_col: str,
    quantity_key: str,
    year_start: int,
    year_end: int,
    exclude_sensitivity: bool,
) -> dict[str, object]:
    if frame.empty:
        return {}
    years = frame["year"].astype(int).tolist()
    return {
        "window_years": years,
        "window_label": f"{years[0]}-{years[-1]}" if years else f"{year_start}-{year_end}",
        "observation_count": int(len(frame)),
        "excludes_sensitivity_years": exclude_sensitivity,
        "preferred_value_mean": float(frame[preferred_col].mean()),
        "preferred_value_median": float(frame[preferred_col].median()),
        "gross_market_value_mean": float(frame[gross_col].mean()),
        quantity_key: float(frame[quantity_col].mean()),
    }


def build_markdown_report(
    price_surface_path: Path,
    demand_iv_path: Path,
    scenario_diagnostics_path: Path,
    scenarios: pd.DataFrame,
    output_path: Path,
    sipp_validation: pd.DataFrame | None = None,
    ce_validation: pd.DataFrame | None = None,
    pipeline_diagnostics_path: Path | None = None,
    demand_iv_strict_path: Path | None = None,
    scenario_sample_comparison_path: Path | None = None,
    scenario_specification_comparison_path: Path | None = None,
    demand_imputation_sweep_path: Path | None = None,
    demand_labor_support_sweep_path: Path | None = None,
    demand_specification_sweep_path: Path | None = None,
    piecewise_supply_demo_path: Path | None = None,
    dual_shift_summary_path: Path | None = None,
    supply_iv_path: Path | None = None,
    satellite_account_path: Path | None = None,
) -> Path:
    price_surface = read_json(price_surface_path)
    demand_iv = read_json(demand_iv_path)
    scenario_diagnostics = read_json(scenario_diagnostics_path)
    comparison = read_json(demand_iv_strict_path) if demand_iv_strict_path is not None and demand_iv_strict_path.exists() else None
    scenario_comparison = (
        read_json(scenario_sample_comparison_path)
        if scenario_sample_comparison_path is not None and scenario_sample_comparison_path.exists()
        else None
    )
    scenario_specification_comparison = (
        read_json(scenario_specification_comparison_path)
        if scenario_specification_comparison_path is not None and scenario_specification_comparison_path.exists()
        else None
    )
    imputation_sweep = (
        read_json(demand_imputation_sweep_path)
        if demand_imputation_sweep_path is not None and demand_imputation_sweep_path.exists()
        else None
    )
    labor_support_sweep = (
        read_json(demand_labor_support_sweep_path)
        if demand_labor_support_sweep_path is not None and demand_labor_support_sweep_path.exists()
        else None
    )
    specification_sweep = (
        read_json(demand_specification_sweep_path)
        if demand_specification_sweep_path is not None and demand_specification_sweep_path.exists()
        else None
    )
    piecewise_supply_demo = (
        read_json(piecewise_supply_demo_path)
        if piecewise_supply_demo_path is not None and piecewise_supply_demo_path.exists()
        else None
    )
    dual_shift_summary = (
        read_json(dual_shift_summary_path)
        if dual_shift_summary_path is not None and dual_shift_summary_path.exists()
        else None
    )
    supply_iv = (
        read_json(supply_iv_path)
        if supply_iv_path is not None and supply_iv_path.exists()
        else None
    )
    satellite_account = (
        read_json(satellite_account_path)
        if satellite_account_path is not None and satellite_account_path.exists()
        else None
    )
    preview = _markdown_table(scenarios.head(6))
    selected_sample = scenario_diagnostics.get("demand_sample_name", demand_iv.get("mode", "broad_complete"))
    current_mode = scenario_diagnostics.get("current_mode", demand_iv.get("current_mode", "unknown"))
    lines = [
        "# unpriced initial report",
        "",
        "## Childcare MVP",
        "- This MVP demonstrates the method on the strongest currently supported public-data sample.",
        "- Broader samples are retained as sensitivity/comparison results and are not the headline estimate.",
        f"- current mode: {current_mode}",
        f"- price-surface holdout year: {price_surface['holdout_year']}",
        f"- price-surface RMSE (test): {price_surface['rmse_test']:.2f}",
        f"- headline childcare sample: {selected_sample}",
        f"- demand estimation mode: {demand_iv.get('mode', 'broad_complete')}",
        f"- demand retained rows: {demand_iv.get('n_obs', '?')} ({demand_iv.get('n_states', '?')} states, {demand_iv.get('year_min', '?')}-{demand_iv.get('year_max', '?')})",
        f"- demand instrument: {demand_iv.get('instrument', 'outside_option_wage')}",
        f"- demand specification profile: {demand_iv.get('specification_profile', 'full_controls')}",
        f"- demand price coefficient: {demand_iv['price_coefficient']:.4f}",
        f"- demand elasticity at mean: {demand_iv['elasticity_at_mean']:.4f}",
        f"- demand economically admissible: {demand_iv.get('economically_admissible', False)}",
        f"- demand first-stage R^2: {demand_iv.get('first_stage_r2', 0):.4f}",
        f"- demand leave-one-state-out R^2: {demand_iv.get('loo_state_fips_r2', 'n/a')}",
        f"- demand leave-one-year-out R^2: {demand_iv.get('loo_year_r2', 'n/a')}",
        f"- scenario rows simulated: {scenario_diagnostics['scenario_rows']}",
        f"- scenario rows on observed NDCP support: {scenario_diagnostics.get('scenario_price_observed_support_rows', 0)}",
        f"- scenario rows labeled post-2022 nowcasts: {scenario_diagnostics.get('scenario_price_nowcast_rows', 0)}",
        f"- median observed gross market price: {scenario_diagnostics.get('baseline_price_p50', 0):.2f}",
        f"- median direct-care-equivalent price: {scenario_diagnostics.get('baseline_direct_care_price_p50', 0):.2f}",
        f"- median non-direct-care residual: {scenario_diagnostics.get('baseline_non_direct_care_price_p50', 0):.2f}",
        f"- median implied direct-care hourly wage: {scenario_diagnostics.get('baseline_implied_wage_p50', 0):.2f}",
        f"- median direct-care labor share: {scenario_diagnostics.get('baseline_direct_care_labor_share_p50', 0):.1%}",
        f"- direct-care clip-binding row share: {scenario_diagnostics.get('baseline_direct_care_clip_binding_row_share', 0):.1%} (raw labor-equivalent cost exceeds gross price in this share of rows; implied wages in clipped rows are mechanically determined by gross price, not independently by the decomposition)",
        f"- median alpha=0.50 gross marketization price: {scenario_diagnostics.get('alpha_50_price_p50', 0):.2f}",
        f"- median alpha=0.50 direct-care-equivalent price: {scenario_diagnostics.get('alpha_50_direct_care_price_p50', 0):.2f}",
        f"- median alpha=0.50 implied hourly wage: {scenario_diagnostics.get('alpha_50_implied_wage_p50', 0):.2f}",
        f"- incomplete state-year rows skipped at simulation time: {scenario_diagnostics['skipped_state_rows']}",
        f"- bootstrap resampling unit: {scenario_diagnostics['bootstrap_resampling_unit']}",
        f"- bootstrap draws requested/accepted/rejected: {scenario_diagnostics.get('bootstrap_draws_requested', 0)} / {scenario_diagnostics.get('bootstrap_draws_accepted', 0)} / {scenario_diagnostics.get('bootstrap_draws_rejected', 0)}",
        f"- bootstrap acceptance rate: {scenario_diagnostics.get('bootstrap_acceptance_rate', 0.0):.1%}",
        f"- bootstrap failed: {scenario_diagnostics.get('bootstrap_failed', False)}",
        f"- headline sample selection reason: {scenario_diagnostics.get('demand_sample_selection_reason', 'n/a')}",
        "- scenario uncertainty: 10th to 90th percentile bootstrap intervals over demand and supply elasticities",
        "- NDCP observed childcare prices end in 2022; any later scenario rows are labeled as nowcasts rather than observed support.",
        "- direct-care-equivalent prices remove the residual non-direct-care component using age/provider staffing assumptions plus observed childcare wages; implied wages are back-solved from that direct-care layer. This is an assumption-based decomposition, not a literal cost-accounting split. A 3x3 staffing-ratio x fringe-multiplier sensitivity sweep is available in `childcare_price_decomposition_sensitivity.json`.",
        "- model assumptions are enumerated in `model_assumption_audit.json`, which distinguishes configured economic assumptions from fallback heuristics and solver tuning constants.",
        "",
    ]
    if comparison is not None:
        ladder_rows = []
        label_map = {
            "broad_complete": "exploratory",
            "observed_core": "headline candidate",
            "observed_core_low_impute": "strict sensitivity",
        }
        samples = comparison.get("samples", {})
        for sample_name in ("broad_complete", "observed_core", "observed_core_low_impute"):
            sample = samples.get(sample_name, {})
            ladder_rows.append(
                {
                    "sample": sample_name,
                    "role": label_map.get(sample_name, "comparison"),
                    "rows": sample.get("n_obs", 0),
                    "states": sample.get("n_states", 0),
                    "years": sample.get("n_years", 0),
                    "elasticity": round(float(sample.get("elasticity_at_mean", float("nan"))), 4)
                    if pd.notna(sample.get("elasticity_at_mean", float("nan")))
                    else "nan",
                    "specification": sample.get("specification_profile", comparison.get("comparison_specification_profile", "unknown")),
                    "admissible": sample.get("economically_admissible", False),
                    "loo_state_r2": sample.get("loo_state_fips_r2", "nan"),
                    "loo_year_r2": sample.get("loo_year_r2", "nan"),
                    "headline_eligible": sample.get("headline_eligible", False),
                }
            )
        lines.extend(
            [
                "## Childcare sample ladder",
                f"- comparison specification profile: {comparison.get('comparison_specification_profile', 'unknown')}",
                _markdown_table(pd.DataFrame(ladder_rows)),
                "",
            ]
        )
    lines.extend(
        [
        "## Scenario preview",
        preview,
        "",
        "## Scenario diagnostics",
        f"- shadow-price interval width p10/p50/p90: {scenario_diagnostics['shadow_width_p10']:.4f} / {scenario_diagnostics['shadow_width_p50']:.4f} / {scenario_diagnostics['shadow_width_p90']:.4f}",
        f"- alpha-price interval width p10/p50/p90: {scenario_diagnostics['alpha_width_p10']:.4f} / {scenario_diagnostics['alpha_width_p50']:.4f} / {scenario_diagnostics['alpha_width_p90']:.4f}",
        f"- zero-width shadow intervals: {scenario_diagnostics['zero_width_shadow_count']}",
        f"- zero-width alpha intervals: {scenario_diagnostics['zero_width_alpha_count']}",
        f"- zero-unpaid scenarios: {scenario_diagnostics['zero_unpaid_quantity_count']}",
        f"- demand first-stage R^2: {scenario_diagnostics['demand_first_stage_r2']:.4f}",
        f"- demand leave-one-state-out R^2: {scenario_diagnostics.get('demand_loo_state_fips_r2', 'n/a')}",
        f"- demand leave-one-year-out R^2: {scenario_diagnostics.get('demand_loo_year_r2', 'n/a')}",
        f"- solver demand elasticity magnitude: {scenario_diagnostics.get('solver_demand_elasticity_magnitude', 0):.4f}",
        f"- supply elasticity: {scenario_diagnostics.get('supply_elasticity', 0):.4f}",
        f"- supply estimation method: {scenario_diagnostics.get('supply_estimation_method', 'unknown')}",
        f"- supply positive-only weighted median slope: {scenario_diagnostics.get('supply_within_state_year_weighted_median_positive_slope', float('nan'))}",
        f"- supply all-slopes weighted median slope: {scenario_diagnostics.get('supply_within_state_year_weighted_median_all_slope', float('nan'))}",
        f"- supply positivity-selection gap: {scenario_diagnostics.get('supply_within_state_year_weighted_median_gap', float('nan'))}",
        f"- demand fit quarantined: {scenario_diagnostics.get('demand_fit_quarantined', False)}",
        "",
        ]
    )
    if scenario_comparison is not None:
        scenario_rows = []
        samples = scenario_comparison.get("samples", {})
        for sample_name in ("broad_complete", "observed_core", "observed_core_low_impute"):
            sample = samples.get(sample_name, {})
            scenario_rows.append(
                {
                    "sample": sample_name,
                    "rows": sample.get("scenario_rows", 0),
                    "states": sample.get("scenario_states", 0),
                    "years": (
                        f"{sample.get('scenario_year_min', 0)}-{sample.get('scenario_year_max', 0)}"
                        if sample.get("scenario_rows", 0)
                        else "n/a"
                    ),
                    "shadow_width_p50": round(float(sample.get("shadow_width_p50", float("nan"))), 4)
                    if pd.notna(sample.get("shadow_width_p50", float("nan")))
                    else "nan",
                    "alpha_width_p50": round(float(sample.get("alpha_width_p50", float("nan"))), 4)
                    if pd.notna(sample.get("alpha_width_p50", float("nan")))
                    else "nan",
                    "nowcast_rows": sample.get("scenario_price_nowcast_rows", 0),
                    "admissible": sample.get("demand_economically_admissible", False),
                    "quarantined": sample.get("demand_fit_quarantined", False),
                    "specification": sample.get("demand_specification_profile", scenario_comparison.get("comparison_specification_profile", "unknown")),
                    "selection_role": "headline"
                    if sample_name == scenario_comparison.get("selected_headline_sample")
                    else "comparison",
                }
            )
        lines.extend(
            [
                "## Scenario sample comparison",
                f"- comparison specification profile: {scenario_comparison.get('comparison_specification_profile', 'unknown')}",
                _markdown_table(pd.DataFrame(scenario_rows)),
                "",
            ]
        )
    if scenario_specification_comparison is not None:
        spec_rows = []
        for profile in ("household_parsimonious", "instrument_only", "labor_parsimonious", "full_controls"):
            sample = scenario_specification_comparison.get("profiles", {}).get(profile, {})
            if not sample:
                continue
            spec_rows.append(
                {
                    "profile": profile,
                    "rows": sample.get("scenario_rows", 0),
                    "shadow_width_p50": round(float(sample.get("shadow_width_p50", float("nan"))), 4)
                    if pd.notna(sample.get("shadow_width_p50", float("nan")))
                    else "nan",
                    "alpha_width_p50": round(float(sample.get("alpha_width_p50", float("nan"))), 4)
                    if pd.notna(sample.get("alpha_width_p50", float("nan")))
                    else "nan",
                    "nowcast_rows": sample.get("scenario_price_nowcast_rows", 0),
                    "loo_state_r2": sample.get("demand_loo_state_fips_r2", "nan"),
                    "loo_year_r2": sample.get("demand_loo_year_r2", "nan"),
                    "elasticity": round(float(sample.get("demand_elasticity_at_mean", float("nan"))), 4)
                    if pd.notna(sample.get("demand_elasticity_at_mean", float("nan")))
                    else "nan",
                    "admissible": sample.get("demand_economically_admissible", False),
                    "quarantined": sample.get("demand_fit_quarantined", False),
                    "role": "canonical"
                    if profile == scenario_diagnostics.get("demand_specification_profile")
                    else "sensitivity",
                }
            )
        lines.extend(
            [
                "## Scenario specification comparison",
                f"- selected-sample comparison specification profile: {scenario_specification_comparison.get('comparison_specification_profile', 'unknown')}",
                _markdown_table(pd.DataFrame(spec_rows)),
                "",
            ]
        )
    if imputation_sweep is not None:
        sweep_rows = []
        for item in imputation_sweep.get("thresholds", []):
            sweep_rows.append(
                {
                    "max_imputed_share": item.get("threshold"),
                    "rows": item.get("n_obs", 0),
                    "states": item.get("n_states", 0),
                    "years": item.get("n_years", 0),
                    "loo_state_r2": item.get("loo_state_fips_r2", "nan"),
                    "loo_year_r2": item.get("loo_year_r2", "nan"),
                    "headline_eligible": item.get("headline_eligible", False),
                }
            )
        best_threshold = imputation_sweep.get("best_headline_eligible_threshold")
        lines.extend(
            [
                "## Observed-core imputation sweep",
                f"- sweep specification profile: {imputation_sweep.get('specification_profile', 'unknown')}",
                _markdown_table(pd.DataFrame(sweep_rows)),
                f"- current headline observed_core: {imputation_sweep.get('current_headline_n_obs', 0)} rows, {imputation_sweep.get('current_headline_n_states', 0)} states, leave-one-state-out R^2 {imputation_sweep.get('current_headline_loo_state_fips_r2', 'nan')}, leave-one-year-out R^2 {imputation_sweep.get('current_headline_loo_year_r2', 'nan')}",
                (
                    f"- best headline-eligible low-impute threshold in the current sweep: <= {best_threshold.get('threshold')} "
                    f"with {best_threshold.get('n_obs')} rows, {best_threshold.get('n_states')} states, "
                    f"LOO state R^2 {best_threshold.get('loo_state_fips_r2')}, LOO year R^2 {best_threshold.get('loo_year_r2')}"
                )
                if best_threshold is not None
                else "- no low-impute threshold in the current sweep clears the headline support rule",
                "",
            ]
        )
    if labor_support_sweep is not None:
        sweep_rows = []
        for item in labor_support_sweep.get("thresholds", []):
            sweep_rows.append(
                {
                    "min_qcew_labor_share": item.get("threshold"),
                    "rows": item.get("n_obs", 0),
                    "states": item.get("n_states", 0),
                    "years": item.get("n_years", 0),
                    "loo_state_r2": item.get("loo_state_fips_r2", "nan"),
                    "loo_year_r2": item.get("loo_year_r2", "nan"),
                    "headline_eligible": item.get("headline_eligible", False),
                }
            )
        best_threshold = labor_support_sweep.get("best_headline_eligible_threshold")
        lines.extend(
            [
                "## Observed-core labor-support sweep",
                f"- sweep specification profile: {labor_support_sweep.get('specification_profile', 'unknown')}",
                _markdown_table(pd.DataFrame(sweep_rows)),
                f"- current headline observed_core: {labor_support_sweep.get('current_headline_n_obs', 0)} rows, {labor_support_sweep.get('current_headline_n_states', 0)} states, leave-one-state-out R^2 {labor_support_sweep.get('current_headline_loo_state_fips_r2', 'nan')}, leave-one-year-out R^2 {labor_support_sweep.get('current_headline_loo_year_r2', 'nan')}",
                (
                    f"- best headline-eligible labor-support threshold in the current sweep: >= {best_threshold.get('threshold')} "
                    f"with {best_threshold.get('n_obs')} rows, {best_threshold.get('n_states')} states, "
                    f"LOO state R^2 {best_threshold.get('loo_state_fips_r2')}, LOO year R^2 {best_threshold.get('loo_year_r2')}"
                )
                if best_threshold is not None
                else "- no labor-support threshold in the current sweep clears the headline support rule",
                "",
            ]
        )
    if specification_sweep is not None:
        spec_rows = []
        for profile in ("full_controls", "household_parsimonious", "labor_parsimonious", "instrument_only"):
            item = specification_sweep.get("profiles", {}).get(profile, {})
            spec_rows.append(
                {
                    "profile": profile,
                    "controls": ", ".join(item.get("exog_features", [])) if item.get("exog_features") else "none",
                    "rows": item.get("n_obs", 0),
                    "first_stage_r2": round(float(item.get("first_stage_r2", float("nan"))), 4)
                    if pd.notna(item.get("first_stage_r2", float("nan")))
                    else "nan",
                    "loo_state_r2": item.get("loo_state_fips_r2", "nan"),
                    "loo_year_r2": item.get("loo_year_r2", "nan"),
                    "elasticity": round(float(item.get("elasticity_at_mean", float("nan"))), 4)
                    if pd.notna(item.get("elasticity_at_mean", float("nan")))
                    else "nan",
                }
            )
        preferred_profile = specification_sweep.get("preferred_holdout_profile")
        lines.extend(
            [
                "## Observed-core specification sweep",
                f"- current comparison profile: {specification_sweep.get('current_profile', 'unknown')}",
                _markdown_table(pd.DataFrame(spec_rows)),
                (
                    f"- admissible specification profiles beating the current {specification_sweep.get('current_profile', 'unknown')} spec on both holdouts: {specification_sweep.get('profiles_beating_current_on_both_holdouts', [])}"
                ),
                (
                    f"- holdout-preferred admissible profile in the current sweep: {preferred_profile}"
                    if preferred_profile is not None
                    else f"- no admissible alternative profile beats the current {specification_sweep.get('current_profile', 'unknown')} spec on both holdouts"
                ),
                "",
            ]
        )
    if piecewise_supply_demo is not None:
        lines.extend(
            [
                "## Piecewise supply demo",
                "- This is a non-canonical methodology demo on the high-labor-support observed-core subset.",
                f"- demo rows/states/years: {piecewise_supply_demo.get('demo_rows', 0)} / {piecewise_supply_demo.get('demo_states', 0)} / {piecewise_supply_demo.get('demo_years', [])}",
                f"- labor-support threshold: >= {piecewise_supply_demo.get('labor_support_threshold', 'n/a')}",
                f"- piecewise method: {piecewise_supply_demo.get('piecewise_method', 'unknown')}",
                f"- supported both-side state-years: {piecewise_supply_demo.get('piecewise_supported_both_sides_groups', 0)} / {piecewise_supply_demo.get('piecewise_group_count', 0)}",
                f"- pooled eta below baseline: {float(piecewise_supply_demo.get('piecewise_pooled_eta_below', float('nan'))):.4f}",
                f"- pooled eta above baseline: {float(piecewise_supply_demo.get('piecewise_pooled_eta_above', float('nan'))):.4f}",
                f"- fallback share on either side: {float(piecewise_supply_demo.get('piecewise_fallback_share_any_side', 0.0)):.1%}",
                f"- median alpha=0.50 piecewise minus constant price effect: {float(piecewise_supply_demo.get('alpha_50_piecewise_minus_constant_p50', 0.0)):.4f}",
                f"- median alpha=1.00 piecewise minus constant price effect: {float(piecewise_supply_demo.get('alpha_100_piecewise_minus_constant_p50', 0.0)):.4f}",
                "",
            ]
        )
    if dual_shift_summary is not None:
        dual_shift_rows = pd.DataFrame(dual_shift_summary.get("headline_alpha_table", []))
        if not dual_shift_rows.empty:
            dual_shift_rows = dual_shift_rows.loc[
                :,
                [
                    column
                    for column in (
                        "kappa_q",
                        "kappa_c",
                        "median_p_alpha",
                        "median_p_alpha_pct_change",
                        "share_price_increase",
                        "share_price_decrease",
                    )
                    if column in dual_shift_rows.columns
                ],
            ]
        frontier_rows = pd.DataFrame(dual_shift_summary.get("frontier_summary", []))
        if not frontier_rows.empty:
            frontier_rows = frontier_rows.loc[
                :,
                [
                    column
                    for column in (
                        "kappa_c",
                        "kappa_q_zero_price_frontier_p10",
                        "kappa_q_zero_price_frontier_p50",
                        "kappa_q_zero_price_frontier_p90",
                    )
                    if column in frontier_rows.columns
                ],
            ]
        lines.extend(
            [
                "## Dual-shift marketization sensitivity",
                "- This additive parallel estimand keeps the canonical short-run benchmark unchanged.",
                "- Short-run fixed-supply benchmark: positive alpha raises price by construction because alpha shifts demand only.",
                "- Medium-run dual-shift sensitivity: price can rise or fall because marketization can expand supply through `kappa_q` and raise costs through `kappa_c`.",
                f"- headline alpha: {float(dual_shift_summary.get('headline_alpha', 0.5)):.2f}",
                f"- short-run fixed-supply median headline price: {float(dual_shift_summary.get('short_run_fixed_supply_headline_alpha_price_p50', 0.0)):.2f}",
                f"- short-run fixed-supply median headline pct change: {float(dual_shift_summary.get('short_run_fixed_supply_headline_alpha_pct_change_p50', 0.0)):.4f}",
                f"- bootstrap draws requested/accepted/rejected: {dual_shift_summary.get('bootstrap_draws_requested', 0)} / {dual_shift_summary.get('bootstrap_draws_accepted', 0)} / {dual_shift_summary.get('bootstrap_draws_rejected', 0)}",
                f"- bootstrap acceptance rate: {float(dual_shift_summary.get('bootstrap_acceptance_rate', 0.0)):.1%}",
            ]
        )
        if not dual_shift_rows.empty:
            lines.extend(
                [
                    "",
                    "### Headline-alpha surface",
                    _markdown_table(dual_shift_rows.round(4)),
                ]
            )
        if not frontier_rows.empty:
            lines.extend(
                [
                    "",
                    "### Zero-price-change frontier",
                    _markdown_table(frontier_rows.round(4)),
                ]
            )
        lines.append("")
    if supply_iv is not None:
        lines.extend(
            [
                "## Supply IV demo",
                f"- status: {supply_iv.get('status', 'unknown')}",
                f"- design: {supply_iv.get('design', 'unknown')}",
                f"- note: {supply_iv.get('note', '')}",
            ]
        )
        if supply_iv.get("status") == "ok":
            secondary_supply = supply_iv.get("secondary_supply_estimate", {}) or {}
            lines.extend(
                [
                    f"- rows/counties/states: {supply_iv.get('n_obs', 0)} / {supply_iv.get('n_counties', 0)} / {supply_iv.get('n_states', 0)}",
                    f"- years: {supply_iv.get('year_min', 'n/a')}-{supply_iv.get('year_max', 'n/a')}",
                    f"- treated states with nonzero licensing shock: {supply_iv.get('treated_state_fips', [])}",
                    f"- first-stage price beta: {supply_iv.get('first_stage_price', {}).get('beta', float('nan'))}",
                    f"- first-stage F-stat: {supply_iv.get('first_stage_price', {}).get('f_stat', float('nan'))}",
                    f"- secondary local IV supply elasticity (provider density): {secondary_supply.get('value', supply_iv.get('local_iv_supply_elasticity_provider_density', float('nan')))}",
                    f"- secondary local IV scope: {secondary_supply.get('scope', 'treated_state_local_wald')}",
                    f"- local IV supply elasticity (employer establishment density): {supply_iv.get('local_iv_supply_elasticity_employer_establishment_density', float('nan'))} (n_obs={supply_iv.get('reduced_form_employer_establishment_density', {}).get('n_obs', '?')} vs {supply_iv.get('reduced_form_provider_density', {}).get('n_obs', '?')} for provider density — sparse CBP coverage)",
                ]
            )
        lines.append("")
    if satellite_account is not None:
        latest = satellite_account.get("latest_year", {})
        methodologies = satellite_account.get("benchmark_methodologies", {})
        under5_method = methodologies.get("under5_child_equivalent_replacement_cost", {})
        under5_latest = under5_method.get("latest_year", {})
        lines.extend(
            [
                "## National childcare benchmarks",
                "- The report now carries two descriptive benchmarks side by side rather than one silently replacing the other.",
                f"- annual-hours account identity: {satellite_account.get('valuation_identity', 'hourly replacement price x unpaid annual childcare hours')}",
                f"- under-5 child-equivalent identity: {under5_method.get('valuation_identity', 'annual replacement price x under5 child-equivalent quantity')}",
                f"- support years: {satellite_account.get('support_years', [])}",
                (
                    f"- latest annual-hours account year ({latest.get('year')}): preferred benchmark {latest.get('preferred_value', 0):.2f}, "
                    f"gross-market upper benchmark {latest.get('gross_market_total_value', 0):.2f}, "
                    f"specialist-wage benchmark {latest.get('specialist_total_value', 0):.2f}"
                )
                if latest
                else "- latest annual-hours account year: n/a",
                (
                    f"- latest under-5 child-equivalent year ({under5_latest.get('year')}): preferred benchmark {under5_latest.get('preferred_value', 0):.2f}, "
                    f"gross-market upper benchmark {under5_latest.get('gross_market_value', 0):.2f}, "
                    f"quantity {under5_latest.get('quantity_slots', 0):.2f}"
                )
                if under5_latest
                else "- latest under-5 child-equivalent year: n/a",
                (
                    f"- latest-year price-support population share: {latest.get('price_support_population_share', 0):.1%}"
                    if latest
                    else "- latest-year price-support population share: n/a"
                ),
                (
                    f"- latest-year active household / nonhousehold / supervisory hours: "
                    f"{latest.get('national_active_household_childcare_hours_total', 0):.2f} / "
                    f"{latest.get('national_active_nonhousehold_childcare_hours_total', 0):.2f} / "
                    f"{latest.get('national_supervisory_childcare_hours_total', 0):.2f}"
                )
                if latest
                else "- latest-year active household / nonhousehold / supervisory hours: n/a",
                "- The direct-care benchmark nets out a pooled residual bucket that can include facilities, administration, meals, transport, advertising, profits, and market power, but those pieces are not separately identified.",
                "- The annual-hours account is broader because supervisory care is still additive and not yet overlap-adjusted.",
                "- Travel is still excluded in this first repair wave, so the broader annual-hours account remains conservative relative to some household-production accounts even while it is much larger than the child-equivalent bridge.",
                "",
            ]
        )
    if bool(scenario_diagnostics.get("demand_first_stage_near_perfect", False)):
        lines.extend(
            [
                "## Warnings",
                "- Demand first-stage fit is near-perfect in the current MVP, so even the clustered bootstrap intervals should be treated as optimistic until the identification layer is tightened.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Warnings",
                "- The current headline childcare sample is more defensible than the broad exploratory sample, but its out-of-sample state/year diagnostics remain weak and should still be treated as demonstration-grade rather than final inference.",
                (
                    f"- Supply elasticity is now estimated from within-state-year county variation (`{scenario_diagnostics.get('supply_estimation_method', 'unknown')}`), with pooled slope {float(scenario_diagnostics.get('supply_pooled_loglog_slope', float('nan'))):.4f} and year-demeaned slope {float(scenario_diagnostics.get('supply_year_demeaned_loglog_slope', float('nan'))):.4f}."
                    if scenario_diagnostics.get("supply_estimation_method") is not None
                    else "- Supply elasticity remains a calibrated scenario input rather than a fully identified causal estimate."
                ),
                (
                    f"- Canonical childcare scenarios include post-2022 nowcast rows for years {scenario_diagnostics.get('scenario_price_nowcast_years', [])}; those rows are extrapolated beyond observed NDCP price support."
                    if scenario_diagnostics.get("scenario_contains_nowcast_rows", False)
                    else "- Canonical childcare scenarios currently stay within observed NDCP price support; any future post-2022 extensions should remain explicitly labeled as nowcasts."
                ),
                "",
            ]
        )
    if sipp_validation is not None and not sipp_validation.empty:
        lines.extend(
            [
                "## SIPP validation preview",
                _markdown_table(sipp_validation.head(6)),
                "",
            ]
        )
    if ce_validation is not None and not ce_validation.empty:
        lines.extend(
            [
                "## CE validation preview",
                _markdown_table(ce_validation.head(6)),
                "",
            ]
        )
    if pipeline_diagnostics_path is not None:
        pd_diag = read_json(pipeline_diagnostics_path)
        n_state = pd_diag.get("state_year_rows", 0)
        n_county = pd_diag.get("county_year_rows", 0)
        lines.extend(
            [
                "## Pipeline data-quality diagnostics",
                "",
                "### State-year panel",
                f"- rows: {n_state} ({pd_diag.get('state_year_unique_states', '?')} states, {pd_diag.get('state_year_min', '?')}-{pd_diag.get('state_year_max', '?')})",
                f"- births observed (CDC WONDER): {pd_diag.get('births_cdc_wonder_observed', 0)} / {n_state}",
                f"- births ATUS-reported fallback: {pd_diag.get('births_atus_reported', 0)} / {n_state}",
                f"- births CDC suppressed fallback: {pd_diag.get('births_cdc_suppressed_fallback', 0)} / {n_state}",
                f"- births unmatched state fallback: {pd_diag.get('births_atus_unmatched_fallback', 0)} / {n_state}",
                f"- controls from ACS observed: {pd_diag.get('state_controls_acs_observed', 0)} / {n_state}",
                f"- controls from ATUS synthetic: {pd_diag.get('state_controls_atus_synthetic', 0)} / {n_state}",
                f"- unemployment from LAUS observed: {pd_diag.get('state_unemployment_laus_observed', 0)} / {n_state}",
                f"- ATUS 2020 sensitivity-year rows: {pd_diag.get('atus_sensitivity_year_rows', 0)}",
                f"- state price support status counts: {pd_diag.get('state_price_observation_status_counts', {})}",
                f"- state rows labeled post-support nowcasts: {pd_diag.get('state_price_post_support_nowcast_rows', 0)} / {n_state}",
                f"- state_price_index NA (no county match): {pd_diag.get('state_state_price_index_na', 0)} / {n_state}",
                f"- mean state QCEW labor observed share: {pd_diag.get('state_qcew_labor_observed_share_mean', 'n/a')}",
                f"- observed_core min state QCEW labor observed share: {pd_diag.get('observed_core_state_qcew_labor_observed_share_min', 'n/a')}",
                f"- broad_complete eligible rows: {pd_diag.get('eligible_broad_complete', 0)} / {n_state}",
                f"- observed_core eligible rows: {pd_diag.get('eligible_observed_core', 0)} / {n_state}",
                f"- observed_core_low_impute eligible rows: {pd_diag.get('eligible_observed_core_low_impute', 0)} / {n_state}",
                f"- observed_core exclusions: {pd_diag.get('observed_core_exclusion_counts', {})}",
                f"- observed_core_low_impute exclusions: {pd_diag.get('observed_core_low_impute_exclusion_counts', {})}",
                "",
                "### County-year panel",
                f"- rows: {n_county} ({pd_diag.get('county_year_unique_counties', '?')} counties, {pd_diag.get('county_year_min', '?')}-{pd_diag.get('county_year_max', '?')})",
                f"- NDCP prices with any imputation: {pd_diag.get('ndcp_any_imputed_rows', 0)} / {n_county} (mean share {pd_diag.get('ndcp_mean_imputed_share', 0):.1%})",
                f"- NDCP fully imputed: {pd_diag.get('ndcp_fully_imputed_rows', 0)} / {n_county}",
                f"- county controls from ACS direct: {pd_diag.get('county_controls_acs_direct', 0)} / {n_county}",
                f"- county controls from state backfill: {pd_diag.get('county_controls_state_backfill', 0)} / {n_county}",
                f"- ACS years available: {pd_diag.get('acs_available_years', [])}",
                f"- ACS years missing for county panel: {pd_diag.get('acs_missing_county_years', [])}",
                f"- county single_parent_share still null: {pd_diag.get('county_single_parent_share_null', 0)} / {n_county}",
                f"- QCEW county-year direct matches: {pd_diag.get('county_qcew_direct', 0)} / {n_county}",
                f"- QCEW years available: {pd_diag.get('qcew_available_years', [])}",
                f"- QCEW years missing for county panel: {pd_diag.get('qcew_missing_county_years', [])}",
                f"- wages from price-derived fallback: {pd_diag.get('county_wage_price_derived', 0)} / {n_county}",
                f"- wages from QCEW observed: {pd_diag.get('county_wage_observed', 0)} / {n_county}",
                f"- employment from synthetic formula: {pd_diag.get('county_employment_synthetic', 0)} / {n_county}",
                f"- employment from QCEW observed: {pd_diag.get('county_employment_observed', 0)} / {n_county}",
                f"- CBP employer data observed: {pd_diag.get('county_cbp_observed', 0)} / {n_county}",
                f"- NES nonemployer data observed: {pd_diag.get('county_nes_observed', 0)} / {n_county}",
                f"- Head Start slots observed: {pd_diag.get('county_head_start_observed', 0)} / {n_county}",
                f"- LAUS unemployment observed: {pd_diag.get('county_laus_observed', 0)} / {n_county}",
                f"- LAUS years available: {pd_diag.get('laus_available_years', [])}",
                f"- LAUS years missing for county panel: {pd_diag.get('laus_missing_county_years', [])}",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "- This first build uses sample-mode data by default for smoke testing.",
            "- Childcare causal results are state-year only; county-year outputs are descriptive and simulation inputs.",
            "- NDCP observed childcare-price support ends in 2022; any later childcare scenario rows must be treated as nowcasts, not observed prices.",
            "- SIPP validation outputs are national because the inspected 2023 public-use file does not expose a clean state identifier in this build.",
            "- CE validation outputs are national interview-family spending benchmarks based on the babysitting/daycare measure in the 2023 interview files.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _weighted_average(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    values = pd.to_numeric(frame.get(value_col), errors="coerce")
    if not isinstance(values, pd.Series):
        values = pd.Series([values] * len(frame), index=frame.index, dtype="float64")
    weights = pd.to_numeric(frame.get(weight_col), errors="coerce").fillna(0.0)
    valid = values.notna() & weights.gt(0)
    if valid.any():
        return float((values.loc[valid] * weights.loc[valid]).sum() / weights.loc[valid].sum())
    if values.notna().any():
        return float(values.loc[values.notna()].mean())
    return float("nan")


def build_childcare_satellite_account(
    county: pd.DataFrame,
    state: pd.DataFrame,
    acs: pd.DataFrame | None,
    childcare_assumptions: dict[str, object],
    output_json_path: Path,
    output_markdown_path: Path,
    output_table_path: Path,
) -> tuple[Path, Path, Path]:
    market_hours = float(childcare_assumptions["market_hours_per_child_per_week"])
    headline_year_start = int(childcare_assumptions.get("static_account_headline_year_start", 2017))
    headline_year_end = int(childcare_assumptions.get("static_account_headline_year_end", 2019))
    exclude_sensitivity = bool(childcare_assumptions.get("static_account_exclude_sensitivity_years", True))
    county_frame = county.copy()
    state_frame = state.copy()

    if "unpaid_active_childcare_hours" not in state_frame.columns and "unpaid_childcare_hours" in state_frame.columns:
        state_frame["unpaid_active_childcare_hours"] = state_frame["unpaid_childcare_hours"]
    if "unpaid_active_household_childcare_hours" not in state_frame.columns and "unpaid_active_childcare_hours" in state_frame.columns:
        state_frame["unpaid_active_household_childcare_hours"] = state_frame["unpaid_active_childcare_hours"]
    if "unpaid_active_nonhousehold_childcare_hours" not in state_frame.columns:
        state_frame["unpaid_active_nonhousehold_childcare_hours"] = 0.0
    if "unpaid_supervisory_childcare_hours" not in state_frame.columns:
        state_frame["unpaid_supervisory_childcare_hours"] = 0.0
    if (
        "unpaid_active_childcare_hours_total" not in state_frame.columns
        and {"unpaid_active_childcare_hours", "atus_weight_sum"} <= set(state_frame.columns)
    ):
        state_frame["unpaid_active_childcare_hours_total"] = (
            pd.to_numeric(state_frame["unpaid_active_childcare_hours"], errors="coerce")
            * pd.to_numeric(state_frame["atus_weight_sum"], errors="coerce")
        )
    if (
        "unpaid_active_household_childcare_hours_total" not in state_frame.columns
        and {"unpaid_active_household_childcare_hours", "atus_weight_sum"} <= set(state_frame.columns)
    ):
        state_frame["unpaid_active_household_childcare_hours_total"] = (
            pd.to_numeric(state_frame["unpaid_active_household_childcare_hours"], errors="coerce")
            * pd.to_numeric(state_frame["atus_weight_sum"], errors="coerce")
        )
    if (
        "unpaid_active_nonhousehold_childcare_hours_total" not in state_frame.columns
        and {"unpaid_active_nonhousehold_childcare_hours", "atus_weight_sum"} <= set(state_frame.columns)
    ):
        state_frame["unpaid_active_nonhousehold_childcare_hours_total"] = (
            pd.to_numeric(state_frame["unpaid_active_nonhousehold_childcare_hours"], errors="coerce")
            * pd.to_numeric(state_frame["atus_weight_sum"], errors="coerce")
        )
    if (
        "unpaid_supervisory_childcare_hours_total" not in state_frame.columns
        and {"unpaid_supervisory_childcare_hours", "atus_weight_sum"} <= set(state_frame.columns)
    ):
        state_frame["unpaid_supervisory_childcare_hours_total"] = (
            pd.to_numeric(state_frame["unpaid_supervisory_childcare_hours"], errors="coerce")
            * pd.to_numeric(state_frame["atus_weight_sum"], errors="coerce")
        )

    for column in (
        "year",
        "under5_population",
        "annual_price",
        "direct_care_price_index",
        "non_direct_care_price_index",
        "benchmark_childcare_wage",
        "specialist_childcare_wage",
    ):
        if column in county_frame.columns:
            county_frame[column] = pd.to_numeric(county_frame[column], errors="coerce")
    for column in (
        "year",
        "unpaid_active_childcare_hours",
        "unpaid_active_household_childcare_hours",
        "unpaid_active_nonhousehold_childcare_hours",
        "unpaid_supervisory_childcare_hours",
        "under5_population",
        "unpaid_quantity_proxy",
        "unpaid_active_childcare_hours_total",
        "unpaid_active_household_childcare_hours_total",
        "unpaid_active_nonhousehold_childcare_hours_total",
        "unpaid_supervisory_childcare_hours_total",
        "atus_weight_sum",
    ):
        if column in state_frame.columns:
            state_frame[column] = pd.to_numeric(state_frame[column], errors="coerce")

    county_valid = county_frame.loc[
        county_frame["year"].notna()
        & county_frame["under5_population"].notna()
        & county_frame["under5_population"].gt(0)
    ].copy()
    state_valid = state_frame.loc[
        state_frame["year"].notna()
        & state_frame["unpaid_active_childcare_hours_total"].notna()
        & state_frame["atus_weight_sum"].notna()
        & state_frame["atus_weight_sum"].gt(0)
    ].copy()

    county_years = []
    for year, group in county_valid.groupby("year", as_index=False):
        county_years.append(
            {
                "year": int(year),
                "price_support_population": float(pd.to_numeric(group["under5_population"], errors="coerce").sum()),
                "gross_market_price": _weighted_average(group, "annual_price", "under5_population"),
                "direct_care_price": _weighted_average(group, "direct_care_price_index", "under5_population"),
                "non_direct_care_price": _weighted_average(group, "non_direct_care_price_index", "under5_population"),
                "benchmark_childcare_wage": _weighted_average(group, "benchmark_childcare_wage", "under5_population"),
                "specialist_childcare_wage": _weighted_average(group, "specialist_childcare_wage", "under5_population"),
            }
        )
    county_years_frame = pd.DataFrame(county_years)
    if not county_years_frame.empty:
        county_years_frame["gross_market_hourly_price"] = county_years_frame["gross_market_price"].div(
            52.0 * market_hours
        )
        county_years_frame["direct_care_hourly_price"] = county_years_frame["direct_care_price"].div(
            52.0 * market_hours
        )
        county_years_frame["non_direct_care_hourly_price"] = county_years_frame["non_direct_care_price"].div(
            52.0 * market_hours
        )

    state_years = []
    for year, group in state_valid.groupby("year", as_index=False):
        sensitivity = group.get("is_sensitivity_year")
        sensitivity_flag = (
            bool(sensitivity.fillna(False).astype(bool).any())
            if sensitivity is not None
            else bool(int(year) == 2020)
        )
        state_years.append(
            {
                "year": int(year),
                "unpaid_active_childcare_hours_per_respondent_year": _weighted_average(
                    group,
                    "unpaid_active_childcare_hours",
                    "atus_weight_sum",
                ),
                "unpaid_supervisory_childcare_hours_per_respondent_year": _weighted_average(
                    group,
                    "unpaid_supervisory_childcare_hours",
                    "atus_weight_sum",
                ),
                "national_active_childcare_hours_total": float(
                    pd.to_numeric(group["unpaid_active_childcare_hours_total"], errors="coerce").sum()
                ),
                "national_active_household_childcare_hours_total": float(
                    pd.to_numeric(group["unpaid_active_household_childcare_hours_total"], errors="coerce").sum()
                ),
                "national_active_nonhousehold_childcare_hours_total": float(
                    pd.to_numeric(group["unpaid_active_nonhousehold_childcare_hours_total"], errors="coerce").sum()
                ),
                "national_supervisory_childcare_hours_total": float(
                    pd.to_numeric(group["unpaid_supervisory_childcare_hours_total"], errors="coerce").sum()
                ),
                "atus_sensitivity_year": sensitivity_flag,
            }
        )
    state_years_frame = pd.DataFrame(state_years)

    annual = county_years_frame.merge(state_years_frame, on="year", how="inner").sort_values("year").reset_index(drop=True)
    if acs is not None and not acs.empty:
        acs_frame = acs.copy()
        acs_frame["year"] = pd.to_numeric(acs_frame.get("year"), errors="coerce")
        acs_frame["under5_population"] = pd.to_numeric(acs_frame.get("under5_population"), errors="coerce")
        acs_years = (
            acs_frame.dropna(subset=["year", "under5_population"])
            .groupby("year", as_index=False)["under5_population"]
            .sum()
            .rename(columns={"under5_population": "national_under5_population"})
        )
        acs_years["year"] = acs_years["year"].astype(int)
        annual = annual.merge(acs_years, on="year", how="left")
        annual = annual.loc[pd.to_numeric(annual.get("national_under5_population"), errors="coerce").notna()].copy()
    annual["national_under5_population"] = pd.to_numeric(
        annual.get("national_under5_population"), errors="coerce"
    ).fillna(annual["price_support_population"])
    annual["price_support_population_share"] = annual["price_support_population"].div(
        annual["national_under5_population"].replace({0: pd.NA})
    )
    annual["under5_child_equivalent_quantity"] = (
        annual["unpaid_active_childcare_hours_per_respondent_year"]
        * annual["national_under5_population"]
        / (52.0 * market_hours)
    )
    annual["under5_average_unpaid_childcare_hours_per_child_year"] = annual[
        "unpaid_active_childcare_hours_per_respondent_year"
    ]
    annual["specialist_child_equivalent_price"] = annual["specialist_childcare_wage"] * 52.0 * market_hours
    annual["under5_direct_care_value"] = annual["under5_child_equivalent_quantity"] * annual["direct_care_price"]
    annual["under5_gross_market_value"] = annual["under5_child_equivalent_quantity"] * annual["gross_market_price"]
    annual["under5_non_direct_value"] = annual["under5_child_equivalent_quantity"] * annual["non_direct_care_price"]
    annual["under5_specialist_value"] = (
        annual["under5_child_equivalent_quantity"] * annual["specialist_child_equivalent_price"]
    )
    annual["national_total_childcare_hours_total"] = (
        annual["national_active_childcare_hours_total"] + annual["national_supervisory_childcare_hours_total"]
    )
    annual["direct_care_active_value"] = (
        annual["national_active_childcare_hours_total"] * annual["direct_care_hourly_price"]
    )
    annual["direct_care_total_value"] = (
        annual["national_total_childcare_hours_total"] * annual["direct_care_hourly_price"]
    )
    annual["gross_market_active_value"] = (
        annual["national_active_childcare_hours_total"] * annual["gross_market_hourly_price"]
    )
    annual["gross_market_total_value"] = (
        annual["national_total_childcare_hours_total"] * annual["gross_market_hourly_price"]
    )
    annual["non_direct_active_value"] = (
        annual["national_active_childcare_hours_total"] * annual["non_direct_care_hourly_price"]
    )
    annual["non_direct_total_value"] = (
        annual["national_total_childcare_hours_total"] * annual["non_direct_care_hourly_price"]
    )
    annual["specialist_active_value"] = (
        annual["national_active_childcare_hours_total"] * annual["specialist_childcare_wage"]
    )
    annual["specialist_total_value"] = (
        annual["national_total_childcare_hours_total"] * annual["specialist_childcare_wage"]
    )
    annual["gross_market_nationalized_value"] = annual["gross_market_total_value"]
    annual["direct_care_nationalized_value"] = annual["direct_care_total_value"]
    annual["non_direct_nationalized_value"] = annual["non_direct_total_value"]
    annual["specialist_nationalized_value"] = annual["specialist_total_value"]
    annual["preferred_value"] = annual["direct_care_total_value"]

    table_columns = [
        "year",
        "atus_sensitivity_year",
        "price_support_population",
        "national_under5_population",
        "price_support_population_share",
        "under5_child_equivalent_quantity",
        "under5_average_unpaid_childcare_hours_per_child_year",
        "unpaid_active_childcare_hours_per_respondent_year",
        "unpaid_supervisory_childcare_hours_per_respondent_year",
        "national_active_household_childcare_hours_total",
        "national_active_nonhousehold_childcare_hours_total",
        "national_active_childcare_hours_total",
        "national_supervisory_childcare_hours_total",
        "national_total_childcare_hours_total",
        "gross_market_hourly_price",
        "direct_care_hourly_price",
        "non_direct_care_hourly_price",
        "benchmark_childcare_wage",
        "specialist_childcare_wage",
        "under5_gross_market_value",
        "under5_direct_care_value",
        "under5_non_direct_value",
        "under5_specialist_value",
        "gross_market_total_value",
        "direct_care_total_value",
        "non_direct_total_value",
        "specialist_total_value",
    ]
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    annual.loc[:, table_columns].to_csv(output_table_path, index=False)

    latest = annual.iloc[-1].to_dict() if not annual.empty else {}
    headline_mask = annual["year"].between(headline_year_start, headline_year_end)
    if exclude_sensitivity and "atus_sensitivity_year" in annual.columns:
        headline_mask = headline_mask & ~annual["atus_sensitivity_year"].fillna(False).astype(bool)
    headline_frame = annual.loc[headline_mask].copy()
    if headline_frame.empty:
        headline_frame = annual.copy()
    headline_summary = _window_summary(
        headline_frame,
        preferred_col="preferred_value",
        gross_col="gross_market_total_value",
        quantity_col="national_active_childcare_hours_total",
        quantity_key="active_hours_mean",
        year_start=headline_year_start,
        year_end=headline_year_end,
        exclude_sensitivity=exclude_sensitivity,
    )
    if headline_summary:
        headline_summary["supervisory_hours_mean"] = float(headline_frame["national_supervisory_childcare_hours_total"].mean())
    child_equivalent_headline_summary = _window_summary(
        headline_frame,
        preferred_col="under5_direct_care_value",
        gross_col="under5_gross_market_value",
        quantity_col="under5_child_equivalent_quantity",
        quantity_key="quantity_slots_mean",
        year_start=headline_year_start,
        year_end=headline_year_end,
        exclude_sensitivity=exclude_sensitivity,
    )
    if child_equivalent_headline_summary:
        child_equivalent_headline_summary["average_unpaid_childcare_hours_per_child_year_mean"] = float(
            headline_frame["under5_average_unpaid_childcare_hours_per_child_year"].mean()
        )
    child_equivalent_latest = {}
    if latest:
        child_equivalent_latest = {
            "year": int(latest["year"]),
            "preferred_value": float(latest["under5_direct_care_value"]),
            "gross_market_value": float(latest["under5_gross_market_value"]),
            "non_direct_value": float(latest["under5_non_direct_value"]),
            "specialist_value": float(latest["under5_specialist_value"]),
            "quantity_slots": float(latest["under5_child_equivalent_quantity"]),
            "direct_care_price": float(latest["direct_care_price"]),
            "gross_market_price": float(latest["gross_market_price"]),
            "average_unpaid_childcare_hours_per_child_year": float(
                latest["under5_average_unpaid_childcare_hours_per_child_year"]
            ),
            "price_support_population_share": float(latest["price_support_population_share"]),
        }
    annual_hours_latest = {}
    if latest:
        annual_hours_latest = {
            "year": int(latest["year"]),
            "preferred_value": float(latest["preferred_value"]),
            "gross_market_value": float(latest["gross_market_total_value"]),
            "non_direct_value": float(latest["non_direct_total_value"]),
            "specialist_value": float(latest["specialist_total_value"]),
            "active_household_hours_total": float(latest["national_active_household_childcare_hours_total"]),
            "active_nonhousehold_hours_total": float(latest["national_active_nonhousehold_childcare_hours_total"]),
            "active_hours_total": float(latest["national_active_childcare_hours_total"]),
            "supervisory_hours_total": float(latest["national_supervisory_childcare_hours_total"]),
            "total_hours_total": float(latest["national_total_childcare_hours_total"]),
            "direct_care_hourly_price": float(latest["direct_care_hourly_price"]),
            "gross_market_hourly_price": float(latest["gross_market_hourly_price"]),
            "price_support_population_share": float(latest["price_support_population_share"]),
        }
    summary = {
        "series_name": "childcare_static_account",
        "preferred_series": "direct_care_total_value",
        "valuation_identity": "hourly replacement price x unpaid annual childcare hours",
        "quantity_identity": "total_childcare_hours = active_household_hours + active_nonhousehold_hours + supervisory_hours",
        "scope": "national_year",
        "support_years": annual["year"].astype(int).tolist(),
        "market_hours_per_child_per_week": market_hours,
        "marketization_bridge_note": "The marketization solver remains a separate active-under-5 lower-bound bridge and is not this static-account quantity object.",
        "annual": annual.to_dict(orient="records"),
        "headline_window": headline_summary,
        "latest_year": latest,
        "benchmark_methodologies": {
            "under5_child_equivalent_replacement_cost": {
                "label": "Under-5 child-equivalent replacement-cost benchmark",
                "preferred_series": "under5_direct_care_value",
                "valuation_identity": "annual replacement price x under5 child-equivalent quantity",
                "quantity_identity": "under5_child_equivalent_quantity = national_under5_population x average_unpaid_active_childcare_hours_per_child_year / (52 x market_hours_per_child_per_week)",
                "scope": "national_year",
                "headline_window": child_equivalent_headline_summary,
                "latest_year": child_equivalent_latest,
                "notes": [
                    "This is the narrower under-5 bridge benchmark closest to the solver quantity object.",
                    "It values active childcare only; supervisory hours remain outside this benchmark.",
                    "The preferred direct-care series removes the pooled non-direct-care residual, but does not separately identify advertising, transport, administration, profits, or market power.",
                ],
            },
            "annual_hours_childcare_account": {
                "label": "Annual-hours childcare account",
                "preferred_series": "direct_care_total_value",
                "valuation_identity": "hourly replacement price x unpaid annual childcare hours",
                "quantity_identity": "total_childcare_hours = active_household_hours + active_nonhousehold_hours + supervisory_hours",
                "scope": "national_year",
                "headline_window": headline_summary,
                "latest_year": annual_hours_latest,
                "notes": [
                    "This is the broader household-production account in annual hours.",
                    "The preferred direct-care series removes the pooled non-direct-care residual from market prices, but remains a decomposition rather than literal cost accounting.",
                    "Supervisory care is currently additive and not yet overlap-adjusted, so this should not yet be read as a full Suh-Folbre-style replication.",
                ],
            },
        },
        "notes": [
            "This is the pooled national static childcare benchmark, not a general-equilibrium outsourcing forecast.",
            "The preferred benchmark values unpaid childcare hours, including supervisory care, at the direct-care-equivalent hourly price.",
            "The gross-market hourly benchmark is an upper valuation layer that retains non-direct-care residual components.",
            "The specialist valuation layer uses the OEWS preschool-teacher wage as a simple public-data specialist proxy, falling back to the childcare-worker wage when needed.",
            "The marketization solver remains a separate active-under-5 bridge; its slot proxies are not the static-account quantity object.",
            "Travel remains excluded in this first repair wave, so the static account should still be read as conservative relative to broader child-care accounts.",
            "ATUS 2020 remains a sensitivity year and is flagged, not dropped, in this descriptive static-account output.",
        ],
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Childcare Benchmarks",
        "",
        "This artifact publishes two descriptive childcare benchmarks side by side:",
        "",
        "- **Under-5 child-equivalent replacement-cost benchmark**",
        "  `value = annual replacement price x under5 child-equivalent quantity`",
        "- **Annual-hours childcare account**",
        "  `value = hourly replacement price x unpaid annual childcare hours`",
        "",
        "Both preferred series use the **direct-care-equivalent** layer, which removes a pooled non-direct-care residual but does not separately identify markup, advertising, transport, or administrative overhead.",
        "",
        f"- Support years: `{summary['support_years'][0]}-{summary['support_years'][-1]}`" if summary["support_years"] else "- Support years: n/a",
        f"- Marketization bridge: `{summary['marketization_bridge_note']}`",
    ]
    if child_equivalent_headline_summary:
        lines.extend(
            [
                f"- Under-5 child-equivalent headline window: `{child_equivalent_headline_summary['window_label']}`",
                f"- Under-5 child-equivalent preferred mean: `{child_equivalent_headline_summary['preferred_value_mean']:.2f}`",
                f"- Under-5 child-equivalent quantity mean: `{child_equivalent_headline_summary['quantity_slots_mean']:.2f}`",
                f"- Under-5 average unpaid childcare hours per child-year mean: `{child_equivalent_headline_summary['average_unpaid_childcare_hours_per_child_year_mean']:.2f}`",
            ]
        )
    if headline_summary:
        lines.extend(
            [
                f"- Annual-hours headline window: `{headline_summary['window_label']}`",
                f"- Annual-hours preferred benchmark mean: `{headline_summary['preferred_value_mean']:.2f}`",
                f"- Annual-hours active hours mean: `{headline_summary['active_hours_mean']:.2f}`",
                f"- Annual-hours supervisory hours mean: `{headline_summary['supervisory_hours_mean']:.2f}`",
            ]
        )
    if child_equivalent_latest:
        lines.extend(
            [
                f"- Latest under-5 child-equivalent year: `{child_equivalent_latest['year']}`",
                f"- Latest under-5 child-equivalent preferred benchmark: `{child_equivalent_latest['preferred_value']:.2f}`",
                f"- Latest under-5 child-equivalent gross-market upper benchmark: `{child_equivalent_latest['gross_market_value']:.2f}`",
                f"- Latest under-5 child-equivalent quantity: `{child_equivalent_latest['quantity_slots']:.2f}`",
                f"- Latest under-5 average unpaid childcare hours per child-year: `{child_equivalent_latest['average_unpaid_childcare_hours_per_child_year']:.2f}`",
            ]
        )
    if annual_hours_latest:
        lines.extend(
            [
                f"- Latest annual-hours year: `{annual_hours_latest['year']}`",
                f"- Latest annual-hours preferred benchmark: `{annual_hours_latest['preferred_value']:.2f}`",
                f"- Latest annual-hours gross-market upper benchmark: `{annual_hours_latest['gross_market_value']:.2f}`",
                f"- Latest annual-hours specialist-wage benchmark: `{annual_hours_latest['specialist_value']:.2f}`",
                f"- Latest annual-hours excluded non-direct residual: `{annual_hours_latest['non_direct_value']:.2f}`",
                f"- Latest annual-hours price-support population share: `{annual_hours_latest['price_support_population_share']:.1%}`",
                f"- Latest annual-hours active household hours: `{annual_hours_latest['active_household_hours_total']:.2f}`",
                f"- Latest annual-hours active nonhousehold hours: `{annual_hours_latest['active_nonhousehold_hours_total']:.2f}`",
                f"- Latest annual-hours supervisory hours: `{annual_hours_latest['supervisory_hours_total']:.2f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Read both measures as accounting benchmarks, not as predictions of what would happen if all unpaid care were marketized.",
            "- The under-5 child-equivalent benchmark is the narrower bridge object; the annual-hours account is much broader because it includes supervisory care.",
            "- The preferred direct-care layer strips out a pooled non-direct residual, but it is not a full home-production purification and does not separately identify markup, advertising, transport, or administrative overhead.",
            "- The specialist layer remains a transparent public-data proxy, not yet a full education-adjusted Suh-Folbre replication.",
            "",
            "## Annual series preview",
            _markdown_table(annual.loc[:, table_columns].round(2)),
        ]
    )
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return output_json_path, output_markdown_path, output_table_path


def build_childcare_headline_summary(
    demand_iv_path: Path,
    scenario_diagnostics_path: Path,
    price_decomposition_path: Path,
    output_json_path: Path,
    output_markdown_path: Path,
    price_decomposition_sensitivity_path: Path | None = None,
) -> tuple[Path, Path]:
    demand_iv = read_json(demand_iv_path)
    scenario_diagnostics = read_json(scenario_diagnostics_path)
    decomposition = read_json(price_decomposition_path)
    sensitivity = (
        read_json(price_decomposition_sensitivity_path)
        if price_decomposition_sensitivity_path is not None and price_decomposition_sensitivity_path.exists()
        else None
    )
    canonical = decomposition.get("canonical", {})
    alpha_half = canonical.get("alphas", {}).get("0.50", {})
    sensitivity_cases = sensitivity.get("cases", []) if sensitivity is not None else []
    direct_care_values = [
        float(case.get("baseline_direct_care_price_p50"))
        for case in sensitivity_cases
        if case.get("baseline_direct_care_price_p50") is not None
    ]
    wage_values = [
        float(case.get("baseline_implied_wage_p50"))
        for case in sensitivity_cases
        if case.get("baseline_implied_wage_p50") is not None
    ]
    summary = {
        "current_mode": scenario_diagnostics.get("current_mode", demand_iv.get("current_mode", "unknown")),
        "headline_sample": scenario_diagnostics.get("demand_sample_name"),
        "demand_instrument": scenario_diagnostics.get("demand_instrument", demand_iv.get("instrument")),
        "demand_specification_profile": scenario_diagnostics.get("demand_specification_profile"),
        "support_years": canonical.get("years", []),
        "scenario_rows": scenario_diagnostics.get("scenario_rows"),
        "scenario_states": scenario_diagnostics.get("scenario_states"),
        "gross_market_price_p50": canonical.get("baseline_price_p50"),
        "direct_care_equivalent_price_p50": canonical.get("baseline_direct_care_price_p50"),
        "non_direct_care_residual_p50": canonical.get("baseline_non_direct_care_price_p50"),
        "implied_direct_care_wage_p50": canonical.get("baseline_implied_wage_p50"),
        "direct_care_labor_share_p50": canonical.get("baseline_direct_care_labor_share_p50"),
        "direct_care_clip_binding_row_share": canonical.get("baseline_direct_care_clip_binding_row_share"),
        "alpha_50_gross_marketization_price_p50": alpha_half.get("price_p50"),
        "alpha_50_direct_care_price_p50": alpha_half.get("direct_care_price_p50"),
        "alpha_50_non_direct_care_residual_p50": alpha_half.get("non_direct_care_price_p50"),
        "alpha_50_implied_wage_p50": alpha_half.get("implied_wage_p50"),
        "demand_elasticity_at_mean": scenario_diagnostics.get("demand_elasticity_at_mean"),
        "demand_first_stage_r2": scenario_diagnostics.get("demand_first_stage_r2"),
        "demand_loo_state_fips_r2": scenario_diagnostics.get("demand_loo_state_fips_r2"),
        "demand_loo_year_r2": scenario_diagnostics.get("demand_loo_year_r2"),
        "bootstrap_draws_requested": scenario_diagnostics.get("bootstrap_draws_requested"),
        "bootstrap_draws_accepted": scenario_diagnostics.get("bootstrap_draws_accepted"),
        "bootstrap_draws_rejected": scenario_diagnostics.get("bootstrap_draws_rejected"),
        "bootstrap_acceptance_rate": scenario_diagnostics.get("bootstrap_acceptance_rate"),
        "bootstrap_failed": scenario_diagnostics.get("bootstrap_failed"),
        "supply_elasticity": scenario_diagnostics.get("supply_elasticity"),
        "supply_estimation_method": scenario_diagnostics.get("supply_estimation_method"),
        "supply_positive_only_weighted_median_slope": scenario_diagnostics.get(
            "supply_within_state_year_weighted_median_positive_slope"
        ),
        "supply_all_slopes_weighted_median_slope": scenario_diagnostics.get(
            "supply_within_state_year_weighted_median_all_slope"
        ),
        "supply_positivity_selection_gap": scenario_diagnostics.get(
            "supply_within_state_year_weighted_median_gap"
        ),
        "direct_care_price_sensitivity_min": min(direct_care_values) if direct_care_values else None,
        "direct_care_price_sensitivity_max": max(direct_care_values) if direct_care_values else None,
        "implied_wage_sensitivity_min": min(wage_values) if wage_values else None,
        "implied_wage_sensitivity_max": max(wage_values) if wage_values else None,
        "notes": [
            "The primary static benchmark is now the pooled national childcare static account; solver quantities are a separate under-5 lower-bound bridge.",
            "Gross market prices remain the canonical solver output.",
            "Direct-care-equivalent prices are a decomposition layer based on observed childcare wages plus staffing and fringe assumptions.",
            "The decomposition is capped at the gross market price; the clip-binding share shows how often the raw labor-equivalent price would otherwise exceed gross price.",
            "Current canonical scenarios stay within observed NDCP support through 2022.",
        ],
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Childcare Headline Readout",
        "",
        f"- Current mode: `{summary['current_mode']}`",
        f"- Headline sample: `{summary['headline_sample']}`",
        f"- Demand instrument: `{summary['demand_instrument']}`",
        f"- Canonical first-stage specification: `{summary['demand_specification_profile']}`",
        f"- Supported years: `{summary['support_years'][0]}-{summary['support_years'][-1]}`" if summary["support_years"] else "- Supported years: n/a",
        f"- Canonical scenario rows: `{summary['scenario_rows']}` across `{summary['scenario_states']}` states",
        f"- Median observed gross market price: `{summary['gross_market_price_p50']:.2f}`" if summary["gross_market_price_p50"] is not None else "- Median observed gross market price: n/a",
        f"- Median direct-care-equivalent price: `{summary['direct_care_equivalent_price_p50']:.2f}`" if summary["direct_care_equivalent_price_p50"] is not None else "- Median direct-care-equivalent price: n/a",
        f"- Median non-direct-care residual: `{summary['non_direct_care_residual_p50']:.2f}`" if summary["non_direct_care_residual_p50"] is not None else "- Median non-direct-care residual: n/a",
        f"- Median implied direct-care hourly wage: `{summary['implied_direct_care_wage_p50']:.2f}`" if summary["implied_direct_care_wage_p50"] is not None else "- Median implied direct-care hourly wage: n/a",
        f"- Median direct-care labor share: `{summary['direct_care_labor_share_p50']:.1%}`" if summary["direct_care_labor_share_p50"] is not None else "- Median direct-care labor share: n/a",
        f"- Direct-care clip-binding row share: `{summary['direct_care_clip_binding_row_share']:.1%}`" if summary["direct_care_clip_binding_row_share"] is not None else "- Direct-care clip-binding row share: n/a",
        f"- Median `alpha=0.50` gross marketization price: `{summary['alpha_50_gross_marketization_price_p50']:.2f}`" if summary["alpha_50_gross_marketization_price_p50"] is not None else "- Median `alpha=0.50` gross marketization price: n/a",
        f"- Median `alpha=0.50` direct-care-equivalent price: `{summary['alpha_50_direct_care_price_p50']:.2f}`" if summary["alpha_50_direct_care_price_p50"] is not None else "- Median `alpha=0.50` direct-care-equivalent price: n/a",
        f"- Median `alpha=0.50` implied wage: `{summary['alpha_50_implied_wage_p50']:.2f}`" if summary["alpha_50_implied_wage_p50"] is not None else "- Median `alpha=0.50` implied wage: n/a",
        f"- Demand elasticity at mean: `{summary['demand_elasticity_at_mean']:.4f}`" if summary["demand_elasticity_at_mean"] is not None else "- Demand elasticity at mean: n/a",
        f"- First-stage `R^2`: `{summary['demand_first_stage_r2']:.4f}`" if summary["demand_first_stage_r2"] is not None else "- First-stage `R^2`: n/a",
        f"- Leave-one-state-out `R^2`: `{summary['demand_loo_state_fips_r2']}`",
        f"- Leave-one-year-out `R^2`: `{summary['demand_loo_year_r2']}`",
        f"- Bootstrap draws requested/accepted/rejected: `{summary['bootstrap_draws_requested']}` / `{summary['bootstrap_draws_accepted']}` / `{summary['bootstrap_draws_rejected']}`",
        f"- Bootstrap acceptance rate: `{summary['bootstrap_acceptance_rate']:.1%}`" if summary["bootstrap_acceptance_rate"] is not None else "- Bootstrap acceptance rate: n/a",
        f"- Bootstrap failed: `{summary['bootstrap_failed']}`",
        f"- Supply elasticity: `{summary['supply_elasticity']:.4f}` via `{summary['supply_estimation_method']}`" if summary["supply_elasticity"] is not None else "- Supply elasticity: n/a",
        f"- Supply positive-only weighted median slope: `{summary['supply_positive_only_weighted_median_slope']:.4f}`" if summary["supply_positive_only_weighted_median_slope"] is not None else "- Supply positive-only weighted median slope: n/a",
        f"- Supply all-slopes weighted median slope: `{summary['supply_all_slopes_weighted_median_slope']:.4f}`" if summary["supply_all_slopes_weighted_median_slope"] is not None else "- Supply all-slopes weighted median slope: n/a",
        f"- Supply positivity-selection gap: `{summary['supply_positivity_selection_gap']:.4f}`" if summary["supply_positivity_selection_gap"] is not None else "- Supply positivity-selection gap: n/a",
    ]
    if direct_care_values and wage_values:
        lines.extend(
            [
                f"- Direct-care price sensitivity range: `{min(direct_care_values):.2f}` to `{max(direct_care_values):.2f}`",
                f"- Implied wage sensitivity range: `{min(wage_values):.2f}` to `{max(wage_values):.2f}`",
            ]
        )
    lines.extend(
        [
            "",
            "Gross market prices remain canonical. Direct-care-equivalent prices are a decomposition layer rather than a replacement canonical series.",
        ]
    )
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return output_json_path, output_markdown_path
