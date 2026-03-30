from __future__ import annotations

import pandas as pd

from unpriced.assumptions import childcare_model_assumptions
from unpriced.clean.atus import build_state_year_panel as build_atus_panel
from unpriced.clean.labor import build_county_labor_panel
from unpriced.clean.ndcp import build_county_year_panel as build_ndcp_panel
from unpriced.config import ProjectPaths
from unpriced.features.demand import aggregate_counties_to_state
from unpriced.features.supply import add_provider_density
from unpriced.features.wages import add_benchmark_wages
from unpriced.storage import read_parquet, write_json, write_parquet

OBSERVED_CORE_START_YEAR = 2014
OBSERVED_CORE_END_YEAR = 2022
STATE_PRICE_STATUS_OBSERVED = "observed_ndcp_support"
STATE_PRICE_STATUS_PRE_SUPPORT = "pre_ndcp_support_gap"
STATE_PRICE_STATUS_POST_SUPPORT = "post_ndcp_nowcast"
CANONICAL_ACTIVE_DEMAND_COLUMNS = [
    "unpaid_childcare_hours",
    "unpaid_active_childcare_hours",
    "state_price_index",
    "outside_option_wage",
    "parent_employment_rate",
    "single_parent_share",
]


def _load_ndcp_staffing_mix(paths: ProjectPaths, assumptions: dict[str, object]) -> pd.DataFrame:
    ndcp_detail_path = paths.interim / "ndcp" / "ndcp.parquet"
    if not ndcp_detail_path.exists():
        return pd.DataFrame(
            columns=[
                "county_fips",
                "state_fips",
                "year",
                "effective_children_per_worker",
                "worker_per_child_equivalent",
            ]
        )

    detail = read_parquet(ndcp_detail_path)
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "county_fips",
                "state_fips",
                "year",
                "effective_children_per_worker",
                "worker_per_child_equivalent",
            ]
        )
    detail = detail.copy()
    detail["sample_weight"] = pd.to_numeric(detail["sample_weight"], errors="coerce").fillna(1.0)
    staffing_children_per_worker = assumptions["staffing_children_per_worker"]
    default_children_per_worker = float(assumptions["default_children_per_worker"])
    detail["staffing_children_per_worker"] = [
        float(staffing_children_per_worker.get((provider_type, child_age), default_children_per_worker))
        for provider_type, child_age in zip(detail["provider_type"], detail["child_age"], strict=False)
    ]
    detail["worker_per_child_equivalent"] = 1.0 / detail["staffing_children_per_worker"]
    grouped = (
        detail.groupby(["county_fips", "state_fips", "year"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "worker_per_child_equivalent": float(
                        (
                            pd.to_numeric(g["worker_per_child_equivalent"], errors="coerce")
                            * pd.to_numeric(g["sample_weight"], errors="coerce").fillna(0.0)
                        ).sum()
                        / max(pd.to_numeric(g["sample_weight"], errors="coerce").fillna(0.0).sum(), 1e-9)
                    )
                }
            )
        )
        .reset_index(drop=True)
    )
    grouped["effective_children_per_worker"] = 1.0 / grouped["worker_per_child_equivalent"].clip(lower=1e-9)
    return grouped


def _add_direct_care_price_equivalents(county: pd.DataFrame, assumptions: dict[str, object]) -> pd.DataFrame:
    enriched = county.copy()
    hours_per_year = float(assumptions["direct_care_hours_per_year"])
    fringe_multiplier = float(assumptions["direct_care_fringe_multiplier"])
    default_children_per_worker = float(assumptions["default_children_per_worker"])
    enriched["effective_children_per_worker"] = pd.to_numeric(
        enriched.get("effective_children_per_worker"), errors="coerce"
    ).fillna(default_children_per_worker)
    enriched["worker_per_child_equivalent"] = (
        1.0 / enriched["effective_children_per_worker"].clip(lower=1e-9)
    )
    annual_price = pd.to_numeric(enriched["annual_price"], errors="coerce")
    wage = pd.to_numeric(enriched["childcare_worker_wage"], errors="coerce")
    raw_direct_care_price = (
        wage * hours_per_year * fringe_multiplier
        / enriched["effective_children_per_worker"].clip(lower=1e-9)
    )
    enriched["direct_care_price_index_raw"] = raw_direct_care_price
    enriched["direct_care_price_index"] = pd.concat([annual_price, raw_direct_care_price], axis=1).min(axis=1)
    enriched["direct_care_price_index"] = enriched["direct_care_price_index"].clip(lower=0.0)
    enriched["direct_care_price_clip_binding"] = raw_direct_care_price.gt(annual_price).fillna(False)
    enriched["non_direct_care_price_index"] = (annual_price - enriched["direct_care_price_index"]).clip(lower=0.0)
    enriched["direct_care_labor_share"] = enriched["direct_care_price_index"].div(
        annual_price.replace({0: pd.NA})
    ).clip(lower=0.0, upper=1.0)
    enriched["non_direct_care_price_share"] = 1.0 - enriched["direct_care_labor_share"].fillna(0.0)
    enriched["implied_direct_care_hourly_wage"] = (
        enriched["direct_care_price_index"]
        * enriched["effective_children_per_worker"]
        / (hours_per_year * fringe_multiplier)
    )
    return enriched


def _state_sample_support_metadata(county: pd.DataFrame, assumptions: dict[str, object]) -> pd.DataFrame:
    support = county.copy()
    if "childcare_worker_wage_source" in support.columns:
        support["county_qcew_wage_observed"] = support["childcare_worker_wage_source"].eq(
            "qcew_county_observed"
        ).astype(float)
    else:
        hours_per_year = float(assumptions["direct_care_hours_per_year"])
        support["county_qcew_wage_observed"] = (
            (
                pd.to_numeric(support["childcare_worker_wage"], errors="coerce")
                - pd.to_numeric(support["annual_price"], errors="coerce") / hours_per_year
            )
            .abs()
            .ge(0.01)
            .astype(float)
        )
    if "employment_source" in support.columns:
        support["county_qcew_employment_observed"] = support["employment_source"].eq(
            "qcew_employment_observed"
        ).astype(float)
    else:
        support["county_qcew_employment_observed"] = 0.0
    support["county_qcew_labor_observed"] = support[
        ["county_qcew_wage_observed", "county_qcew_employment_observed"]
    ].min(axis=1)
    state_meta = aggregate_counties_to_state(
        support,
        [
            "annual_price",
            "provider_density",
            "benchmark_childcare_wage",
            "specialist_childcare_wage",
            "outside_option_wage",
            "imputed_share",
            "direct_care_price_index",
            "direct_care_price_index_raw",
            "non_direct_care_price_index",
            "direct_care_labor_share",
            "direct_care_price_clip_binding",
            "worker_per_child_equivalent",
            "implied_direct_care_hourly_wage",
            "county_qcew_wage_observed",
            "county_qcew_employment_observed",
            "county_qcew_labor_observed",
        ],
    ).rename(
        columns={
            "annual_price": "state_price_index",
            "imputed_share": "state_ndcp_imputed_share",
            "direct_care_price_index": "state_direct_care_price_index",
            "direct_care_price_index_raw": "state_direct_care_price_index_raw",
            "non_direct_care_price_index": "state_non_direct_care_price_index",
            "specialist_childcare_wage": "state_specialist_childcare_wage",
            "direct_care_labor_share": "state_direct_care_labor_share",
            "direct_care_price_clip_binding": "state_direct_care_price_clip_binding_share",
            "implied_direct_care_hourly_wage": "state_implied_direct_care_hourly_wage",
            "county_qcew_wage_observed": "state_qcew_wage_observed_share",
            "county_qcew_employment_observed": "state_qcew_employment_observed_share",
            "county_qcew_labor_observed": "state_qcew_labor_observed_share",
        }
    )
    state_meta["state_effective_children_per_worker"] = 1.0 / pd.to_numeric(
        state_meta["worker_per_child_equivalent"], errors="coerce"
    ).clip(lower=1e-9)
    state_meta["state_direct_care_price_clip_binding"] = pd.to_numeric(
        state_meta["state_direct_care_price_clip_binding_share"], errors="coerce"
    ).gt(0)
    state_meta = state_meta.drop(columns=["worker_per_child_equivalent"])
    return state_meta


def _assign_exclusion_reason(
    state: pd.DataFrame,
    low_impute: bool = False,
    low_impute_threshold: float = 0.25,
) -> pd.Series:
    reasons = pd.Series(index=state.index, dtype="object")
    year = pd.to_numeric(state["year"], errors="coerce")
    reasons.loc[~year.between(OBSERVED_CORE_START_YEAR, OBSERVED_CORE_END_YEAR)] = "outside_year_window"
    reasons.loc[reasons.isna() & state["is_sensitivity_year"].fillna(False).astype(bool)] = "sensitivity_year"
    reasons.loc[
        reasons.isna()
        & (
            pd.to_numeric(state["state_price_index"], errors="coerce").isna()
            | ~pd.to_numeric(state["state_price_index"], errors="coerce").gt(0)
        )
    ] = "missing_state_price"
    reasons.loc[
        reasons.isna() & ~state["births_value_source"].fillna("").eq("cdc_wonder_observed")
    ] = "births_not_cdc_observed"
    reasons.loc[
        reasons.isna() & ~state["state_controls_source"].fillna("").eq("acs_observed")
    ] = "state_controls_not_observed"
    reasons.loc[
        reasons.isna() & ~state["state_unemployment_source"].fillna("").eq("laus_observed")
    ] = "state_unemployment_not_observed"
    if low_impute:
        reasons.loc[
            reasons.isna()
            & (
                pd.to_numeric(state["state_ndcp_imputed_share"], errors="coerce").isna()
                | pd.to_numeric(state["state_ndcp_imputed_share"], errors="coerce").gt(low_impute_threshold)
            )
        ] = "imputation_share_above_threshold"
    return reasons.fillna("eligible")


def _assign_observed_core_exclusion_reason(
    state: pd.DataFrame,
    required_columns: list[str],
) -> pd.Series:
    reasons = pd.Series(index=state.index, dtype="object")
    year = pd.to_numeric(state["year"], errors="coerce")
    in_support = (
        state["state_price_observation_status"].eq(STATE_PRICE_STATUS_OBSERVED)
        if "state_price_observation_status" in state.columns
        else year.between(OBSERVED_CORE_START_YEAR, OBSERVED_CORE_END_YEAR)
    )
    if "state_price_support_window" in state.columns:
        in_support = state["state_price_support_window"].eq("in_support")
    in_year_window = year.between(OBSERVED_CORE_START_YEAR, OBSERVED_CORE_END_YEAR)
    reasons.loc[~in_year_window] = "outside_year_window"
    sensitivity = (
        state["is_sensitivity_year"].fillna(False).astype(bool)
        if "is_sensitivity_year" in state.columns
        else year.eq(2020)
    )
    reasons.loc[reasons.isna() & sensitivity] = "sensitivity_year"
    reasons.loc[
        reasons.isna()
        & (
            pd.to_numeric(state["state_price_index"], errors="coerce").isna()
            | ~pd.to_numeric(state["state_price_index"], errors="coerce").gt(0)
        )
    ] = "missing_state_price"
    missing_required = pd.DataFrame(
        {
            col: pd.to_numeric(state[col], errors="coerce").notna() if col != "outside_option_wage" else state[col].notna()
            for col in required_columns
        }
    )
    reasons.loc[reasons.isna() & ~missing_required.all(axis=1)] = "missing_demand_specification_fields"
    reasons.loc[reasons.isna() & ~in_support] = "outside_price_support"
    return reasons.fillna("eligible")


def _annotate_state_sample_ladder(
    state: pd.DataFrame,
    county: pd.DataFrame,
    low_impute_threshold: float,
) -> pd.DataFrame:
    result = state.copy()
    county_year_min = int(pd.to_numeric(county["year"], errors="coerce").min())
    county_year_max = int(pd.to_numeric(county["year"], errors="coerce").max())
    result["state_price_support_window"] = "out_of_support"
    result["state_price_observation_status"] = STATE_PRICE_STATUS_PRE_SUPPORT
    year = pd.to_numeric(result["year"], errors="coerce")
    result.loc[year.between(county_year_min, county_year_max), "state_price_support_window"] = "in_support"
    result.loc[year.between(county_year_min, county_year_max), "state_price_observation_status"] = (
        STATE_PRICE_STATUS_OBSERVED
    )
    result.loc[year.gt(county_year_max), "state_price_observation_status"] = STATE_PRICE_STATUS_POST_SUPPORT
    result["state_price_nowcast"] = result["state_price_observation_status"].eq(STATE_PRICE_STATUS_POST_SUPPORT)

    broad_complete_mask = result[
        [
            "unpaid_childcare_hours",
            "state_price_index",
            "outside_option_wage",
            "parent_employment_rate",
            "single_parent_share",
            "median_income",
            "unemployment_rate",
        ]
    ].notna().all(axis=1) & pd.to_numeric(result["state_price_index"], errors="coerce").gt(0)
    result["eligible_broad_complete"] = broad_complete_mask

    observed_archival_reason = _assign_exclusion_reason(result, low_impute=False, low_impute_threshold=low_impute_threshold)
    result["observed_archival_exclusion_reason"] = observed_archival_reason
    result["eligible_observed_archival"] = observed_archival_reason.eq("eligible")

    low_impute_reason = _assign_exclusion_reason(
        result,
        low_impute=True,
        low_impute_threshold=low_impute_threshold,
    )
    result["observed_archival_low_impute_exclusion_reason"] = low_impute_reason
    result["eligible_observed_archival_low_impute"] = low_impute_reason.eq("eligible")

    active_core_reason = _assign_observed_core_exclusion_reason(result, required_columns=CANONICAL_ACTIVE_DEMAND_COLUMNS)
    result["observed_core_exclusion_reason"] = active_core_reason
    result["eligible_observed_core"] = active_core_reason.eq("eligible")
    result["eligible_observed_core_low_impute"] = result["eligible_observed_core"].copy()
    result["observed_core_low_impute_exclusion_reason"] = active_core_reason.copy()
    if "state_ndcp_imputed_share" in result.columns:
        observed_core_low_impute = result["eligible_observed_core"] & (
            pd.to_numeric(result["state_ndcp_imputed_share"], errors="coerce").le(low_impute_threshold)
        )
        result.loc[:, "eligible_observed_core_low_impute"] = (
            observed_core_low_impute.fillna(False).astype(bool)
        )
        result.loc[
            result["eligible_observed_core"].fillna(False).astype(bool)
            & ~result["eligible_observed_core_low_impute"].fillna(False).astype(bool),
            "observed_core_low_impute_exclusion_reason",
        ] = "imputation_share_above_threshold"
    return result


def _add_head_start_capacity(county: pd.DataFrame, paths: ProjectPaths, assumptions: dict[str, object]) -> pd.DataFrame:
    head_start_path = paths.interim / "head_start" / "head_start.parquet"
    county["head_start_capacity_source"] = "missing"
    if not head_start_path.exists():
        county["head_start_slots"] = 0.0
        county["head_start_capacity"] = pd.NA
        county = _fill_from_group_medians(
            county,
            value_col="head_start_capacity",
            base_col="under5_population",
            source_col="head_start_capacity_source",
            exact_label="head_start_observed",
            derived_prefix="head_start_slot_share",
            group_levels=[([], "national")],
        )
        county["head_start_capacity"] = pd.to_numeric(county["head_start_capacity"], errors="coerce").fillna(0.0)
        county["head_start_slot_share"] = county["head_start_capacity"].div(
            county["under5_population"].replace({0: pd.NA})
        ).fillna(0.0)
        return county

    head_start = read_parquet(head_start_path)
    county = county.merge(head_start, on=["county_fips", "state_fips"], how="left")
    county["head_start_slots"] = pd.to_numeric(county["head_start_slots"], errors="coerce").fillna(0.0)
    county["head_start_capacity"] = county["head_start_slots"]
    county.loc[county["head_start_slots"].gt(0), "head_start_capacity_source"] = "head_start_observed"
    county["head_start_slot_share"] = county["head_start_slots"].div(
        county["under5_population"].replace({0: pd.NA})
    ).fillna(0.0)
    county = _fill_from_group_medians(
        county,
        value_col="head_start_capacity",
        base_col="under5_population",
        source_col="head_start_capacity_source",
        exact_label="head_start_observed",
        derived_prefix="head_start_slot_share",
        group_levels=[
            (["state_fips"], "state"),
            ([], "national"),
        ],
    )
    county["head_start_capacity"] = pd.to_numeric(county["head_start_capacity"], errors="coerce").fillna(0.0)
    county["head_start_slot_share"] = county["head_start_capacity"].div(
        county["under5_population"].replace({0: pd.NA})
    ).fillna(0.0)
    return county


def _add_public_school_options(county: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    ccd_path = paths.interim / "nces_ccd" / "nces_ccd.parquet"
    if not ccd_path.exists():
        county["public_school_option_index"] = 1.0 - county["single_parent_share"]
        return county

    ccd = read_parquet(ccd_path)
    keep = [
        "state_fips",
        "public_school_option_index",
        "prek_schools",
        "kg_schools",
        "pk_or_kg_schools",
        "operational_schools",
    ]
    county = county.merge(ccd[keep], on="state_fips", how="left")
    county["public_school_option_index"] = pd.to_numeric(
        county["public_school_option_index"], errors="coerce"
    ).fillna(1.0 - county["single_parent_share"])
    return county


def _fill_from_group_medians(
    frame: pd.DataFrame,
    value_col: str,
    base_col: str,
    source_col: str,
    exact_label: str,
    derived_prefix: str,
    group_levels: list[tuple[list[str], str]],
) -> pd.DataFrame:
    result = frame.copy()
    ratio_col = f"{value_col}_per_{base_col}"
    result[ratio_col] = pd.to_numeric(result[value_col], errors="coerce").div(
        pd.to_numeric(result[base_col], errors="coerce").replace({0: pd.NA})
    )
    result[source_col] = result[source_col].fillna("missing")
    missing = pd.to_numeric(result[value_col], errors="coerce").isna() & pd.to_numeric(result[base_col], errors="coerce").notna()

    for columns, label in group_levels:
        observed = result.loc[
            pd.to_numeric(result[ratio_col], errors="coerce").notna()
            & pd.to_numeric(result[ratio_col], errors="coerce").gt(0),
            columns + [ratio_col],
        ].copy()
        if observed.empty:
            continue
        fill_name = f"{ratio_col}_{label}"
        if columns:
            grouped = observed.groupby(columns, as_index=False)[ratio_col].median()
            grouped = grouped.rename(columns={ratio_col: fill_name})
            result = result.merge(grouped, on=columns, how="left")
            fill_values = pd.to_numeric(result[fill_name], errors="coerce")
        else:
            result[fill_name] = float(pd.to_numeric(observed[ratio_col], errors="coerce").median())
            fill_values = pd.to_numeric(result[fill_name], errors="coerce")
        use_mask = missing & fill_values.notna()
        result.loc[use_mask, value_col] = (
            pd.to_numeric(result.loc[use_mask, base_col], errors="coerce") * fill_values.loc[use_mask]
        )
        result.loc[use_mask, source_col] = f"{derived_prefix}_{label}"
        missing = pd.to_numeric(result[value_col], errors="coerce").isna() & pd.to_numeric(result[base_col], errors="coerce").notna()
        result = result.drop(columns=[fill_name])

    value_present = pd.to_numeric(result[value_col], errors="coerce").notna()
    result.loc[value_present, source_col] = result.loc[value_present, source_col].replace({"missing": exact_label})
    result = result.drop(columns=[ratio_col])
    return result


def _add_market_structure(county: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    cbp_path = paths.interim / "cbp" / "cbp.parquet"
    if cbp_path.exists():
        cbp = read_parquet(cbp_path)
        keep = [
            "county_fips",
            "state_fips",
            "year",
            "employer_establishments",
            "employer_employment",
            "employer_annual_payroll",
        ]
        county = county.merge(cbp[keep], on=["county_fips", "state_fips", "year"], how="left")

    nes_path = paths.interim / "nes" / "nes.parquet"
    if nes_path.exists():
        nes = read_parquet(nes_path)
        keep = [
            "county_fips",
            "state_fips",
            "year",
            "nonemployer_firms",
            "receipts",
        ]
        county = county.merge(nes[keep], on=["county_fips", "state_fips", "year"], how="left")
    return county


def _add_laus_controls(county: pd.DataFrame, state: pd.DataFrame, paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    laus_path = paths.interim / "laus" / "laus.parquet"
    if not laus_path.exists():
        return county, state

    laus = read_parquet(laus_path)
    county_laus = laus.loc[laus["geography"].eq("county")].copy()
    state_laus = laus.loc[laus["geography"].eq("state")].copy()

    county_keep = [
        "county_fips",
        "state_fips",
        "year",
        "laus_unemployment_rate",
        "laus_unemployed",
        "laus_employment",
        "laus_labor_force",
    ]
    state_keep = [
        "state_fips",
        "year",
        "laus_unemployment_rate",
        "laus_unemployed",
        "laus_employment",
        "laus_labor_force",
    ]
    county = county.merge(county_laus[county_keep], on=["county_fips", "state_fips", "year"], how="left")
    state = state.merge(state_laus[state_keep], on=["state_fips", "year"], how="left")
    county["unemployment_rate"] = pd.to_numeric(county["laus_unemployment_rate"], errors="coerce").fillna(
        county["unemployment_rate"]
    )
    state["unemployment_rate"] = pd.to_numeric(state["laus_unemployment_rate"], errors="coerce").fillna(
        state["unemployment_rate"]
    )
    return county, state


def _fill_county_covariates(county: pd.DataFrame, atus: pd.DataFrame, assumptions: dict[str, object]) -> pd.DataFrame:
    state_controls = atus[
        ["state_fips", "year", "single_parent_share", "median_income", "unemployment_rate"]
    ].drop_duplicates().rename(
        columns={
            "single_parent_share": "single_parent_share_state",
            "median_income": "median_income_state",
            "unemployment_rate": "unemployment_rate_state",
        }
    )
    county = county.merge(
        state_controls,
        on=["state_fips", "year"],
        how="left",
    )

    for base_name in ("single_parent_share", "median_income", "unemployment_rate"):
        state_name = f"{base_name}_state"
        if base_name not in county.columns:
            county[base_name] = county[state_name]
        else:
            county[base_name] = county[base_name].fillna(county[state_name])
    county = county.drop(
        columns=["single_parent_share_state", "median_income_state", "unemployment_rate_state"]
    )

    county["under5_population"] = county["under5_population"].fillna(1.0)
    if "outside_option_wage_source" not in county.columns:
        county["outside_option_wage_source"] = pd.NA
    county["outside_option_wage_source"] = county["outside_option_wage_source"].fillna("outside_option_observed")
    county["childcare_worker_wage"] = county["childcare_worker_wage"].fillna(
        county["annual_price"] / 52.0 / 40.0
    )
    county = _fill_from_group_medians(
        county,
        value_col="outside_option_wage",
        base_col="childcare_worker_wage",
        source_col="outside_option_wage_source",
        exact_label="outside_option_observed",
        derived_prefix="outside_option_ratio",
        group_levels=[
            (["state_fips", "year"], "state_year"),
            (["state_fips"], "state"),
            (["year"], "year"),
            ([], "national"),
        ],
    )
    county["employment_source"] = pd.to_numeric(county["employment"], errors="coerce").notna().map(
        lambda observed: "qcew_employment_observed" if observed else "missing"
    )
    county = _fill_from_group_medians(
        county,
        value_col="employment",
        base_col="under5_population",
        source_col="employment_source",
        exact_label="qcew_employment_observed",
        derived_prefix="employment_ratio",
        group_levels=[
            (["state_fips", "year"], "state_year"),
            (["state_fips"], "state"),
            (["year"], "year"),
            ([], "national"),
        ],
    )
    county["employment"] = pd.to_numeric(county["employment"], errors="coerce").clip(lower=1.0)
    return county


def build_childcare_panels(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    assumptions = childcare_model_assumptions(paths)
    ndcp = build_ndcp_panel(paths)
    labor = build_county_labor_panel(paths)
    acs = read_parquet(paths.interim / "acs" / "acs.parquet")
    atus = build_atus_panel(paths)
    staffing_mix = _load_ndcp_staffing_mix(paths, assumptions)

    county = ndcp.merge(acs, on=["county_fips", "state_fips", "year"], how="left")
    county = county.merge(labor, on=["county_fips", "state_fips", "year"], how="left")
    county = county.merge(staffing_mix, on=["county_fips", "state_fips", "year"], how="left")
    county = _fill_county_covariates(county, atus, assumptions)
    county = _add_direct_care_price_equivalents(county, assumptions)
    county = _add_market_structure(county, paths)
    county = _add_head_start_capacity(county, paths, assumptions)
    county = _add_public_school_options(county, paths)
    county = add_provider_density(add_benchmark_wages(county))
    county["observed_market_hours"] = county["under5_population"] * float(assumptions["market_hours_per_child_per_week"])
    write_parquet(county, paths.processed / "childcare_county_year_price_panel.parquet")

    state_meta = _state_sample_support_metadata(county, assumptions)
    state = atus.merge(state_meta, on=["state_fips", "year"], how="left")
    county, state = _add_laus_controls(county, state, paths)
    state["market_quantity_proxy"] = state["under5_population"] * float(assumptions["market_hours_per_child_per_week"])
    state["unpaid_quantity_proxy"] = state["unpaid_active_childcare_hours"] * state["under5_population"] / 52.0
    state["market_quantity_proxy_basis"] = "under5_population_x_market_hours_per_child_per_week"
    state["unpaid_quantity_proxy_basis"] = "active_childcare_population_scaled_bridge"
    state["childcare_bridge_estimand"] = "marketization_bridge"
    state = _annotate_state_sample_ladder(state, county, float(assumptions["low_impute_threshold"]))
    write_parquet(state, paths.processed / "childcare_state_year_panel.parquet")
    return county, state


def diagnose_childcare_pipeline(
    county: pd.DataFrame,
    state: pd.DataFrame,
    paths: ProjectPaths,
) -> dict[str, object]:
    """Quantify how much of the childcare pipeline relies on fallback or synthetic inputs.

    Returns a dict of diagnostic counts and writes it to
    ``outputs/reports/childcare_pipeline_diagnostics.json``.
    """
    diag: dict[str, object] = {}
    assumptions = childcare_model_assumptions(paths)
    diag["assumptions"] = {
        "direct_care_hours_per_year": float(assumptions["direct_care_hours_per_year"]),
        "direct_care_fringe_multiplier": float(assumptions["direct_care_fringe_multiplier"]),
        "market_hours_per_child_per_week": float(assumptions["market_hours_per_child_per_week"]),
        "low_impute_threshold": float(assumptions["low_impute_threshold"]),
    }

    # --- State-year panel ---
    n_state = len(state)
    diag["state_year_rows"] = n_state
    diag["state_year_unique_states"] = int(state["state_fips"].nunique())
    diag["state_year_min"] = int(state["year"].min())
    diag["state_year_max"] = int(state["year"].max())

    # Births provenance
    if "births_value_source" in state.columns:
        births_counts = state["births_value_source"].value_counts().to_dict()
        diag["births_cdc_wonder_observed"] = int(births_counts.get("cdc_wonder_observed", 0))
        diag["births_atus_reported"] = int(births_counts.get("atus_reported", 0))
        diag["births_cdc_suppressed_fallback"] = int(births_counts.get("cdc_wonder_suppressed_fallback", 0))
        diag["births_atus_unmatched_fallback"] = int(births_counts.get("atus_unmatched_state_fallback", 0))
    else:
        diag["births_cdc_wonder_observed"] = 0
        diag["births_atus_reported"] = n_state
        diag["births_cdc_suppressed_fallback"] = 0
        diag["births_atus_unmatched_fallback"] = 0

    # ATUS 2020 sensitivity year
    if "sensitivity_year" in state.columns:
        diag["atus_sensitivity_year_rows"] = int(state["sensitivity_year"].sum())
    else:
        diag["atus_sensitivity_year_rows"] = int((state["year"] == 2020).sum())
    if "state_price_observation_status" in state.columns:
        support_counts = state["state_price_observation_status"].fillna("missing").value_counts().to_dict()
        diag["state_price_observation_status_counts"] = {str(key): int(value) for key, value in support_counts.items()}
        diag["state_price_observed_support_rows"] = int(
            state["state_price_observation_status"].eq(STATE_PRICE_STATUS_OBSERVED).sum()
        )
        diag["state_price_pre_support_rows"] = int(
            state["state_price_observation_status"].eq(STATE_PRICE_STATUS_PRE_SUPPORT).sum()
        )
        diag["state_price_post_support_nowcast_rows"] = int(
            state["state_price_observation_status"].eq(STATE_PRICE_STATUS_POST_SUPPORT).sum()
        )
    if "state_price_nowcast" in state.columns:
        diag["state_price_nowcast_rows"] = int(state["state_price_nowcast"].fillna(False).astype(bool).sum())
    if "state_direct_care_price_index" in state.columns:
        diag["state_direct_care_price_index_na"] = int(pd.to_numeric(state["state_direct_care_price_index"], errors="coerce").isna().sum())
        diag["state_direct_care_price_p50"] = float(pd.to_numeric(state["state_direct_care_price_index"], errors="coerce").median())
    if "state_direct_care_price_index_raw" in state.columns:
        diag["state_direct_care_price_raw_p50"] = float(
            pd.to_numeric(state["state_direct_care_price_index_raw"], errors="coerce").median()
        )
    if "state_non_direct_care_price_index" in state.columns:
        diag["state_non_direct_care_price_p50"] = float(pd.to_numeric(state["state_non_direct_care_price_index"], errors="coerce").median())
    if "state_direct_care_labor_share" in state.columns:
        diag["state_direct_care_labor_share_mean"] = float(pd.to_numeric(state["state_direct_care_labor_share"], errors="coerce").mean())
        diag["state_direct_care_labor_share_p50"] = float(pd.to_numeric(state["state_direct_care_labor_share"], errors="coerce").median())
    if "state_direct_care_price_clip_binding_share" in state.columns:
        diag["state_direct_care_clip_binding_share_mean"] = float(
            pd.to_numeric(state["state_direct_care_price_clip_binding_share"], errors="coerce").mean()
        )
        diag["state_direct_care_clip_binding_share_p50"] = float(
            pd.to_numeric(state["state_direct_care_price_clip_binding_share"], errors="coerce").median()
        )
    if "state_implied_direct_care_hourly_wage" in state.columns:
        diag["state_implied_direct_care_hourly_wage_p50"] = float(
            pd.to_numeric(state["state_implied_direct_care_hourly_wage"], errors="coerce").median()
        )

    # Scenario input completeness (state_price_index comes from county aggregation)
    for col in ("state_price_index", "market_quantity_proxy", "unpaid_quantity_proxy"):
        if col in state.columns:
            diag[f"state_{col}_na"] = int(state[col].isna().sum())
            diag[f"state_{col}_zero"] = int((state[col] == 0).sum())

    # Observed vs ATUS-synthetic state controls
    acs_path = paths.interim / "acs" / "acs.parquet"
    if "state_controls_source" in state.columns:
        diag["state_controls_acs_observed"] = int(state["state_controls_source"].eq("acs_observed").sum())
    else:
        diag["state_controls_acs_observed"] = 0
    diag["state_controls_atus_synthetic"] = n_state - diag["state_controls_acs_observed"]

    # LAUS unemployment overlap
    laus_path = paths.interim / "laus" / "laus.parquet"
    if "state_unemployment_source" in state.columns:
        diag["state_unemployment_laus_observed"] = int(state["state_unemployment_source"].eq("laus_observed").sum())
    else:
        diag["state_unemployment_laus_observed"] = 0
    diag["state_unemployment_atus_synthetic"] = n_state - diag["state_unemployment_laus_observed"]
    if "state_ndcp_imputed_share" in state.columns:
        diag["state_ndcp_imputed_share_mean"] = round(
            float(pd.to_numeric(state["state_ndcp_imputed_share"], errors="coerce").mean()),
            3,
        )
    if "state_qcew_labor_observed_share" in state.columns:
        diag["state_qcew_labor_observed_share_mean"] = round(
            float(pd.to_numeric(state["state_qcew_labor_observed_share"], errors="coerce").mean()),
            3,
        )
        observed_core = state.loc[state["eligible_observed_core"].fillna(False).astype(bool)].copy()
        if not observed_core.empty:
            diag["observed_core_state_qcew_labor_observed_share_min"] = round(
                float(pd.to_numeric(observed_core["state_qcew_labor_observed_share"], errors="coerce").min()),
                3,
            )
    for column in (
        "eligible_broad_complete",
        "eligible_observed_core",
        "eligible_observed_core_low_impute",
        "eligible_observed_archival",
        "eligible_observed_archival_low_impute",
    ):
        if column in state.columns:
            diag[column] = int(state[column].fillna(False).astype(bool).sum())
    for reason_col in (
        "observed_core_exclusion_reason",
        "observed_core_low_impute_exclusion_reason",
        "observed_archival_exclusion_reason",
        "observed_archival_low_impute_exclusion_reason",
    ):
        if reason_col in state.columns:
            counts = state[reason_col].fillna("missing").value_counts().to_dict()
            prefix = reason_col.replace("_exclusion_reason", "_exclusion")
            diag[f"{prefix}_counts"] = {str(key): int(value) for key, value in counts.items()}

    # --- County-year panel ---
    n_county = len(county)
    diag["county_year_rows"] = n_county
    diag["county_year_unique_counties"] = int(county["county_fips"].nunique())
    diag["county_year_min"] = int(county["year"].min())
    diag["county_year_max"] = int(county["year"].max())

    # NDCP imputation prevalence
    if "imputed_share" in county.columns:
        diag["ndcp_any_imputed_rows"] = int((county["imputed_share"] > 0).sum())
        diag["ndcp_fully_imputed_rows"] = int((county["imputed_share"] >= 0.99).sum())
        diag["ndcp_mean_imputed_share"] = round(float(county["imputed_share"].mean()), 3)
    else:
        diag["ndcp_any_imputed_rows"] = 0
        diag["ndcp_fully_imputed_rows"] = 0
        diag["ndcp_mean_imputed_share"] = 0.0

    # County covariates from ACS vs state-level ATUS backfill
    if acs_path.exists():
        acs = read_parquet(acs_path)
        acs_cy = set(
            zip(acs["county_fips"].astype(str).str.zfill(5), acs["year"].astype(int))
        )
        county_years = list(zip(county["county_fips"].astype(str), county["year"].astype(int)))
        diag["county_controls_acs_direct"] = sum(1 for cy in county_years if cy in acs_cy)
        # Year-level coverage detail
        acs_available_years = sorted(set(acs["year"].astype(int).tolist()))
        county_panel_years = sorted(set(county["year"].astype(int).tolist()))
        diag["acs_available_years"] = acs_available_years
        diag["county_panel_years"] = county_panel_years
        diag["acs_missing_county_years"] = sorted(set(county_panel_years) - set(acs_available_years))
    else:
        diag["county_controls_acs_direct"] = 0
        diag["acs_available_years"] = []
        diag["county_panel_years"] = sorted(set(county["year"].astype(int).tolist()))
        diag["acs_missing_county_years"] = diag["county_panel_years"]
    diag["county_controls_state_backfill"] = n_county - diag["county_controls_acs_direct"]

    # County controls still null after backfill
    for col in ("single_parent_share", "median_income"):
        if col in county.columns:
            diag[f"county_{col}_null"] = int(county[col].isna().sum())

    # QCEW wage/employment coverage
    qcew_path = paths.interim / "qcew" / "qcew.parquet"
    if qcew_path.exists():
        qcew_data = read_parquet(qcew_path)
        qcew_cy = set(
            zip(qcew_data["county_fips"].astype(str).str.zfill(5), qcew_data["year"].astype(int))
        )
        county_years = list(zip(county["county_fips"].astype(str), county["year"].astype(int)))
        diag["county_qcew_direct"] = sum(1 for cy in county_years if cy in qcew_cy)
        qcew_available = sorted(set(qcew_data["year"].astype(int).tolist()))
        diag["qcew_available_years"] = qcew_available
        diag["qcew_missing_county_years"] = sorted(set(county["year"].astype(int).tolist()) - set(qcew_available))
    else:
        diag["county_qcew_direct"] = 0
        diag["qcew_available_years"] = []
        diag["qcew_missing_county_years"] = sorted(set(county["year"].astype(int).tolist()))

    # Wage source: price-derived fallback
    if "childcare_worker_wage_source" in county.columns:
        wage_sources = county["childcare_worker_wage_source"].fillna("missing")
        diag["county_wage_observed"] = int(wage_sources.eq("qcew_county_observed").sum())
        diag["county_wage_oews_state_observed"] = int(wage_sources.eq("oews_state_observed").sum())
        diag["county_wage_price_derived"] = int(wage_sources.eq("missing").sum())
    elif "childcare_worker_wage" in county.columns and "annual_price" in county.columns:
        price_derived = (
            county["childcare_worker_wage"]
            - county["annual_price"] / float(assumptions["direct_care_hours_per_year"])
        ).abs() < 0.01
        diag["county_wage_price_derived"] = int(price_derived.sum())
        diag["county_wage_observed"] = n_county - int(price_derived.sum())

    # Employment source: synthetic fallback
    if "employment_source" in county.columns:
        employment_sources = county["employment_source"].fillna("missing")
        diag["county_employment_observed"] = int(employment_sources.eq("qcew_employment_observed").sum())
        diag["county_employment_synthetic"] = int(
            employment_sources.str.startswith("employment_ratio_").sum()
        )
    elif "employment" in county.columns and "under5_population" in county.columns:
        observed_ratio = (
            pd.to_numeric(county["employment"], errors="coerce")
            .div(pd.to_numeric(county["under5_population"], errors="coerce").replace({0: pd.NA}))
            .dropna()
        )
        fallback_ratio = float(observed_ratio.median()) if not observed_ratio.empty else 0.0
        synthetic_emp = (
            county["employment"]
            - (
                county["under5_population"] * fallback_ratio
            ).clip(lower=1.0)
        ).abs() < 0.5
        diag["county_employment_synthetic"] = int(synthetic_emp.sum())
        diag["county_employment_observed"] = n_county - int(synthetic_emp.sum())

    # Market structure coverage
    if "employer_establishments" in county.columns:
        diag["county_cbp_observed"] = int(county["employer_establishments"].notna().sum())
    if "nonemployer_firms" in county.columns:
        diag["county_nes_observed"] = int(county["nonemployer_firms"].notna().sum())
    if "head_start_slots" in county.columns:
        diag["county_head_start_observed"] = int(county["head_start_slots"].gt(0).sum())
    if "head_start_capacity_source" in county.columns:
        source = county["head_start_capacity_source"].fillna("missing")
        diag["county_head_start_state_ratio_fallback"] = int(source.eq("head_start_slot_share_state").sum())
        diag["county_head_start_national_ratio_fallback"] = int(source.eq("head_start_slot_share_national").sum())

    # LAUS county coverage
    if "laus_unemployment_rate" in county.columns:
        diag["county_laus_observed"] = int(county["laus_unemployment_rate"].notna().sum())
    laus_path = paths.interim / "laus" / "laus.parquet"
    if laus_path.exists():
        laus_data = read_parquet(laus_path)
        laus_available = sorted(set(laus_data["year"].astype(int).tolist()))
        diag["laus_available_years"] = laus_available
        diag["laus_missing_county_years"] = sorted(set(county["year"].astype(int).tolist()) - set(laus_available))
    else:
        diag["county_laus_observed"] = 0

    # Write artifact
    output_path = paths.outputs_reports / "childcare_pipeline_diagnostics.json"
    write_json(diag, output_path)
    return diag
