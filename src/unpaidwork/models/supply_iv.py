from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    dataset = frame.copy()
    for column in (
        "year",
        "annual_price",
        "provider_density",
        "under5_population",
        "employer_establishments",
        "nonemployer_firms",
        "center_labor_intensity_index",
        "center_labor_intensity_shock",
    ):
        if column in dataset.columns:
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    return dataset


def _compute_labor_intensity_index(shocks: pd.DataFrame) -> pd.DataFrame:
    dataset = shocks.copy()
    if "center_labor_intensity_index" in dataset.columns and dataset["center_labor_intensity_index"].notna().any():
        return dataset
    components = []
    for column in (
        "center_infant_ratio",
        "center_toddler_ratio",
        "center_infant_group_size",
        "center_toddler_group_size",
    ):
        if column in dataset.columns:
            values = pd.to_numeric(dataset[column], errors="coerce")
            components.append(1.0 / values.replace({0: pd.NA}))
    if not components:
        dataset["center_labor_intensity_index"] = np.nan
        return dataset
    component_frame = pd.concat(components, axis=1)
    dataset["center_labor_intensity_index"] = component_frame.mean(axis=1, skipna=True)
    return dataset


def normalize_licensing_supply_shocks(shocks: pd.DataFrame) -> pd.DataFrame:
    if shocks.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "center_labor_intensity_index",
                "center_labor_intensity_shock",
            ]
        )
    dataset = _coerce(shocks)
    dataset["state_fips"] = dataset["state_fips"].astype(str).str.zfill(2)
    dataset = _compute_labor_intensity_index(dataset)
    dataset = dataset.dropna(subset=["state_fips", "year", "center_labor_intensity_index"]).copy()
    dataset = dataset.sort_values(["state_fips", "year"]).reset_index(drop=True)
    baseline = dataset.groupby("state_fips")["center_labor_intensity_index"].transform("first")
    dataset["center_labor_intensity_shock"] = dataset["center_labor_intensity_index"] - baseline
    keep = [
        column
        for column in (
            "state_fips",
            "year",
            "center_labor_intensity_index",
            "center_labor_intensity_shock",
            "effective_date",
            "shock_label",
            "source_url",
        )
        if column in dataset.columns
    ]
    return dataset[keep].drop_duplicates(["state_fips", "year"]).reset_index(drop=True)


def build_supply_iv_panel(county: pd.DataFrame, shocks: pd.DataFrame) -> pd.DataFrame:
    county_panel = _coerce(county)
    county_panel["state_fips"] = county_panel["state_fips"].astype(str).str.zfill(2)
    county_panel["county_fips"] = county_panel["county_fips"].astype(str).str.zfill(5)
    shock_panel = normalize_licensing_supply_shocks(shocks)
    if shock_panel.empty:
        return pd.DataFrame()

    county_panel["total_provider_firms"] = (
        pd.to_numeric(county_panel.get("employer_establishments"), errors="coerce").fillna(0.0)
        + pd.to_numeric(county_panel.get("nonemployer_firms"), errors="coerce").fillna(0.0)
    )
    county_panel["center_exposure_share_raw"] = pd.to_numeric(
        county_panel.get("employer_establishments"), errors="coerce"
    ).div(county_panel["total_provider_firms"].replace({0: pd.NA}))
    county_panel["center_exposure_share_raw"] = county_panel["center_exposure_share_raw"].clip(lower=0.0, upper=1.0)

    exposure_source = (
        county_panel.sort_values(["county_fips", "year"])
        .groupby("county_fips", as_index=False)
        .first()[["county_fips", "center_exposure_share_raw", "year"]]
        .rename(
            columns={
                "center_exposure_share_raw": "center_exposure_share_pre",
                "year": "center_exposure_base_year",
            }
        )
    )
    panel = county_panel.merge(exposure_source, on="county_fips", how="left")
    panel = panel.merge(shock_panel, on=["state_fips", "year"], how="inner")
    panel["reg_shock_ct"] = (
        pd.to_numeric(panel["center_exposure_share_pre"], errors="coerce")
        * pd.to_numeric(panel["center_labor_intensity_shock"], errors="coerce")
    )
    panel["employer_establishment_density"] = pd.to_numeric(
        panel.get("employer_establishments"), errors="coerce"
    ).div(pd.to_numeric(panel["under5_population"], errors="coerce").replace({0: pd.NA})).mul(1000.0)
    panel["log_annual_price"] = np.log(pd.to_numeric(panel["annual_price"], errors="coerce"))
    panel["log_provider_density"] = np.log(pd.to_numeric(panel["provider_density"], errors="coerce"))
    panel["log_employer_establishment_density"] = np.log(
        pd.to_numeric(panel["employer_establishment_density"], errors="coerce")
    )
    panel["state_year"] = panel["state_fips"].astype(str) + "_" + panel["year"].astype(int).astype(str)
    panel = panel.dropna(
        subset=[
            "county_fips",
            "state_fips",
            "state_year",
            "reg_shock_ct",
            "log_annual_price",
            "log_provider_density",
        ]
    ).copy()
    return panel


def _two_way_demean(series: pd.Series, fe_a: pd.Series, fe_b: pd.Series, max_iter: int = 50, tol: float = 1e-10) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float).copy()
    demeaned = values - values.mean()
    for _ in range(max_iter):
        previous = demeaned.copy()
        demeaned = demeaned - demeaned.groupby(fe_a).transform("mean")
        demeaned = demeaned - demeaned.groupby(fe_b).transform("mean")
        if float((demeaned - previous).abs().max()) < tol:
            break
    return demeaned


def _clustered_scalar_regression(y: pd.Series, x: pd.Series, cluster: pd.Series) -> dict[str, float | int | None]:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    y = pd.to_numeric(y, errors="coerce").astype(float)
    valid = x.notna() & y.notna() & cluster.notna()
    x = x.loc[valid]
    y = y.loc[valid]
    cluster = cluster.loc[valid].astype(str)
    n_obs = int(len(x))
    n_clusters = int(cluster.nunique())
    xx = float(np.dot(x, x))
    if n_obs < 3 or xx <= 0:
        return {"beta": float("nan"), "se_cluster_state": float("nan"), "t_cluster_state": float("nan"), "f_stat": float("nan"), "within_r2": float("nan"), "n_obs": n_obs, "n_clusters": n_clusters}
    beta = float(np.dot(x, y) / xx)
    residual = y - beta * x
    sst = float(np.dot(y, y))
    sse = float(np.dot(residual, residual))
    within_r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    meat = 0.0
    for _, idx in cluster.groupby(cluster).groups.items():
        xu = float(np.dot(x.loc[idx], residual.loc[idx]))
        meat += xu * xu
    se = math.sqrt(meat / (xx * xx)) if meat >= 0 else float("nan")
    t_stat = beta / se if se and np.isfinite(se) and se > 0 else float("nan")
    f_stat = t_stat * t_stat if np.isfinite(t_stat) else float("nan")
    return {
        "beta": beta,
        "se_cluster_state": float(se),
        "t_cluster_state": float(t_stat),
        "f_stat": float(f_stat),
        "within_r2": within_r2,
        "n_obs": n_obs,
        "n_clusters": n_clusters,
    }


def fit_supply_iv_exposure_design(county: pd.DataFrame, shocks: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    panel = build_supply_iv_panel(county, shocks)
    if panel.empty:
        return {
            "status": "insufficient_overlap",
            "design": "county_fe_state_year_fe_exposure_shock",
            "note": "No overlapping county-year rows after joining the licensing shock panel to the childcare county panel.",
        }, panel

    shock = _two_way_demean(panel["reg_shock_ct"], panel["county_fips"], panel["state_year"])
    log_price = _two_way_demean(panel["log_annual_price"], panel["county_fips"], panel["state_year"])
    log_provider_density = _two_way_demean(panel["log_provider_density"], panel["county_fips"], panel["state_year"])
    log_employer_density = _two_way_demean(
        panel["log_employer_establishment_density"], panel["county_fips"], panel["state_year"]
    )

    first_stage = _clustered_scalar_regression(log_price, shock, panel["state_fips"])
    reduced_form_provider = _clustered_scalar_regression(log_provider_density, shock, panel["state_fips"])
    reduced_form_employer = _clustered_scalar_regression(log_employer_density, shock, panel["state_fips"])

    price_beta = float(first_stage["beta"]) if np.isfinite(first_stage["beta"]) else float("nan")
    provider_beta = (
        float(reduced_form_provider["beta"]) if np.isfinite(reduced_form_provider["beta"]) else float("nan")
    )
    employer_beta = (
        float(reduced_form_employer["beta"]) if np.isfinite(reduced_form_employer["beta"]) else float("nan")
    )
    shock_state_count = int(
        panel.loc[pd.to_numeric(panel["center_labor_intensity_shock"], errors="coerce").abs().gt(0), "state_fips"].nunique()
    )
    has_identifying_variation = shock_state_count > 0 and np.isfinite(price_beta)
    has_multi_state_treatment = shock_state_count >= 2
    first_stage_strength_flag = (
        "strong"
        if has_multi_state_treatment
        and int(first_stage["n_clusters"]) >= 2
        and np.isfinite(first_stage["f_stat"])
        and first_stage["f_stat"] >= 10
        else "weak_or_unknown"
    )
    pilot_scope = "multi_state_demo" if has_multi_state_treatment else "single_state_pilot"
    treated_states = sorted(
        panel.loc[
            pd.to_numeric(panel["center_labor_intensity_shock"], errors="coerce").abs().gt(0),
            "state_fips",
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    local_iv_provider_elasticity = (
        provider_beta / price_beta
        if np.isfinite(provider_beta) and np.isfinite(price_beta) and abs(price_beta) > 1e-12
        else float("nan")
    )
    local_iv_employer_elasticity = (
        employer_beta / price_beta
        if np.isfinite(employer_beta) and np.isfinite(price_beta) and abs(price_beta) > 1e-12
        else float("nan")
    )
    summary = {
        "status": "ok" if has_identifying_variation else "insufficient_identification",
        "design": "county_fe_state_year_fe_exposure_shock",
        "pilot_scope": pilot_scope,
        "shock_variable": "center_exposure_share_pre * center_labor_intensity_shock",
        "n_obs": int(len(panel)),
        "n_counties": int(panel["county_fips"].nunique()),
        "n_states": int(panel["state_fips"].nunique()),
        "year_min": int(pd.to_numeric(panel["year"], errors="coerce").min()),
        "year_max": int(pd.to_numeric(panel["year"], errors="coerce").max()),
        "center_exposure_base_year_min": int(pd.to_numeric(panel["center_exposure_base_year"], errors="coerce").min()),
        "center_exposure_base_year_max": int(pd.to_numeric(panel["center_exposure_base_year"], errors="coerce").max()),
        "shock_state_count": shock_state_count,
        "treated_state_fips": treated_states,
        "first_stage_price": first_stage,
        "reduced_form_provider_density": reduced_form_provider,
        "reduced_form_employer_establishment_density": reduced_form_employer,
        "iv_supply_elasticity_provider_density": local_iv_provider_elasticity,
        "iv_supply_elasticity_employer_establishment_density": local_iv_employer_elasticity,
        "local_iv_supply_elasticity_provider_density": local_iv_provider_elasticity,
        "local_iv_supply_elasticity_employer_establishment_density": local_iv_employer_elasticity,
        "secondary_supply_estimate": {
            "name": "local_iv_supply_elasticity_provider_density",
            "value": local_iv_provider_elasticity,
            "outcome": "provider_density",
            "scope": "treated_state_local_wald",
            "treated_state_count": shock_state_count,
            "treated_state_fips": treated_states,
            "interpretation": (
                "Local treated-state Wald elasticity from the licensing-shock exposure design. "
                "This is a non-canonical secondary supply estimate, not the headline national supply curve."
            ),
        },
        "first_stage_strength_flag": first_stage_strength_flag,
        "note": (
            "This is a non-canonical supply-shock exposure design. It should be interpreted as an experimental reduced-form / Wald supply-IV demo, not as the canonical childcare supply estimate."
            if has_identifying_variation
            and pilot_scope != "single_state_pilot"
            else "This is a non-canonical single-treated-state supply-shock pilot. The joined panel may include additional comparison states, but only one state contributes nonzero licensing-shock variation. Treat the sign and overlap diagnostics as the main output, not the cluster-robust inference."
            if has_identifying_variation
            else "The joined panel exists, but the merged licensing shock has no usable first-stage variation after overlap and fixed-effects residualization."
        ),
    }
    return summary, panel
