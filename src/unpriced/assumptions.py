from __future__ import annotations

from pathlib import Path
from typing import Any

from unpriced.config import ProjectPaths, load_yaml
from unpriced.storage import write_json


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_root(paths_or_root: ProjectPaths | Path) -> Path:
    if isinstance(paths_or_root, ProjectPaths):
        return paths_or_root.root
    return Path(paths_or_root)


def _assumptions_path(paths_or_root: ProjectPaths | Path) -> Path:
    root = _as_root(paths_or_root)
    candidate = root / "configs" / "assumptions.yaml"
    if candidate.exists():
        return candidate
    return _repo_root() / "configs" / "assumptions.yaml"


def load_assumptions(paths_or_root: ProjectPaths | Path) -> dict[str, Any]:
    return load_yaml(_assumptions_path(paths_or_root))


def childcare_model_assumptions(paths_or_root: ProjectPaths | Path) -> dict[str, Any]:
    config = load_assumptions(paths_or_root).get("childcare", {})
    staffing = config.get("direct_care", {}).get("staffing_children_per_worker", {})
    staffing_pairs = {
        (provider_type, child_age): float(value)
        for provider_type, ages in staffing.items()
        for child_age, value in (ages or {}).items()
    }
    default_children_per_worker = config.get("direct_care", {}).get("default_children_per_worker")
    if default_children_per_worker is None:
        default_children_per_worker = sorted(staffing_pairs.values())[len(staffing_pairs) // 2]
    return {
        "low_impute_threshold": float(config.get("sample_ladder", {}).get("low_impute_threshold", 0.25)),
        "direct_care_hours_per_year": float(config.get("direct_care", {}).get("hours_per_year", 2080.0)),
        "direct_care_fringe_multiplier": float(
            config.get("direct_care", {}).get("fringe_multiplier_default", 1.3763)
        ),
        "default_children_per_worker": float(default_children_per_worker),
        "staffing_children_per_worker": staffing_pairs,
        "market_hours_per_child_per_week": float(
            config.get("fallbacks", {}).get("market_hours_per_child_per_week", 18.24)
        ),
        "sensitivity_staffing_cases": {
            str(key): float(value)
            for key, value in config.get("sensitivity", {}).get("staffing_scale_cases", {}).items()
        },
        "sensitivity_fringe_cases": {
            str(key): float(value)
            for key, value in config.get("sensitivity", {}).get("fringe_cases", {}).items()
        },
        "piecewise_supply_labor_support_threshold": float(
            config.get("demos", {}).get("piecewise_supply_labor_support_threshold", 0.99)
        ),
        "dual_shift_headline_alpha": float(
            config.get("demos", {}).get("dual_shift", {}).get("headline_alpha", 0.50)
        ),
        "dual_shift_kappa_q_grid": [
            float(value)
            for value in config.get("demos", {}).get("dual_shift", {}).get(
                "kappa_q_grid",
                [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50],
            )
        ],
        "dual_shift_kappa_c_grid": [
            float(value)
            for value in config.get("demos", {}).get("dual_shift", {}).get(
                "kappa_c_grid",
                [0.00, 0.05, 0.10, 0.15, 0.20],
            )
        ],
    }


def home_maintenance_assumptions(paths_or_root: ProjectPaths | Path) -> dict[str, Any]:
    config = load_assumptions(paths_or_root).get("home_maintenance", {})
    return {
        "cbsa_unemployment_base": float(config.get("fallbacks", {}).get("cbsa_unemployment_base", 0.05)),
        "cbsa_unemployment_storm_slope": float(
            config.get("fallbacks", {}).get("cbsa_unemployment_storm_slope", 0.10)
        ),
    }


def solver_assumptions(paths_or_root: ProjectPaths | Path) -> dict[str, Any]:
    config = load_assumptions(paths_or_root).get("solver", {})
    return {
        "price_search_lower_bound_ratio": float(
            config.get("price_search", {}).get("lower_bound_ratio", 0.25)
        ),
        "price_search_upper_bound_base_ratio": float(
            config.get("price_search", {}).get("upper_bound_base_ratio", 1.50)
        ),
        "bootstrap_n_boot": int(config.get("bootstrap", {}).get("n_boot", 100)),
        "bootstrap_seed": int(config.get("bootstrap", {}).get("seed", 42)),
    }


def build_assumption_audit(paths: ProjectPaths) -> dict[str, Any]:
    childcare = childcare_model_assumptions(paths)
    home = home_maintenance_assumptions(paths)
    solver = solver_assumptions(paths)
    audit = {
        "childcare": [
            {
                "name": "direct_care_fringe_multiplier",
                "value": childcare["direct_care_fringe_multiplier"],
                "status": "source_backed_scalar",
                "used_in": "canonical decomposition",
                "formula": "26.48 / 19.24",
                "source_name": "BLS ECEC Table 4, June 2025, Health care and social assistance industry, Service occupations",
                "source_url": "https://www.bls.gov/news.release/ecec.t04.htm",
                "source_quote": "Health care and social assistance industry ... Service occupations",
                "note": "Compensation-to-wages benchmark from BLS ECEC. This is source-backed, but it is still a benchmark for relevant service work rather than a childcare-only compensation estimate.",
            },
            {
                "name": "direct_care_hours_per_year",
                "value": childcare["direct_care_hours_per_year"],
                "status": "source_backed_normalization",
                "used_in": "canonical decomposition",
                "formula": "52 weeks * 40 hours",
                "source_name": "BLS OEWS FAQ / Technical Notes",
                "source_url": "https://www.bls.gov/oes/oes_ques.htm",
                "source_quote": "full-time, year-round schedule of 2,080 hours",
                "note": "BLS annualization convention for converting hourly wages to annual wages. This is a normalization constant, not a claim about realized childcare-worker hours.",
            },
            {
                "name": "staffing_children_per_worker_center",
                "value": {
                    child_age: value
                    for (provider_type, child_age), value in childcare["staffing_children_per_worker"].items()
                    if provider_type == "center"
                },
                "status": "source_backed_lookup",
                "used_in": "canonical decomposition",
                "source_name": "ACF/NCECQA Research Brief #1: Trends in Child Care Center Licensing Regulations and Policies for 2014, Table 4",
                "source_url": "https://childcareta.acf.hhs.gov/sites/default/files/new-occ/resource/files/center_licensing_trends_brief_2014.pdf",
                "source_quote": "Range of State Child-staff Ratio Requirements for Child Care Centers, 2014",
                "note": "Center infant and preschool values follow the brief's most-common state ratios. The single toddler bucket is set to the midpoint of the brief's 18-month and 35-month most-common ratios because the model keeps toddler collapsed into one category.",
            },
            {
                "name": "staffing_children_per_worker_home",
                "value": {
                    child_age: value
                    for (provider_type, child_age), value in childcare["staffing_children_per_worker"].items()
                    if provider_type == "home"
                },
                "status": "source_backed_benchmark_mapping",
                "used_in": "canonical decomposition",
                "source_name": "Head Start 45 C.F.R. 1302.23 Family Child Care Option",
                "source_url": "https://eclkc.ohs.acf.hhs.gov/policy/45-cfr-chap-xiii/1302-23-family-child-care-option",
                "source_quote": "maximum group size is six children ... maximum group size is four children",
                "note": "Official public sources distinguish family child care homes from group homes, so the repo's single 'home' provider type uses a transparent Head Start family child care benchmark mapping of 2/4/6 rather than claiming a national market-average licensing table.",
            },
            {
                "name": "outside_option_wage_fallback_series",
                "value": "OEWS-derived state/year ratio fallback",
                "status": "derived_from_observed_public_data",
                "used_in": "demand instrument fallback only",
                "note": "Outside-option fallback is now derived from observed OEWS childcare/outside-option ratios rather than a fixed scalar.",
            },
            {
                "name": "employment_fallback_series",
                "value": "QCEW/ACS employment-per-under5 medians",
                "status": "derived_from_observed_public_data",
                "used_in": "county labor fallback only",
                "note": "Missing county employment is now filled from observed employment-to-under5 ratios by state/year when available.",
            },
            {
                "name": "head_start_capacity_fallback_series",
                "value": "Head Start / under5 slot-share medians",
                "status": "derived_from_observed_public_data",
                "used_in": "county feature fallback only",
                "note": "Missing Head Start capacity is now filled from observed slot-share medians rather than a fixed scalar.",
            },
            {
                "name": "market_hours_per_child_per_week",
                "value": childcare["market_hours_per_child_per_week"],
                "status": "source_backed_formula",
                "used_in": "scenario quantity proxy",
                "formula": "(12594 / 21195) * 30.7",
                "source_name": "NCES Early Childhood Program Participation: 2019 + Digest Table 202.30",
                "source_urls": [
                    "https://nces.ed.gov/pubs2020/2020075REV.pdf",
                    "https://nces.ed.gov/programs/digest/d22/tables/dt22_202.30.asp",
                ],
                "source_quote": "Approximately 59 percent",
                "note": "Computed weekly nonparental-care hours per child from NCES participation and conditional-hours tables. This remains a market-size proxy, not a direct state-year paid-enrollment estimate.",
            },
            {
                "name": "decomposition_sensitivity_staffing_scales",
                "value": childcare["sensitivity_staffing_cases"],
                "status": "stress_test_design",
                "used_in": "decomposition sensitivity only",
                "formula": "canonical sourced staffing table multiplied by +/- 10%",
                "note": "These are not claimed as outside-source estimates. They are a bounded stress envelope around the source-backed canonical staffing lookup and do not affect canonical outputs.",
            },
            {
                "name": "decomposition_sensitivity_fringe_cases",
                "value": childcare["sensitivity_fringe_cases"],
                "status": "stress_test_design",
                "used_in": "decomposition sensitivity only",
                "formula": "canonical sourced fringe multiplier multiplied by +/- 10%",
                "note": "These are not claimed as outside-source estimates. They are a bounded stress envelope around the source-backed canonical fringe benchmark and do not affect canonical outputs.",
            },
        ],
        "home_maintenance": [
            {
                "name": "cbsa_unemployment_fallback_rule",
                "value": {
                    "base": home["cbsa_unemployment_base"],
                    "storm_slope": home["cbsa_unemployment_storm_slope"],
                },
                "status": "fallback_assumption",
                "used_in": "home panel only when LAUS is unavailable",
                "note": "Not used in normal real-data runs when LAUS ingest succeeds.",
            }
        ],
        "solver": [
            {
                "name": "price_search_bounds",
                "value": {
                    "lower_bound_ratio": solver["price_search_lower_bound_ratio"],
                    "upper_bound_base_ratio": solver["price_search_upper_bound_base_ratio"],
                },
                "status": "algorithmic_tuning",
                "used_in": "root finding",
                "note": "Numerical search bounds, not economic parameters.",
            },
            {
                "name": "bootstrap_defaults",
                "value": {
                    "n_boot": solver["bootstrap_n_boot"],
                    "seed": solver["bootstrap_seed"],
                },
                "status": "algorithmic_tuning",
                "used_in": "uncertainty simulation",
                "note": "Computation defaults, not estimands.",
            },
        ],
    }
    return audit


def write_assumption_audit(paths: ProjectPaths) -> dict[str, Any]:
    audit = build_assumption_audit(paths)
    write_json(audit, paths.outputs_reports / "model_assumption_audit.json")
    return audit
