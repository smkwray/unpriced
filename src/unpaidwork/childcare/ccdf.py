from __future__ import annotations

import re
from typing import Any

import pandas as pd

STATE_NAME_TO_FIPS = {
    "alabama": "01",
    "alaska": "02",
    "arizona": "04",
    "arkansas": "05",
    "california": "06",
    "colorado": "08",
    "connecticut": "09",
    "delaware": "10",
    "district of columbia": "11",
    "florida": "12",
    "georgia": "13",
    "hawaii": "15",
    "idaho": "16",
    "illinois": "17",
    "indiana": "18",
    "iowa": "19",
    "kansas": "20",
    "kentucky": "21",
    "louisiana": "22",
    "maine": "23",
    "maryland": "24",
    "massachusetts": "25",
    "michigan": "26",
    "minnesota": "27",
    "mississippi": "28",
    "missouri": "29",
    "montana": "30",
    "nebraska": "31",
    "nevada": "32",
    "new hampshire": "33",
    "new jersey": "34",
    "new mexico": "35",
    "new york": "36",
    "north carolina": "37",
    "north dakota": "38",
    "ohio": "39",
    "oklahoma": "40",
    "oregon": "41",
    "pennsylvania": "42",
    "rhode island": "44",
    "south carolina": "45",
    "south dakota": "46",
    "tennessee": "47",
    "texas": "48",
    "utah": "49",
    "vermont": "50",
    "virginia": "51",
    "washington": "53",
    "west virginia": "54",
    "wisconsin": "55",
    "wyoming": "56",
    "puerto rico": "72",
}

STATE_ABBR_TO_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
    "PR": "72",
}

ADMIN_REQUIRED_COLUMNS = {
    "source_component",
    "raw_relpath",
    "source_sheet",
    "row_number",
    "column_name",
    "value_text",
    "value_numeric",
    "table_year",
    "parse_status",
}

POLICY_REQUIRED_COLUMNS = ADMIN_REQUIRED_COLUMNS

CHILDREN_SERVED_TOKENS = (
    "children_served",
    "children served",
    "average_monthly_children",
    "average monthly children",
    "served_average_monthly",
)
EXPENDITURE_TOKENS = (
    "expenditure",
    "expenditures",
    "amount_paid",
    "payments",
    "subsidy_spending",
    "total_spending",
)
ADMIN_EXPENDITURE_TOKENS = ("ccdf_admin_expenditures", "admin_expenditures")
QUALITY_EXPENDITURE_TOKENS = (
    "ccdf_quality_activities_expenditures",
    "quality_activities_expenditures",
)
INFANT_TODDLER_QUALITY_EXPENDITURE_TOKENS = (
    "ccdf_infant_toddler_quality_expenditures",
    "infant_toddler_quality_expenditures",
)
DIRECT_SERVICES_EXPENDITURE_TOKENS = (
    "ccdf_direct_services_expenditures",
    "direct_services_expenditures",
)
NONDIRECT_SYSTEMS_EXPENDITURE_TOKENS = (
    "ccdf_nondirect_systems_expenditures",
    "nondirect_systems_expenditures",
)
NONDIRECT_ELIGIBILITY_EXPENDITURE_TOKENS = (
    "ccdf_nondirect_cert_prog_eligibility_expenditures",
    "nondirect_cert_prog_eligibility_expenditures",
)
NONDIRECT_OTHER_EXPENDITURE_TOKENS = (
    "ccdf_nondirect_other_expenditures",
    "nondirect_other_expenditures",
)
CARE_TYPE_CHILD_HOME_TOKENS = ("ccdf_care_type_child_home_share", "care_type_child_home_share")
CARE_TYPE_FAMILY_HOME_TOKENS = ("ccdf_care_type_family_home_share", "care_type_family_home_share")
CARE_TYPE_GROUP_HOME_TOKENS = ("ccdf_care_type_group_home_share", "care_type_group_home_share")
CARE_TYPE_CENTER_TOKENS = ("ccdf_care_type_center_share", "care_type_center_share")
CARE_TYPE_INVALID_TOKENS = ("ccdf_care_type_invalid_share", "care_type_invalid_share")
REGULATED_SHARE_TOKENS = (
    "ccdf_regulated_share",
    "regulated_share",
    "ccdf_setting_detail_regulated_total_share",
)
UNREGULATED_SHARE_TOKENS = (
    "ccdf_unregulated_share",
    "unregulated_share",
    "ccdf_setting_detail_unregulated_total_share",
)
REGULATION_INVALID_SHARE_TOKENS = (
    "ccdf_regulation_invalid_share",
    "regulation_invalid_share",
    "ccdf_setting_detail_invalid_share",
)
UNREGULATED_RELATIVE_SHARE_TOKENS = ("ccdf_setting_detail_unregulated_relative_total_share",)
UNREGULATED_NONRELATIVE_SHARE_TOKENS = ("ccdf_setting_detail_unregulated_nonrelative_total_share",)
RELATIVE_CARE_TOTAL_CHILDREN_TOKENS = ("ccdf_relative_care_total_children",)
RELATIVE_CARE_RELATIVE_CHILDREN_TOKENS = ("ccdf_relative_care_relative_children",)
RELATIVE_CARE_NONRELATIVE_CHILDREN_TOKENS = ("ccdf_relative_care_nonrelative_children",)
SUBSIDIZED_PRIVATE_TOKENS = (
    "subsidized_private_slots",
    "subsidized_private_children",
    "ccdf_subsidized_private_slots",
    "ccdf_subsidized_private_children",
    "subsidized_private",
    "subsidized_slots",
    "subsidized_children",
    "subsidized private slots",
    "subsidized private children",
)
PUBLIC_ADMIN_TOKENS = (
    "public_admin_slots",
    "public_admin_children",
    "ccdf_public_admin_slots",
    "ccdf_public_admin_children",
    "public_admin",
    "public_slots",
    "public_children",
    "public admin slots",
    "public admin children",
)
SUBSIDIZED_PRIVATE_SHARE_TOKENS = (
    "subsidized_private_share",
    "subsidized_private_percent",
    "subsidized_private_percentage",
    "subsidized_share",
    "subsidized_percent",
    "subsidized_percentage",
    "private_subsidy_share",
    "private_subsidy_percent",
)
PUBLIC_ADMIN_SHARE_TOKENS = (
    "public_admin_share",
    "public_admin_percent",
    "public_admin_percentage",
    "public_share",
    "public_percent",
    "public_percentage",
    "administrative_public_share",
    "administrative_public_percent",
)
YEAR_PATTERN = re.compile(r"(20\d{2})")
PAYMENT_PROXY_CLOSE_RATIO_GAP_MAX = 0.25
PAYMENT_PROXY_MODERATE_RATIO_GAP_MAX = 0.75
PAYMENT_PROXY_CLOSE_ABS_GAP_MAX = 25_000.0
PAYMENT_PROXY_MODERATE_ABS_GAP_MAX = 75_000.0
PAYMENT_METHOD_BOUNDARY_SHARE_EPS = 0.02
PAYMENT_PROXY_LARGE_GAP_SUPPORT_FLAG = "ccdf_split_proxy_from_payment_method_shares_large_gap"
PAYMENT_PROXY_LARGE_GAP_SUPPORT_STATUS = "observed_long_payment_method_share_proxy_large_gap"
PAYMENT_PROXY_LARGE_GAP_DOWNGRADED_SUPPORT_FLAG = (
    "ccdf_split_proxy_from_payment_method_shares_large_gap_downgraded_to_children_served_proxy"
)
PAYMENT_PROXY_LARGE_GAP_DOWNGRADED_SUPPORT_STATUS = (
    "observed_long_payment_method_share_proxy_large_gap_downgraded_to_children_served_proxy"
)
POLICY_CONTROL_SPECS = {
    "ccdf_control_copayment_required": {
        "kind": "text",
        "aliases": (
            "copayment_required",
            "copay_required",
            "family_copayment_required",
            "family_copay_required",
            "copaycollect",
            "copay_collect",
            "copaymin",
            "copay_min",
        ),
    },
}


def _normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _resolve_state_fips(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if digits:
        if len(digits) == 1:
            return digits.zfill(2)
        if len(digits) == 2:
            return digits
    upper = text.upper()
    if upper in STATE_ABBR_TO_FIPS:
        return STATE_ABBR_TO_FIPS[upper]
    lower = text.lower()
    return STATE_NAME_TO_FIPS.get(lower)


def _to_numeric(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = (
        text.replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .replace("(", "-")
        .replace(")", "")
    )
    numeric = pd.to_numeric(pd.Series([normalized]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def _first_numeric(values: list[object]) -> float | None:
    for value in values:
        numeric = _to_numeric(value)
        if numeric is not None:
            return numeric
    return None


def _first_int(values: list[object]) -> int | None:
    for value in values:
        if value is None or pd.isna(value):
            continue
        digits = YEAR_PATTERN.search(str(value))
        if digits:
            return int(digits.group(1))
        numeric = _to_numeric(value)
        if numeric is not None:
            candidate = int(numeric)
            if 1900 <= candidate <= 2100:
                return candidate
    return None


def _first_code_int(values: list[object]) -> int | None:
    for value in values:
        numeric = _to_numeric(value)
        if numeric is None:
            continue
        return int(numeric)
    return None


def _pivot_long_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    group_columns = ["raw_relpath", "source_sheet", "row_number"]
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        row: dict[str, Any] = {
            "raw_relpath": keys[0],
            "source_sheet": keys[1],
            "row_number": keys[2],
            "table_year": group["table_year"].dropna().iloc[0] if group["table_year"].notna().any() else pd.NA,
        }
        for item in group.to_dict(orient="records"):
            key = _normalize_label(item.get("column_name"))
            if not key:
                continue
            if key not in row or row[key] in (None, "", pd.NA):
                row[key] = item.get("value_text")
            numeric_key = f"{key}__numeric"
            if numeric_key not in row and pd.notna(item.get("value_numeric")):
                row[numeric_key] = float(item["value_numeric"])
        rows.append(row)
    return rows


def _extract_state_year(row: dict[str, Any]) -> tuple[str | None, int | None]:
    state_candidates = [
        row.get("state_fips"),
        row.get("state"),
        row.get("state_name"),
        row.get("state_code"),
        row.get("state_abbreviation"),
        row.get("st"),
    ]
    year_candidates = [
        row.get("year"),
        row.get("fiscal_year"),
        row.get("ffy"),
        row.get("report_year"),
        row.get("table_year"),
        row.get("source_sheet"),
        row.get("raw_relpath"),
    ]
    return _resolve_state_fips(next((value for value in state_candidates if _resolve_state_fips(value)), None)), _first_int(
        year_candidates
    )


def _parse_effective_year(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = YEAR_PATTERN.search(text)
    if match:
        return int(match.group(1))
    try:
        timestamp = pd.to_datetime(text, errors="coerce")
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    return int(timestamp.year)


def _policy_effective_years(row: dict[str, Any]) -> list[int]:
    explicit_year = _first_int(
        [
            row.get("year"),
            row.get("fiscal_year"),
            row.get("ffy"),
            row.get("report_year"),
        ]
    )
    if explicit_year is not None:
        return [explicit_year]

    workbook_year = _first_int([row.get("table_year"), row.get("raw_relpath"), row.get("source_sheet")])
    start_year = (
        _parse_effective_year(row.get("beginmajority"))
        or _parse_effective_year(row.get("begindat"))
        or workbook_year
    )
    end_year = (
        _parse_effective_year(row.get("endmajority"))
        or _parse_effective_year(row.get("enddat"))
        or start_year
        or workbook_year
    )
    if start_year is None:
        return [year for year in [workbook_year] if year is not None]
    if end_year is None:
        return [start_year]
    if end_year >= 2100 and workbook_year is not None:
        end_year = workbook_year
    if end_year < start_year:
        end_year = start_year
    if end_year - start_year > 30:
        return [start_year]
    return list(range(start_year, end_year + 1))


def _policy_row_is_general_scope(row: dict[str, Any]) -> bool:
    family_group = _first_code_int([row.get("familygroup"), row.get("family_group")])
    provider_type = _first_code_int([row.get("providertype"), row.get("provider_type")])
    provider_subtype = _first_code_int([row.get("providersubtype"), row.get("provider_subtype")])

    if family_group not in {None, 1, 98, 99}:
        return False
    if provider_type not in {None, 98, 99}:
        return False
    if provider_subtype not in {None, 0}:
        return False
    return True


def _normalize_yes_no_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    return None


def _normalize_yes_no_code(values: list[object]) -> str | None:
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip().lower()
        if text in {"yes", "y", "true"}:
            return "yes"
        if text in {"no", "n", "false"}:
            return "no"
    code = _first_code_int(values)
    if code == 1:
        return "yes"
    if code == 2:
        return "no"
    return None


def _extract_curated_policy_features(row: dict[str, Any]) -> list[dict[str, Any]]:
    if str(row.get("source_sheet", "")).startswith("0_"):
        return []

    state_fips = _resolve_state_fips(
        next(
            (
                value
                for value in (
                    row.get("state_fips"),
                    row.get("state"),
                    row.get("state_name"),
                    row.get("state_code"),
                    row.get("state_abbreviation"),
                    row.get("st"),
                )
                if _resolve_state_fips(value)
            ),
            None,
        )
    )
    if state_fips is None:
        return []

    direct_copayment = next(
        (
            _normalize_yes_no_text(value)
            for key, value in row.items()
            if _normalize_label(key) in {"copayment_required", "copay_required", "family_copayment_required"}
            and _normalize_yes_no_text(value) is not None
        ),
        None,
    )
    copay_collect = _first_code_int([row.get("copaycollect"), row.get("copaycollect__numeric")])
    copay_min = _first_numeric([row.get("copaymin"), row.get("copaymin__numeric")])
    copayment_required = direct_copayment
    if copayment_required is None:
        if copay_collect in {1, 2, 99}:
            copayment_required = "yes"
        elif copay_min is not None:
            copayment_required = "yes" if copay_min > 0 else "no"

    redeterm_period = _first_numeric([row.get("redetermperiod"), row.get("redetermperiod__numeric")])
    redeterm_app_doc_new = _normalize_yes_no_code(
        [row.get("redetermappdocnew"), row.get("redetermappdocnew__numeric")]
    )
    redeterm_doc_method_online = _normalize_yes_no_code(
        [row.get("redetermdocmethodonline"), row.get("redetermdocmethodonline__numeric")]
    )
    waitlist_active = _normalize_yes_no_code([row.get("waitlist"), row.get("waitlist__numeric")])
    copay_poverty_exempt = _normalize_yes_no_code(
        [row.get("copaypovertyexempt"), row.get("copaypovertyexempt__numeric")]
    )
    copay_tanf_exempt = _normalize_yes_no_code(
        [row.get("copaytanfexempt"), row.get("copaytanfexempt__numeric")]
    )
    years = _policy_effective_years(row)
    if not years:
        return []

    source_sheet = str(row.get("source_sheet") or "")
    selection_priority = 1 if _policy_row_is_general_scope(row) else 0
    features: list[dict[str, Any]] = []
    for year in years:
        if copayment_required is not None:
            features.append(
                {
                    "state_fips": state_fips,
                    "year": int(year),
                    "feature_name": "copayment_required",
                    "feature_value_text": copayment_required,
                    "feature_value_numeric": None,
                    "raw_relpath": row.get("raw_relpath"),
                    "source_sheet": source_sheet,
                    "row_number": row.get("row_number"),
                    "feature_support_status": "observed_policy_long_effective_dated",
                    "selection_priority": selection_priority,
                }
            )
        if redeterm_period is not None:
            features.append(
                {
                    "state_fips": state_fips,
                    "year": int(year),
                    "feature_name": "redetermination_period_months",
                    "feature_value_text": str(int(redeterm_period) if float(redeterm_period).is_integer() else redeterm_period),
                    "feature_value_numeric": float(redeterm_period),
                    "raw_relpath": row.get("raw_relpath"),
                    "source_sheet": source_sheet,
                    "row_number": row.get("row_number"),
                    "feature_support_status": "observed_policy_long_effective_dated",
                    "selection_priority": selection_priority,
                }
            )
        if redeterm_app_doc_new is not None:
            features.append(
                {
                    "state_fips": state_fips,
                    "year": int(year),
                    "feature_name": "redetermination_requires_new_documentation",
                    "feature_value_text": redeterm_app_doc_new,
                    "feature_value_numeric": None,
                    "raw_relpath": row.get("raw_relpath"),
                    "source_sheet": source_sheet,
                    "row_number": row.get("row_number"),
                    "feature_support_status": "observed_policy_long_effective_dated",
                    "selection_priority": selection_priority,
                }
            )
        if redeterm_doc_method_online is not None:
            features.append(
                {
                    "state_fips": state_fips,
                    "year": int(year),
                    "feature_name": "redetermination_online_submission_available",
                    "feature_value_text": redeterm_doc_method_online,
                    "feature_value_numeric": None,
                    "raw_relpath": row.get("raw_relpath"),
                    "source_sheet": source_sheet,
                    "row_number": row.get("row_number"),
                    "feature_support_status": "observed_policy_long_effective_dated",
                    "selection_priority": selection_priority,
                }
            )
        if waitlist_active is not None:
            features.append(
                {
                    "state_fips": state_fips,
                    "year": int(year),
                    "feature_name": "waitlist_active",
                    "feature_value_text": waitlist_active,
                    "feature_value_numeric": None,
                    "raw_relpath": row.get("raw_relpath"),
                    "source_sheet": source_sheet,
                    "row_number": row.get("row_number"),
                    "feature_support_status": "observed_policy_long_effective_dated",
                    "selection_priority": selection_priority,
                }
            )
        if copay_poverty_exempt is not None:
            features.append(
                {
                    "state_fips": state_fips,
                    "year": int(year),
                    "feature_name": "copay_poverty_exempt",
                    "feature_value_text": copay_poverty_exempt,
                    "feature_value_numeric": None,
                    "raw_relpath": row.get("raw_relpath"),
                    "source_sheet": source_sheet,
                    "row_number": row.get("row_number"),
                    "feature_support_status": "observed_policy_long_effective_dated",
                    "selection_priority": selection_priority,
                }
            )
        if copay_tanf_exempt is not None:
            features.append(
                {
                    "state_fips": state_fips,
                    "year": int(year),
                    "feature_name": "copay_tanf_exempt",
                    "feature_value_text": copay_tanf_exempt,
                    "feature_value_numeric": None,
                    "raw_relpath": row.get("raw_relpath"),
                    "source_sheet": source_sheet,
                    "row_number": row.get("row_number"),
                    "feature_support_status": "observed_policy_long_effective_dated",
                    "selection_priority": selection_priority,
                }
            )
    return features


def _find_metric(
    row: dict[str, Any],
    tokens: tuple[str, ...],
    *,
    exclude_tokens: tuple[str, ...] = (),
) -> float | None:
    candidates: list[float | None] = []
    for key, value in row.items():
        normalized_key = _normalize_label(key)
        if exclude_tokens and any(token in normalized_key for token in exclude_tokens):
            continue
        if any(token in normalized_key for token in tokens):
            candidates.append(row.get(f"{key}__numeric"))
            candidates.append(value)
    return _first_numeric(candidates)


def _normalize_share(value: float | None) -> float | None:
    if value is None:
        return None
    share = float(value)
    if share < 0:
        return None
    if share > 1.0 and share <= 100.0:
        share = share / 100.0
    if share < 0.0 or share > 1.0:
        return None
    return share


def _resolve_admin_split_support(
    *,
    children_served: float | None,
    subsidized_private_slots: float | None,
    public_admin_slots: float | None,
) -> tuple[float | None, float | None, str, str]:
    if subsidized_private_slots is not None and public_admin_slots is not None:
        return (
            subsidized_private_slots,
            public_admin_slots,
            "ccdf_explicit_split_observed",
            "observed_long_explicit_split",
        )
    if subsidized_private_slots is None and public_admin_slots is None:
        if children_served is not None:
            return (
                children_served,
                0.0,
                "ccdf_children_served_proxy_for_subsidized_private",
                "observed_long_proxy_subsidized_private",
            )
        return (None, None, "ccdf_metric_missing", "state_year_detected_metric_missing")

    if children_served is None:
        return (
            subsidized_private_slots,
            public_admin_slots,
            "ccdf_partial_split_missing_children_served",
            "observed_long_partial_split_missing_children_served",
        )

    if subsidized_private_slots is not None and public_admin_slots is None:
        inferred_public_admin = children_served - subsidized_private_slots
        if inferred_public_admin >= 0:
            return (
                subsidized_private_slots,
                inferred_public_admin,
                "ccdf_inferred_public_admin_from_children_served_minus_subsidized_private",
                "observed_long_inferred_public_admin_complement",
            )
        return (
            subsidized_private_slots,
            None,
            "ccdf_partial_split_inconsistent_children_served",
            "observed_long_partial_split_inconsistent_children_served",
        )

    inferred_subsidized_private = children_served - float(public_admin_slots)
    if inferred_subsidized_private >= 0:
        return (
            inferred_subsidized_private,
            public_admin_slots,
            "ccdf_inferred_subsidized_private_from_children_served_minus_public_admin",
            "observed_long_inferred_subsidized_private_complement",
        )
    return (
        None,
        public_admin_slots,
        "ccdf_partial_split_inconsistent_children_served",
        "observed_long_partial_split_inconsistent_children_served",
    )


def _classify_payment_method_proxy_support(
    *,
    payment_method_ratio: float | None,
    payment_method_gap: float | None,
) -> tuple[str, str]:
    if payment_method_ratio is None or payment_method_gap is None:
        return (
            "ccdf_split_proxy_from_payment_method_shares_unknown_gap",
            "observed_long_payment_method_share_proxy_unknown_gap",
        )
    ratio_gap = abs(float(payment_method_ratio) - 1.0)
    absolute_gap = abs(float(payment_method_gap))
    if ratio_gap <= PAYMENT_PROXY_CLOSE_RATIO_GAP_MAX and absolute_gap <= PAYMENT_PROXY_CLOSE_ABS_GAP_MAX:
        return (
            "ccdf_split_proxy_from_payment_method_shares_close_gap",
            "observed_long_payment_method_share_proxy_close_gap",
        )
    if ratio_gap <= PAYMENT_PROXY_MODERATE_RATIO_GAP_MAX and absolute_gap <= PAYMENT_PROXY_MODERATE_ABS_GAP_MAX:
        return (
            "ccdf_split_proxy_from_payment_method_shares_moderate_gap",
            "observed_long_payment_method_share_proxy_moderate_gap",
        )
    return (
        PAYMENT_PROXY_LARGE_GAP_SUPPORT_FLAG,
        PAYMENT_PROXY_LARGE_GAP_SUPPORT_STATUS,
    )


def _resolve_payment_method_boundary_split_support(
    *,
    children_served: float | None,
    grants_contracts_share: float | None,
    certificates_share: float | None,
    cash_share: float | None,
) -> tuple[float, float, str, str] | None:
    if children_served is None:
        return None
    if grants_contracts_share is None and certificates_share is None and cash_share is None:
        return None

    grants = float(grants_contracts_share or 0.0)
    private_share = float(certificates_share or 0.0) + float(cash_share or 0.0)
    if abs(grants) <= PAYMENT_METHOD_BOUNDARY_SHARE_EPS and abs(private_share - 1.0) <= PAYMENT_METHOD_BOUNDARY_SHARE_EPS:
        return (
            max(float(children_served), 0.0),
            0.0,
            "ccdf_inferred_zero_public_admin_from_payment_method_mix",
            "observed_long_inferred_zero_public_admin_payment_method_mix",
        )
    if abs(grants - 1.0) <= PAYMENT_METHOD_BOUNDARY_SHARE_EPS and abs(private_share) <= PAYMENT_METHOD_BOUNDARY_SHARE_EPS:
        return (
            0.0,
            max(float(children_served), 0.0),
            "ccdf_inferred_zero_subsidized_private_from_payment_method_mix",
            "observed_long_inferred_zero_subsidized_private_payment_method_mix",
        )
    return None


def _first_non_null(values: pd.Series) -> object:
    non_null = values.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def _max_non_null(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.max())


def _quantity_source_priority(raw_relpath: object) -> tuple[int, str]:
    path_text = str(raw_relpath or "").lower()
    if "ccdf-data-tables" in path_text or "ccdf_data_tables" in path_text:
        if "final" in path_text:
            return (0, path_text)
        if "preliminary" in path_text:
            return (1, path_text)
        return (2, path_text)
    return (9, path_text)


def _expenditure_source_priority(raw_relpath: object) -> tuple[int, str]:
    path_text = str(raw_relpath or "").lower()
    if "expenditures" in path_text:
        return (0, path_text)
    if "ccdf-data-tables" in path_text or "ccdf_data_tables" in path_text:
        if "final" in path_text:
            return (1, path_text)
        if "preliminary" in path_text:
            return (2, path_text)
    return (9, path_text)


def build_ccdf_admin_state_year(admin_long: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(ADMIN_REQUIRED_COLUMNS - set(admin_long.columns))
    if missing:
        raise KeyError(f"CCDF admin long frame missing required columns: {', '.join(missing)}")
    working = admin_long.loc[admin_long["source_component"].astype(str).eq("admin")].copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "ccdf_children_served",
                "ccdf_total_expenditures",
                "ccdf_admin_expenditures",
                "ccdf_quality_activities_expenditures",
                "ccdf_infant_toddler_quality_expenditures",
                "ccdf_direct_services_expenditures",
                "ccdf_nondirect_systems_expenditures",
                "ccdf_nondirect_cert_prog_eligibility_expenditures",
                "ccdf_nondirect_other_expenditures",
                "ccdf_subsidized_private_slots",
                "ccdf_public_admin_slots",
                "ccdf_grants_contracts_share",
                "ccdf_certificates_share",
                "ccdf_cash_share",
                "ccdf_payment_method_total_children",
                "ccdf_payment_method_gap_vs_children_served",
                "ccdf_payment_method_ratio_vs_children_served",
                "ccdf_care_type_child_home_share",
                "ccdf_care_type_family_home_share",
                "ccdf_care_type_group_home_share",
                "ccdf_care_type_center_share",
                "ccdf_care_type_invalid_share",
                "ccdf_regulated_share",
                "ccdf_unregulated_share",
                "ccdf_regulation_invalid_share",
                "ccdf_relative_care_total_children",
                "ccdf_relative_care_relative_children",
                "ccdf_relative_care_nonrelative_children",
                "ccdf_unregulated_relative_share",
                "ccdf_unregulated_nonrelative_share",
                "ccdf_admin_rows_mapped",
                "ccdf_children_served_support_status",
                "ccdf_expenditures_support_status",
                "ccdf_admin_support_status",
                "ccdf_care_type_support_status",
                "ccdf_regulation_support_status",
            ]
        )

    normalized_rows: list[dict[str, Any]] = []
    for row in _pivot_long_rows(working):
        state_fips, year = _extract_state_year(row)
        if not state_fips or year is None:
            continue
        children_served = _find_metric(row, CHILDREN_SERVED_TOKENS)
        total_expenditures = _find_metric(row, EXPENDITURE_TOKENS)
        admin_expenditures = _find_metric(row, ADMIN_EXPENDITURE_TOKENS)
        quality_activities_expenditures = _find_metric(row, QUALITY_EXPENDITURE_TOKENS)
        infant_toddler_quality_expenditures = _find_metric(row, INFANT_TODDLER_QUALITY_EXPENDITURE_TOKENS)
        direct_services_expenditures = _find_metric(row, DIRECT_SERVICES_EXPENDITURE_TOKENS)
        nondirect_systems_expenditures = _find_metric(row, NONDIRECT_SYSTEMS_EXPENDITURE_TOKENS)
        nondirect_cert_prog_eligibility_expenditures = _find_metric(
            row,
            NONDIRECT_ELIGIBILITY_EXPENDITURE_TOKENS,
        )
        nondirect_other_expenditures = _find_metric(row, NONDIRECT_OTHER_EXPENDITURE_TOKENS)
        subsidized_private_slots = _find_metric(
            row,
            SUBSIDIZED_PRIVATE_TOKENS,
            exclude_tokens=("share", "percent", "percentage"),
        )
        public_admin_slots = _find_metric(
            row,
            PUBLIC_ADMIN_TOKENS,
            exclude_tokens=("share", "percent", "percentage"),
        )
        subsidized_private_share = _normalize_share(_find_metric(row, SUBSIDIZED_PRIVATE_SHARE_TOKENS))
        public_admin_share = _normalize_share(_find_metric(row, PUBLIC_ADMIN_SHARE_TOKENS))
        grants_contracts_share = _normalize_share(_find_metric(row, ("grants_contracts_percent",)))
        certificates_share = _normalize_share(_find_metric(row, ("certificates_percent",)))
        cash_share = _normalize_share(_find_metric(row, ("cash_percent",)))
        payment_method_total_children = _find_metric(row, ("payment_method_total_children",))
        care_type_child_home_share = _normalize_share(_find_metric(row, CARE_TYPE_CHILD_HOME_TOKENS))
        care_type_family_home_share = _normalize_share(_find_metric(row, CARE_TYPE_FAMILY_HOME_TOKENS))
        care_type_group_home_share = _normalize_share(_find_metric(row, CARE_TYPE_GROUP_HOME_TOKENS))
        care_type_center_share = _normalize_share(_find_metric(row, CARE_TYPE_CENTER_TOKENS))
        care_type_invalid_share = _normalize_share(_find_metric(row, CARE_TYPE_INVALID_TOKENS))
        regulated_share = _normalize_share(_find_metric(row, REGULATED_SHARE_TOKENS))
        unregulated_share = _normalize_share(_find_metric(row, UNREGULATED_SHARE_TOKENS))
        regulation_invalid_share = _normalize_share(_find_metric(row, REGULATION_INVALID_SHARE_TOKENS))
        relative_care_total_children = _find_metric(row, RELATIVE_CARE_TOTAL_CHILDREN_TOKENS)
        relative_care_relative_children = _find_metric(row, RELATIVE_CARE_RELATIVE_CHILDREN_TOKENS)
        relative_care_nonrelative_children = _find_metric(row, RELATIVE_CARE_NONRELATIVE_CHILDREN_TOKENS)
        unregulated_relative_share = _normalize_share(_find_metric(row, UNREGULATED_RELATIVE_SHARE_TOKENS))
        unregulated_nonrelative_share = _normalize_share(_find_metric(row, UNREGULATED_NONRELATIVE_SHARE_TOKENS))
        split_proxy_source = _normalize_label(row.get("split_proxy_source"))

        if children_served is not None:
            if subsidized_private_slots is None and subsidized_private_share is not None:
                subsidized_private_slots = children_served * subsidized_private_share
            if public_admin_slots is None and public_admin_share is not None:
                public_admin_slots = children_served * public_admin_share
            if relative_care_total_children is not None and unregulated_share is None:
                unregulated_share = _normalize_share(relative_care_total_children / children_served)
            if relative_care_relative_children is not None and unregulated_relative_share is None:
                derived_relative_share = _normalize_share(relative_care_relative_children / children_served)
                if derived_relative_share is not None:
                    unregulated_relative_share = derived_relative_share
            if relative_care_nonrelative_children is not None and unregulated_nonrelative_share is None:
                derived_nonrelative_share = _normalize_share(relative_care_nonrelative_children / children_served)
                if derived_nonrelative_share is not None:
                    unregulated_nonrelative_share = derived_nonrelative_share

        normalized_rows.append(
            {
                "raw_relpath": row.get("raw_relpath"),
                "state_fips": state_fips,
                "year": year,
                "ccdf_children_served": children_served,
                "ccdf_total_expenditures": total_expenditures,
                "ccdf_admin_expenditures": admin_expenditures,
                "ccdf_quality_activities_expenditures": quality_activities_expenditures,
                "ccdf_infant_toddler_quality_expenditures": infant_toddler_quality_expenditures,
                "ccdf_direct_services_expenditures": direct_services_expenditures,
                "ccdf_nondirect_systems_expenditures": nondirect_systems_expenditures,
                "ccdf_nondirect_cert_prog_eligibility_expenditures": nondirect_cert_prog_eligibility_expenditures,
                "ccdf_nondirect_other_expenditures": nondirect_other_expenditures,
                "ccdf_subsidized_private_slots": subsidized_private_slots,
                "ccdf_public_admin_slots": public_admin_slots,
                "ccdf_grants_contracts_share": grants_contracts_share,
                "ccdf_certificates_share": certificates_share,
                "ccdf_cash_share": cash_share,
                "ccdf_payment_method_total_children": payment_method_total_children,
                "subsidized_private_share": subsidized_private_share,
                "public_admin_share": public_admin_share,
                "ccdf_care_type_child_home_share": care_type_child_home_share,
                "ccdf_care_type_family_home_share": care_type_family_home_share,
                "ccdf_care_type_group_home_share": care_type_group_home_share,
                "ccdf_care_type_center_share": care_type_center_share,
                "ccdf_care_type_invalid_share": care_type_invalid_share,
                "ccdf_regulated_share": regulated_share,
                "ccdf_unregulated_share": unregulated_share,
                "ccdf_regulation_invalid_share": regulation_invalid_share,
                "ccdf_relative_care_total_children": relative_care_total_children,
                "ccdf_relative_care_relative_children": relative_care_relative_children,
                "ccdf_relative_care_nonrelative_children": relative_care_nonrelative_children,
                "ccdf_unregulated_relative_share": unregulated_relative_share,
                "ccdf_unregulated_nonrelative_share": unregulated_nonrelative_share,
                "split_proxy_source": split_proxy_source if split_proxy_source else pd.NA,
            }
        )
    if not normalized_rows:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "ccdf_children_served",
                "ccdf_total_expenditures",
                "ccdf_admin_expenditures",
                "ccdf_quality_activities_expenditures",
                "ccdf_infant_toddler_quality_expenditures",
                "ccdf_direct_services_expenditures",
                "ccdf_nondirect_systems_expenditures",
                "ccdf_nondirect_cert_prog_eligibility_expenditures",
                "ccdf_nondirect_other_expenditures",
                "ccdf_subsidized_private_slots",
                "ccdf_public_admin_slots",
                "ccdf_grants_contracts_share",
                "ccdf_certificates_share",
                "ccdf_cash_share",
                "ccdf_payment_method_total_children",
                "ccdf_payment_method_gap_vs_children_served",
                "ccdf_payment_method_ratio_vs_children_served",
                "ccdf_care_type_child_home_share",
                "ccdf_care_type_family_home_share",
                "ccdf_care_type_group_home_share",
                "ccdf_care_type_center_share",
                "ccdf_care_type_invalid_share",
                "ccdf_regulated_share",
                "ccdf_unregulated_share",
                "ccdf_regulation_invalid_share",
                "ccdf_relative_care_total_children",
                "ccdf_relative_care_relative_children",
                "ccdf_relative_care_nonrelative_children",
                "ccdf_unregulated_relative_share",
                "ccdf_unregulated_nonrelative_share",
                "ccdf_admin_rows_mapped",
                "ccdf_children_served_support_status",
                "ccdf_expenditures_support_status",
                "ccdf_admin_support_status",
                "ccdf_support_flag",
                "ccdf_care_type_support_status",
                "ccdf_regulation_support_status",
            ]
        )
    frame = pd.DataFrame(normalized_rows)
    per_source = (
        frame.groupby(["raw_relpath", "state_fips", "year"], as_index=False)
        .agg(
            ccdf_children_served=("ccdf_children_served", _max_non_null),
            ccdf_total_expenditures=("ccdf_total_expenditures", _max_non_null),
            ccdf_admin_expenditures=("ccdf_admin_expenditures", _max_non_null),
            ccdf_quality_activities_expenditures=("ccdf_quality_activities_expenditures", _max_non_null),
            ccdf_infant_toddler_quality_expenditures=("ccdf_infant_toddler_quality_expenditures", _max_non_null),
            ccdf_direct_services_expenditures=("ccdf_direct_services_expenditures", _max_non_null),
            ccdf_nondirect_systems_expenditures=("ccdf_nondirect_systems_expenditures", _max_non_null),
            ccdf_nondirect_cert_prog_eligibility_expenditures=(
                "ccdf_nondirect_cert_prog_eligibility_expenditures",
                _max_non_null,
            ),
            ccdf_nondirect_other_expenditures=("ccdf_nondirect_other_expenditures", _max_non_null),
            ccdf_subsidized_private_slots=("ccdf_subsidized_private_slots", _max_non_null),
            ccdf_public_admin_slots=("ccdf_public_admin_slots", _max_non_null),
            ccdf_grants_contracts_share=("ccdf_grants_contracts_share", _max_non_null),
            ccdf_certificates_share=("ccdf_certificates_share", _max_non_null),
            ccdf_cash_share=("ccdf_cash_share", _max_non_null),
            ccdf_payment_method_total_children=("ccdf_payment_method_total_children", _max_non_null),
            subsidized_private_share=("subsidized_private_share", _max_non_null),
            public_admin_share=("public_admin_share", _max_non_null),
            ccdf_care_type_child_home_share=("ccdf_care_type_child_home_share", _max_non_null),
            ccdf_care_type_family_home_share=("ccdf_care_type_family_home_share", _max_non_null),
            ccdf_care_type_group_home_share=("ccdf_care_type_group_home_share", _max_non_null),
            ccdf_care_type_center_share=("ccdf_care_type_center_share", _max_non_null),
            ccdf_care_type_invalid_share=("ccdf_care_type_invalid_share", _max_non_null),
            ccdf_regulated_share=("ccdf_regulated_share", _max_non_null),
            ccdf_unregulated_share=("ccdf_unregulated_share", _max_non_null),
            ccdf_regulation_invalid_share=("ccdf_regulation_invalid_share", _max_non_null),
            ccdf_relative_care_total_children=("ccdf_relative_care_total_children", _max_non_null),
            ccdf_relative_care_relative_children=("ccdf_relative_care_relative_children", _max_non_null),
            ccdf_relative_care_nonrelative_children=("ccdf_relative_care_nonrelative_children", _max_non_null),
            ccdf_unregulated_relative_share=("ccdf_unregulated_relative_share", _max_non_null),
            ccdf_unregulated_nonrelative_share=("ccdf_unregulated_nonrelative_share", _max_non_null),
            split_proxy_source=("split_proxy_source", _first_non_null),
        )
    )

    quantity_candidates = per_source.loc[
        per_source[
            [
                "ccdf_children_served",
                "ccdf_subsidized_private_slots",
                "ccdf_public_admin_slots",
                "ccdf_grants_contracts_share",
                "ccdf_certificates_share",
                "ccdf_cash_share",
                "ccdf_payment_method_total_children",
                "subsidized_private_share",
                "public_admin_share",
                "ccdf_care_type_child_home_share",
                "ccdf_care_type_family_home_share",
                "ccdf_care_type_group_home_share",
                "ccdf_care_type_center_share",
                "ccdf_care_type_invalid_share",
                "ccdf_regulated_share",
                "ccdf_unregulated_share",
                "ccdf_regulation_invalid_share",
                "ccdf_relative_care_total_children",
                "ccdf_relative_care_relative_children",
                "ccdf_relative_care_nonrelative_children",
                "ccdf_unregulated_relative_share",
                "ccdf_unregulated_nonrelative_share",
            ]
        ]
        .notna()
        .any(axis=1)
    ].copy()
    quantity_candidates["source_priority"] = quantity_candidates["raw_relpath"].map(_quantity_source_priority)
    quantity_selected = (
        quantity_candidates.sort_values(["state_fips", "year", "source_priority"], kind="stable")
        .drop_duplicates(["state_fips", "year"], keep="first")
        .reset_index(drop=True)
    )

    expenditure_candidates = per_source.loc[per_source["ccdf_total_expenditures"].notna()].copy()
    expenditure_candidates["source_priority"] = expenditure_candidates["raw_relpath"].map(_expenditure_source_priority)
    expenditure_selected = (
        expenditure_candidates.sort_values(["state_fips", "year", "source_priority"], kind="stable")
        .drop_duplicates(["state_fips", "year"], keep="first")
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    for keys in sorted(
        set(zip(quantity_selected["state_fips"], quantity_selected["year"]))
        | set(zip(expenditure_selected["state_fips"], expenditure_selected["year"]))
    ):
        state_fips, year = keys
        quantity_row = quantity_selected.loc[
            quantity_selected["state_fips"].eq(state_fips) & quantity_selected["year"].eq(year)
        ]
        expenditure_row = expenditure_selected.loc[
            expenditure_selected["state_fips"].eq(state_fips) & expenditure_selected["year"].eq(year)
        ]
        quantity_base = quantity_row.iloc[0].to_dict() if not quantity_row.empty else {}
        expenditure_base = expenditure_row.iloc[0].to_dict() if not expenditure_row.empty else {}

        children_served = quantity_base.get("ccdf_children_served")
        subsidized_private_slots = quantity_base.get("ccdf_subsidized_private_slots")
        public_admin_slots = quantity_base.get("ccdf_public_admin_slots")
        grants_contracts_share = _normalize_share(quantity_base.get("ccdf_grants_contracts_share"))
        certificates_share = _normalize_share(quantity_base.get("ccdf_certificates_share"))
        cash_share = _normalize_share(quantity_base.get("ccdf_cash_share"))
        payment_method_total_children = quantity_base.get("ccdf_payment_method_total_children")
        subsidized_private_share = _normalize_share(quantity_base.get("subsidized_private_share"))
        public_admin_share = _normalize_share(quantity_base.get("public_admin_share"))
        unregulated_share = _normalize_share(quantity_base.get("ccdf_unregulated_share"))
        unregulated_relative_share = _normalize_share(quantity_base.get("ccdf_unregulated_relative_share"))
        unregulated_nonrelative_share = _normalize_share(
            quantity_base.get("ccdf_unregulated_nonrelative_share")
        )
        relative_care_total_children = quantity_base.get("ccdf_relative_care_total_children")
        relative_care_relative_children = quantity_base.get("ccdf_relative_care_relative_children")
        relative_care_nonrelative_children = quantity_base.get("ccdf_relative_care_nonrelative_children")
        if children_served is not None:
            if subsidized_private_slots is None and subsidized_private_share is not None:
                subsidized_private_slots = float(children_served) * subsidized_private_share
            if public_admin_slots is None and public_admin_share is not None:
                public_admin_slots = float(children_served) * public_admin_share
            # Relative-care counts often arrive on a separate sheet from children served.
            if relative_care_total_children is not None and unregulated_share is None:
                unregulated_share = _normalize_share(float(relative_care_total_children) / float(children_served))
            if relative_care_relative_children is not None and unregulated_relative_share is None:
                unregulated_relative_share = _normalize_share(
                    float(relative_care_relative_children) / float(children_served)
                )
            if relative_care_nonrelative_children is not None and unregulated_nonrelative_share is None:
                unregulated_nonrelative_share = _normalize_share(
                    float(relative_care_nonrelative_children) / float(children_served)
                )

        metrics_present = any(
            value is not None
            for value in (
                children_served,
                expenditure_base.get("ccdf_total_expenditures"),
                expenditure_base.get("ccdf_admin_expenditures"),
                expenditure_base.get("ccdf_quality_activities_expenditures"),
                expenditure_base.get("ccdf_infant_toddler_quality_expenditures"),
                expenditure_base.get("ccdf_direct_services_expenditures"),
                expenditure_base.get("ccdf_nondirect_systems_expenditures"),
                expenditure_base.get("ccdf_nondirect_cert_prog_eligibility_expenditures"),
                expenditure_base.get("ccdf_nondirect_other_expenditures"),
                subsidized_private_slots,
                public_admin_slots,
                subsidized_private_share,
                public_admin_share,
            )
        )
        care_type_metrics_present = any(
            quantity_base.get(column) is not None
            for column in (
                "ccdf_care_type_child_home_share",
                "ccdf_care_type_family_home_share",
                "ccdf_care_type_group_home_share",
                "ccdf_care_type_center_share",
                "ccdf_care_type_invalid_share",
            )
        )
        regulation_metrics_present = any(
            quantity_base.get(column) is not None
            for column in (
                "ccdf_regulated_share",
                "ccdf_unregulated_share",
                "ccdf_regulation_invalid_share",
                "ccdf_unregulated_relative_share",
                "ccdf_unregulated_nonrelative_share",
            )
        )
        (
            resolved_subsidized_private_slots,
            resolved_public_admin_slots,
            support_flag,
            admin_support_status,
        ) = _resolve_admin_split_support(
            children_served=children_served,
            subsidized_private_slots=subsidized_private_slots,
            public_admin_slots=public_admin_slots,
        )
        if (
            children_served is not None
            and (
                (subsidized_private_share is not None and public_admin_share is not None)
                or (
                    subsidized_private_share is not None
                    and public_admin_slots is not None
                    and _normalize_share(public_admin_slots / children_served if children_served else None) is not None
                )
                or (
                    public_admin_share is not None
                    and subsidized_private_slots is not None
                    and _normalize_share(subsidized_private_slots / children_served if children_served else None) is not None
                )
            )
            and support_flag == "ccdf_explicit_split_observed"
        ):
            support_flag = "ccdf_split_observed_from_shares_or_mixed_components"
            admin_support_status = "observed_long_split_from_shares_or_mixed_components"
        payment_method_gap = None
        payment_method_ratio = None
        if children_served is not None and payment_method_total_children is not None:
            payment_method_gap = float(payment_method_total_children) - float(children_served)
            if float(children_served) != 0.0:
                payment_method_ratio = float(payment_method_total_children) / float(children_served)
        if _normalize_label(quantity_base.get("split_proxy_source")) == "payment_method_shares" and children_served is not None:
            support_flag, admin_support_status = _classify_payment_method_proxy_support(
                payment_method_ratio=payment_method_ratio,
                payment_method_gap=payment_method_gap,
            )
            if support_flag == PAYMENT_PROXY_LARGE_GAP_SUPPORT_FLAG:
                # Large payment-method gaps are treated as weak evidence; avoid split-proxy behavior.
                resolved_subsidized_private_slots = max(float(children_served), 0.0)
                resolved_public_admin_slots = 0.0
                support_flag = PAYMENT_PROXY_LARGE_GAP_DOWNGRADED_SUPPORT_FLAG
                admin_support_status = PAYMENT_PROXY_LARGE_GAP_DOWNGRADED_SUPPORT_STATUS
            else:
                boundary_support = _resolve_payment_method_boundary_split_support(
                    children_served=children_served,
                    grants_contracts_share=grants_contracts_share,
                    certificates_share=certificates_share,
                    cash_share=cash_share,
                )
                if boundary_support is not None:
                    (
                        resolved_subsidized_private_slots,
                        resolved_public_admin_slots,
                        support_flag,
                        admin_support_status,
                    ) = boundary_support

        rows.append(
            {
                "state_fips": state_fips,
                "year": year,
                "ccdf_children_served": float(children_served) if children_served is not None else 0.0,
                "ccdf_total_expenditures": float(expenditure_base.get("ccdf_total_expenditures"))
                if expenditure_base.get("ccdf_total_expenditures") is not None
                else 0.0,
                "ccdf_admin_expenditures": float(expenditure_base.get("ccdf_admin_expenditures"))
                if expenditure_base.get("ccdf_admin_expenditures") is not None
                else 0.0,
                "ccdf_quality_activities_expenditures": float(
                    expenditure_base.get("ccdf_quality_activities_expenditures")
                )
                if expenditure_base.get("ccdf_quality_activities_expenditures") is not None
                else 0.0,
                "ccdf_infant_toddler_quality_expenditures": float(
                    expenditure_base.get("ccdf_infant_toddler_quality_expenditures")
                )
                if expenditure_base.get("ccdf_infant_toddler_quality_expenditures") is not None
                else 0.0,
                "ccdf_direct_services_expenditures": float(
                    expenditure_base.get("ccdf_direct_services_expenditures")
                )
                if expenditure_base.get("ccdf_direct_services_expenditures") is not None
                else 0.0,
                "ccdf_nondirect_systems_expenditures": float(
                    expenditure_base.get("ccdf_nondirect_systems_expenditures")
                )
                if expenditure_base.get("ccdf_nondirect_systems_expenditures") is not None
                else 0.0,
                "ccdf_nondirect_cert_prog_eligibility_expenditures": float(
                    expenditure_base.get("ccdf_nondirect_cert_prog_eligibility_expenditures")
                )
                if expenditure_base.get("ccdf_nondirect_cert_prog_eligibility_expenditures") is not None
                else 0.0,
                "ccdf_nondirect_other_expenditures": float(
                    expenditure_base.get("ccdf_nondirect_other_expenditures")
                )
                if expenditure_base.get("ccdf_nondirect_other_expenditures") is not None
                else 0.0,
                "ccdf_subsidized_private_slots": float(resolved_subsidized_private_slots)
                if resolved_subsidized_private_slots is not None
                else 0.0,
                "ccdf_public_admin_slots": float(resolved_public_admin_slots)
                if resolved_public_admin_slots is not None
                else 0.0,
                "ccdf_grants_contracts_share": float(grants_contracts_share) if grants_contracts_share is not None else 0.0,
                "ccdf_certificates_share": float(certificates_share) if certificates_share is not None else 0.0,
                "ccdf_cash_share": float(cash_share) if cash_share is not None else 0.0,
                "ccdf_payment_method_total_children": float(payment_method_total_children)
                if payment_method_total_children is not None
                else 0.0,
                "ccdf_payment_method_gap_vs_children_served": float(payment_method_gap)
                if payment_method_gap is not None
                else 0.0,
                "ccdf_payment_method_ratio_vs_children_served": float(payment_method_ratio)
                if payment_method_ratio is not None
                else 0.0,
                "ccdf_care_type_child_home_share": float(quantity_base.get("ccdf_care_type_child_home_share"))
                if quantity_base.get("ccdf_care_type_child_home_share") is not None
                else 0.0,
                "ccdf_care_type_family_home_share": float(quantity_base.get("ccdf_care_type_family_home_share"))
                if quantity_base.get("ccdf_care_type_family_home_share") is not None
                else 0.0,
                "ccdf_care_type_group_home_share": float(quantity_base.get("ccdf_care_type_group_home_share"))
                if quantity_base.get("ccdf_care_type_group_home_share") is not None
                else 0.0,
                "ccdf_care_type_center_share": float(quantity_base.get("ccdf_care_type_center_share"))
                if quantity_base.get("ccdf_care_type_center_share") is not None
                else 0.0,
                "ccdf_care_type_invalid_share": float(quantity_base.get("ccdf_care_type_invalid_share"))
                if quantity_base.get("ccdf_care_type_invalid_share") is not None
                else 0.0,
                "ccdf_regulated_share": float(quantity_base.get("ccdf_regulated_share"))
                if quantity_base.get("ccdf_regulated_share") is not None
                else 0.0,
                "ccdf_unregulated_share": float(unregulated_share)
                if unregulated_share is not None
                else 0.0,
                "ccdf_regulation_invalid_share": float(quantity_base.get("ccdf_regulation_invalid_share"))
                if quantity_base.get("ccdf_regulation_invalid_share") is not None
                else 0.0,
                "ccdf_relative_care_total_children": float(relative_care_total_children)
                if relative_care_total_children is not None
                else 0.0,
                "ccdf_relative_care_relative_children": float(relative_care_relative_children)
                if relative_care_relative_children is not None
                else 0.0,
                "ccdf_relative_care_nonrelative_children": float(relative_care_nonrelative_children)
                if relative_care_nonrelative_children is not None
                else 0.0,
                "ccdf_unregulated_relative_share": float(unregulated_relative_share)
                if unregulated_relative_share is not None
                else 0.0,
                "ccdf_unregulated_nonrelative_share": float(unregulated_nonrelative_share)
                if unregulated_nonrelative_share is not None
                else 0.0,
                "ccdf_admin_rows_mapped": int(bool(quantity_base)) + int(bool(expenditure_base)),
                "ccdf_children_served_support_status": "observed_long_mapped"
                if children_served is not None
                else "missing_metric",
                "ccdf_expenditures_support_status": "observed_long_mapped"
                if expenditure_base.get("ccdf_total_expenditures") is not None
                else "missing_metric",
                "ccdf_admin_support_status": admin_support_status
                if metrics_present
                else "state_year_detected_metric_missing",
                "ccdf_support_flag": support_flag if metrics_present else "ccdf_metric_missing",
                "ccdf_care_type_support_status": "observed_long_mapped"
                if care_type_metrics_present
                else "missing_metric",
                "ccdf_regulation_support_status": "observed_long_mapped"
                if regulation_metrics_present
                else "missing_metric",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "ccdf_children_served",
                "ccdf_total_expenditures",
                "ccdf_admin_expenditures",
                "ccdf_quality_activities_expenditures",
                "ccdf_infant_toddler_quality_expenditures",
                "ccdf_direct_services_expenditures",
                "ccdf_nondirect_systems_expenditures",
                "ccdf_nondirect_cert_prog_eligibility_expenditures",
                "ccdf_nondirect_other_expenditures",
                "ccdf_subsidized_private_slots",
                "ccdf_public_admin_slots",
                "ccdf_care_type_child_home_share",
                "ccdf_care_type_family_home_share",
                "ccdf_care_type_group_home_share",
                "ccdf_care_type_center_share",
                "ccdf_care_type_invalid_share",
                "ccdf_regulated_share",
                "ccdf_unregulated_share",
                "ccdf_regulation_invalid_share",
                "ccdf_unregulated_relative_share",
                "ccdf_unregulated_nonrelative_share",
                "ccdf_admin_rows_mapped",
                "ccdf_children_served_support_status",
                "ccdf_expenditures_support_status",
                "ccdf_admin_support_status",
                "ccdf_support_flag",
                "ccdf_care_type_support_status",
                "ccdf_regulation_support_status",
            ]
        )
    grouped = pd.DataFrame(rows)
    return grouped.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def _build_policy_feature_rows(policy_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in _pivot_long_rows(policy_long):
        curated = _extract_curated_policy_features(row)
        if curated:
            rows.extend(curated)
            continue
        if str(row.get("source_sheet", "")) != "__default__" and not (
            "policy_key" in row and "policy_value" in row
        ):
            continue
        state_fips, year = _extract_state_year(row)
        if not state_fips or year is None:
            continue
        if "policy_key" in row and "policy_value" in row:
            feature_name = _normalize_label(row.get("policy_key"))
            feature_value = row.get("policy_value")
            rows.append(
                {
                    "state_fips": state_fips,
                    "year": year,
                    "feature_name": feature_name,
                    "feature_value_text": None if feature_value is None else str(feature_value),
                    "feature_value_numeric": _to_numeric(feature_value),
                    "raw_relpath": row.get("raw_relpath"),
                    "source_sheet": row.get("source_sheet"),
                    "row_number": row.get("row_number"),
                    "feature_support_status": "observed_policy_long",
                    "selection_priority": 0,
                }
            )
            continue
        for key, value in row.items():
            normalized_key = _normalize_label(key)
            if normalized_key in {"raw_relpath", "source_sheet", "row_number", "table_year", "state_fips", "state", "state_name", "state_code", "state_abbreviation", "st", "year", "fiscal_year", "ffy", "report_year"}:
                continue
            if normalized_key.endswith("__numeric") or normalized_key.endswith("_numeric"):
                continue
            rows.append(
                {
                    "state_fips": state_fips,
                    "year": year,
                    "feature_name": normalized_key,
                    "feature_value_text": None if value is None else str(value),
                    "feature_value_numeric": row.get(f"{key}__numeric"),
                    "raw_relpath": row.get("raw_relpath"),
                    "source_sheet": row.get("source_sheet"),
                    "row_number": row.get("row_number"),
                    "feature_support_status": "observed_policy_long",
                    "selection_priority": 0,
                }
            )
    return pd.DataFrame(rows)


def _policy_feature_rows_from_policy_long(policy_long: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(POLICY_REQUIRED_COLUMNS - set(policy_long.columns))
    if missing:
        raise KeyError(f"CCDF policy long frame missing required columns: {', '.join(missing)}")
    working = policy_long.loc[policy_long["source_component"].astype(str).eq("policies")].copy()
    if working.empty:
        return pd.DataFrame()
    curated_columns = {
        "state",
        "state_fips",
        "state_name",
        "state_code",
        "state_abbreviation",
        "st",
        "year",
        "fiscal_year",
        "ffy",
        "report_year",
        "policy_key",
        "policy_value",
        "copayment_required",
        "copay_required",
        "family_copayment_required",
        "begindat",
        "beginmajority",
        "enddat",
        "endmajority",
        "familygroup",
        "family_group",
        "providertype",
        "provider_type",
        "providersubtype",
        "provider_subtype",
        "copaycollect",
        "copaymin",
        "redetermperiod",
        "redetermappdocnew",
        "redetermdocmethodonline",
        "waitlist",
        "copaypovertyexempt",
        "copaytanfexempt",
    }
    normalized_columns = working["column_name"].map(_normalize_label)
    sample_like_rows = working["source_sheet"].astype(str).eq("__default__")
    working = working.loc[
        sample_like_rows
        | normalized_columns.isin(curated_columns)
        | normalized_columns.str.endswith("__numeric")
    ].copy()
    if working.empty:
        return pd.DataFrame()
    return _build_policy_feature_rows(working)


def _canonical_policy_control_name(feature_name: object) -> str | None:
    normalized = _normalize_label(feature_name)
    for canonical_name, spec in POLICY_CONTROL_SPECS.items():
        if normalized in {_normalize_label(alias) for alias in spec["aliases"]}:
            return canonical_name
    return None


def _policy_features_state_year_from_feature_rows(feature_rows: pd.DataFrame) -> pd.DataFrame:
    if feature_rows is None or feature_rows.empty:
        return pd.DataFrame(columns=["state_fips", "year", "ccdf_policy_support_status", "ccdf_policy_feature_count"])

    features = feature_rows.copy()
    if "selection_priority" not in features.columns:
        features["selection_priority"] = 0
    features = features.sort_values(
        ["state_fips", "year", "feature_name", "selection_priority", "row_number"],
        kind="stable",
    )
    features = features.drop_duplicates(
        subset=["state_fips", "year", "feature_name", "feature_value_text"],
        keep="last",
    ).reset_index(drop=True)
    wide = (
        features.pivot_table(
            index=["state_fips", "year"],
            columns="feature_name",
            values="feature_value_text",
            aggfunc="last",
        )
        .reset_index()
    )
    wide.columns = [
        str(column) if not isinstance(column, tuple) else "_".join(str(item) for item in column if item)
        for column in wide.columns
    ]
    rename_map = {
        column: f"ccdf_policy_{_normalize_label(column)}"
        for column in wide.columns
        if column not in {"state_fips", "year"}
    }
    wide = wide.rename(columns=rename_map)
    counts = (
        features.groupby(["state_fips", "year"], as_index=False)
        .agg(ccdf_policy_feature_count=("feature_name", "nunique"))
    )
    result = wide.merge(counts, on=["state_fips", "year"], how="left")
    result["ccdf_policy_support_status"] = "observed_policy_long"
    return result.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def _policy_controls_state_year_from_feature_rows(feature_rows: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=[
            "state_fips",
            "year",
            "ccdf_policy_control_count",
            "ccdf_policy_control_support_status",
        ]
        + sorted(POLICY_CONTROL_SPECS)
    )
    if feature_rows is None or feature_rows.empty:
        return empty

    controls = feature_rows.copy()
    controls["canonical_control"] = controls["feature_name"].map(_canonical_policy_control_name)
    controls = controls.loc[controls["canonical_control"].notna()].copy()
    if controls.empty:
        return empty

    rows: list[dict[str, Any]] = []
    for (state_fips, year), group in controls.groupby(["state_fips", "year"], as_index=False, sort=True):
        record: dict[str, Any] = {"state_fips": state_fips, "year": year}
        control_count = 0
        for canonical_name, spec in POLICY_CONTROL_SPECS.items():
            subset = group.loc[group["canonical_control"].eq(canonical_name)].copy()
            if subset.empty:
                record[canonical_name] = pd.NA if spec["kind"] == "text" else float("nan")
                continue
            if "selection_priority" not in subset.columns:
                subset["selection_priority"] = 0
            subset = subset.sort_values(["selection_priority", "row_number", "feature_name"], kind="stable")
            numeric_values = subset["feature_value_numeric"].dropna()
            text_values = subset["feature_value_text"].dropna()
            value = (
                numeric_values.iloc[-1]
                if spec["kind"] == "numeric" and not numeric_values.empty
                else (text_values.iloc[-1] if not text_values.empty else pd.NA)
            )
            if pd.notna(value):
                control_count += 1
            record[canonical_name] = value
        record["ccdf_policy_control_count"] = int(control_count)
        record["ccdf_policy_control_support_status"] = "observed_policy_controls"
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def _policy_controls_coverage_from_feature_rows(feature_rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "control_name",
        "control_kind",
        "policy_state_year_rows",
        "observed_state_year_rows",
        "missing_state_year_rows",
        "state_year_coverage_rate",
        "observed_states",
        "observed_years",
        "coverage_support_status",
    ]
    if feature_rows is None or feature_rows.empty:
        return pd.DataFrame(columns=columns)

    state_years = feature_rows[["state_fips", "year"]].drop_duplicates().reset_index(drop=True)
    total_state_year_rows = int(len(state_years))
    controls = _policy_controls_state_year_from_feature_rows(feature_rows)
    rows: list[dict[str, Any]] = []
    for control_name, spec in POLICY_CONTROL_SPECS.items():
        if control_name in controls.columns:
            observed = controls.loc[controls[control_name].notna(), ["state_fips", "year"]].drop_duplicates()
        else:
            observed = state_years.iloc[0:0].copy()
        observed_state_year_rows = int(len(observed))
        if total_state_year_rows == 0:
            coverage_rate = float("nan")
            support_status = "no_policy_state_years"
        else:
            coverage_rate = observed_state_year_rows / total_state_year_rows
            if observed_state_year_rows == 0:
                support_status = "control_unobserved"
            elif observed_state_year_rows == total_state_year_rows:
                support_status = "full_state_year_coverage"
            else:
                support_status = "partial_state_year_coverage"
        rows.append(
            {
                "control_name": control_name,
                "control_kind": spec["kind"],
                "policy_state_year_rows": total_state_year_rows,
                "observed_state_year_rows": observed_state_year_rows,
                "missing_state_year_rows": max(total_state_year_rows - observed_state_year_rows, 0),
                "state_year_coverage_rate": coverage_rate,
                "observed_states": int(observed["state_fips"].nunique()),
                "observed_years": int(observed["year"].nunique()),
                "coverage_support_status": support_status,
            }
        )
    return pd.DataFrame(rows).sort_values(["control_name"], kind="stable").reset_index(drop=True)


def _policy_promoted_controls_state_year_from_feature_rows(
    feature_rows: pd.DataFrame,
    min_state_year_coverage: float,
) -> pd.DataFrame:
    threshold = float(min_state_year_coverage)
    controls = _policy_controls_state_year_from_feature_rows(feature_rows)
    columns = [
        "state_fips",
        "year",
        "ccdf_policy_control_count",
        "ccdf_policy_control_support_status",
        "ccdf_policy_promoted_control_rule",
        "ccdf_policy_promoted_min_state_year_coverage",
        "ccdf_policy_promoted_controls_selected",
    ]
    if controls.empty:
        return pd.DataFrame(columns=columns)

    coverage = _policy_controls_coverage_from_feature_rows(feature_rows)
    promoted_controls = (
        coverage.loc[
            pd.to_numeric(coverage["state_year_coverage_rate"], errors="coerce").ge(threshold),
            "control_name",
        ]
        .astype(str)
        .tolist()
    )
    promoted_controls = [name for name in promoted_controls if name in controls.columns]
    result = controls[["state_fips", "year"]].copy()
    for control_name in promoted_controls:
        result[control_name] = controls[control_name]
    if promoted_controls:
        promoted_count = controls[promoted_controls].notna().sum(axis=1).astype(int)
        promoted_status = promoted_count.gt(0).map(
            {
                True: "observed_policy_promoted_controls",
                False: "promoted_controls_selected_but_missing_state_year_values",
            }
        )
    else:
        promoted_count = pd.Series(0, index=controls.index, dtype="int64")
        promoted_status = pd.Series(
            ["no_controls_passed_promotion_rule"] * len(controls),
            index=controls.index,
            dtype="object",
        )
    result["ccdf_policy_control_count"] = promoted_count.values
    result["ccdf_policy_control_support_status"] = promoted_status.values
    result["ccdf_policy_promoted_control_rule"] = "state_year_coverage_gte_threshold"
    result["ccdf_policy_promoted_min_state_year_coverage"] = threshold
    result["ccdf_policy_promoted_controls_selected"] = ",".join(promoted_controls)
    order = columns + promoted_controls
    return result[order].sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def _policy_feature_audit_from_feature_rows(feature_rows: pd.DataFrame) -> pd.DataFrame:
    if feature_rows is None or feature_rows.empty:
        return pd.DataFrame(
            columns=[
                "state_fips",
                "year",
                "feature_name",
                "feature_value_text",
                "feature_value_numeric",
                "raw_relpath",
                "source_sheet",
                "row_number",
                "feature_support_status",
            ]
        )
    audit = feature_rows.copy()
    if "selection_priority" in audit.columns:
        audit = audit.drop(columns=["selection_priority"])
    return audit.sort_values(["state_fips", "year", "feature_name", "row_number"], kind="stable").reset_index(
        drop=True
    )


def build_ccdf_policy_features_state_year(policy_long: pd.DataFrame) -> pd.DataFrame:
    feature_rows = _policy_feature_rows_from_policy_long(policy_long)
    return _policy_features_state_year_from_feature_rows(feature_rows)


def build_ccdf_policy_controls_state_year(policy_long: pd.DataFrame) -> pd.DataFrame:
    feature_rows = _policy_feature_rows_from_policy_long(policy_long)
    return _policy_controls_state_year_from_feature_rows(feature_rows)


def build_ccdf_policy_controls_coverage(policy_long: pd.DataFrame) -> pd.DataFrame:
    feature_rows = _policy_feature_rows_from_policy_long(policy_long)
    return _policy_controls_coverage_from_feature_rows(feature_rows)


def build_ccdf_policy_promoted_controls_state_year(
    policy_long: pd.DataFrame,
    min_state_year_coverage: float = 0.75,
) -> pd.DataFrame:
    feature_rows = _policy_feature_rows_from_policy_long(policy_long)
    return _policy_promoted_controls_state_year_from_feature_rows(feature_rows, min_state_year_coverage)


def build_ccdf_policy_feature_audit(policy_long: pd.DataFrame) -> pd.DataFrame:
    feature_rows = _policy_feature_rows_from_policy_long(policy_long)
    return _policy_feature_audit_from_feature_rows(feature_rows)
