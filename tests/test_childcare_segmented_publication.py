from __future__ import annotations

import pandas as pd

from unpriced.childcare.segmented_publication import (
    build_segmented_publication_admin_sheet_targets,
    build_childcare_segmented_publication_outputs,
    build_segmented_publication_parser_action_plan,
    build_segmented_publication_channel_table,
    build_segmented_publication_fallback_table,
    build_segmented_publication_issue_breakdown,
    build_segmented_publication_parser_focus_areas,
    build_segmented_publication_priority_states,
    build_segmented_publication_priority_summary,
    build_segmented_publication_support_quality_summary,
)


def _headline_summary() -> dict[str, object]:
    return {
        "headline_alpha": 0.5,
        "state_year_count": 2,
        "fallback_state_year_count": 1,
        "fallback_state_count": 1,
        "promoted_control_observed_state_year_count": 1,
        "proxy_ccdf_state_year_count": 1,
        "public_admin_invariant_prices": True,
        "channel_price_response": {
            "private_unsubsidized": {
                "state_year_count": 2,
                "price_responsive": True,
                "mean_p_baseline": 97.5,
                "mean_p_alpha": 105.0,
                "mean_p_shadow_marginal": 112.5,
                "mean_p_alpha_pct_change_vs_baseline": 0.0769,
            },
            "private_subsidized": {
                "state_year_count": 2,
                "price_responsive": True,
                "mean_p_baseline": 90.0,
                "mean_p_alpha": 93.0,
                "mean_p_shadow_marginal": 95.5,
                "mean_p_alpha_pct_change_vs_baseline": 0.0333,
            },
            "public_admin": {
                "state_year_count": 2,
                "price_responsive": False,
                "mean_p_baseline": 75.0,
                "mean_p_alpha": 75.0,
                "mean_p_shadow_marginal": 75.0,
                "mean_p_alpha_pct_change_vs_baseline": 0.0,
            },
        },
    }


def _channel_response_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "solver_channel": "private_unsubsidized",
                "alpha": 0.5,
                "market_quantity_proxy": 20.0,
                "unpaid_quantity_proxy": 6.0,
                "price_responsive": True,
                "p_baseline": 100.0,
                "p_shadow_marginal": 120.0,
                "p_alpha": 110.0,
                "p_shadow_delta_vs_baseline": 20.0,
                "p_shadow_pct_change_vs_baseline": 0.2,
                "p_alpha_delta_vs_baseline": 10.0,
                "p_alpha_pct_change_vs_baseline": 0.1,
                "public_admin_price_invariant": False,
            },
            {
                "state_fips": "06",
                "year": 2021,
                "solver_channel": "public_admin",
                "alpha": 0.5,
                "market_quantity_proxy": 15.0,
                "unpaid_quantity_proxy": 0.0,
                "price_responsive": False,
                "p_baseline": 80.0,
                "p_shadow_marginal": 80.0,
                "p_alpha": 80.0,
                "p_shadow_delta_vs_baseline": 0.0,
                "p_shadow_pct_change_vs_baseline": 0.0,
                "p_alpha_delta_vs_baseline": 0.0,
                "p_alpha_pct_change_vs_baseline": 0.0,
                "public_admin_price_invariant": True,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "solver_channel": "private_subsidized",
                "alpha": 0.5,
                "market_quantity_proxy": 5.0,
                "unpaid_quantity_proxy": 1.0,
                "price_responsive": True,
                "p_baseline": 90.0,
                "p_shadow_marginal": 96.0,
                "p_alpha": 93.0,
                "p_shadow_delta_vs_baseline": 6.0,
                "p_shadow_pct_change_vs_baseline": 0.0667,
                "p_alpha_delta_vs_baseline": 3.0,
                "p_alpha_pct_change_vs_baseline": 0.0333,
                "public_admin_price_invariant": False,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "solver_channel": "private_unsubsidized",
                "alpha": 0.25,
                "market_quantity_proxy": 25.0,
                "unpaid_quantity_proxy": 4.0,
                "price_responsive": True,
                "p_baseline": 95.0,
                "p_shadow_marginal": 105.0,
                "p_alpha": 98.0,
                "p_shadow_delta_vs_baseline": 10.0,
                "p_shadow_pct_change_vs_baseline": 0.1053,
                "p_alpha_delta_vs_baseline": 3.0,
                "p_alpha_pct_change_vs_baseline": 0.0316,
                "public_admin_price_invariant": False,
            },
        ]
    )


def _fallback_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "headline_alpha": 0.5,
                "has_support_row": True,
                "any_segment_allocation_fallback": False,
                "any_private_allocation_fallback": False,
                "ccdf_support_flag": "explicit",
                "ccdf_admin_support_status": "explicit_admin_support",
                "q0_support_flag": "explicit_q0_support",
                "public_program_support_status": "supported",
                "promoted_control_observed": True,
                "proxy_ccdf_row_count": 0,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "headline_alpha": 0.5,
                "has_support_row": True,
                "any_segment_allocation_fallback": True,
                "any_private_allocation_fallback": True,
                "ccdf_support_flag": "proxy",
                "ccdf_admin_support_status": "proxy_admin_support",
                "q0_support_flag": "fallback_q0_support",
                "public_program_support_status": "ccdf_observed",
                "promoted_control_observed": False,
                "proxy_ccdf_row_count": 1,
            },
        ]
    )


def _ccdf_parse_inventory() -> pd.DataFrame:
    return pd.DataFrame(
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
    )


def _ccdf_admin_long() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"filename": "fy2023_table_1.xlsx", "source_sheet": "Table 1", "column_name": "state"},
            {"filename": "fy2023_table_1.xlsx", "source_sheet": "Table 1", "column_name": "children_served_average_monthly"},
        ]
    )


def _ccdf_admin_state_year() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2023,
                "ccdf_expenditures_support_status": "missing_metric",
            }
        ]
    )


def test_publication_channel_table_filters_to_headline_alpha():
    table = build_segmented_publication_channel_table(_channel_response_summary(), _headline_summary())

    assert len(table) == 3
    assert set(table["solver_channel"]) == {"private_unsubsidized", "private_subsidized", "public_admin"}
    assert table["alpha"].eq(0.5).all()


def test_publication_fallback_table_sorts_fallback_rows_first():
    table = build_segmented_publication_fallback_table(_fallback_summary())

    assert table.iloc[0]["state_fips"] == "48"
    assert bool(table.iloc[0]["any_segment_allocation_fallback"]) is True
    assert bool(table.iloc[1]["promoted_control_observed"]) is True
    assert table.iloc[0]["support_quality_tier"] == "proxy_or_fallback_support"
    assert table.iloc[1]["support_quality_tier"] == "explicit_split_support"


def test_support_quality_summary_counts_tiers_and_fallback_rows():
    summary = build_segmented_publication_support_quality_summary(
        build_segmented_publication_fallback_table(_fallback_summary())
    )

    proxy_row = summary.loc[summary["support_quality_tier"] == "proxy_or_fallback_support"].iloc[0]
    explicit_row = summary.loc[summary["support_quality_tier"] == "explicit_split_support"].iloc[0]

    assert int(proxy_row["state_year_count"]) == 1
    assert int(proxy_row["fallback_state_year_count"]) == 1
    assert int(proxy_row["proxy_ccdf_state_year_count"]) == 1
    assert int(explicit_row["promoted_control_observed_state_year_count"]) == 1


def test_priority_states_and_summary_rank_proxy_states_first():
    fallback = build_segmented_publication_fallback_table(_fallback_summary())
    support_quality = build_segmented_publication_support_quality_summary(fallback)
    priority_states = build_segmented_publication_priority_states(fallback)
    priority_summary = build_segmented_publication_priority_summary(support_quality, priority_states)

    assert priority_states.iloc[0]["state_fips"] == "48"
    assert priority_states.iloc[0]["primary_support_issue"] == "proxy_or_fallback_support"
    assert priority_states.iloc[1]["primary_support_issue"] == "stable_supported"
    assert priority_summary["weak_support_state_year_count"] == 1
    assert priority_summary["priority_state_count"] == 1
    assert priority_summary["priority_states"][0]["state_fips"] == "48"


def test_issue_breakdown_tracks_support_flags_and_counts():
    fallback = build_segmented_publication_fallback_table(_fallback_summary())
    issue_breakdown = build_segmented_publication_issue_breakdown(fallback)

    proxy_row = issue_breakdown.loc[issue_breakdown["support_quality_tier"] == "proxy_or_fallback_support"].iloc[0]
    explicit_row = issue_breakdown.loc[issue_breakdown["support_quality_tier"] == "explicit_split_support"].iloc[0]

    assert proxy_row["ccdf_support_flag"] == "proxy"
    assert int(proxy_row["state_year_count"]) == 1
    assert explicit_row["ccdf_admin_support_status"] == "explicit_admin_support"
    assert int(explicit_row["promoted_control_observed_state_year_count"]) == 1


def test_parser_focus_and_admin_targets_surface_missing_split_columns():
    fallback = build_segmented_publication_fallback_table(_fallback_summary())
    support_quality = build_segmented_publication_support_quality_summary(fallback)
    parser_focus = build_segmented_publication_parser_focus_areas(
        support_quality_summary=support_quality,
        ccdf_admin_long=_ccdf_admin_long(),
        ccdf_admin_state_year=_ccdf_admin_state_year(),
    )
    admin_targets = build_segmented_publication_admin_sheet_targets(
        parser_focus_areas=parser_focus,
        ccdf_parse_inventory=_ccdf_parse_inventory(),
        ccdf_admin_long=_ccdf_admin_long(),
    )

    high_priority = parser_focus.loc[parser_focus["priority_tier"] == "high", "focus_area"].tolist()
    assert "subsidized_private_split_columns" in high_priority
    assert "public_admin_split_columns" in high_priority
    assert bool(
        parser_focus.loc[parser_focus["focus_area"] == "children_served_columns", "current_signal_present"].iloc[0]
    ) is True
    assert "subsidized_private_split_columns" in admin_targets.iloc[0]["missing_priority_focus_areas"]


def test_parser_action_plan_ranks_high_priority_missing_signals_and_notes_limited_inventory():
    fallback = build_segmented_publication_fallback_table(_fallback_summary())
    support_quality = build_segmented_publication_support_quality_summary(fallback)
    priority_states = build_segmented_publication_priority_states(fallback)
    issue_breakdown = build_segmented_publication_issue_breakdown(fallback)
    parser_focus = build_segmented_publication_parser_focus_areas(
        support_quality_summary=support_quality,
        ccdf_admin_long=_ccdf_admin_long(),
        ccdf_admin_state_year=_ccdf_admin_state_year(),
    )
    admin_targets = build_segmented_publication_admin_sheet_targets(
        parser_focus_areas=parser_focus,
        ccdf_parse_inventory=_ccdf_parse_inventory(),
        ccdf_admin_long=_ccdf_admin_long(),
    )

    action_plan = build_segmented_publication_parser_action_plan(
        priority_states=priority_states,
        issue_breakdown=issue_breakdown,
        parser_focus_areas=parser_focus,
        admin_sheet_targets=admin_targets,
    )

    assert not action_plan.empty
    assert int(action_plan.iloc[0]["action_rank"]) == 1
    assert action_plan.iloc[0]["priority_tier"] == "high"
    assert bool(action_plan.iloc[0]["current_signal_present"]) is False
    assert action_plan.iloc[0]["inventory_scope"] == "sample_or_limited_inventory"
    assert "sample-sized or single-sheet" in action_plan.iloc[0]["inventory_scope_note"]
    assert "48" in action_plan.iloc[0]["top_priority_states"]


def test_publication_outputs_include_markdown_and_tables():
    outputs = build_childcare_segmented_publication_outputs(
        segmented_headline_summary=_headline_summary(),
        segmented_channel_response_summary=_channel_response_summary(),
        segmented_state_fallback_summary=_fallback_summary(),
        ccdf_parse_inventory=_ccdf_parse_inventory(),
        ccdf_admin_long=_ccdf_admin_long(),
        ccdf_admin_state_year=_ccdf_admin_state_year(),
    )

    assert set(outputs) == {
        "segmented_publication_headline_summary",
        "segmented_publication_channel_table",
        "segmented_publication_fallback_table",
        "segmented_publication_support_quality_summary",
        "segmented_publication_priority_states",
        "segmented_publication_priority_summary",
        "segmented_publication_issue_breakdown",
        "segmented_publication_parser_focus_areas",
        "segmented_publication_admin_sheet_targets",
        "segmented_publication_parser_action_plan",
        "segmented_publication_parser_action_plan_summary",
        "segmented_publication_readout_markdown",
        "segmented_publication_report_markdown",
    }
    assert not outputs["segmented_publication_channel_table"].empty
    assert not outputs["segmented_publication_fallback_table"].empty
    assert not outputs["segmented_publication_support_quality_summary"].empty
    assert not outputs["segmented_publication_priority_states"].empty
    assert isinstance(outputs["segmented_publication_priority_summary"], dict)
    assert not outputs["segmented_publication_issue_breakdown"].empty
    assert not outputs["segmented_publication_parser_focus_areas"].empty
    assert not outputs["segmented_publication_admin_sheet_targets"].empty
    assert not outputs["segmented_publication_parser_action_plan"].empty
    assert outputs["segmented_publication_parser_action_plan_summary"]["row_count"] > 0
    assert "# Childcare Segmented Headline Readout" in outputs["segmented_publication_readout_markdown"]
    assert "# childcare_segmented_report" in outputs["segmented_publication_report_markdown"]
    assert "## Support quality" in outputs["segmented_publication_readout_markdown"]
    assert "## Support quality summary" in outputs["segmented_publication_report_markdown"]
    assert "## Priority states" in outputs["segmented_publication_readout_markdown"]
    assert "## Parser-target priority states" in outputs["segmented_publication_report_markdown"]
    assert "## Support issue mix" in outputs["segmented_publication_readout_markdown"]
    assert "## Support issue breakdown" in outputs["segmented_publication_report_markdown"]
    assert "## Parser focus areas" in outputs["segmented_publication_readout_markdown"]
    assert "## Admin sheet targets" in outputs["segmented_publication_report_markdown"]
    assert "## Parser action plan" in outputs["segmented_publication_readout_markdown"]
    assert "## Parser action plan" in outputs["segmented_publication_report_markdown"]
