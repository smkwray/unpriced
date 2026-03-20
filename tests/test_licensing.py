from __future__ import annotations

import pandas as pd
import pytest

from unpriced.errors import SourceAccessError
from unpriced.ingest.licensing import ingest as ingest_licensing
from unpriced.storage import read_parquet


def test_licensing_sample_ingest_writes_expected_columns(project_paths):
    result = ingest_licensing(project_paths, sample=True, refresh=True)

    assert result.normalized_path.exists()
    frame = read_parquet(result.normalized_path)
    assert {
        "state_fips",
        "year",
        "center_infant_ratio",
        "center_toddler_ratio",
        "center_infant_group_size",
        "center_toddler_group_size",
        "shock_label",
    } <= set(frame.columns)


def test_licensing_real_ingest_normalizes_curated_csv(project_paths):
    raw_path = project_paths.raw / "licensing" / "licensing_supply_shocks.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "state_fips": "6",
                "year": 2021,
                "center_infant_ratio": 3,
                "center_toddler_ratio": 6,
                "center_infant_group_size": 6,
                "center_toddler_group_size": 12,
                "shock_label": "demo",
            }
        ]
    ).to_csv(raw_path, index=False)

    result = ingest_licensing(project_paths, sample=False, refresh=True)

    frame = read_parquet(result.normalized_path)
    assert result.normalized_path.name == "licensing_supply_shocks.parquet"
    assert frame.loc[0, "state_fips"] == "06"
    assert int(frame.loc[0, "year"]) == 2021


def test_licensing_real_ingest_accepts_optional_rule_level_file(project_paths):
    licensing_dir = project_paths.raw / "licensing"
    licensing_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "state_fips": "6",
                "year": 2021,
                "center_infant_ratio": 3,
                "center_toddler_ratio": 6,
                "center_infant_group_size": 6,
                "center_toddler_group_size": 12,
                "shock_label": "demo",
            }
        ]
    ).to_csv(licensing_dir / "licensing_supply_shocks.csv", index=False)
    pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "provider_type": "center",
                "age_group": "infant",
                "rule_family": "max_children_per_staff",
                "rule_value": 3,
                "source_note": "optional richer rule-level contract",
            }
        ]
    ).to_csv(licensing_dir / "licensing_rules_long.csv", index=False)

    result = ingest_licensing(project_paths, sample=False, refresh=True)

    frame = read_parquet(result.normalized_path)
    rules_long = read_parquet(project_paths.interim / "licensing" / "licensing_rules_long.parquet")
    assert result.normalized_path.name == "licensing_supply_shocks.parquet"
    assert frame.loc[0, "state_fips"] == "06"
    assert int(frame.loc[0, "year"]) == 2021
    assert not rules_long.empty
    assert rules_long.loc[0, "provider_type"] == "center"
    assert float(rules_long.loc[0, "rule_value"]) == 3.0


def test_licensing_real_ingest_accepts_rule_level_only_file(project_paths):
    licensing_dir = project_paths.raw / "licensing"
    licensing_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "state_fips": "6",
                "year": 2021,
                "provider_type": "center",
                "age_group": "infant",
                "rule_family": "max_children_per_staff",
                "rule_name": "center_infant_ratio",
                "rule_value": 3,
            }
        ]
    ).to_csv(licensing_dir / "licensing_rules_long.csv", index=False)

    result = ingest_licensing(project_paths, sample=False, refresh=True)

    assert result.normalized_path.name == "licensing_rules_long.parquet"
    rules_long = read_parquet(result.normalized_path)
    assert rules_long.loc[0, "state_fips"] == "06"
    assert int(rules_long.loc[0, "year"]) == 2021
    assert rules_long.loc[0, "rule_column_source"] == "center_infant_ratio"


def test_licensing_real_ingest_builds_rule_level_from_icpsr_studies(project_paths):
    licensing_dir = project_paths.raw / "licensing"
    path_2017 = licensing_dir / "icpsr_2017" / "ICPSR_37700" / "DS0001"
    path_2020 = licensing_dir / "icpsr_2020" / "ICPSR_38539" / "DS0001"
    path_2017.mkdir(parents=True, exist_ok=True)
    path_2020.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "STATE": "CA",
                "C_CHR_ALLSTAFF": 1,
                "F_CHR_ALLSTAFF": 0,
                "LF_CHR_ALLSTAFF": 1,
                "C_FINGERPRT_ALLSTAFF": 1,
                "F_FINGERPRT_ALLSTAFF": 1,
                "LF_FINGERPRT_ALLSTAFF": 1,
                "C_CAN_ALLSTAFF": 1,
                "F_CAN_ALLSTAFF": 1,
                "LF_CAN_ALLSTAFF": 0,
                "C_SX_ALLSTAFF": 1,
                "F_SX_ALLSTAFF": 1,
                "LF_SX_ALLSTAFF": 0,
            }
        ]
    ).to_csv(path_2017 / "37700-0001-Data.tsv", sep="\t", index=False)
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
                "PS_LGGROUP": 0,
                "PS_GSEXCEED": 0,
                "SPND_CAPACITY": 1,
                "SPND_RATIO": 1,
                "SPND_GROUPSIZE": 0,
            }
        ]
    ).to_csv(path_2020 / "38539-0001-Data.tsv", sep="\t", index=False)

    result = ingest_licensing(project_paths, sample=False, refresh=True)

    assert result.normalized_path.name == "licensing_rules_long.parquet"
    rules_long = read_parquet(result.normalized_path)
    assert not rules_long.empty
    assert set(rules_long["year"]) == {2017, 2020}
    assert "teacher_min_age" in set(rules_long["rule_family"])
    assert "criminal_history_check_all_staff" in set(rules_long["rule_family"])
    assert set(rules_long["provider_type"]) >= {"center", "family_home", "large_group_home"}


def test_licensing_real_ingest_requires_manual_file(project_paths):
    raw_path = project_paths.raw / "licensing" / "licensing_supply_shocks.csv"
    placeholder_path = project_paths.raw / "licensing" / "licensing_placeholder.txt"

    with pytest.raises(SourceAccessError) as exc_info:
        ingest_licensing(project_paths, sample=False, refresh=True)

    message = str(exc_info.value)
    assert str(raw_path) in message
    assert "licensingregulations.acf.hhs.gov" in message
    assert not raw_path.exists()
    assert not placeholder_path.exists()
