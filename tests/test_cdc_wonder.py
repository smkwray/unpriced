from __future__ import annotations

import pandas as pd

from unpaidwork.errors import SourceAccessError
from unpaidwork.ingest import cdc_wonder
from unpaidwork.ingest.cdc_wonder import _parse_wonder_tsv, ingest
from unpaidwork.storage import read_parquet, write_parquet


def _build_wonder_tsv() -> str:
    return "\n".join(
        [
            '"Notes"\t"State"\t"State Code"\t"Year"\t"Year Code"\tBirths',
            '\t"Alabama"\t"01"\t"2022"\t"2022"\t58149',
            '"Suppressed for confidentiality"\t"Alaska"\t"02"\t"2022"\t"2022"\tSuppressed',
            '"---"',
            '"Dataset: Natality, 2007-2024"',
            '"Help: See https://wonder.cdc.gov/wonder/help/natality.html for more information."',
            '"http://wonder.cdc.gov/natality-current.html on Mar 10, 2026 1:51:52 PM"',
        ]
    )


def test_parse_wonder_tsv_preserves_suppression_metadata():
    frame = _parse_wonder_tsv(_build_wonder_tsv(), "https://wonder.cdc.gov/natality-current.html")

    assert len(frame) == 2
    assert {
        "state_fips",
        "year",
        "births",
        "source_url",
        "births_raw",
        "births_suppressed",
        "suppression_note",
    } <= set(frame.columns)

    alabama = frame.loc[frame["state_fips"] == "01"].iloc[0]
    alaska = frame.loc[frame["state_fips"] == "02"].iloc[0]

    assert alabama["births"] == 58149.0
    assert bool(alabama["births_suppressed"]) is False
    assert pd.isna(alabama["suppression_note"])

    assert pd.isna(alaska["births"])
    assert bool(alaska["births_suppressed"]) is True
    assert "Suppressed" in str(alaska["suppression_note"])


def test_cdc_wonder_real_ingest_writes_normalized_parquet(project_paths, monkeypatch):
    tsv_text = _build_wonder_tsv()
    monkeypatch.setattr(
        cdc_wonder,
        "_download_wonder_tsv",
        lambda year: (tsv_text, "https://wonder.cdc.gov/natality-current.html"),
    )

    result = ingest(project_paths, sample=False, refresh=True, year=2022)
    frame = read_parquet(result.normalized_path)

    assert not frame.empty
    assert {"state_fips", "year", "births", "source_url"} <= set(frame.columns)
    assert "entry_name" not in frame.columns
    assert set(frame["year"].astype(int).tolist()) == {2022}
    assert frame["source_url"].str.contains("wonder.cdc.gov").all()


def test_cdc_wonder_real_ingest_uses_local_fallback_tsv(project_paths, monkeypatch):
    raw_path = project_paths.raw / "cdc_wonder" / "cdc_wonder_2022.tsv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(_build_wonder_tsv(), encoding="utf-8")

    def _raise_download(year: int | None = None):  # noqa: ANN001
        raise SourceAccessError("network blocked")

    monkeypatch.setattr(cdc_wonder, "_download_wonder_tsv", _raise_download)

    result = ingest(project_paths, sample=False, refresh=True, year=2022)
    frame = read_parquet(result.normalized_path)

    assert not frame.empty
    assert {"state_fips", "year", "births", "source_url"} <= set(frame.columns)
    assert set(frame["state_fips"].tolist()) == {"01", "02"}


def test_cdc_wonder_year_refresh_preserves_other_existing_years(project_paths, monkeypatch):
    existing = pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "births": 420786.0,
                "source_url": "https://wonder.cdc.gov/natality-current.html",
                "births_raw": "420786",
                "births_suppressed": False,
                "suppression_note": pd.NA,
            }
        ]
    )
    normalized_path = project_paths.interim / "cdc_wonder" / "cdc_wonder.parquet"
    write_parquet(existing, normalized_path)

    tsv_text = _build_wonder_tsv()
    monkeypatch.setattr(
        cdc_wonder,
        "_download_wonder_tsv",
        lambda year: (tsv_text, "https://wonder.cdc.gov/natality-current.html"),
    )

    result = ingest(project_paths, sample=False, refresh=True, year=2022)
    frame = read_parquet(result.normalized_path)

    assert set(frame["year"].astype(int).tolist()) == {2021, 2022}
    assert frame.loc[frame["year"].astype(int) == 2021, "births"].iloc[0] == 420786.0
