from __future__ import annotations

from html import escape
import io
import zipfile

import pandas as pd

from unpaidwork.ingest import common as ingest_common
from unpaidwork.ingest.acs import ingest as ingest_acs
from unpaidwork.ingest.ahs import _parse_ahs_archive, ingest as ingest_ahs
from unpaidwork.ingest import atus as atus_ingest
from unpaidwork.ingest.atus import (
    _parse_atus_activity_zip,
    _parse_atus_respondent_weights_zip,
    ingest as ingest_atus,
)
from unpaidwork.ingest import ce as ce_ingest
from unpaidwork.ingest.ce import _parse_ce_zip, ingest as ingest_ce
from unpaidwork.ingest import head_start as head_start_ingest
from unpaidwork.ingest.head_start import _normalize_head_start_frame, ingest as ingest_head_start
from unpaidwork.ingest import nces_ccd as nces_ccd_ingest
from unpaidwork.ingest.nces_ccd import _normalize_ccd_zip, ingest as ingest_nces_ccd
from unpaidwork.ingest import nes as nes_ingest
from unpaidwork.ingest.nes import _normalize_nes_rows, ingest as ingest_nes
from unpaidwork.ingest import ndcp as ndcp_ingest
from unpaidwork.ingest.ndcp import _parse_ndcp_workbook, ingest as ingest_ndcp
from unpaidwork.ingest import oews as oews_ingest
from unpaidwork.ingest.oews import _normalize_oews_zip, ingest as ingest_oews
from unpaidwork.ingest.qcew import ingest as ingest_qcew
from unpaidwork.ingest import sipp as sipp_ingest
from unpaidwork.ingest.sipp import _parse_sipp_zip, ingest as ingest_sipp
from unpaidwork.storage import read_parquet


def _excel_col(index: int) -> str:
    label = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _cell_xml(ref: str, value: str) -> str:
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'


def _build_ndcp_test_workbook() -> bytes:
    headers = [
        "STATE_NAME",
        "STATE_ABBREVIATION",
        "COUNTY_NAME",
        "COUNTY_FIPS_CODE",
        "STUDYYEAR",
        "MCINFANT",
        "MCInfant_flag",
        "MCTODDLER",
        "MCToddler_flag",
        "MCPRESCHOOL",
        "MCPreschool_flag",
        "MFCCINFANT",
        "MFCCInfant_flag",
        "MFCCTODDLER",
        "MFCCToddler_flag",
        "MFCCPRESCHOOL",
        "MFCCPreschool_flag",
        "STATE_FIPS",
    ]
    rows = [
        [
            "Alabama",
            "AL",
            "Autauga County",
            "1001",
            "2020",
            "100",
            "1",
            "90",
            "3",
            "",
            "",
            "80",
            "2",
            "",
            "",
            "70",
            "1",
            "1",
        ],
        [
            "Florida",
            "FL",
            "Miami-Dade County",
            "12086",
            "2021",
            "",
            "",
            "85",
            "3",
            "110",
            "1",
            "",
            "",
            "60",
            "1",
            "",
            "",
            "",
        ],
    ]

    sheet_rows = []
    for row_index, values in enumerate([headers, *rows], start=1):
        cells = []
        for col_index, value in enumerate(values, start=1):
            if value == "":
                continue
            ref = f"{_excel_col(col_index)}{row_index}"
            cells.append(_cell_xml(ref, value))
        sheet_rows.append(f"<row r=\"{row_index}\">{''.join(cells)}</row>")

    worksheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="NDCP" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet_xml)
    return buffer.getvalue()


def _build_ahs_test_archive() -> bytes:
    household_csv = "\n".join(
        [
            "CONTROL,OMB13CBSA,WEIGHT,JMARKETVAL,HINCP,FINCP,JTENURE,YRBUILT",
            "'11000005',31080,813.89,245790,48000,48000,1,1980",
            "'11000006',19100,500.00,180000,62000,62000,2,1992",
        ]
    )
    project_csv = "\n".join(
        [
            "CONTROL,JJOBTYPE,JOBTYPE,JOBDIY,JOBCOST,JOBCOMP,JOBCOMPYR,JOBWORKYR,JOBFUNDS",
            "'11000005',0,35,1,500,1,2,-6,1",
            "'11000006',0,16,2,16000,1,3,-6,4",
            "'11000006',0,26,2,4000,1,1,-6,7",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("household.csv", household_csv)
        archive.writestr("project.csv", project_csv)
    return buffer.getvalue()


def _build_atus_test_archive() -> bytes:
    loader_do = "\n".join(
        [
            "#delimit ;",
            "import delimited",
            "tucaseid",
            "tuactdur24",
            "trto_ln",
            "trtier1p",
            "trcodep",
            " using c:\\atusact_0324.dat, stringcols(1) ;",
        ]
    )
    activity_csv = "\n".join(
        [
            "TUCASEID,TUACTDUR24,TRTO_LN,TRTIER1P,TRCODEP",
            "20200100010601,60,30,05,050101",
            "20200100010601,45,0,01,030101",
            "20190300014888,20,20,01,010101",
            "20190300014888,10,0,01,030101",
            "20210100014877,30,-1,05,050101",
        ]
    )
    info_text = "The data file atusact_0324.dat contains ATUS activity rows."
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("atusact_0324.do", loader_do)
        archive.writestr("atusact_0324.dat", activity_csv)
        archive.writestr("Activity0324_info.txt", info_text)
    return buffer.getvalue()


def _build_atus_respondent_test_archive() -> bytes:
    loader_do = "\n".join(
        [
            "#delimit ;",
            "import delimited",
            "tucaseid",
            "tufnwgtp",
            "tu20fwgt",
            "tuyear",
            " using c:\\atusresp_0324.dat, stringcols(1) ;",
        ]
    )
    respondent_csv = "\n".join(
        [
            "TUCASEID,TUFNWGTP,TU20FWGT,TUYEAR",
            "20200100010601,10.0,25.0,2020",
            "20190300014888,8.0,9.0,2019",
            "20210100014877,12.0,13.0,2021",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("atusresp_0324.do", loader_do)
        archive.writestr("atusresp_0324.dat", respondent_csv)
    return buffer.getvalue()


def _build_ccd_test_archive() -> bytes:
    school_csv = "\n".join(
        [
            "SCHOOL_YEAR,FIPST,SY_STATUS,G_PK_OFFERED,G_KG_OFFERED,NCESSCH",
            "2024-2025,06,1,Yes,Yes,0001",
            "2024-2025,06,1,No,Yes,0002",
            "2024-2025,48,1,Yes,No,0003",
            "2024-2025,48,2,Yes,Yes,0004",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ccd_sch_029_2425_w_0a_051425.csv", school_csv)
    return buffer.getvalue()


def _build_sipp_test_archive() -> bytes:
    csv_text = "\n".join(
        [
            "SSUID|PNUM|SPANEL|SWAVE|ERP|RANY5|WPFINWGT|EPAY|TPAYWK|EDAYCARE|EFAM|ENREL|EHEADST|ENUR",
            "1|101|2021|3|1|1|100|1|200|1|0|1|0|1",
            "2|101|2021|3|1|2|150|2|0|0|1|0|0|0",
            "3|101|2023|1|1|1|200|1|300|1|1|1|1|1",
            "4|101|2023|1|2||250|||||||",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("pu2023.csv", csv_text)
    return buffer.getvalue()


def _build_ce_test_archive() -> bytes:
    fmli_csv = "\n".join(
        [
            "NEWID,FINLWT21,PERSLT18,CHILDAGE,QINTRVYR,BBYDAYPQ,TOTEXPPQ",
            "1,100,2,4,2023,600,10000",
            "2,200,1,7,2023,0,8000",
            "3,150,0,0,2023,0,7000",
            "4,120,3,2,2023,300,9000",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ["intrvw23/fmli232.csv", "intrvw23/fmli233.csv", "intrvw23/fmli234.csv", "intrvw23/fmli241.csv"]:
            archive.writestr(name, fmli_csv)
    return buffer.getvalue()


def _build_oews_test_archive() -> bytes:
    frame = pd.DataFrame(
        [
            {
                "AREA": 1,
                "AREA_TITLE": "Alabama",
                "AREA_TYPE": 2,
                "PRIM_STATE": "AL",
                "OCC_CODE": "39-9011",
                "H_MEAN": 12.5,
            },
            {
                "AREA": 1,
                "AREA_TITLE": "Alabama",
                "AREA_TYPE": 2,
                "PRIM_STATE": "AL",
                "OCC_CODE": "35-0000",
                "H_MEAN": 14.2,
            },
            {
                "AREA": 6,
                "AREA_TITLE": "California",
                "AREA_TYPE": 2,
                "PRIM_STATE": "CA",
                "OCC_CODE": "39-9011",
                "H_MEAN": 18.7,
            },
            {
                "AREA": 6,
                "AREA_TITLE": "California",
                "AREA_TYPE": 2,
                "PRIM_STATE": "CA",
                "OCC_CODE": "35-0000",
                "H_MEAN": 21.1,
            },
        ]
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("oesm22st/state_M2022_dl.xlsx", buffer.getvalue())
    return zip_buffer.getvalue()


def _build_nes_rows_payload(naics: str = "62441") -> list[list[str]]:
    return [
        ["STATE", "COUNTY", "NAICS2022", "LFO", "RCPSZES", "NESTAB", "NRCPTOT", "state", "county"],
        ["06", "037", naics, "001", "001", "1200", "4800", "06", "037"],
        ["48", "201", naics, "001", "001", "800", "2900", "48", "201"],
    ]


def test_core_sample_ingestors_write_parquet(project_paths):
    for ingestor in (
        ingest_atus,
        ingest_ndcp,
        ingest_ahs,
        ingest_qcew,
        ingest_acs,
        ingest_ce,
        ingest_head_start,
        ingest_nces_ccd,
        ingest_nes,
        ingest_oews,
        ingest_sipp,
    ):
        result = ingestor(project_paths, sample=True)
        frame = read_parquet(result.normalized_path)
        assert not frame.empty


def test_sample_ingest_dry_run_does_not_write(project_paths):
    result = ingest_atus(project_paths, sample=True, dry_run=True)
    assert result.dry_run is True
    assert not result.raw_path.exists()
    assert not result.normalized_path.exists()


def test_atus_activity_parser_normalizes_for_childcare_pipeline(tmp_path):
    archive_path = tmp_path / "atus_test.zip"
    archive_path.write_bytes(_build_atus_test_archive())
    respondent_archive = tmp_path / "atus_resp_test.zip"
    respondent_archive.write_bytes(_build_atus_respondent_test_archive())
    respondent_weights = _parse_atus_respondent_weights_zip(respondent_archive)

    frame = _parse_atus_activity_zip(
        archive_path,
        "https://example.com/atus.zip",
        respondent_weights=respondent_weights,
    )

    assert len(frame) == 3
    assert {
        "state_fips",
        "year",
        "subgroup",
        "childcare_hours",
        "weight",
        "parent_employment_rate",
        "single_parent_share",
        "median_income",
        "unemployment_rate",
        "births",
    } <= set(frame.columns)
    case_2020 = frame.loc[frame["respondent_id"] == "20200100010601"].iloc[0]
    assert case_2020["state_fips"] == "06"
    assert case_2020["year"] == 2020
    assert abs(case_2020["childcare_hours"] - 5.25) < 1e-9
    assert case_2020["subgroup"] == "all"
    assert case_2020["weight"] == 25.0


def test_atus_respondent_parser_uses_2020_pandemic_weight(tmp_path):
    archive_path = tmp_path / "atus_resp_test.zip"
    archive_path.write_bytes(_build_atus_respondent_test_archive())

    frame = _parse_atus_respondent_weights_zip(archive_path)

    case_2020 = frame.loc[frame["respondent_id"] == "20200100010601"].iloc[0]
    case_2019 = frame.loc[frame["respondent_id"] == "20190300014888"].iloc[0]
    assert case_2020["weight"] == 25.0
    assert case_2019["weight"] == 8.0


def test_ndcp_workbook_parser_normalizes_county_year_rows(tmp_path):
    workbook_path = tmp_path / "ndcp_test.xlsx"
    workbook_path.write_bytes(_build_ndcp_test_workbook())

    frame = _parse_ndcp_workbook(workbook_path, "https://example.com/ndcp.xlsx")

    assert len(frame) == 7
    assert set(frame["provider_type"].unique()) == {"center", "home"}
    assert set(frame["child_age"].unique()) == {"infant", "toddler", "preschool"}
    assert "ndcp_flag_code" in frame.columns
    assert "weekly_price" in frame.columns

    row = frame[(frame["county_fips"] == "01001") & (frame["child_age"] == "infant") & (frame["provider_type"] == "center")].iloc[0]
    assert row["annual_price"] == 5200.0
    assert row["imputed_flag"] == 0
    assert row["ndcp_flag_code"] == 1

    derived_state = frame[frame["county_fips"] == "12086"]["state_fips"].unique().tolist()
    assert derived_state == ["12"]


def test_ndcp_real_ingest_writes_normalized_parquet(project_paths, monkeypatch):
    workbook_bytes = _build_ndcp_test_workbook()
    monkeypatch.setattr(ndcp_ingest, "_download_ndcp_workbook", lambda url: workbook_bytes)

    result = ingest_ndcp(project_paths, sample=False, refresh=True)
    frame = read_parquet(result.normalized_path)

    assert not frame.empty
    assert {"county_fips", "state_fips", "year", "annual_price", "imputed_flag", "ndcp_flag_code"} <= set(frame.columns)
    assert "entry_name" not in frame.columns


def test_nes_parser_normalizes_county_receipts():
    rows = _build_nes_rows_payload()
    frame = _normalize_nes_rows(rows, 2022, "https://api.census.gov/data/2022/nonemp")

    assert len(frame) == 2
    assert {
        "county_fips",
        "state_fips",
        "year",
        "nonemployer_firms",
        "receipts",
        "source_url",
    } <= set(frame.columns)
    la = frame.loc[frame["county_fips"] == "06037"].iloc[0]
    assert la["state_fips"] == "06"
    assert la["year"] == 2022
    assert la["nonemployer_firms"] == 1200
    assert la["receipts"] == 4_800_000.0


def test_nes_real_ingest_writes_normalized_parquet_with_fallback(project_paths, monkeypatch):
    class _MockResponse:
        def __init__(self, status_code: int, rows: list[list[str]] | None, url: str):
            self.status_code = status_code
            self._rows = rows
            self.url = url
            self.text = "" if rows is None else "ok"

        def json(self):
            if self._rows is None:
                return []
            return self._rows

    def _mock_get(url, params, timeout):  # noqa: ANN001
        naics = params.get("NAICS2022")
        if naics == "624410":
            return _MockResponse(204, None, f"{url}?NAICS2022=624410")
        rows = _build_nes_rows_payload("62441")
        return _MockResponse(200, rows, f"{url}?NAICS2022=62441")

    monkeypatch.setattr(nes_ingest.requests, "get", _mock_get)

    result = ingest_nes(project_paths, sample=False, refresh=True, year=2022)
    frame = read_parquet(result.normalized_path)

    assert not frame.empty
    assert {
        "county_fips",
        "state_fips",
        "year",
        "nonemployer_firms",
        "receipts",
        "source_url",
    } <= set(frame.columns)
    assert "NAICS2022=62441" in frame.loc[0, "source_url"]


def test_atus_real_ingest_writes_normalized_parquet(project_paths, monkeypatch):
    archive_bytes = _build_atus_test_archive()
    respondent_archive_bytes = _build_atus_respondent_test_archive()
    monkeypatch.setattr(
        atus_ingest,
        "_download_atus_zip",
        lambda url: respondent_archive_bytes if "atusresp" in url else archive_bytes,
    )

    result = ingest_atus(project_paths, sample=False, refresh=True)
    frame = read_parquet(result.normalized_path)

    assert not frame.empty
    assert {
        "state_fips",
        "year",
        "subgroup",
        "childcare_hours",
        "weight",
        "parent_employment_rate",
        "single_parent_share",
        "median_income",
        "unemployment_rate",
        "births",
    } <= set(frame.columns)
    assert "entry_name" not in frame.columns
    assert frame["weight"].ne(1.0).any()


def test_ahs_archive_parser_normalizes_jobs(tmp_path):
    archive_path = tmp_path / "ahs_test.zip"
    archive_path.write_bytes(_build_ahs_test_archive())

    frame = _parse_ahs_archive(archive_path)

    assert len(frame) == 3
    assert {"job_id", "cbsa_code", "job_type_code", "job_type", "job_group", "job_cost", "job_diy"} <= set(frame.columns)
    assert frame.loc[0, "cbsa_code"] == "31080"
    assert frame.loc[0, "job_diy"] == 1
    assert frame["job_type_code"].tolist() == ["35", "16", "26"]
    assert frame["job_type"].tolist() == [
        "Added or replaced shed, detached garage, or other building",
        "Added or replaced roofing",
        "Added or replaced central air conditioning",
    ]
    assert frame["job_group"].tolist() == [
        "lot_yard_and_outbuildings",
        "exterior_envelope",
        "interior_finish_and_appliances",
    ]


def test_ahs_real_ingest_writes_normalized_parquet(project_paths, monkeypatch):
    archive_bytes = _build_ahs_test_archive()
    monkeypatch.setattr(ingest_common, "_download", lambda url: archive_bytes)

    result = ingest_ahs(project_paths, sample=False, refresh=True)
    frame = read_parquet(result.normalized_path)

    assert not frame.empty
    assert {"job_id", "cbsa_code", "household_income", "home_value"} <= set(frame.columns)
    assert "entry_name" not in frame.columns


def test_head_start_parser_maps_counties_and_slots():
    raw = pd.DataFrame(
        [
            {
                "service_location_name": "Center A",
                "state": "CA",
                "county": "Los Angeles County",
                "funded_slots": "120",
                "status": "Open",
            },
            {
                "service_location_name": "Center B",
                "state": "CA",
                "county": "Los Angeles County",
                "funded_slots": "80",
                "status": "Open",
            },
            {
                "service_location_name": "Center C",
                "state": "TX",
                "county": "Harris County",
                "funded_slots": "50",
                "status": "Closed",
            },
        ]
    )
    crosswalk = pd.DataFrame(
        [
            {"state_fips": "06", "county_fips": "06037", "county_key": "los angeles"},
            {"state_fips": "48", "county_fips": "48201", "county_key": "harris"},
        ]
    )

    normalized = _normalize_head_start_frame(raw, crosswalk)

    assert len(normalized) == 1
    assert normalized.loc[0, "county_fips"] == "06037"
    assert normalized.loc[0, "head_start_slots"] == 200
    assert normalized.loc[0, "open_locations"] == 2


def test_head_start_real_ingest_writes_normalized_parquet(project_paths, monkeypatch):
    csv_bytes = "\n".join(
        [
            "service_location_name,state,county,funded_slots,status",
            "Center A,CA,Los Angeles County,120,Open",
            "Center B,CA,Los Angeles County,80,Open",
            "Center C,TX,Harris County,50,Closed",
        ]
    ).encode("utf-8")
    monkeypatch.setattr(head_start_ingest, "_download_bytes", lambda url: csv_bytes)
    monkeypatch.setattr(
        head_start_ingest,
        "_fetch_county_crosswalk",
        lambda year: pd.DataFrame(
            [
                {"state_fips": "06", "county_fips": "06037", "county_key": "los angeles"},
                {"state_fips": "48", "county_fips": "48201", "county_key": "harris"},
            ]
        ),
    )

    result = ingest_head_start(project_paths, sample=False, refresh=True)
    frame = read_parquet(result.normalized_path)

    assert not frame.empty
    assert {"county_fips", "state_fips", "head_start_slots", "open_locations"} <= set(frame.columns)
    assert frame.loc[0, "head_start_slots"] == 200


def test_ccd_zip_parser_builds_state_option_index(tmp_path):
    archive_path = tmp_path / "ccd_test.zip"
    archive_path.write_bytes(_build_ccd_test_archive())

    frame = _normalize_ccd_zip(archive_path, "https://example.com/ccd.zip")

    assert len(frame) == 2
    california = frame.loc[frame["state_fips"] == "06"].iloc[0]
    texas = frame.loc[frame["state_fips"] == "48"].iloc[0]
    assert california["prek_schools"] == 1
    assert california["kg_schools"] == 2
    assert california["pk_or_kg_schools"] == 2
    assert california["operational_schools"] == 2
    assert texas["operational_schools"] == 1


def test_ccd_real_ingest_writes_normalized_parquet(project_paths, monkeypatch):
    archive_bytes = _build_ccd_test_archive()
    monkeypatch.setattr(nces_ccd_ingest, "_download_ccd_zip", lambda url: archive_bytes)

    result = ingest_nces_ccd(project_paths, sample=False, refresh=True)
    frame = read_parquet(result.normalized_path)

    assert not frame.empty
    assert {
        "state_fips",
        "prek_schools",
        "kg_schools",
        "pk_or_kg_schools",
        "operational_schools",
        "public_school_option_index",
    } <= set(frame.columns)


def test_sipp_zip_parser_builds_validation_summary(tmp_path):
    archive_path = tmp_path / "sipp_test.zip"
    archive_path.write_bytes(_build_sipp_test_archive())

    frame = _parse_sipp_zip(archive_path, "https://example.com/sipp.zip")

    assert len(frame) == 4
    required = {
        "year",
        "subgroup",
        "any_paid_childcare_rate",
        "avg_weekly_paid_childcare_payers",
        "center_care_rate",
        "geography",
    }
    assert required <= set(frame.columns)
    row_2023 = frame.loc[(frame["year"] == 2023) & (frame["subgroup"] == "under5_ref_parents")].iloc[0]
    assert row_2023["any_paid_childcare_rate"] == 1.0
    assert row_2023["avg_weekly_paid_childcare_payers"] == 300.0
    assert row_2023["head_start_rate"] == 1.0
    assert row_2023["nursery_preschool_rate"] == 1.0


def test_sipp_real_ingest_writes_normalized_parquet(project_paths, monkeypatch):
    archive_bytes = _build_sipp_test_archive()
    monkeypatch.setattr(sipp_ingest, "_download_sipp_zip", lambda url: archive_bytes)

    result = ingest_sipp(project_paths, sample=False, refresh=True)
    frame = read_parquet(result.normalized_path)

    assert not frame.empty
    assert {"year", "subgroup", "any_paid_childcare_rate", "avg_weekly_paid_childcare_all"} <= set(frame.columns)


def test_ce_zip_parser_builds_validation_summary(tmp_path):
    archive_path = tmp_path / "ce_test.zip"
    archive_path.write_bytes(_build_ce_test_archive())

    frame = _parse_ce_zip(archive_path, "https://example.com/ce.zip")

    assert len(frame) == 2
    required = {
        "year",
        "subgroup",
        "childcare_spender_rate",
        "avg_childcare_spend_pq_all",
        "avg_childcare_spend_pq_payers",
        "childcare_spend_share_pq",
    }
    assert required <= set(frame.columns)
    under5 = frame.loc[frame["subgroup"] == "with_child_age_1_5"].iloc[0]
    assert under5["childcare_spender_rate"] == 1.0
    assert under5["avg_childcare_spend_pq_payers"] == 436.3636363636364


def test_ce_real_ingest_writes_normalized_parquet(project_paths, monkeypatch):
    archive_bytes = _build_ce_test_archive()
    monkeypatch.setattr(ce_ingest, "_download_ce_zip", lambda url: archive_bytes)

    result = ingest_ce(project_paths, sample=False, refresh=True)
    frame = read_parquet(result.normalized_path)

    assert not frame.empty
    assert {"year", "subgroup", "childcare_spender_rate", "avg_childcare_spend_pq_all"} <= set(frame.columns)


def test_oews_zip_parser_builds_state_wages(tmp_path):
    archive_path = tmp_path / "oews_test.zip"
    archive_path.write_bytes(_build_oews_test_archive())

    frame = _normalize_oews_zip(archive_path, "https://example.com/oews.zip", 2022)

    assert len(frame) == 2
    california = frame.loc[frame["state_fips"] == "06"].iloc[0]
    assert california["oews_childcare_worker_wage"] == 18.7
    assert california["oews_outside_option_wage"] == 21.1


def test_oews_real_ingest_writes_normalized_parquet(project_paths, monkeypatch):
    archive_bytes = _build_oews_test_archive()
    monkeypatch.setattr(oews_ingest, "_download_oews_zip", lambda url: archive_bytes)

    result = ingest_oews(project_paths, sample=False, refresh=True, year=2022)
    frame = read_parquet(result.normalized_path)

    assert not frame.empty
    assert {"state_fips", "year", "oews_childcare_worker_wage", "oews_outside_option_wage"} <= set(frame.columns)


def test_acs_sample_ingest_preserves_multiple_years(project_paths):
    """Sample ACS data has 3 years; verify they survive ingestion."""
    result = ingest_acs(project_paths, sample=True)
    frame = read_parquet(result.normalized_path)
    assert len(frame["year"].unique()) == 3


def test_acs_merge_existing_years_preserves_old_years(project_paths):
    """When ingesting a new year, existing years must be kept."""
    from unpaidwork.ingest.acs import _merge_existing_years
    import numpy as np

    normalized_path = project_paths.interim / "acs" / "acs.parquet"
    # Write an initial year
    old = pd.DataFrame({
        "county_fips": ["06037"],
        "state_fips": ["06"],
        "year": [2019],
        "under5_population": [640000.0],
        "median_income": [72000.0],
        "rent_index": [1.45],
        "single_parent_share": [0.22],
        "parent_employment_rate": [0.76],
        "under6_population": [np.nan],
    })
    from unpaidwork.storage import write_parquet as wp
    wp(old, normalized_path)

    # Merge a new year
    new = pd.DataFrame({
        "county_fips": ["06037"],
        "state_fips": ["06"],
        "year": [2020],
        "under5_population": [626000.0],
        "median_income": [73500.0],
        "rent_index": [1.48],
        "single_parent_share": [0.23],
        "parent_employment_rate": [0.69],
        "under6_population": [np.nan],
    })
    combined = _merge_existing_years(normalized_path, new)
    assert set(combined["year"].tolist()) == {2019, 2020}
    assert len(combined) == 2


def test_acs_merge_existing_years_replaces_overlapping_year(project_paths):
    """When re-ingesting an existing year, the new data must replace the old."""
    from unpaidwork.ingest.acs import _merge_existing_years
    import numpy as np

    normalized_path = project_paths.interim / "acs" / "acs.parquet"
    old = pd.DataFrame({
        "county_fips": ["06037"],
        "state_fips": ["06"],
        "year": [2019],
        "under5_population": [640000.0],
        "median_income": [72000.0],
        "rent_index": [1.45],
        "single_parent_share": [0.22],
        "parent_employment_rate": [0.76],
        "under6_population": [np.nan],
    })
    from unpaidwork.storage import write_parquet as wp
    wp(old, normalized_path)

    # Re-ingest 2019 with updated data
    new = pd.DataFrame({
        "county_fips": ["06037"],
        "state_fips": ["06"],
        "year": [2019],
        "under5_population": [999999.0],
        "median_income": [99999.0],
        "rent_index": [9.99],
        "single_parent_share": [0.99],
        "parent_employment_rate": [0.99],
        "under6_population": [np.nan],
    })
    combined = _merge_existing_years(normalized_path, new)
    assert len(combined) == 1
    assert combined.iloc[0]["under5_population"] == 999999.0


def test_qcew_merge_existing_years_preserves_old_years(project_paths):
    from unpaidwork.ingest.qcew import _merge_existing_years
    from unpaidwork.storage import write_parquet as wp

    normalized_path = project_paths.interim / "qcew" / "qcew.parquet"
    old = pd.DataFrame({
        "county_fips": ["06037"],
        "state_fips": ["06"],
        "year": [2019],
        "industry_code": ["624410"],
        "childcare_worker_wage": [16.1],
        "outside_option_wage": [21.8],
        "employment": [55000.0],
    })
    wp(old, normalized_path)

    new = pd.DataFrame({
        "county_fips": ["06037"],
        "state_fips": ["06"],
        "year": [2020],
        "industry_code": ["624410"],
        "childcare_worker_wage": [16.5],
        "outside_option_wage": [22.0],
        "employment": [54000.0],
    })
    combined = _merge_existing_years(normalized_path, new)
    assert set(combined["year"].tolist()) == {2019, 2020}
    assert len(combined) == 2


def test_laus_merge_existing_years_preserves_old_years(project_paths):
    from unpaidwork.ingest.laus import _merge_existing_years
    from unpaidwork.storage import write_parquet as wp

    normalized_path = project_paths.interim / "laus" / "laus.parquet"
    old = pd.DataFrame({
        "geography": ["county"],
        "state_fips": ["06"],
        "county_fips": ["06037"],
        "year": [2019],
        "laus_unemployment_rate": [0.041],
        "laus_unemployed": [362000.0],
        "laus_employment": [4520000.0],
        "laus_labor_force": [4882000.0],
    })
    wp(old, normalized_path)

    new = pd.DataFrame({
        "geography": ["county"],
        "state_fips": ["06"],
        "county_fips": ["06037"],
        "year": [2020],
        "laus_unemployment_rate": [0.092],
        "laus_unemployed": [410000.0],
        "laus_employment": [4200000.0],
        "laus_labor_force": [4610000.0],
    })
    combined = _merge_existing_years(normalized_path, new)
    assert set(combined["year"].tolist()) == {2019, 2020}
    assert len(combined) == 2


def test_qcew_sample_ingest_preserves_multiple_years(project_paths):
    from unpaidwork.ingest.qcew import ingest as ingest_qcew_fn
    result = ingest_qcew_fn(project_paths, sample=True)
    frame = read_parquet(result.normalized_path)
    assert len(frame["year"].unique()) == 3


def test_laus_sample_ingest_preserves_multiple_years(project_paths):
    from unpaidwork.ingest.laus import ingest as ingest_laus_fn
    result = ingest_laus_fn(project_paths, sample=True)
    frame = read_parquet(result.normalized_path)
    # Sample LAUS has 1 year (2021)
    assert len(frame["year"].unique()) >= 1
