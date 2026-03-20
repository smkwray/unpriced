from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import requests

from unpriced.config import ProjectPaths
from unpriced.errors import SourceAccessError
from unpriced.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpriced.registry import append_registry, build_record
from unpriced.sample_data import ndcp as sample_ndcp
from unpriced.storage import write_parquet

SPEC = SourceSpec(
    name="ndcp",
    citation="https://www.dol.gov/agencies/wb/topics/featured-childcare",
    license_name="Public data",
    retrieval_method="download",
    landing_page="https://www.dol.gov/agencies/wb/topics/featured-childcare",
)

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
MAIN = f"{{{MAIN_NS}}}"
DOC_REL = f"{{{DOC_REL_NS}}}"
CELL_REF_RE = re.compile(r"([A-Z]+)")

PRICE_SPECS = (
    ("MCINFANT", "MCInfant_flag", "infant", "center"),
    ("MCTODDLER", "MCToddler_flag", "toddler", "center"),
    ("MCPRESCHOOL", "MCPreschool_flag", "preschool", "center"),
    ("MFCCINFANT", "MFCCInfant_flag", "infant", "home"),
    ("MFCCTODDLER", "MFCCToddler_flag", "toddler", "home"),
    ("MFCCPRESCHOOL", "MFCCPreschool_flag", "preschool", "home"),
)


def _column_index(cell_ref: str) -> int | None:
    match = CELL_REF_RE.match(cell_ref)
    if not match:
        return None
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - 64)
    return index


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _coerce_fips(value: str | None, width: int) -> str:
    if value is None:
        return ""
    text = value.strip()
    if not text:
        return ""
    numeric = _to_int(text)
    if numeric is not None:
        return str(numeric).zfill(width)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return digits.zfill(width)[-width:]
    return ""


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values = []
    for node in root.findall(f".//{MAIN}si"):
        values.append("".join((text.text or "") for text in node.findall(f".//{MAIN}t")))
    return values


def _worksheet_target(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find(f".//{MAIN}sheets/{MAIN}sheet")
    if sheet is None:
        raise ValueError("NDCP workbook has no sheets")
    rel_id = sheet.attrib.get(f"{DOC_REL}id")
    if not rel_id:
        raise ValueError("NDCP workbook is missing a worksheet relationship id")
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = ""
    for rel in rels.findall(f".//{{{PKG_REL_NS}}}Relationship"):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib.get("Target", "")
            break
    if not target:
        raise ValueError("NDCP workbook worksheet target not found")
    if target.startswith("/"):
        return target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    value = cell.find(f"{MAIN}v")
    if value is not None:
        text = value.text or ""
        if cell_type == "s":
            index = _to_int(text)
            if index is None or index < 0 or index >= len(shared_strings):
                return ""
            return shared_strings[index]
        return text
    inline = cell.find(f"{MAIN}is")
    if inline is not None:
        return "".join((text.text or "") for text in inline.findall(f".//{MAIN}t"))
    return ""


def _normalize_imputed_flag(flag_code: int | None) -> int | object:
    if flag_code is None:
        return pd.NA
    return 0 if flag_code == 1 else 1


def _parse_ndcp_workbook(path: Path, url: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _shared_strings(archive)
        worksheet_path = _worksheet_target(archive)

        rows: list[dict[str, object]] = []
        header_map: dict[int, str] = {}
        column_index_to_name: dict[int, str] = {}

        required = {"COUNTY_FIPS_CODE", "STATE_FIPS", "STUDYYEAR"}
        for price_col, flag_col, _, _ in PRICE_SPECS:
            required.add(price_col)
            required.add(flag_col)

        with archive.open(worksheet_path) as worksheet:
            for _, element in ET.iterparse(worksheet, events=("end",)):
                if element.tag != f"{MAIN}row":
                    continue

                row_number = _to_int(element.attrib.get("r", ""))
                if row_number == 1:
                    for cell in element.findall(f"{MAIN}c"):
                        cell_index = _column_index(cell.attrib.get("r", ""))
                        if cell_index is None:
                            continue
                        header_map[cell_index] = _cell_text(cell, shared_strings)
                    column_index_to_name = {
                        column_index: name for column_index, name in header_map.items() if name in required
                    }
                    missing = sorted(required - set(column_index_to_name.values()))
                    if missing:
                        missing_text = ", ".join(missing)
                        raise ValueError(f"NDCP workbook missing expected columns: {missing_text}")
                    element.clear()
                    continue

                if not column_index_to_name:
                    element.clear()
                    continue

                row_values: dict[str, str] = {}
                for cell in element.findall(f"{MAIN}c"):
                    cell_index = _column_index(cell.attrib.get("r", ""))
                    if cell_index is None:
                        continue
                    column_name = column_index_to_name.get(cell_index)
                    if column_name is None:
                        continue
                    row_values[column_name] = _cell_text(cell, shared_strings)

                county_fips = _coerce_fips(row_values.get("COUNTY_FIPS_CODE"), width=5)
                year = _to_int(row_values.get("STUDYYEAR"))
                if not county_fips or year is None:
                    element.clear()
                    continue

                state_fips = _coerce_fips(row_values.get("STATE_FIPS"), width=2)
                if not state_fips:
                    state_fips = county_fips[:2]

                for price_col, flag_col, child_age, provider_type in PRICE_SPECS:
                    weekly_price = _to_float(row_values.get(price_col))
                    if weekly_price is None:
                        continue
                    flag_code = _to_int(row_values.get(flag_col))
                    rows.append(
                        {
                            "county_fips": county_fips,
                            "state_fips": state_fips,
                            "year": year,
                            "child_age": child_age,
                            "provider_type": provider_type,
                            "weekly_price": weekly_price,
                            "annual_price": weekly_price * 52.0,
                            "ndcp_flag_code": flag_code,
                            "imputed_flag": _normalize_imputed_flag(flag_code),
                            "sample_weight": 1.0,
                            "source_url": url,
                        }
                    )
                element.clear()

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "county_fips",
                "state_fips",
                "year",
                "child_age",
                "provider_type",
                "weekly_price",
                "annual_price",
                "ndcp_flag_code",
                "imputed_flag",
                "sample_weight",
                "source_url",
            ]
        )

    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["ndcp_flag_code"] = pd.array(frame["ndcp_flag_code"], dtype="Int64")
    frame["imputed_flag"] = pd.array(frame["imputed_flag"], dtype="Int64")
    frame["sample_weight"] = pd.to_numeric(frame["sample_weight"], errors="coerce")
    return frame.sort_values(
        ["county_fips", "year", "provider_type", "child_age"],
        kind="stable",
    ).reset_index(drop=True)


def _download_ndcp_workbook(url: str) -> bytes:
    # DOL serves a bot challenge for some user-agent strings; use a generic requests UA.
    headers = {"User-Agent": "python-requests/2.32.3"}
    try:
        response = requests.get(url, headers=headers, timeout=180)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to fetch {url}: {exc}") from exc
    content_type = (response.headers.get("content-type") or "").lower()
    if "text/html" in content_type:
        snippet = response.content[:1500].decode("utf-8", errors="ignore")
        if "Challenge Validation" in snippet or "<!DOCTYPE html" in snippet:
            raise SourceAccessError(
                f"source returned a challenge page instead of data for {url}; this source likely needs a manual browser download"
            )
    return response.content


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_ndcp, refresh=refresh, dry_run=dry_run)
    url = "https://www.dol.gov/sites/dolgov/files/WB/NDCP2022.xlsx"
    raw_path = paths.raw / SPEC.name / "NDCP2022.xlsx"
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"
    if dry_run:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=url)
    if raw_path.exists() and normalized_path.exists() and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")
    try:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists() or refresh:
            raw_path.write_bytes(_download_ndcp_workbook(url))
        frame = _parse_ndcp_workbook(raw_path, url)
        write_parquet(frame, normalized_path)
        append_registry(
            paths,
            build_record(
                source_name=SPEC.name,
                raw_path=raw_path,
                normalized_path=normalized_path,
                license_name=SPEC.license_name,
                retrieval_method=SPEC.retrieval_method,
                citation=SPEC.citation,
                sample_mode=False,
            ),
        )
        return IngestResult(SPEC.name, raw_path, normalized_path, False)
    except SourceAccessError as exc:
        raise SourceAccessError(
            f"{exc}. Manual fallback: open the DOL NDCP page in a browser, download the 2008-2022 data file, "
            f"save it as '{paths.raw / SPEC.name / 'NDCP2022.xlsx'}', then rerun without --refresh."
        ) from exc
