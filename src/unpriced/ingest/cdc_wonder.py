from __future__ import annotations

import csv
import io
from html.parser import HTMLParser
import re
from urllib.parse import urljoin

import pandas as pd
import requests

from unpriced.config import ProjectPaths
from unpriced.errors import SourceAccessError
from unpriced.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpriced.registry import append_registry, build_record
from unpriced.storage import read_parquet, write_parquet

SPEC = SourceSpec(
    name="cdc_wonder",
    citation="https://wonder.cdc.gov/natality-current.html",
    license_name="Public data",
    retrieval_method="http-form-export",
    landing_page="https://wonder.cdc.gov/natality-current.html",
)

WONDER_DATAREQUEST_URL = "https://wonder.cdc.gov/controller/datarequest/D66"
WONDER_LANDING_PAGE = "https://wonder.cdc.gov/natality-current.html"
SUPPRESSION_TERMS = (
    "suppressed",
    "not available",
    "not reported",
    "missing",
    "unreliable",
    "too small",
)


class _WonderFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_form = False
        self.form_action = ""
        self.inputs: list[dict[str, str]] = []
        self.selects: list[dict[str, object]] = []
        self._current_select: dict[str, object] | None = None
        self.textareas: list[dict[str, str]] = []
        self._current_textarea: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: (value or "") for key, value in attrs if key}
        if tag == "form":
            action = attributes.get("action", "")
            self._in_form = "/controller/datarequest/D66" in action
            if self._in_form:
                self.form_action = action
            return
        if not self._in_form:
            return
        if tag == "input":
            self.inputs.append(attributes)
            return
        if tag == "select":
            self._current_select = {"attrs": attributes, "options": []}
            return
        if tag == "option" and self._current_select is not None:
            options = self._current_select["options"]
            assert isinstance(options, list)
            options.append(attributes)
            return
        if tag == "textarea":
            self._current_textarea = {"attrs": attributes, "text": ""}

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._in_form = False
            return
        if not self._in_form:
            return
        if tag == "select" and self._current_select is not None:
            self.selects.append(self._current_select)
            self._current_select = None
            return
        if tag == "textarea" and self._current_textarea is not None:
            self.textareas.append(self._current_textarea)
            self._current_textarea = None

    def handle_data(self, data: str) -> None:
        if not self._in_form or self._current_textarea is None:
            return
        self._current_textarea["text"] += data


def _empty_wonder_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "state_fips",
            "year",
            "births",
            "source_url",
            "births_raw",
            "births_suppressed",
            "suppression_note",
        ]
    )


def _sample_cdc_wonder() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": "06",
                "year": 2021,
                "births": 420_786.0,
                "source_url": WONDER_LANDING_PAGE,
                "births_raw": "420786",
                "births_suppressed": False,
                "suppression_note": pd.NA,
            },
            {
                "state_fips": "48",
                "year": 2021,
                "births": 377_397.0,
                "source_url": WONDER_LANDING_PAGE,
                "births_raw": "377397",
                "births_suppressed": False,
                "suppression_note": pd.NA,
            },
            {
                "state_fips": "36",
                "year": 2021,
                "births": 219_375.0,
                "source_url": WONDER_LANDING_PAGE,
                "births_raw": "219375",
                "births_suppressed": False,
                "suppression_note": pd.NA,
            },
        ]
    )


def _extract_error_messages(html: str) -> list[str]:
    messages = []
    for match in re.finditer(r'<div class="error-message">\s*(.*?)</div>', html, flags=re.IGNORECASE | re.DOTALL):
        text = re.sub(r"<[^>]+>", " ", match.group(1))
        text = " ".join(text.split())
        if text:
            messages.append(text)
    return messages


def _replace_payload_value(
    payload: list[tuple[str, str]],
    key: str,
    values: str | list[str],
) -> list[tuple[str, str]]:
    result = [(name, value) for name, value in payload if name != key]
    if isinstance(values, list):
        result.extend((key, value) for value in values)
    else:
        result.append((key, values))
    return result


def _parse_form_defaults(html: str) -> tuple[str, list[tuple[str, str]], list[str]]:
    parser = _WonderFormParser()
    parser.feed(html)
    parser.close()

    if not parser.form_action:
        raise SourceAccessError("CDC WONDER response is missing the natality request form action URL")

    payload: list[tuple[str, str]] = []
    for item in parser.inputs:
        name = item.get("name", "")
        if not name or "disabled" in item:
            continue
        input_type = item.get("type", "text").lower()
        if input_type in {"submit", "button", "image", "file", "reset"}:
            continue
        if input_type in {"checkbox", "radio"} and "checked" not in item:
            continue
        payload.append((name, item.get("value", "")))

    year_values: list[str] = []
    for select in parser.selects:
        attrs = select["attrs"]
        assert isinstance(attrs, dict)
        name = str(attrs.get("name", ""))
        if not name or "disabled" in attrs:
            continue
        options = select["options"]
        assert isinstance(options, list)
        selected_values = [str(opt.get("value", "")) for opt in options if "selected" in opt]
        if not selected_values and options:
            selected_values = [str(options[0].get("value", ""))]
        payload.extend((name, value) for value in selected_values)
        if name == "V_D66.V20":
            year_values = [
                str(opt.get("value", "")).strip()
                for opt in options
                if re.fullmatch(r"\d{4}", str(opt.get("value", "")).strip())
            ]

    for text_area in parser.textareas:
        attrs = text_area.get("attrs", {})
        name = str(attrs.get("name", ""))
        if not name or "disabled" in attrs:
            continue
        payload.append((name, text_area.get("text", "").strip()))

    action_url = urljoin(WONDER_DATAREQUEST_URL, parser.form_action)
    return action_url, payload, sorted(set(year_values))


def _resolve_years(available: list[str], year: int | None) -> list[str]:
    if year is None:
        if not available:
            raise SourceAccessError("CDC WONDER request form did not expose selectable natality years")
        return available
    requested = str(int(year))
    if available and requested not in set(available):
        raise SourceAccessError(
            f"CDC WONDER natality year {requested} is unavailable; choose one of: {', '.join(available)}"
        )
    return [requested]


def _download_wonder_tsv(year: int | None = None) -> tuple[str, str]:
    session = requests.Session()
    headers = {"User-Agent": "unpriced/0.1 (+research repo)"}

    try:
        request_page = session.post(
            WONDER_DATAREQUEST_URL,
            data={"stage": "about", "action-I Agree": "I Agree"},
            headers=headers,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to open CDC WONDER natality form: {exc}") from exc
    if request_page.status_code >= 400:
        raise SourceAccessError(
            f"CDC WONDER natality form request failed (HTTP {request_page.status_code})"
        )

    request_action, request_payload, available_years = _parse_form_defaults(request_page.text)
    target_years = _resolve_years(available_years, year)
    request_payload = _replace_payload_value(request_payload, "B_1", "D66.V21-level1")
    request_payload = _replace_payload_value(request_payload, "B_2", "D66.V20")
    request_payload = _replace_payload_value(request_payload, "B_3", "*None*")
    request_payload = _replace_payload_value(request_payload, "B_4", "*None*")
    request_payload = _replace_payload_value(request_payload, "B_5", "*None*")
    request_payload = _replace_payload_value(request_payload, "O_location", "D66.V21")
    request_payload = _replace_payload_value(request_payload, "V_D66.V20", target_years)
    request_payload = _replace_payload_value(request_payload, "O_show_totals", "false")
    request_payload = _replace_payload_value(request_payload, "O_show_suppressed", "true")
    request_payload = _replace_payload_value(request_payload, "action-Send", "Send")

    try:
        results_page = session.post(
            request_action,
            data=request_payload,
            headers=headers,
            timeout=180,
        )
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to submit CDC WONDER natality query: {exc}") from exc
    if results_page.status_code >= 400:
        raise SourceAccessError(
            f"CDC WONDER natality query request failed (HTTP {results_page.status_code})"
        )
    result_errors = _extract_error_messages(results_page.text)
    if result_errors:
        raise SourceAccessError(
            "CDC WONDER natality query returned errors: " + " | ".join(result_errors)
        )

    results_action, results_payload, _ = _parse_form_defaults(results_page.text)
    results_payload = _replace_payload_value(results_payload, "O_export-format", "tsv")
    results_payload = _replace_payload_value(results_payload, "action-Export", "Download")

    try:
        export_response = session.post(
            results_action,
            data=results_payload,
            headers=headers,
            timeout=180,
        )
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to export CDC WONDER natality query as TSV: {exc}") from exc
    if export_response.status_code >= 400:
        raise SourceAccessError(
            f"CDC WONDER natality export request failed (HTTP {export_response.status_code})"
        )
    text = export_response.text
    if "<!DOCTYPE html" in text[:500]:
        export_errors = _extract_error_messages(text)
        if export_errors:
            raise SourceAccessError(
                "CDC WONDER natality export returned errors: " + " | ".join(export_errors)
            )
        raise SourceAccessError("CDC WONDER natality export returned HTML instead of TSV data")
    return text, export_response.url or WONDER_LANDING_PAGE


def _extract_source_url(tsv_text: str) -> str:
    match = re.search(r"https?://wonder\.cdc\.gov/[^\s\"']+", tsv_text)
    if not match:
        return WONDER_LANDING_PAGE
    return match.group(0).rstrip(".,")


def _parse_births_value(raw_value: str) -> float | pd._libs.missing.NAType:
    token = raw_value.strip().replace(",", "")
    token = token.rstrip("*")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
        return float(token)
    return pd.NA


def _suppression_note(raw_value: str, notes_value: str) -> str | None:
    text = f"{raw_value} {notes_value}".strip().lower()
    if any(term in text for term in SUPPRESSION_TERMS):
        return notes_value.strip() or raw_value.strip() or "suppressed"
    return None


def _parse_wonder_tsv(tsv_text: str, source_url: str) -> pd.DataFrame:
    reader = csv.reader(io.StringIO(tsv_text), delimiter="\t", quotechar='"')
    try:
        header = next(reader)
    except StopIteration:
        return _empty_wonder_frame()
    header = [column.replace("\ufeff", "").strip() for column in header]
    lookup = {name: idx for idx, name in enumerate(header)}
    required = {"State Code", "Year", "Births"}
    missing = sorted(required - set(lookup))
    if missing:
        raise ValueError(f"CDC WONDER export missing required columns: {', '.join(missing)}")

    notes_idx = lookup.get("Notes")
    records: list[dict[str, object]] = []
    for row in reader:
        if not row:
            continue
        if row[0].strip() == "---":
            break
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        state_fips = row[lookup["State Code"]].strip().zfill(2)
        year_token = row[lookup["Year"]].strip()
        births_raw = row[lookup["Births"]].strip()
        notes_value = row[notes_idx].strip() if notes_idx is not None and notes_idx < len(row) else ""
        if not re.fullmatch(r"\d{2}", state_fips) or not re.fullmatch(r"\d{4}", year_token):
            continue
        births_value = _parse_births_value(births_raw)
        suppression_note = _suppression_note(births_raw, notes_value)
        records.append(
            {
                "state_fips": state_fips,
                "year": int(year_token),
                "births": births_value,
                "source_url": source_url,
                "births_raw": births_raw or pd.NA,
                "births_suppressed": bool(suppression_note),
                "suppression_note": suppression_note if suppression_note else pd.NA,
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        return _empty_wonder_frame()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["births"] = pd.to_numeric(frame["births"], errors="coerce")
    frame["births_suppressed"] = frame["births_suppressed"].astype(bool)
    return frame.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)


def _fallback_error(exc: SourceAccessError, raw_path: str) -> SourceAccessError:
    return SourceAccessError(
        f"{exc}. Documented flat-file fallback: in a browser, run the CDC WONDER natality query "
        f"grouped by State and Year (show suppressed values), export as TSV, and save it to '{raw_path}'."
    )


def _merge_existing_years(
    normalized_path,
    new_frame: pd.DataFrame,
    target_year: int | None,
    refresh: bool,
) -> pd.DataFrame:
    if target_year is None or not normalized_path.exists():
        return new_frame
    if not refresh and normalized_path.exists():
        return new_frame

    existing = read_parquet(normalized_path)
    if existing.empty:
        return new_frame

    years = set(pd.to_numeric(new_frame["year"], errors="coerce").dropna().astype(int).tolist())
    if not years:
        return existing

    keep_existing = existing.loc[
        ~pd.to_numeric(existing["year"], errors="coerce").isin(sorted(years))
    ].copy()
    combined = pd.concat([keep_existing, new_frame], ignore_index=True)
    combined = combined.sort_values(["state_fips", "year"], kind="stable").reset_index(drop=True)
    return combined


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, _sample_cdc_wonder, refresh=refresh, dry_run=dry_run)

    target_year = int(year) if year is not None else None
    suffix = f"{target_year}" if target_year is not None else "state_year"
    raw_path = paths.raw / SPEC.name / f"cdc_wonder_{suffix}.tsv"
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"

    if dry_run:
        detail = str(target_year) if target_year is not None else WONDER_LANDING_PAGE
        return IngestResult(SPEC.name, raw_path, normalized_path, False, dry_run=True, detail=detail)
    if raw_path.exists() and normalized_path.exists() and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    source_url = WONDER_LANDING_PAGE
    raw_text = ""
    if refresh or not raw_path.exists():
        try:
            raw_text, source_url = _download_wonder_tsv(target_year)
            raw_path.write_text(raw_text, encoding="utf-8")
        except SourceAccessError as exc:
            if not raw_path.exists():
                raise _fallback_error(exc, str(raw_path)) from exc
    if not raw_text:
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    source_url = _extract_source_url(raw_text) or source_url
    frame = _parse_wonder_tsv(raw_text, source_url)
    frame = _merge_existing_years(normalized_path, frame, target_year, refresh)

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
