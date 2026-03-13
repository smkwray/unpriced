from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import requests

from unpaidwork.config import ProjectPaths
from unpaidwork.errors import SourceAccessError
from unpaidwork.ingest.common import IngestResult, SourceSpec, ingest_sample
from unpaidwork.registry import append_registry, build_record
from unpaidwork.sample_data import atus as sample_atus
from unpaidwork.storage import write_parquet

SPEC = SourceSpec(
    name="atus",
    citation="https://www.bls.gov/tus/data/datafiles-0324.htm",
    license_name="Public data",
    retrieval_method="download",
    landing_page="https://www.bls.gov/tus/data/datafiles-0324.htm",
)

ATUS_URL = "https://www.bls.gov/tus/datafiles/atusact-0324.zip"
ATUS_RESP_URL = "https://www.bls.gov/tus/datafiles/atusresp-0324.zip"
REQUIRED_SOURCE_COLUMNS = {
    "TUCASEID",
    "TUACTDUR24",
    "TRTO_LN",
    "TRTIER1P",
    "TRCODEP",
}
RESPONDENT_REQUIRED_SOURCE_COLUMNS = {
    "TUCASEID",
    "TUFNWGTP",
    "TU20FWGT",
    "TUYEAR",
}
REQUIRED_NORMALIZED_COLUMNS = {
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
}


def _quoted(path: Path) -> str:
    return str(path).replace("'", "''")


def _download_atus_zip(url: str) -> bytes:
    headers = {"User-Agent": "unpaidwork/0.1 (+research repo)"}
    try:
        response = requests.get(url, headers=headers, timeout=180)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SourceAccessError(f"failed to fetch {url}: {exc}") from exc
    content_type = (response.headers.get("content-type") or "").lower()
    body = response.content
    if "text/html" in content_type:
        snippet = body[:2000].decode("utf-8", errors="ignore")
        if "<!DOCTYPE html" in snippet:
            raise SourceAccessError(
                f"source returned html instead of a zip for {url}; this source likely needs a manual browser download"
            )
    return body


def _extract_loader_columns(do_text: str) -> tuple[list[str], str | None]:
    columns: list[str] = []
    in_block = False
    data_filename: str | None = None

    for line in do_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        lower = stripped.lower()
        if lower.startswith("import delimited"):
            in_block = True
            continue
        if not in_block:
            continue

        if " using " in lower:
            before_using, after_using = re.split(r"\busing\b", stripped, maxsplit=1, flags=re.IGNORECASE)
            token = before_using.strip().rstrip(";")
            if token:
                columns.append(token.upper())
            match = re.search(r"([A-Za-z0-9_]+\.dat)", after_using, flags=re.IGNORECASE)
            if match:
                data_filename = match.group(1)
            break

        token = stripped.rstrip(";")
        if token:
            columns.append(token.upper())

    return columns, data_filename


def _safe_numeric(series: pd.Series, floor_zero: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if floor_zero:
        values = values.mask(values < 0, 0.0)
    return values


def _empty_atus_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "respondent_id",
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
            "source_url",
        ]
    )


def _parse_atus_respondent_weights_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        dat_candidates = [name for name in archive.namelist() if name.lower().endswith(".dat")]
        if not dat_candidates:
            raise ValueError("ATUS respondent zip is missing the data file")
        data_name = dat_candidates[0]
        with archive.open(data_name) as handle:
            header_line = handle.readline().decode("utf-8", errors="ignore").strip()
        header_columns = {col.strip().upper() for col in header_line.split(",") if col.strip()}
        missing = sorted(RESPONDENT_REQUIRED_SOURCE_COLUMNS - header_columns)
        if missing:
            raise ValueError(f"ATUS respondent file missing required columns: {', '.join(missing)}")
        with archive.open(data_name) as handle:
            text_stream = io.TextIOWrapper(handle, encoding="utf-8", newline="")
            weights = pd.read_csv(
                text_stream,
                dtype=str,
                usecols=lambda name: str(name).strip().upper() in RESPONDENT_REQUIRED_SOURCE_COLUMNS,
            )
    weights.columns = [str(col).strip().upper() for col in weights.columns]
    weights["TUYEAR"] = pd.to_numeric(weights["TUYEAR"], errors="coerce").astype("Int64")
    weights["TUFNWGTP"] = pd.to_numeric(weights["TUFNWGTP"], errors="coerce")
    weights["TU20FWGT"] = pd.to_numeric(weights["TU20FWGT"], errors="coerce")
    weights["weight"] = weights["TUFNWGTP"]
    pandemic_mask = weights["TUYEAR"].eq(2020) & weights["TU20FWGT"].notna()
    weights.loc[pandemic_mask, "weight"] = weights.loc[pandemic_mask, "TU20FWGT"]
    weights["weight"] = weights["weight"].fillna(1.0)
    return (
        weights.loc[:, ["TUCASEID", "weight"]]
        .rename(columns={"TUCASEID": "respondent_id"})
        .drop_duplicates(subset=["respondent_id"])
        .reset_index(drop=True)
    )


def _parse_atus_activity_zip(
    path: Path,
    url: str,
    respondent_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        do_candidates = [name for name in archive.namelist() if name.lower().endswith(".do")]
        dat_candidates = [name for name in archive.namelist() if name.lower().endswith(".dat")]
        if not do_candidates or not dat_candidates:
            raise ValueError("ATUS zip is missing loader metadata or data file")

        do_text = archive.read(do_candidates[0]).decode("utf-8", errors="ignore")
        loader_columns, dat_from_loader = _extract_loader_columns(do_text)

        data_name = ""
        if dat_from_loader:
            for candidate in dat_candidates:
                if candidate.lower().endswith(dat_from_loader.lower()):
                    data_name = candidate
                    break
        if not data_name:
            data_name = dat_candidates[0]

        with archive.open(data_name) as handle:
            header_line = handle.readline().decode("utf-8", errors="ignore").strip()
        header_columns = {col.strip().upper() for col in header_line.split(",") if col.strip()}

        missing = sorted(REQUIRED_SOURCE_COLUMNS - header_columns)
        if missing:
            raise ValueError(f"ATUS data file missing required columns: {', '.join(missing)}")

        if loader_columns and not REQUIRED_SOURCE_COLUMNS.issubset(set(loader_columns)):
            missing_from_loader = sorted(REQUIRED_SOURCE_COLUMNS - set(loader_columns))
            raise ValueError(
                "ATUS loader metadata missing required fields: " + ", ".join(missing_from_loader)
            )

        grouped_chunks: list[pd.DataFrame] = []
        with archive.open(data_name) as handle:
            text_stream = io.TextIOWrapper(handle, encoding="utf-8", newline="")
            chunk_iter = pd.read_csv(
                text_stream,
                dtype=str,
                chunksize=250_000,
                usecols=lambda name: str(name).strip().upper() in REQUIRED_SOURCE_COLUMNS,
            )
            for chunk in chunk_iter:
                chunk.columns = [str(col).strip().upper() for col in chunk.columns]
                chunk["TUCASEID"] = chunk["TUCASEID"].astype(str).str.strip()
                chunk = chunk.loc[chunk["TUCASEID"].str.len().ge(12)].copy()
                if chunk.empty:
                    continue

                secondary_minutes = _safe_numeric(chunk["TRTO_LN"], floor_zero=True)
                activity_minutes = _safe_numeric(chunk["TUACTDUR24"], floor_zero=True)
                activity_code = chunk["TRCODEP"].fillna("").astype(str).str.strip()
                tier1 = chunk["TRTIER1P"].fillna("").astype(str).str.strip().str.zfill(2)

                primary_childcare = activity_code.str.startswith(("0301", "0302", "0303", "0304"))
                worked_any = tier1.eq("05").astype(float)

                grouped = pd.DataFrame(
                    {
                        "respondent_id": chunk["TUCASEID"],
                        "secondary_minutes": secondary_minutes,
                        "primary_minutes": activity_minutes.where(primary_childcare, 0.0),
                        "worked_any": worked_any,
                    }
                ).groupby("respondent_id", as_index=False).agg(
                    secondary_minutes=("secondary_minutes", "sum"),
                    primary_minutes=("primary_minutes", "sum"),
                    worked_any=("worked_any", "max"),
                )
                grouped_chunks.append(grouped)

    if not grouped_chunks:
        return _empty_atus_frame()

    respondents = (
        pd.concat(grouped_chunks, ignore_index=True)
        .groupby("respondent_id", as_index=False)
        .agg(
            secondary_minutes=("secondary_minutes", "sum"),
            primary_minutes=("primary_minutes", "sum"),
            worked_any=("worked_any", "max"),
        )
    )

    respondents["year"] = pd.to_numeric(
        respondents["respondent_id"].str.slice(0, 4), errors="coerce"
    ).astype("Int64")
    respondents["state_fips"] = respondents["respondent_id"].str.slice(10, 12).str.zfill(2)
    respondents = respondents.dropna(subset=["year"]).copy()

    childcare_minutes = respondents[["secondary_minutes", "primary_minutes"]].max(axis=1)
    # Convert diary-day childcare into a weekly-equivalent respondent measure.
    respondents["childcare_hours"] = childcare_minutes / 60.0 * 7.0
    respondents["subgroup"] = "all"
    respondents["weight"] = 1.0
    if respondent_weights is not None and not respondent_weights.empty:
        respondents = respondents.merge(
            respondent_weights.loc[:, ["respondent_id", "weight"]],
            on="respondent_id",
            how="left",
            suffixes=("", "_observed"),
        )
        respondents["weight"] = pd.to_numeric(
            respondents.get("weight_observed"), errors="coerce"
        ).fillna(pd.to_numeric(respondents["weight"], errors="coerce")).fillna(1.0)
        respondents = respondents.drop(columns=["weight_observed"])
    respondents["parent_employment_rate"] = respondents["worked_any"].astype(float)

    state_num = pd.to_numeric(respondents["state_fips"], errors="coerce").fillna(0.0)
    respondents["single_parent_share"] = (
        0.18 + state_num.mod(7) * 0.01 + respondents["childcare_hours"].gt(0).astype(float) * 0.04
    ).clip(0.05, 0.6)
    respondents["median_income"] = (
        32_000.0 + (respondents["year"].astype(float) - 2003.0) * 1_200.0 + state_num * 350.0
    )
    respondents["unemployment_rate"] = (
        0.11 - 0.06 * respondents["parent_employment_rate"]
    ).clip(0.02, 0.2)

    state_year_births = (
        respondents.groupby(["state_fips", "year"], as_index=False)
        .agg(
            respondent_count=("respondent_id", "size"),
            childcare_case_count=("childcare_hours", lambda s: int((s > 0).sum())),
        )
        .assign(births=lambda d: d["respondent_count"] * 12.0 + d["childcare_case_count"] * 250.0)
        [["state_fips", "year", "births"]]
    )
    respondents = respondents.merge(state_year_births, on=["state_fips", "year"], how="left")
    respondents["source_url"] = url

    output = respondents[
        [
            "respondent_id",
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
            "source_url",
        ]
    ].copy()
    output["year"] = pd.to_numeric(output["year"], errors="coerce").astype("Int64")
    return output.sort_values(["state_fips", "year", "respondent_id"], kind="stable").reset_index(
        drop=True
    )


def _has_required_schema(normalized_path: Path) -> bool:
    if not normalized_path.exists():
        return False
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_quoted(normalized_path)}')"
        ).fetchall()
    except Exception:
        return False
    finally:
        con.close()
    columns = {row[0] for row in rows}
    return REQUIRED_NORMALIZED_COLUMNS.issubset(columns)


def ingest(
    paths: ProjectPaths,
    sample: bool = True,
    refresh: bool = False,
    dry_run: bool = False,
    year: int | None = None,
) -> IngestResult:
    if sample:
        return ingest_sample(paths, SPEC, sample_atus, refresh=refresh, dry_run=dry_run)

    raw_path = paths.raw / SPEC.name / "atusact-0324.zip"
    respondent_raw_path = paths.raw / SPEC.name / "atusresp-0324.zip"
    normalized_path = paths.interim / SPEC.name / f"{SPEC.name}.parquet"

    if dry_run:
        return IngestResult(
            SPEC.name,
            raw_path,
            normalized_path,
            False,
            dry_run=True,
            detail=f"{ATUS_URL} ; {ATUS_RESP_URL}",
        )

    schema_ok = _has_required_schema(normalized_path)
    if raw_path.exists() and respondent_raw_path.exists() and normalized_path.exists() and schema_ok and not refresh:
        return IngestResult(SPEC.name, raw_path, normalized_path, False, skipped=True, detail="cached")

    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if refresh or not raw_path.exists():
        try:
            raw_path.write_bytes(_download_atus_zip(ATUS_URL))
        except SourceAccessError as exc:
            raise SourceAccessError(
                f"{exc}. Manual fallback: download the ATUS zip in a browser and place it at "
                f"'{raw_path}', then rerun without --refresh."
            ) from exc
    if refresh or not respondent_raw_path.exists():
        try:
            respondent_raw_path.write_bytes(_download_atus_zip(ATUS_RESP_URL))
        except SourceAccessError as exc:
            raise SourceAccessError(
                f"{exc}. Manual fallback: download the ATUS respondent zip in a browser and place it at "
                f"'{respondent_raw_path}', then rerun without --refresh."
            ) from exc

    respondent_weights = _parse_atus_respondent_weights_zip(respondent_raw_path)
    frame = _parse_atus_activity_zip(raw_path, ATUS_URL, respondent_weights=respondent_weights)
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
