from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def _quoted(path: Path) -> str:
    return str(path).replace("'", "''")


def write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("frame", frame)
        con.execute(f"COPY frame TO '{_quoted(path)}' (FORMAT PARQUET)")
    finally:
        con.close()
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"SELECT * FROM read_parquet('{_quoted(path)}')").df()


def write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
    return path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
