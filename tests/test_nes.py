from __future__ import annotations

from unpaidwork.ingest import nes as nes_ingest
from unpaidwork.ingest.nes import _fetch_nes_rows, _naics_field


def test_naics_field_switches_by_year():
    assert _naics_field(2021) == "NAICS2017"
    assert _naics_field(2022) == "NAICS2022"


def test_fetch_nes_rows_uses_year_specific_naics_field(monkeypatch):
    seen_params: list[dict[str, str]] = []

    class _MockResponse:
        def __init__(self, status_code: int, rows: list[list[str]] | None, url: str):
            self.status_code = status_code
            self._rows = rows
            self.url = url
            self.text = "" if rows is None else "ok"

        def json(self):
            return self._rows or []

    def _mock_get(url, params, timeout):  # noqa: ANN001
        seen_params.append(dict(params))
        rows = [
            ["STATE", "COUNTY", params.get("NAICS2017", params.get("NAICS2022", "")), "NESTAB", "NRCPTOT", "state", "county"],
            ["06", "037", params.get("NAICS2017", params.get("NAICS2022", "")), "1200", "4800", "06", "037"],
        ]
        return _MockResponse(200, rows, f"{url}?ok=1")

    monkeypatch.setattr(nes_ingest.requests, "get", _mock_get)

    rows_2021, _ = _fetch_nes_rows(2021)
    rows_2022, _ = _fetch_nes_rows(2022)

    assert rows_2021[1][2] == "624410"
    assert rows_2022[1][2] == "624410"
    assert "NAICS2017" in seen_params[0]
    assert "NAICS2022" not in seen_params[0]
    assert "NAICS2022" in seen_params[1]
    assert "NAICS2017" not in seen_params[1]
