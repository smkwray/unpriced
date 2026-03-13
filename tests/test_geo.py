from __future__ import annotations

from unpaidwork.geo.crosswalks import ensure_fips, harmonize_cbsa, state_from_county


def test_geo_helpers():
    assert ensure_fips(6037, 5) == "06037"
    assert state_from_county("36061") == "36"
    assert harmonize_cbsa(35620) == "35620"
