from __future__ import annotations

import pandas as pd


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[name], errors="coerce")


def add_provider_density(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    under5 = pd.to_numeric(enriched["under5_population"], errors="coerce")
    employment = _numeric_column(enriched, "employment")
    employer_establishments = _numeric_column(enriched, "employer_establishments")
    nonemployer_firms = _numeric_column(enriched, "nonemployer_firms")

    total_provider_firms = employer_establishments.fillna(0.0).add(nonemployer_firms.fillna(0.0))
    has_business_counts = employer_establishments.notna() | nonemployer_firms.notna()
    total_provider_firms = total_provider_firms.where(has_business_counts)
    provider_base = total_provider_firms.fillna(employer_establishments).fillna(employment)

    enriched["total_provider_firms"] = total_provider_firms
    enriched["provider_density"] = provider_base.div(under5.replace({0: pd.NA})).mul(1000.0).fillna(0.0)
    return enriched
