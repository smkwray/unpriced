from __future__ import annotations


class UnpaidWorkError(RuntimeError):
    """Base project exception."""


class SourceAccessError(UnpaidWorkError):
    """Raised when a public source cannot be fetched cleanly."""


class DataSchemaError(UnpaidWorkError):
    """Raised when a normalized dataset does not meet the expected schema."""
