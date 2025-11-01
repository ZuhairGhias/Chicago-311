"""Lightweight cleaning helpers."""
from typing import Iterable


def clean_features(df, columns_to_drop: Iterable[str] | None = None):
    """Return a copy of ``df`` with duplicates dropped and optional columns removed."""
    cleaned = df.drop_duplicates().copy()

    if columns_to_drop:
        cleaned = cleaned.drop(columns=list(columns_to_drop), errors="ignore")

    return cleaned
