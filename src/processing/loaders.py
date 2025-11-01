"""Basic data loading utilities."""
from pathlib import Path
from typing import Any


def load_raw_data(path: Path, **read_csv_kwargs: Any):
    """Load a CSV file into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    import pandas as pd  # Imported lazily to keep the scaffolding lightweight.

    return pd.read_csv(path, **read_csv_kwargs)
