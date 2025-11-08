"""Minimal configuration helpers for the Chicago 311 project."""
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DEFAULT_TARGET_COLUMN = "estimated_response_time"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


class DataPaths(NamedTuple):
    raw: Path
    interim: Path
    processed: Path


DATA_PATHS = DataPaths(
    raw=RAW_DATA_DIR,
    interim=INTERIM_DATA_DIR,
    processed=PROCESSED_DATA_DIR
)
