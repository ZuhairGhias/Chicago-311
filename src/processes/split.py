"""Helpers for creating train/test splits."""
from typing import Tuple

from src.config import DEFAULT_RANDOM_STATE, DEFAULT_TARGET_COLUMN, DEFAULT_TEST_SIZE


def split_train_test(
    df,
    target_column: str = DEFAULT_TARGET_COLUMN,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple:
    """Split ``df`` into train/test feature and target sets."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe")

    features = df.drop(columns=[target_column])
    target = df[target_column]

    from sklearn.model_selection import train_test_split  # Lazy import

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
