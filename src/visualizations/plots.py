"""Minimal plotting helper."""
from typing import Optional


def plot_response_distribution(df, response_column: str, bins: int = 40, output_path: Optional[str] = None):
    """Plot a histogram of ``response_column`` and optionally save it."""
    import matplotlib.pyplot as plt

    ax = df[response_column].plot.hist(bins=bins, grid=False)
    ax.set_title(f"Distribution of {response_column}")
    ax.set_xlabel(response_column)
    ax.set_ylabel("Frequency")

    if output_path:
        figure = ax.get_figure()
        figure.tight_layout()
        figure.savefig(output_path)

    return ax
