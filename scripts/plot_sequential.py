"""
plot_sequential.py — Plot CSV values sequentially by index (row-major order).

Usage:
    python plot_sequential.py data/raw_data_500cGy.csv
    python plot_sequential.py data/raw_data_500cGy.csv --title "500 cGy Raw OD Readings"
    python plot_sequential.py data/raw_data_500cGy.csv --ylabel "Net OD" --no-grid

Output saved to: outputs/<stem>_sequential_<timestamp>.png
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_csv_flat(path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Load a CSV file and return a 1D array of values in row-major (horizontal) order.
    Automatically handles numeric-header rows (e.g., '1,2,3,...') by including or
    skipping them based on whether they look like column labels vs. data.

    Returns:
        values: 1D numpy array
        col_names: list of column name strings
    """
    # Try reading with header row
    df_header = pd.read_csv(path, header=0)
    # Try reading without header
    df_no_header = pd.read_csv(path, header=None)

    # Check if all values in df_header are numeric
    try:
        df_header.values.astype(float)
        header_all_numeric = True
    except (ValueError, TypeError):
        header_all_numeric = False

    # If column names look like sequential integers (1,2,3...) treat as labels, not data
    col_names = list(df_header.columns.astype(str))
    try:
        col_ints = [int(c) for c in col_names]
        is_sequential_int_header = col_ints == list(range(col_ints[0], col_ints[0] + len(col_ints)))
    except (ValueError, TypeError):
        is_sequential_int_header = False

    if header_all_numeric and is_sequential_int_header:
        # Header row is just column labels — skip it, use df_header (data below)
        df = df_header
    elif not header_all_numeric:
        # Real string header
        df = df_header
    else:
        # All rows are data
        df = df_no_header
        col_names = [str(c) for c in df.columns]

    values = df.values.astype(float).ravel()  # row-major flatten
    return values, col_names


def _repo_root() -> Path:
    """Walk up from this script's location to find the repo root (contains CLAUDE.md)."""
    candidate = Path(__file__).resolve().parent
    for p in [candidate, *candidate.parents]:
        if (p / "CLAUDE.md").exists():
            return p
    return candidate  # fallback: use script's directory


def make_output_path(input_path: Path, suffix: str = "sequential") -> Path:
    outputs_dir = _repo_root() / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{input_path.stem}_{suffix}_{timestamp}.png"
    return outputs_dir / filename


def plot_sequential(
    csv_path: str,
    title: str | None = None,
    ylabel: str = "Value",
    xlabel: str = "Index",
    grid: bool = True,
    marker_size: float = 3.0,
    line: bool = True,
) -> Path:
    path = Path(csv_path)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    values, col_names = load_csv_flat(path)
    n_cols = len(col_names)
    n_points = len(values)
    n_rows = n_points // n_cols

    indices = np.arange(1, n_points + 1)

    fig, ax = plt.subplots(figsize=(14, 5))

    plot_kwargs = dict(markersize=marker_size, linewidth=0.8 if line else 0, color="#2563EB")
    fmt = "o-" if line else "o"
    ax.plot(indices, values, fmt, **plot_kwargs, alpha=0.7, label="Measured value")

    # Draw faint vertical separators between rows
    for row_i in range(1, n_rows):
        ax.axvline(row_i * n_cols + 0.5, color="gray", linewidth=0.4, linestyle="--", alpha=0.4)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title or f"Indexed OD — {path.name}", fontsize=13)
    if grid:
        ax.grid(True, which="major", linestyle=":", alpha=0.5)

    # Annotate stats
    mean, std = values.mean(), values.std()
    ax.axhline(mean, color="red", linewidth=1.0, linestyle="--", label=f"Mean = {mean:.6f}")
    ax.fill_between(indices, mean - std, mean + std, color="red", alpha=0.08, label=f"±1σ = {std:.6f}")
    ax.legend(fontsize=10, loc="upper left")

    fig.tight_layout()
    out_path = make_output_path(path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot CSV values sequentially by index (row-major).")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("--title", default=None, help="Plot title (auto-generated if omitted)")
    parser.add_argument("--ylabel", default="OD", help="Y-axis label (default: OD)")
    parser.add_argument("--xlabel", default="Index", help="X-axis label (default: Index)")
    parser.add_argument("--no-grid", dest="grid", action="store_false", help="Disable grid")
    parser.add_argument("--no-line", dest="line", action="store_false", help="Scatter only (no connecting line)")
    parser.add_argument("--marker-size", type=float, default=3.0, help="Marker size (default: 3.0)")
    args = parser.parse_args()

    plot_sequential(
        csv_path=args.csv,
        title=args.title,
        ylabel=args.ylabel,
        xlabel=args.xlabel,
        grid=args.grid,
        line=args.line,
        marker_size=args.marker_size,
    )


if __name__ == "__main__":
    main()
