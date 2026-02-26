#!/usr/bin/env python3
"""
RGB Linear Combination Regression
==================================
Fits OD_dektronics = a*OD_R + b*OD_G + c*OD_B using least squares.
Averages two post-scan CSVs (no pre subtraction, no net OD).
Computes R² and RMS. Saves a 2-panel plot and CSV to outputs/.

Usage:
    python rgb_combination_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Two post-irradiation flatbed scan CSVs to average
POST_CSV_PATHS = [
    Path("outputs/Post-003/film_analysis_results.csv"),
    Path("outputs/Post-004/film_analysis_results.csv"),
]

# Dektronics reference OD per dose
DEKTRONICS_CSV_PATH = Path("data/regression data.csv")

# Dose (cGy) for each region, in the order they appear in the scan (1-indexed)
# Two film pieces per dose level → 2 entries per dose
REGION_DOSES = [500, 500, 200, 200, 50, 50, 10, 10, 0, 0]


def fit_linear_combination(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = a*X[:,0] + b*X[:,1] + c*X[:,2] with no intercept.

    Args:
        X: shape (n, 3) — columns are OD_R, OD_G, OD_B
        y: shape (n,)   — OD_dektronics

    Returns:
        (a, b, c) as floats
    """
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None) # coefficients, residuals, rank, and singular values
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def compute_r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Return R² between measured and predicted values."""
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1 - ss_res / ss_tot)


def compute_rms(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean square error."""
    return float(np.sqrt(np.mean((y - y_pred) ** 2)))


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average two post-scan CSVs by region, assign doses, merge with Dektronics OD.

    Returns:
        doses: shape (n,)
        X: shape (n, 3) — averaged OD_R, OD_G, OD_B from flatbed post scans
        y: shape (n,)   — OD_dektronics
    """
    dfs = [pd.read_csv(p) for p in POST_CSV_PATHS]
    avg = (
        pd.concat(dfs)
        .groupby("region")[["red_od", "green_od", "blue_od"]]
        .mean()
        .reset_index()
        .sort_values("region")
        .reset_index(drop=True)
    )

    if len(avg) != len(REGION_DOSES):
        raise ValueError(
            f"REGION_DOSES has {len(REGION_DOSES)} entries but scan has {len(avg)} regions. "
            "Update REGION_DOSES to match."
        )

    avg["dose_cgy"] = REGION_DOSES
    avg_by_dose = avg.groupby("dose_cgy")[["red_od", "green_od", "blue_od"]].mean().reset_index()

    dek = pd.read_csv(DEKTRONICS_CSV_PATH)
    dek = dek.rename(columns={"Dose (cGy)": "dose_cgy", "Avg Net OD (Dektronics)": "od_dektronics"})

    merged = pd.merge(avg_by_dose, dek[["dose_cgy", "od_dektronics"]], on="dose_cgy") # Meres both csv files together
    if len(merged) == 0:
        raise ValueError(
            "Merge produced no rows — check dose values in REGION_DOSES align with "
            f"{DEKTRONICS_CSV_PATH}."
        )

    doses = merged["dose_cgy"].to_numpy(dtype=float)
    X = merged[["red_od", "green_od", "blue_od"]].to_numpy(dtype=float) # Optical density of RGB channels
    y = merged["od_dektronics"].to_numpy(dtype=float) # OD taken from the densitometer
    return doses, X, y


def save_results_csv(
    output_dir: Path,
    doses: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
) -> None:
    df = pd.DataFrame(
        {
            "dose_cgy": doses,
            "od_dektronics": y,
            "od_predicted": y_pred,
            "red_od": X[:, 0],
            "green_od": X[:, 1],
            "blue_od": X[:, 2],
        }
    )
    path = output_dir / "rgb_combination_results.csv"
    df.to_csv(path, index=False)
    print(f"CSV saved to: {path}")


def save_plot(
    output_dir: Path,
    doses: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    a: float,
    b: float,
    c: float,
    r2: float,
    rms: float,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: predicted vs measured (identity line)
    ax1.scatter(y, y_pred, color="black", zorder=3, label="Data points")
    lims = [min(y.min(), y_pred.min()) * 0.9, max(y.max(), y_pred.max()) * 1.1]
    ax1.plot(lims, lims, "r--", label="Identity (y = x)")
    ax1.set_xlabel("Measured OD (Dektronics)")
    ax1.set_ylabel("Predicted OD (a·R + b·G + c·B)")
    ax1.set_title(
        f"Predicted vs Measured\n"
        f"a={a:.3f}, b={b:.3f}, c={c:.3f}\n"
        f"R²={r2:.4f}, RMS={rms:.4f}"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: OD vs Dose — all curves
    ax2.plot(doses, X[:, 0], "r-o", label="Flatbed Red OD")
    ax2.plot(doses, X[:, 1], "g-o", label="Flatbed Green OD")
    ax2.plot(doses, X[:, 2], "b-o", label="Flatbed Blue OD")
    ax2.plot(doses, y, "k-s", linewidth=2, label="Dektronics OD (measured)")
    ax2.plot(doses, y_pred, "k--^", linewidth=2, label="Combined OD (predicted)")
    ax2.set_xlabel("Dose (cGy)")
    ax2.set_ylabel("OD")
    ax2.set_title("OD vs Dose")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "rgb_combination_analysis.png"
    plt.savefig(path, dpi=150)
    print(f"Plot saved to: {path}")
    plt.close()


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    doses, X, y = load_data()
    a, b, c = fit_linear_combination(X, y)
    y_pred = a * X[:, 0] + b * X[:, 1] + c * X[:, 2]
    r2 = compute_r2(y, y_pred)
    rms = compute_rms(y, y_pred)

    print(f"Coefficients: a(Red)={a:.4f}, b(Green)={b:.4f}, c(Blue)={c:.4f}")
    print(f"R²  = {r2:.4f}")
    print(f"RMS = {rms:.4f}")

    save_results_csv(output_dir, doses, y, y_pred, X)
    save_plot(output_dir, doses, y, y_pred, X, a, b, c, r2, rms)


if __name__ == "__main__":
    main()
