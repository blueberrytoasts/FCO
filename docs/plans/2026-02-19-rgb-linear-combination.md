# RGB Linear Combination Regression Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `rgb_combination_analysis.py` that fits OD_dektronics = a·OD_R + b·OD_G + c·OD_B using least squares, reports R² and RMS, and saves a 2-panel plot and CSV.

**Architecture:** Standalone script with no CLI args. Loads two existing CSVs, pairs rows by dose, runs `numpy.linalg.lstsq` with no intercept, computes R² and RMS, saves plot and CSV to `outputs/`.

**Tech Stack:** Python 3, numpy, pandas, matplotlib, pytest

---

### Task 1: Write failing tests

**Files:**
- Create: `tests/test_rgb_combination.py`

**Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from rgb_combination_analysis import fit_linear_combination, compute_r2, compute_rms


def test_fit_returns_three_coefficients():
    X = np.array([[0.1, 0.05, 0.02],
                  [0.3, 0.15, 0.06],
                  [0.6, 0.30, 0.12]])
    y = np.array([0.12, 0.36, 0.72])
    a, b, c = fit_linear_combination(X, y)
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert isinstance(c, float)


def test_fit_perfect_red_only():
    # If y = 2 * OD_R exactly, we expect a≈2, b≈0, c≈0
    X = np.array([[0.1, 0.0, 0.0],
                  [0.3, 0.0, 0.0],
                  [0.6, 0.0, 0.0]])
    y = np.array([0.2, 0.6, 1.2])
    a, b, c = fit_linear_combination(X, y)
    assert abs(a - 2.0) < 1e-6
    assert abs(b) < 1e-6
    assert abs(c) < 1e-6


def test_compute_r2_perfect_fit():
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert abs(compute_r2(y, y_pred) - 1.0) < 1e-10


def test_compute_r2_zero_fit():
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])  # predicts the mean
    assert abs(compute_r2(y, y_pred)) < 1e-10


def test_compute_rms_known_value():
    y = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert abs(compute_rms(y, y_pred) - 1.0) < 1e-10


def test_fit_no_intercept_passes_through_origin():
    # With no intercept, zero input must give zero output
    X = np.array([[0.0, 0.0, 0.0],
                  [0.1, 0.05, 0.02],
                  [0.5, 0.25, 0.10]])
    y = np.array([0.0, 0.13, 0.65])
    a, b, c = fit_linear_combination(X, y)
    predicted_zero = a * 0.0 + b * 0.0 + c * 0.0
    assert abs(predicted_zero) < 1e-10
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_rgb_combination.py -v
```

Expected: FAIL with `ImportError: cannot import name 'fit_linear_combination'`

---

### Task 2: Implement core functions

**Files:**
- Create: `rgb_combination_analysis.py`

**Step 1: Write minimal implementation**

```python
#!/usr/bin/env python3
"""
RGB Linear Combination Regression
==================================
Fits OD_dektronics = a*OD_R + b*OD_G + c*OD_B using least squares.
Computes R² and RMS. Saves a 2-panel plot and CSV to outputs/.

Usage:
    python rgb_combination_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def fit_linear_combination(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = a*X[:,0] + b*X[:,1] + c*X[:,2] with no intercept.

    Args:
        X: shape (n, 3) — columns are OD_R, OD_G, OD_B
        y: shape (n,)   — OD_dektronics

    Returns:
        (a, b, c) as floats
    """
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
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
    """Load and pair Dektronics OD with flatbed RGB Net OD by dose.

    Returns:
        doses: shape (n,)
        X: shape (n, 3) — OD_R, OD_G, OD_B from flatbed
        y: shape (n,)   — OD_dektronics
    """
    dek = pd.read_csv("data/regression data.csv")
    rgb = pd.read_csv(
        "outputs/Pre-001_Pre-002_Post-003_Post-004/net_od_plot.csv"
    )

    # Merge on dose
    dek = dek.rename(columns={"Dose (cGy)": "dose_cgy"})
    merged = pd.merge(
        dek[["dose_cgy", "Avg Net OD (Dektronics)"]],
        rgb[["dose_cgy", "net_red_od", "net_green_od", "net_blue_od"]],
        on="dose_cgy",
    )

    doses = merged["dose_cgy"].to_numpy(dtype=float)
    X = merged[["net_red_od", "net_green_od", "net_blue_od"]].to_numpy(dtype=float)
    y = merged["Avg Net OD (Dektronics)"].to_numpy(dtype=float)
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
            "net_red_od": X[:, 0],
            "net_green_od": X[:, 1],
            "net_blue_od": X[:, 2],
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

    # Right: Net OD vs Dose — all curves
    ax2.plot(doses, X[:, 0], "r-o", label="Flatbed Red OD")
    ax2.plot(doses, X[:, 1], "g-o", label="Flatbed Green OD")
    ax2.plot(doses, X[:, 2], "b-o", label="Flatbed Blue OD")
    ax2.plot(doses, y, "k-s", linewidth=2, label="Dektronics OD (measured)")
    ax2.plot(doses, y_pred, "k--^", linewidth=2, label="Combined OD (predicted)")
    ax2.set_xlabel("Dose (cGy)")
    ax2.set_ylabel("Net OD")
    ax2.set_title("Net OD vs Dose")
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
```

**Step 2: Run tests to verify they pass**

```bash
pytest tests/test_rgb_combination.py -v
```

Expected: All 6 tests PASS.

**Step 3: Commit**

```bash
git add rgb_combination_analysis.py tests/test_rgb_combination.py
git commit -m "feat: add RGB linear combination regression"
```

---

### Task 3: Run the script on real data

**Step 1: Run the script**

```bash
python rgb_combination_analysis.py
```

Expected output (values will vary):
```
Coefficients: a(Red)=X.XXXX, b(Green)=X.XXXX, c(Blue)=X.XXXX
R²  = 0.XXXX
RMS = 0.XXXX
CSV saved to: outputs/rgb_combination_results.csv
Plot saved to: outputs/rgb_combination_analysis.png
```

**Step 2: Inspect outputs**

- Open `outputs/rgb_combination_analysis.png` — left panel points should cluster near the identity line; right panel combined curve should track the Dektronics curve closely.
- Open `outputs/rgb_combination_results.csv` — verify 5 rows (one per dose level).

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: verify rgb_combination_analysis runs on real data"
```

---

### Task 4: Write changelog entry

**Files:**
- Create: `changelog/2026-02-19-rgb-linear-combination.md`

```markdown
# RGB Linear Combination Regression

**Date:** 2026-02-19

## Summary

Created `rgb_combination_analysis.py` to fit OD_dektronics = a·OD_R + b·OD_G + c·OD_B
using least squares (no intercept). Reports R², RMS, and saves a 2-panel plot and CSV.

## Changes

### 1. Created `rgb_combination_analysis.py`
- `fit_linear_combination(X, y)`: numpy lstsq, no intercept, returns (a, b, c)
- `compute_r2(y, y_pred)`: standard R² formula
- `compute_rms(y, y_pred)`: root mean square error
- `load_data()`: merges regression data.csv and net_od_plot.csv on dose_cgy
- Outputs: `outputs/rgb_combination_analysis.png`, `outputs/rgb_combination_results.csv`

### 2. Created `tests/test_rgb_combination.py`
- 6 unit tests covering coefficient count, perfect single-channel fit,
  R² edge cases, RMS known value, and zero-input passthrough.

## Usage

    python rgb_combination_analysis.py

## Files Modified
- `rgb_combination_analysis.py` (created)
- `tests/test_rgb_combination.py` (created)
```

**Commit:**

```bash
git add changelog/2026-02-19-rgb-linear-combination.md
git commit -m "docs: add changelog for RGB linear combination regression"
```
