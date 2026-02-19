# RGB Linear Combination Regression

**Date:** 2026-02-19

## Summary

Created `rgb_combination_analysis.py` to fit OD_dektronics = a·OD_R + b·OD_G + c·OD_B
using least squares (no intercept). Reports R², RMS, and saves a 2-panel plot and CSV.

## Results (02-03-26 scan session)

- a(Red) = 0.6700, b(Green) = 2.4207, c(Blue) = −3.2946
- R² = 1.0000, RMS = 0.0009

## Changes

### 1. Created `rgb_combination_analysis.py`
- `fit_linear_combination(X, y)`: numpy lstsq, no intercept, returns (a, b, c)
- `compute_r2(y, y_pred)`: standard R² formula
- `compute_rms(y, y_pred)`: root mean square error
- `load_data()`: merges `data/regression data.csv` and net_od_plot.csv on dose_cgy; input paths exposed as module-level constants
- Post-merge validation raises ValueError on empty result
- Outputs: `outputs/rgb_combination_analysis.png`, `outputs/rgb_combination_results.csv`

### 2. Created `tests/test_rgb_combination.py`
- 6 unit tests covering: coefficient count and type, perfect single-channel fit,
  R² edge cases (perfect and zero), RMS known value, no-intercept constraint.

## Usage

    python rgb_combination_analysis.py

## Files Modified
- `rgb_combination_analysis.py` (created)
- `tests/test_rgb_combination.py` (created)
