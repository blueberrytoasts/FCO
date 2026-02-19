# RGB Linear Combination Regression — Design

**Date:** 2026-02-19

## Goal

Find coefficients a, b, c such that the Dektronics densitometer OD is modeled as a linear combination of the flatbed scanner's RGB Net OD channels:

```
OD_dektronics = a·OD_R + b·OD_G + c·OD_B
```

This lets us understand what spectral weighting the densitometer effectively applies, and potentially reconstruct "artificial" densitometer curves from flatbed scanner data alone.

## Inputs

| File | Contents |
|------|----------|
| `data/regression data.csv` | Dektronics Net OD averaged per dose (y) |
| `outputs/Pre-001_Pre-002_Post-003_Post-004/net_od_plot.csv` | Flatbed Net OD per channel per dose (X) |

Rows are paired by dose level: 0, 10, 50, 200, 500 cGy.

## Regression Method

- Multiple linear regression via `numpy.linalg.lstsq`, no intercept
- Design matrix X: columns are `net_red_od`, `net_green_od`, `net_blue_od` from flatbed CSV
- Target vector y: `Avg Net OD (Dektronics)` from regression data CSV
- Outputs: coefficients a, b, c — R² — RMS error

No intercept is used because at zero dose, net OD should be zero for both devices.

## Output Script

**`rgb_combination_analysis.py`** — standalone CLI script, no arguments needed.

## Output Plot

Single figure, 2 panels, saved to `outputs/rgb_combination_analysis.png`:

- **Left panel:** Scatter plot of predicted (`a·OD_R + b·OD_G + c·OD_B`) vs measured Dektronics OD, with identity line (y=x) for reference. Annotated with R² and RMS.
- **Right panel:** Net OD vs Dose (cGy) — flatbed Red, Green, Blue curves + Dektronics measured + combined predicted, all on one axes.

## Output CSV

`outputs/rgb_combination_results.csv`

Columns: `dose_cgy, od_dektronics, od_predicted, net_red_od, net_green_od, net_blue_od`

## Spectral Interpretation

The fitted a, b, c weights are compared qualitatively to the LED source spectrum (available as a graph image). No digitization of the spectrum is performed at this stage. The relative magnitudes of a, b, c indicate which wavelength band dominates the densitometer's effective spectral response.

## Success Criteria

- Script runs without error on existing CSV data
- R² and RMS printed to console
- Plot and CSV saved to `outputs/`
- Coefficients a, b, c are physically plausible (e.g., positive, consistent with LED spectrum shape)
