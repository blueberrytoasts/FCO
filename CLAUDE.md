# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for film dosimetry analysis in radiation therapy. Goal: develop and validate the IR-FCO (Immediate-Read Film Cut Out) protocol using a handheld Dektronics point densitometer, comparing it against flatbed scanner methods and TLD/OSLD systems.

**Clinical motivation:** Flatbed scanners are centralized (one per clinic); the handheld densitometer is portable, enabling satellite sites to read films on-site without transport.

## Scripts

All scripts live in `scripts/`.

| File | Purpose |
|---|---|
| `film_rgb_analysis.py` | Main analysis — loads TIFFs, detects film regions, extracts ROIs, computes OD per RGB channel |
| `net_od_analysis.py` | Computes Net OD (Post − Pre) for matched scan pairs, plots Net OD vs. Dose |
| `rgb_combination_analysis.py` | Fits `OD_dektronics = a·OD_R + b·OD_G + c·OD_B` via linear regression |
| `regression_analysis.py` | 2nd-order polynomial regression: `Flatbed_OD = A·(Dek_OD)² + B·(Dek_OD) + C` |
| `log_reader.py` | Parses temperature values from densitometer log files; copies tab-separated blocks to clipboard |
| `plot_normalized_od.py` | Plots normalized OD vs. time from a normalized data CSV (e.g., `normalized_data_0cGy.csv`) |
| `plot_sequential.py` | Plots CSV values sequentially by index (row-major order) for reproducibility/drift analysis |

## Core Architecture: FilmAnalyzer (`film_rgb_analysis.py`)

**Workflow:**
1. Load 48-bit TIFF (16-bit/channel) using `tifffile`
2. `detect_film_regions()` — horizontal pixel profiling on red channel, threshold at `0.98 × pv_unexposed`
3. For each region: `find_film_center()` → `extract_roi_circular()` → compute mean PV and OD for R/G/B
4. Export CSV + visualization

**Critical parameters:**
- `threshold_factor = 0.98` — empirically determined; lower values miss lightly exposed films (e.g., 0.9 detects only 6/10)
- `MAX_PV_16BIT = 65536`
- `roi_radius = 25` px default
- `pv_unexposed` (I₀): defaults to 65536

**OD formula:** `OD = log₁₀(I₀ / I)`, clipped to [1, pv_unexposed] to avoid log(0)

**Uncertainty propagation:** `σ_OD = (1/ln(10)) × (σ_PV / PV_mean)`

## RGB Linear Combination (`rgb_combination_analysis.py`)

Characterizes the densitometer's effective spectral response:
```
OD_dektronics = a·OD_R + b·OD_G + c·OD_B
```
No intercept. Input CSV paths are module-level constants (`DEKTRONICS_CSV_PATH`, `RGB_CSV_PATH`) — update when switching scan sessions.

**Latest results (02-26-26, post OD):** a=0.663, b=1.651, c=−1.379 | R²=1.000, RMS=0.0008

Note: coefficients are fit against **post OD** (not net OD), per advisor feedback. Negative c is physically expected — see `notes/rgb_linear_combination_observations.md` for spectral physics rationale.

Outputs: `outputs/rgb_combination_analysis.png`, `outputs/rgb_combination_results.csv`

## Common Commands

```bash
# Film analysis
python scripts/film_rgb_analysis.py "path/to/scan.tif"
python scripts/film_rgb_analysis.py "path/to/scan.tif" --roi-radius 50
python scripts/film_rgb_analysis.py "path/to/scan.tif" --pv-unexposed 60000

# Net OD (pre/post pairs, matched by position)
python scripts/net_od_analysis.py --pre Pre-001 Pre-002 --post Post-003 Post-004 --doses 0 10 50 200 500

# RGB combination regression
python scripts/rgb_combination_analysis.py

# Polynomial regression
python scripts/regression_analysis.py

# Plot normalized OD vs time
python scripts/plot_normalized_od.py data/normalized_data_0cGy.csv

# Plot sequential readings (reproducibility/drift)
python scripts/plot_sequential.py data/raw_data_500cGy.csv

# Parse temperatures from densitometer log
python scripts/log_reader.py

# Tests
python -m pytest tests/  # 13/13 tests passing
```

## Data Organization

**Inputs:**
- TIFFs in dated directories (e.g., `02.03.26 15 MeV Scans/`)
- Naming: `Pre-###.tif`, `Post-###.tif`, `Test #.tif`
- `data/regression data.csv`: columns `Dose (cGy)`, `Avg Net OD (Dektronics)`, `Avg Net OD (Flatbed)`
- `data/normalized_data_0cGy.csv`, `data/normalized_data_500cGy.csv`: time-series normalized OD (trials as columns, time in minutes as rows; value at t=0 is 1.0)
- `data/raw_data_500cGy.csv`, `data/2nd_raw_data_500cGy.csv`: raw sequential OD readings for reproducibility analysis
- `data/log_text.txt`: raw densitometer log output for temperature extraction

**Outputs** (gitignored, auto-created):
- `outputs/<filename>/film_analysis_plot.png` + `film_analysis_results.csv`
- `outputs/<pre>_<post>/net_od_plot.csv` + `net_od_plot.png`
- `outputs/rgb_combination_analysis.png` + `rgb_combination_results.csv`

## Notes

`notes/` contains research observations and physics rationale:
- `rgb_linear_combination_observations.md` — detailed explanation of why c (blue coefficient) is negative, spectral physics of the WW LED + TSL2585 photopic sensor, and fit methodology discussion.

## Tests

`tests/` contains unit tests for `net_od_analysis` and `rgb_combination_analysis`. Run with `pytest`. All 13 tests pass as of 2026-02-19.

Note: import is `net_od_analysis` (not `net_od_plot` — module was renamed).

## Project Specific Aims

1. **Aim 1:** Protocol validation — IR-FCO vs. flatbed scanner
2. **Aim 2:** Therapeutic dose range (100–1000 cGy)
3. **Aim 3:** Low-dose out-of-field monitoring (0.1–100 cGy)

Uncertainty analysis follows AAPM TG-191 guidelines.

## Dependencies

```bash
pip install tifffile numpy matplotlib pandas scipy
```

Falls back to PIL if `tifffile` unavailable (loses 16-bit depth).
