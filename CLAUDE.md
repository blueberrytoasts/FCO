# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for film dosimetry analysis in radiation therapy. Goal: develop and validate the IR-FCO (Immediate-Read Film Cut Out) protocol using a handheld Dektronics point densitometer, comparing it against flatbed scanner methods and TLD/OSLD systems.

**Clinical motivation:** Flatbed scanners are centralized (one per clinic); the handheld densitometer is portable, enabling satellite sites to read films on-site without transport.

## Scripts

| File | Purpose |
|---|---|
| `film_rgb_analysis.py` | Main analysis — loads TIFFs, detects film regions, extracts ROIs, computes OD per RGB channel |
| `net_od_analysis.py` | Computes Net OD (Post − Pre) for matched scan pairs, plots Net OD vs. Dose |
| `rgb_combination_analysis.py` | Fits `OD_dektronics = a·OD_R + b·OD_G + c·OD_B` via linear regression |
| `regression_analysis.py` | 2nd-order polynomial regression: `Flatbed_OD = A·(Dek_OD)² + B·(Dek_OD) + C` |

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

**Latest results (02-03-26):** a=0.670, b=2.421, c=−2.295 | R²=1.000, RMS=0.0009

Outputs: `outputs/rgb_combination_analysis.png`, `outputs/rgb_combination_results.csv`

## Common Commands

```bash
# Film analysis
python film_rgb_analysis.py "path/to/scan.tif"
python film_rgb_analysis.py "path/to/scan.tif" --roi-radius 50
python film_rgb_analysis.py "path/to/scan.tif" --pv-unexposed 60000

# Net OD (pre/post pairs, matched by position)
python net_od_analysis.py --pre Pre-001 Pre-002 --post Post-003 Post-004 --doses 0 10 50 200 500

# RGB combination regression
python rgb_combination_analysis.py

# Polynomial regression
python regression_analysis.py

# Tests
python -m pytest tests/  # 13/13 tests passing
```

## Data Organization

**Inputs:**
- TIFFs in dated directories (e.g., `02.03.26 15 MeV Scans/`)
- Naming: `Pre-###.tif`, `Post-###.tif`, `Test #.tif`
- `data/regression data.csv`: columns `Dose (cGy)`, `Avg Net OD (Dektronics)`, `Avg Net OD (Flatbed)`

**Outputs** (gitignored, auto-created):
- `outputs/<filename>/film_analysis_plot.png` + `film_analysis_results.csv`
- `outputs/<pre>_<post>/net_od_plot.csv` + `net_od_plot.png`
- `outputs/rgb_combination_analysis.png` + `rgb_combination_results.csv`

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
