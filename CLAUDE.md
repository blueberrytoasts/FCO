# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for film dosimetry analysis in radiation therapy. The primary goal is to develop and validate the IR-FCO (Immediate-Read Film Cut Out) dosimetry protocol using a handheld point densitometer, comparing it against standard flatbed scanner methods and established TLD/OSLD systems.

## Core Architecture

### FilmAnalyzer Class (`film_rgb_analysis.py`)

The main analysis pipeline is built around the `FilmAnalyzer` class which processes 48-bit TIFF scans from flatbed scanners (e.g., Epson Expression 10000XL).

**Key workflow:**
1. Load 16-bit TIFF image (preserves full bit depth using `tifffile` library)
2. Auto-detect film regions via `detect_film_regions()` using horizontal pixel value profiling
3. For each detected region:
   - Find center of mass with `find_film_center()`
   - Extract circular ROI with `extract_roi_circular()` (current method) or full-height ROI with `extract_roi()` (legacy)
   - Calculate mean pixel values and optical density for R/G/B channels
4. Export results to CSV + visualization plots

**Critical parameters:**
- `threshold_factor = 0.98` for film detection (values < 0.98 will miss lightly exposed films)
- `MAX_PV_16BIT = 65536` (maximum pixel value for 16-bit scanner)
- `roi_radius = 30` pixels (default circular ROI size)
- `pv_unexposed` (I₀): reference pixel value for unexposed film, defaults to 65536

**Optical density calculation:**
```
OD = log₁₀(I₀ / I)
```
where I₀ is unexposed pixel value and I is measured pixel value.

### Calibration/Regression (`regression.py`)

Performs 2nd-order polynomial regression to calibrate Dektronics densitometer readings against flatbed scanner measurements:

```
Flatbed_OD = A*(Dek_OD)² + B*(Dek_OD) + C
```

This allows converting handheld densitometer readings to scanner-equivalent optical density values.

## Common Development Tasks

### Running Film Analysis

**Basic usage:**
```bash
python film_rgb_analysis.py "path/to/scan.tif"
```

**Custom ROI radius:**
```bash
python film_rgb_analysis.py "path/to/scan.tif" --roi-radius 50
```

**Legacy mode (full-height ROI):**
```bash
python film_rgb_analysis.py "path/to/scan.tif" --legacy-mode
```

**Custom unexposed pixel value:**
```bash
python film_rgb_analysis.py "path/to/scan.tif" --pv-unexposed 60000
```

### Running Regression Analysis

```bash
python regression.py
```

Expects `regression data.csv` with columns: `Dose (cGy)`, `Avg Net OD (Dektronics)`, `Avg Net OD (Flatbed)`

Outputs: `outputs/dosimetry_analysis.png` with comparison plots and calibration curve.

### Installing Dependencies

```bash
pip install tifffile numpy matplotlib pandas scipy --break-system-packages
```

Note: `--break-system-packages` may be needed on some systems. Falls back to PIL if `tifffile` unavailable (but may lose 16-bit depth).

## Data Organization

### Input Data
- TIFF scans stored in dated directories (e.g., `02.03.26 15 MeV Scans/`)
- 48-bit TIFF files with naming convention: `Pre-###.tif`, `Post-###.tif`, `Test #.tif`
- CSV calibration data: `regression data.csv`

### Output Structure
- All analysis outputs go to `outputs/<filename>/` (auto-created per input file)
- Each output folder contains:
  - `film_analysis_plot.png`: visualization with detected regions, pixel values, optical density
  - `film_analysis_results.csv`: tabular results with PV and OD for each region

The `outputs/` directory is gitignored to prevent committing generated results.

## Important Implementation Details

### Film Detection Algorithm
The `detect_film_regions()` method uses horizontal pixel profiling on the red channel:
1. Calculate mean pixel value for each column
2. Threshold at 0.98 × pv_unexposed
3. Find continuous regions where mean PV < threshold
4. Filter out regions narrower than `min_width` (default 20 pixels)

**Critical:** The threshold factor of 0.98 was determined empirically. Values too low (e.g., 0.9) will miss lightly exposed films, detecting only 6/10 films instead of all 10.

### ROI Extraction Methods
Two methods available (configured via `use_circular_roi` parameter):

1. **Circular ROI (current method):**
   - Finds center of mass of film piece within detected x-boundaries
   - Extracts circular region of specified radius
   - Uses `nanmean`/`nanstd` to ignore pixels outside circle
   - More accurate for non-uniform film pieces

2. **Legacy full-height ROI:**
   - Uses entire height of detected region with 10% margin
   - Backward compatibility mode

### Pixel Value to Optical Density Conversion
The conversion handles edge cases:
- Clips pixel values to range [1, pv_unexposed] to avoid log(0) or log(negative)
- Propagates uncertainty: `σ_OD = (1/ln(10)) × (σ_PV / PV_mean)`

## Project Specific Aims

See README.md for full details, but analysis should support:

1. **Specific Aim 1:** Protocol validation comparing IR-FCO vs flatbed scanner
2. **Specific Aim 2:** Therapeutic dose range (100-1000 cGy) validation
3. **Specific Aim 3:** Low-dose out-of-field monitoring (0.1-100 cGy)

Uncertainty analysis follows AAPM TG-191 guidelines. Comparisons with TLD and OSLD systems are planned.
