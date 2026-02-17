# Output Organization and Film Detection Improvements

**Date:** 2026-02-17

## Summary

Reorganized output file handling to prevent overwrites and fixed film region detection algorithm to properly detect all films.

## Changes

### 1. Removed hardcoded output files
- Deleted `film_analysis_plot.png` and `film_analysis_results.csv` from repo root

### 2. Added per-file output directories
- Modified `main()` in `film_rgb_analysis.py` to create output folders based on input filename
- Outputs now saved to `outputs/<filename>/` (e.g., `outputs/Post-003/`)
- Each analysis gets its own folder, preventing overwrites

### 3. Created `.gitignore`
- Added `outputs/` to prevent generated results from being tracked

### 4. Fixed film detection threshold
- Changed `threshold_factor` from `0.9` to `0.98` in `detect_film_regions()`
- Previously only detected 6/10 films (missed lighter/less exposed films)
- Now correctly detects all 10 film regions

### 5. Corrected max pixel value constant
- Changed `MAX_PV_16BIT` from `65535` to `65536`

## Files Modified
- `film_rgb_analysis.py`
- `.gitignore` (created)
