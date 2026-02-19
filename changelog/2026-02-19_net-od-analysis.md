# Net OD Analysis Script

**Date:** 2026-02-19

## Summary

Created `net_od_analysis.py`, a standalone CLI script that reads per-scan CSV outputs from `film_rgb_analysis.py`, computes Net Optical Density (Post − Pre) and Net Pixel Values for matched pre/post scan pairs, and produces a Net OD vs. Dose plot and CSV export.

## Changes

### 1. Created `net_od_analysis.py`
- CLI interface: `--pre`, `--post`, `--doses`, `--output` arguments
- Matched pre/post pairs by position (e.g. Pre-001 ↔ Post-003)
- Dose-to-region mapping reversed internally (highest dose = region 1, lowest = last)
- Replicates per dose computed dynamically: `n_regions // len(doses)`
- Averages replicates within each dose level per channel
- Averages Net OD across all pairs into a single result (3 lines on plot)
- Raises `ValueError` on uneven film/dose count or mismatched dose keys between pre/post

### 2. Net OD plot (`outputs/net_od_plot.png`)
- Single figure, 3 lines: Red, Green, Blue (color-coded)
- X-axis: Dose (cGy), linear; Y-axis: Net OD, linear
- Pairs averaged together before plotting

### 3. Net OD CSV export (`outputs/net_od_plot.csv`)
- Auto-exported every run alongside the plot
- Columns: `dose_cgy, net_red_od, net_green_od, net_blue_od, net_red_pv, net_green_pv, net_blue_pv`

### 4. Created `tests/test_net_od_plot.py`
- 7 unit tests covering: dose-region mapping, uneven divide guard, per-dose averaging, Net OD calculation, mismatched dose key guard

### 5. Design doc
- Saved to `docs/plans/2026-02-19-net-od-plot-design.md`

## Usage

```bash
python net_od_analysis.py --pre Pre-001 Pre-002 --post Post-003 Post-004 --doses 0 10 50 200 500
```

## Files Modified
- `net_od_analysis.py` (created; renamed from `net_od_plot.py`)
- `tests/test_net_od_plot.py` (created)
- `tests/__init__.py` (created)
- `docs/plans/2026-02-19-net-od-plot-design.md` (created)
- `docs/plans/2026-02-19-net-od-plot.md` (created)
