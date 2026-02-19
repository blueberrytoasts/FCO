# Net Optical Density Plot — Design

**Date:** 2026-02-19

---

## Overview

A standalone script `net_od_plot.py` that reads per-scan CSV results produced by
`film_rgb_analysis.py`, computes Net Optical Density (Net OD = Post OD − Pre OD) for
matched pre/post scan pairs, and produces a single Net OD vs. Dose plot for all three
RGB channels.

---

## CLI Interface

```bash
python net_od_plot.py --pre Pre-001 Pre-002 --post Post-003 Post-004 --doses 0 10 50 200 500
```

| Argument | Description | Default |
|---|---|---|
| `--pre` | One or more pre-scan names (matched by position to `--post`) | required |
| `--post` | One or more post-scan names | required |
| `--doses` | Dose levels in cGy, listed least-to-greatest | required |
| `--output` | Output PNG path | `outputs/net_od_plot.png` |

Pre/post pairs are matched positionally: `--pre Pre-001 Pre-002 --post Post-003 Post-004`
means Pre-001↔Post-003 and Pre-002↔Post-004.

---

## Data Flow

1. For each scan name, load `outputs/<name>/film_analysis_results.csv`
2. Determine replicates per dose level:
   ```
   replicates_per_dose = total_regions / len(doses)
   ```
   Assumes equal replicates per dose level.
3. Map regions to doses — **reversed**, because films are ordered most-to-least exposed
   (highest dose first, lowest dose last):
   - Regions 1–N → highest dose
   - Regions (total−N+1)–total → 0 cGy (largest PV, lightest film)
4. Average the replicates within each dose level per channel.
5. Compute Net OD per pair: `post_od − pre_od` for each channel at each dose level.
6. Plot and save.

---

## Region-to-Dose Mapping Example

With 10 films and doses `[0, 10, 50, 200, 500]` (2 replicates each):

| Regions | Dose (cGy) |
|---------|-----------|
| 1–2 | 500 |
| 3–4 | 200 |
| 5–6 | 50 |
| 7–8 | 10 |
| 9–10 | 0 |

The `--doses` argument is provided least-to-greatest; the script reverses internally.

---

## Plot Design

- **Single figure**, one set of axes
- **X-axis:** Dose (cGy), linear scale
- **Y-axis:** Net OD, linear scale
- **6 lines total:** R, G, B × 2 pairs
  - Color-coded by channel: red, green, blue
  - Solid line = pair 1, dashed line = pair 2
- Markers at each dose point
- Legend identifying channel and pair
- Output saved to `outputs/net_od_plot.png` (fixed name)

---

## Variable Film/Dose Support

Both scripts handle variable counts:

- **`film_rgb_analysis.py`:** `detect_film_regions()` auto-detects however many films are
  present — no hardcoded film count.
- **`net_od_plot.py`:** Number of dose levels driven by `--doses` argument; replicates per
  dose computed dynamically. Works for any equal-replicate configuration.

---

## Files

| File | Role |
|------|------|
| `net_od_plot.py` | New standalone script (to be created) |
| `film_rgb_analysis.py` | Unchanged — produces the CSV inputs |
| `outputs/<name>/film_analysis_results.csv` | Input CSVs (one per scan) |
| `outputs/net_od_plot.png` | Output plot |
