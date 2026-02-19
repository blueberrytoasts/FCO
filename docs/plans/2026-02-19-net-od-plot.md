# Net OD Plot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `net_od_plot.py` — a standalone CLI script that reads per-scan CSV outputs, computes Net OD (Post − Pre) for matched scan pairs, and saves a single Net OD vs. Dose plot with all 3 RGB channels.

**Architecture:** CLI script reads existing `outputs/<name>/film_analysis_results.csv` files produced by `film_rgb_analysis.py`. Core logic (dose mapping, net OD calculation) is separated into pure functions for testability. Plot is generated with matplotlib and saved to `outputs/net_od_plot.png`.

**Tech Stack:** Python 3, pandas, matplotlib, argparse. No new dependencies.

---

## Context

### File paths
- **New script:** `net_od_plot.py` (repo root)
- **Input CSVs:** `outputs/<scan-name>/film_analysis_results.csv` (one per scan)
- **Output plot:** `outputs/net_od_plot.png`
- **Design doc:** `docs/plans/2026-02-19-net-od-plot-design.md`

### CSV structure (from `film_rgb_analysis.py` output)
Columns: `region, x_start, x_end, center_x, center_y, roi_radius, red_pv, red_od, green_pv, green_od, blue_pv, blue_od`

Regions are ordered **most-to-least exposed** (highest dose first). With 10 films and 5 doses, regions 1-2 → 500 cGy ... regions 9-10 → 0 cGy.

### CLI usage
```bash
python net_od_plot.py --pre Pre-001 Pre-002 --post Post-003 Post-004 --doses 0 10 50 200 500
```

### Plot design
- 6 lines: R/G/B × pair 1/pair 2
- Color = channel (red, green, blue)
- Solid line = pair 1, dashed = pair 2
- Linear x and y axes
- Output: `outputs/net_od_plot.png`

---

## Task 1: Set up script skeleton and CLI

**Files:**
- Create: `net_od_plot.py`

**Step 1: Create the file with imports and CLI parser**

```python
#!/usr/bin/env python3
"""
Net Optical Density Plot
========================
Reads per-scan CSV results from film_rgb_analysis.py, computes Net OD
(Post OD - Pre OD) for matched pre/post scan pairs, and plots Net OD vs. Dose.

Usage:
    python net_od_plot.py --pre Pre-001 Pre-002 --post Post-003 Post-004 --doses 0 10 50 200 500
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot Net OD vs. Dose from matched pre/post film scans'
    )
    parser.add_argument('--pre', nargs='+', required=True,
                        help='Pre-scan names (e.g. Pre-001 Pre-002)')
    parser.add_argument('--post', nargs='+', required=True,
                        help='Post-scan names, matched by position to --pre')
    parser.add_argument('--doses', nargs='+', type=float, required=True,
                        help='Dose levels in cGy, least-to-greatest (e.g. 0 10 50 200 500)')
    parser.add_argument('--output', default='outputs/net_od_plot.png',
                        help='Output PNG path (default: outputs/net_od_plot.png)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Pre scans:  {args.pre}")
    print(f"Post scans: {args.post}")
    print(f"Doses:      {args.doses}")
```

**Step 2: Verify CLI runs**

```bash
python net_od_plot.py --pre Pre-001 --post Post-003 --doses 0 10 50 200 500
```

Expected output:
```
Pre scans:  ['Pre-001']
Post scans: ['Post-003']
Doses:      [0.0, 10.0, 50.0, 200.0, 500.0]
```

**Step 3: Commit**

```bash
git add net_od_plot.py
git commit -m "feat: add net_od_plot.py skeleton with CLI parser"
```

---

## Task 2: Implement dose-region mapping

**Files:**
- Modify: `net_od_plot.py`

**Step 1: Write the failing test**

Create `tests/test_net_od_plot.py`:

```python
import pytest
import pandas as pd
from net_od_plot import map_regions_to_doses


def test_map_regions_to_doses_basic():
    """10 films, 5 doses -> 2 replicates each, reversed order."""
    doses = [0, 10, 50, 200, 500]
    n_regions = 10
    result = map_regions_to_doses(n_regions, doses)
    # regions are 1-indexed; highest dose is first
    assert result[1] == 500
    assert result[2] == 500
    assert result[3] == 200
    assert result[9] == 0
    assert result[10] == 0


def test_map_regions_to_doses_variable():
    """6 films, 3 doses -> 2 replicates each."""
    doses = [0, 50, 500]
    n_regions = 6
    result = map_regions_to_doses(n_regions, doses)
    assert result[1] == 500
    assert result[2] == 500
    assert result[3] == 50
    assert result[4] == 50
    assert result[5] == 0
    assert result[6] == 0


def test_map_regions_to_doses_three_replicates():
    """9 films, 3 doses -> 3 replicates each."""
    doses = [0, 50, 500]
    n_regions = 9
    result = map_regions_to_doses(n_regions, doses)
    assert result[1] == 500
    assert result[3] == 500
    assert result[4] == 50
    assert result[6] == 50
    assert result[7] == 0
    assert result[9] == 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_net_od_plot.py -v
```

Expected: `ImportError` or `AttributeError` — `map_regions_to_doses` not defined yet.

**Step 3: Implement `map_regions_to_doses`**

Add to `net_od_plot.py`:

```python
def map_regions_to_doses(n_regions: int, doses: list) -> dict:
    """
    Map 1-indexed region numbers to dose levels.

    Regions are ordered most-to-least exposed (highest dose first).
    Doses are provided least-to-greatest and reversed internally.

    Args:
        n_regions: Total number of film regions detected.
        doses: Dose levels in cGy, least-to-greatest.

    Returns:
        Dict mapping region number (1-indexed) -> dose value.
    """
    doses_desc = list(reversed(doses))  # highest dose first
    replicates = n_regions // len(doses_desc)
    mapping = {}
    for i, dose in enumerate(doses_desc):
        for r in range(replicates):
            region_num = i * replicates + r + 1
            mapping[region_num] = dose
    return mapping
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_net_od_plot.py -v
```

Expected: all 3 tests PASS.

**Step 5: Commit**

```bash
git add net_od_plot.py tests/test_net_od_plot.py
git commit -m "feat: implement region-to-dose mapping with tests"
```

---

## Task 3: Implement CSV loading and per-dose averaging

**Files:**
- Modify: `net_od_plot.py`
- Modify: `tests/test_net_od_plot.py`

**Step 1: Write the failing test**

Add to `tests/test_net_od_plot.py`:

```python
def test_load_and_average_od():
    """Load a CSV and average OD values per dose level."""
    # Build a minimal fake DataFrame (4 regions, 2 doses)
    df = pd.DataFrame({
        'region': [1, 2, 3, 4],
        'red_od':   [0.50, 0.52, 0.20, 0.22],
        'green_od': [0.40, 0.42, 0.10, 0.12],
        'blue_od':  [0.30, 0.32, 0.05, 0.07],
    })
    doses = [0, 500]
    result = average_od_by_dose(df, doses)
    # result should be a dict: dose -> {channel: mean_od}
    assert result[500]['red'] == pytest.approx(0.51)
    assert result[500]['green'] == pytest.approx(0.41)
    assert result[0]['red'] == pytest.approx(0.21)
    assert result[0]['blue'] == pytest.approx(0.06)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_net_od_plot.py::test_load_and_average_od -v
```

Expected: FAIL — `average_od_by_dose` not defined.

**Step 3: Implement `average_od_by_dose`**

Add to `net_od_plot.py`:

```python
def average_od_by_dose(df: pd.DataFrame, doses: list) -> dict:
    """
    Average OD values per dose level across replicates.

    Args:
        df: DataFrame from film_analysis_results.csv (must have region, *_od columns).
        doses: Dose levels in cGy, least-to-greatest.

    Returns:
        Dict mapping dose -> {'red': mean_od, 'green': mean_od, 'blue': mean_od}
    """
    n_regions = len(df)
    region_to_dose = map_regions_to_doses(n_regions, doses)
    df = df.copy()
    df['dose'] = df['region'].map(region_to_dose)

    result = {}
    for dose, group in df.groupby('dose'):
        result[dose] = {
            'red':   group['red_od'].mean(),
            'green': group['green_od'].mean(),
            'blue':  group['blue_od'].mean(),
        }
    return result
```

Also add the CSV loader:

```python
def load_csv(scan_name: str) -> pd.DataFrame:
    """Load film_analysis_results.csv for a given scan name."""
    path = Path('outputs') / scan_name / 'film_analysis_results.csv'
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}\nRun film_rgb_analysis.py on '{scan_name}' first.")
    return pd.read_csv(path)
```

**Step 4: Run all tests**

```bash
pytest tests/test_net_od_plot.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add net_od_plot.py tests/test_net_od_plot.py
git commit -m "feat: implement CSV loading and per-dose OD averaging"
```

---

## Task 4: Implement Net OD calculation

**Files:**
- Modify: `net_od_plot.py`
- Modify: `tests/test_net_od_plot.py`

**Step 1: Write the failing test**

Add to `tests/test_net_od_plot.py`:

```python
def test_compute_net_od():
    """Net OD = post - pre for each dose and channel."""
    pre = {
        0:   {'red': 0.10, 'green': 0.15, 'blue': 0.20},
        500: {'red': 0.50, 'green': 0.55, 'blue': 0.60},
    }
    post = {
        0:   {'red': 0.12, 'green': 0.17, 'blue': 0.23},
        500: {'red': 0.80, 'green': 0.90, 'blue': 1.00},
    }
    result = compute_net_od(pre, post)
    assert result[0]['red']    == pytest.approx(0.02)
    assert result[0]['green']  == pytest.approx(0.02)
    assert result[500]['red']  == pytest.approx(0.30)
    assert result[500]['blue'] == pytest.approx(0.40)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_net_od_plot.py::test_compute_net_od -v
```

Expected: FAIL — `compute_net_od` not defined.

**Step 3: Implement `compute_net_od`**

Add to `net_od_plot.py`:

```python
def compute_net_od(pre: dict, post: dict) -> dict:
    """
    Compute Net OD = post_od - pre_od for each dose and channel.

    Args:
        pre:  Dict from average_od_by_dose for the pre scan.
        post: Dict from average_od_by_dose for the post scan.

    Returns:
        Dict mapping dose -> {'red': net_od, 'green': net_od, 'blue': net_od}
    """
    result = {}
    for dose in post:
        result[dose] = {
            ch: post[dose][ch] - pre[dose][ch]
            for ch in ('red', 'green', 'blue')
        }
    return result
```

**Step 4: Run all tests**

```bash
pytest tests/test_net_od_plot.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add net_od_plot.py tests/test_net_od_plot.py
git commit -m "feat: implement Net OD calculation (post - pre)"
```

---

## Task 5: Implement plotting and wire up main()

**Files:**
- Modify: `net_od_plot.py`

**Step 1: Implement `plot_net_od`**

Add to `net_od_plot.py`:

```python
CHANNEL_COLORS = {'red': 'red', 'green': 'green', 'blue': 'blue'}
LINE_STYLES = ['solid', 'dashed']


def plot_net_od(pairs_net_od: list, pair_labels: list, doses: list, output_path: str):
    """
    Plot Net OD vs. Dose for all RGB channels across multiple pairs.

    Args:
        pairs_net_od: List of net_od dicts (one per pair), each from compute_net_od().
        pair_labels:  List of label strings for each pair (e.g. ['Pre-001/Post-003', ...]).
        doses:        Sorted dose levels in cGy.
        output_path:  Path to save the PNG.
    """
    doses_sorted = sorted(doses)

    fig, ax = plt.subplots(figsize=(8, 6))

    for pair_idx, (net_od, label) in enumerate(zip(pairs_net_od, pair_labels)):
        linestyle = LINE_STYLES[pair_idx % len(LINE_STYLES)]
        for ch in ('red', 'green', 'blue'):
            y = [net_od[dose][ch] for dose in doses_sorted]
            ax.plot(
                doses_sorted, y,
                color=CHANNEL_COLORS[ch],
                linestyle=linestyle,
                marker='o',
                label=f'{ch.capitalize()} — {label}'
            )

    ax.set_xlabel('Dose (cGy)')
    ax.set_ylabel('Net Optical Density')
    ax.set_title('Net OD vs. Dose')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.show()
```

**Step 2: Wire up `main()`**

Replace the `if __name__ == '__main__'` block:

```python
if __name__ == '__main__':
    args = parse_args()

    if len(args.pre) != len(args.post):
        raise ValueError(f"--pre and --post must have the same number of entries "
                         f"(got {len(args.pre)} pre, {len(args.post)} post)")

    pairs_net_od = []
    pair_labels = []

    for pre_name, post_name in zip(args.pre, args.post):
        print(f"Processing pair: {pre_name} / {post_name}")
        pre_df  = load_csv(pre_name)
        post_df = load_csv(post_name)

        pre_avg  = average_od_by_dose(pre_df,  args.doses)
        post_avg = average_od_by_dose(post_df, args.doses)

        net_od = compute_net_od(pre_avg, post_avg)
        pairs_net_od.append(net_od)
        pair_labels.append(f"{pre_name}/{post_name}")

    plot_net_od(pairs_net_od, pair_labels, args.doses, args.output)
```

**Step 3: Run end-to-end with real data**

Make sure all 4 scans have been analyzed first (CSVs exist in `outputs/`). Then:

```bash
python net_od_plot.py --pre Pre-001 Pre-002 --post Post-003 Post-004 --doses 0 10 50 200 500
```

Expected: plot window opens, `outputs/net_od_plot.png` saved.

**Step 4: Run all tests one final time**

```bash
pytest tests/test_net_od_plot.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add net_od_plot.py
git commit -m "feat: implement Net OD plot and wire up main()"
```

---

## Done

All 4 scans must be processed with `film_rgb_analysis.py` before running `net_od_plot.py`:

```bash
python film_rgb_analysis.py "02.03.26 15 MeV Scans/Pre-001.tif"
python film_rgb_analysis.py "02.03.26 15 MeV Scans/Pre-002.tif"
python film_rgb_analysis.py "02.03.26 15 MeV Scans/Post-003.tif"
python film_rgb_analysis.py "02.03.26 15 MeV Scans/Post-004.tif"
```

Then:

```bash
python net_od_plot.py --pre Pre-001 Pre-002 --post Post-003 Post-004 --doses 0 10 50 200 500
```
