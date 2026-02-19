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



def map_regions_to_doses(n_regions: int, doses: list) -> dict:
    """
    Map 1-indexed region numbers to dose levels.

    Regions are ordered most-to-least exposed (highest dose first).
    Doses are provided least-to-greatest and reversed internally.

    Precondition: n_regions must be evenly divisible by len(doses).

    Args:
        n_regions: Total number of film regions detected.
        doses: Dose levels in cGy, least-to-greatest.

    Returns:
        Dict mapping region number (1-indexed) -> dose value.

    Raises:
        ValueError: If n_regions is not evenly divisible by len(doses).
    """
    if len(doses) == 0:
        raise ValueError("doses list must not be empty.")
    if n_regions % len(doses) != 0:
        raise ValueError(
            f"n_regions ({n_regions}) must be evenly divisible by the number of "
            f"dose levels ({len(doses)}). Got remainder {n_regions % len(doses)}."
        )
    doses_desc = list(reversed(doses))  # highest dose first
    replicates = n_regions // len(doses_desc)
    mapping = {}
    for i, dose in enumerate(doses_desc):
        for r in range(replicates):
            region_num = i * replicates + r + 1
            mapping[region_num] = dose
    return mapping



def load_csv(scan_name: str) -> pd.DataFrame:
    """Load film_analysis_results.csv for a given scan name."""
    path = Path('outputs') / scan_name / 'film_analysis_results.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            f"Run film_rgb_analysis.py on '{scan_name}' first."
        )
    return pd.read_csv(path)


def average_od_by_dose(df: pd.DataFrame, doses: list) -> dict:
    """
    Average OD values per dose level across replicate films.

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


def compute_net_od(pre: dict, post: dict) -> dict:
    """
    Compute Net OD = post_od - pre_od for each dose and channel.

    Args:
        pre:  Dict from average_od_by_dose for the pre scan.
        post: Dict from average_od_by_dose for the post scan.

    Returns:
        Dict mapping dose -> {'red': net_od, 'green': net_od, 'blue': net_od}

    Raises:
        ValueError: If pre and post do not have identical dose keys.
    """
    if set(pre.keys()) != set(post.keys()):
        only_pre  = set(pre.keys())  - set(post.keys())
        only_post = set(post.keys()) - set(pre.keys())
        raise ValueError(
            f"pre and post must have identical dose levels. "
            f"Only in pre: {sorted(only_pre)}. Only in post: {sorted(only_post)}."
        )
    result = {}
    for dose in post:
        result[dose] = {
            ch: post[dose][ch] - pre[dose][ch]
            for ch in ('red', 'green', 'blue')
        }
    return result


CHANNEL_COLORS = {'red': 'red', 'green': 'green', 'blue': 'blue'}
LINE_STYLES = ['solid', 'dashed']


def plot_net_od(pairs_net_od: list, pair_labels: list, doses: list, output_path: str):
    """
    Plot Net OD vs. Dose for all RGB channels across multiple pairs.

    Args:
        pairs_net_od: List of net_od dicts (one per pair), each from compute_net_od().
        pair_labels:  List of label strings for each pair (e.g. ['Pre-001/Post-003', ...]).
        doses:        Dose levels in cGy (any order, will be sorted ascending for x-axis).
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


if __name__ == '__main__':
    args = parse_args()
    if len(args.pre) != len(args.post):
        raise SystemExit(f"Error: --pre ({len(args.pre)} items) and --post ({len(args.post)} items) "
                         f"must have the same number of entries.")

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
