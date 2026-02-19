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
    parser.add_argument('--output', default=None,
                        help='Output PNG path (default: outputs/<pre_names>_<post_names>/net_od_plot.png)')
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


def average_pv_by_dose(df: pd.DataFrame, doses: list) -> dict:
    """
    Average pixel values per dose level across replicate films.

    Args:
        df: DataFrame from film_analysis_results.csv (must have region, *_pv columns).
        doses: Dose levels in cGy, least-to-greatest.

    Returns:
        Dict mapping dose -> {'red': mean_pv, 'green': mean_pv, 'blue': mean_pv}
    """
    n_regions = len(df)
    region_to_dose = map_regions_to_doses(n_regions, doses)
    df = df.copy()
    df['dose'] = df['region'].map(region_to_dose)

    result = {}
    for dose, group in df.groupby('dose'):
        result[dose] = {
            'red':   group['red_pv'].mean(),
            'green': group['green_pv'].mean(),
            'blue':  group['blue_pv'].mean(),
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


def average_pairs(pairs_net_od: list) -> dict:
    """
    Average Net OD values across multiple pairs at each dose and channel.

    Args:
        pairs_net_od: List of net_od dicts, each from compute_net_od().

    Returns:
        Single dict: dose -> {'red': mean_net_od, 'green': mean_net_od, 'blue': mean_net_od}
    """
    doses = list(pairs_net_od[0].keys())
    result = {}
    for dose in doses:
        result[dose] = {
            ch: np.mean([p[dose][ch] for p in pairs_net_od])
            for ch in ('red', 'green', 'blue')
        }
    return result


def export_csv(net_od: dict, net_pv: dict, doses: list, output_path: str):
    """
    Export net OD and net pixel values to a CSV file.

    Args:
        net_od:      Averaged net_od dict: dose -> {'red': val, 'green': val, 'blue': val}
        net_pv:      Averaged net_pv dict: dose -> {'red': val, 'green': val, 'blue': val}
        doses:       Dose levels in cGy.
        output_path: Path to save the CSV.
    """
    rows = []
    for dose in sorted(doses):
        rows.append({
            'dose_cgy':    dose,
            'net_red_od':  net_od[dose]['red'],
            'net_green_od': net_od[dose]['green'],
            'net_blue_od': net_od[dose]['blue'],
            'net_red_pv':  net_pv[dose]['red'],
            'net_green_pv': net_pv[dose]['green'],
            'net_blue_pv': net_pv[dose]['blue'],
        })
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"CSV saved to:  {output_path}")


def plot_net_od(net_od: dict, doses: list, output_path: str):
    """
    Plot Net OD vs. Dose for RGB channels (averaged across pairs).

    Args:
        net_od:      Averaged net_od dict from average_pairs(), dose -> {channel: net_od}.
        doses:       Dose levels in cGy (any order, will be sorted ascending for x-axis).
        output_path: Path to save the PNG.
    """
    doses_sorted = sorted(doses)

    fig, ax = plt.subplots(figsize=(8, 6))

    for ch in ('red', 'green', 'blue'):
        y = [net_od[dose][ch] for dose in doses_sorted]
        ax.plot(
            doses_sorted, y,
            color=CHANNEL_COLORS[ch],
            marker='o',
            label=ch.capitalize()
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

    if args.output is None:
        folder_name = '_'.join(args.pre + args.post)
        args.output = f'outputs/{folder_name}/net_od_plot.png'

    pairs_net_od = []
    pairs_net_pv = []

    for pre_name, post_name in zip(args.pre, args.post):
        print(f"Processing pair: {pre_name} / {post_name}")
        pre_df  = load_csv(pre_name)
        post_df = load_csv(post_name)

        pre_od_avg  = average_od_by_dose(pre_df,  args.doses)
        post_od_avg = average_od_by_dose(post_df, args.doses)
        pre_pv_avg  = average_pv_by_dose(pre_df,  args.doses)
        post_pv_avg = average_pv_by_dose(post_df, args.doses)

        pairs_net_od.append(compute_net_od(pre_od_avg, post_od_avg))
        pairs_net_pv.append(compute_net_od(pre_pv_avg, post_pv_avg))

    averaged_od = average_pairs(pairs_net_od)
    averaged_pv = average_pairs(pairs_net_pv)

    csv_path = Path(args.output).with_suffix('.csv')
    export_csv(averaged_od, averaged_pv, args.doses, str(csv_path))
    plot_net_od(averaged_od, args.doses, args.output)
