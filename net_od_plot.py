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


if __name__ == '__main__':
    args = parse_args()
    if len(args.pre) != len(args.post):
        raise SystemExit(f"Error: --pre ({len(args.pre)} items) and --post ({len(args.post)} items) "
                         f"must have the same number of entries.")
    print(f"Pre scans:  {args.pre}")
    print(f"Post scans: {args.post}")
    print(f"Doses:      {args.doses}")
