#!/usr/bin/env python3
"""
Film Dosimetry RGB Channel Analysis
====================================
This script analyzes 48-bit TIFF scans from flatbed scanners (like Epson Expression 10000XL)
to extract RGB channel data for radiochromic film dosimetry.

Requirements:
    pip install tifffile numpy matplotlib pandas scipy --break-system-packages
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import ndimage
from typing import Tuple, List, Dict, Optional
import warnings

# Try to import tifffile, fall back to PIL if not available
try:
    import tifffile
    USE_TIFFFILE = True
except ImportError:
    from PIL import Image
    USE_TIFFFILE = False
    warnings.warn("tifffile not found, using PIL. 16-bit depth may not be preserved.")


class FilmAnalyzer:
    """
    Analyzes RGB channels from scanned radiochromic film images.
    """
    
    # Maximum pixel value for 16-bit scanner
    MAX_PV_16BIT = 65536
    
    def __init__(self, filepath: str, pv_unexposed: Optional[int] = None):
        self.filepath = Path(filepath)
        self.pv_unexposed = pv_unexposed if pv_unexposed is not None else self.MAX_PV_16BIT
        
        # Load the image
        self.img_array = self._load_image()
        
        # Store image properties
        self.height, self.width, self.n_channels = self.img_array.shape
        self.bit_depth = 16 if self.img_array.dtype == np.uint16 else 8
        
        print(f"Loaded: {self.filepath.name}")
        print(f"Dimensions: {self.width} x {self.height} pixels")
        print(f"Bit depth: {self.bit_depth}-bit per channel")
        print(f"I0 (unexposed PV): {self.pv_unexposed}")
    
    def _load_image(self) -> np.ndarray:
        """Load TIFF image preserving 16-bit depth if possible."""
        if USE_TIFFFILE:
            return tifffile.imread(str(self.filepath))
        else:
            img = Image.open(str(self.filepath))
            return np.array(img)
    
    def get_channel(self, channel: str) -> np.ndarray:
        channel_map = {'red': 0, 'r': 0, 'green': 1, 'g': 1, 'blue': 2, 'b': 2}
        idx = channel_map.get(channel.lower())
        if idx is None:
            raise ValueError(f"Unknown channel: {channel}. Use 'red', 'green', or 'blue'.")
        return self.img_array[:, :, idx]
    
    def pixel_value_to_od(self, pv: np.ndarray) -> np.ndarray:
        """Convert pixel values to optical density."""
        pv_clipped = np.clip(pv, 1, self.pv_unexposed)
        return np.log10(self.pv_unexposed / pv_clipped)
    
    def find_film_center(self, x_start: int, x_end: int) -> Tuple[int, int]:
        """Find the center of mass of the film piece in the detected x-region."""
        # Extract the vertical slice for this x-region
        slice_data = self.img_array[:, x_start:x_end, 0]  # Use red channel

        # Threshold to find film pixels (darker than background)
        threshold = self.pv_unexposed * 0.95
        film_mask = slice_data < threshold

        # Find center of mass
        y_coords, x_coords = np.where(film_mask)
        if len(y_coords) == 0:
            # Fallback to geometric center if no film detected
            center_y = self.height // 2
            center_x = (x_start + x_end) // 2
        else:
            center_y = int(y_coords.mean())
            center_x = int(x_coords.mean()) + x_start

        return center_x, center_y

    def extract_roi_circular(self, center_x: int, center_y: int, radius: int) -> np.ndarray:
        """Extract a circular ROI around the specified center point."""
        # Create bounding box
        y_min = max(0, center_y - radius)
        y_max = min(self.height, center_y + radius)
        x_min = max(0, center_x - radius)
        x_max = min(self.width, center_x + radius)

        # Extract rectangular region
        roi_rect = self.img_array[y_min:y_max, x_min:x_max, :]

        # Create circular mask
        y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
        circle_mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= radius**2

        # Apply mask to each channel (keep only pixels inside circle)
        roi_masked = np.zeros_like(roi_rect, dtype=np.float64)
        for ch in range(3):
            roi_masked[:, :, ch] = np.where(circle_mask, roi_rect[:, :, ch], np.nan)

        return roi_masked

    def extract_roi(self, x_start: int, x_end: int,
                    y_start: Optional[int] = None, y_end: Optional[int] = None,
                    margin_percent: float = 10.0) -> np.ndarray:
        """Legacy method for backward compatibility. Use extract_roi_circular for center-based sampling."""
        x_margin = int((x_end - x_start) * margin_percent / 100)
        x_start_adj = x_start + x_margin
        x_end_adj = x_end - x_margin

        if y_start is None:
            y_margin = int(self.height * margin_percent / 100)
            y_start_adj = y_margin
            y_end_adj = self.height - y_margin
        else:
            y_margin = int((y_end - y_start) * margin_percent / 100)
            y_start_adj = y_start + y_margin
            y_end_adj = y_end - y_margin

        return self.img_array[y_start_adj:y_end_adj, x_start_adj:x_end_adj, :]
    
    def analyze_roi(self, roi: np.ndarray, masked: bool = False) -> Dict:
        """Analyze ROI and compute statistics for each channel.

        Args:
            roi: ROI array (may contain NaN values if masked=True)
            masked: If True, uses nanmean/nanstd to ignore NaN values
        """
        channels = ['red', 'green', 'blue']
        results = {}
        for i, ch in enumerate(channels):
            channel_data = roi[:, :, i].astype(np.float64)

            if masked:
                # Use nanmean/nanstd to ignore pixels outside the circle
                pv_mean = np.nanmean(channel_data)
                pv_std = np.nanstd(channel_data)
            else:
                pv_mean = channel_data.mean()
                pv_std = channel_data.std()

            od_mean = self.pixel_value_to_od(pv_mean)
            od_std = (1 / np.log(10)) * (pv_std / pv_mean) if pv_mean > 0 else 0

            results[ch] = {
                'pv_mean': pv_mean, 'pv_std': pv_std,
                'od_mean': od_mean, 'od_std': od_std
            }
        return results
    
    def detect_film_regions(self, threshold_factor: float = 0.98, min_width: int = 20) -> List[Tuple[int, int]]:
        red_profile = self.get_channel('red').mean(axis=0)
        threshold = self.pv_unexposed * threshold_factor
        below_threshold = red_profile < threshold
        transitions = np.diff(below_threshold.astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        
        if below_threshold[0]: starts = np.concatenate([[0], starts])
        if below_threshold[-1]: ends = np.concatenate([ends, [len(red_profile) - 1]])
        
        regions = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_width]
        print(f"Detected {len(regions)} film regions")
        return regions
    
    def analyze_all_regions(self, regions: Optional[List[Tuple[int, int]]] = None,
                           roi_radius: int = 30, use_circular_roi: bool = True) -> pd.DataFrame:
        """Analyze all detected film regions.

        Args:
            regions: List of (x_start, x_end) tuples. If None, auto-detect.
            roi_radius: Radius in pixels for circular ROI (default: 30)
            use_circular_roi: If True, use center-based circular ROI. If False, use legacy full-height method.

        Returns:
            DataFrame with PV and OD values for each region
        """
        if regions is None:
            regions = self.detect_film_regions()

        results_list = []
        for i, (x_start, x_end) in enumerate(regions):
            if use_circular_roi:
                # Find film center and extract circular ROI
                center_x, center_y = self.find_film_center(x_start, x_end)
                roi = self.extract_roi_circular(center_x, center_y, roi_radius)
                stats = self.analyze_roi(roi, masked=True)
                row = {
                    'region': i + 1,
                    'x_start': x_start,
                    'x_end': x_end,
                    'center_x': center_x,
                    'center_y': center_y,
                    'roi_radius': roi_radius
                }
            else:
                # Legacy method: full-height ROI
                roi = self.extract_roi(x_start, x_end, margin_percent=10.0)
                stats = self.analyze_roi(roi, masked=False)
                row = {'region': i + 1, 'x_start': x_start, 'x_end': x_end}

            for ch in ['red', 'green', 'blue']:
                row[f'{ch}_pv'] = stats[ch]['pv_mean']
                row[f'{ch}_od'] = stats[ch]['od_mean']
            results_list.append(row)

        return pd.DataFrame(results_list)
    
    def plot_analysis(self, results_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """Simplified visualization: 1x3 grid, no channel correlation."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Image with Regions
        ax1 = axes[0]
        rgb_display = (self.img_array / 256).astype(np.uint8)
        ax1.imshow(rgb_display)
        ax1.set_title('Detected Regions & ROIs')

        # Check if circular ROI data is available
        has_circular_roi = 'center_x' in results_df.columns

        for _, row in results_df.iterrows():
            # Show detected x-boundaries
            ax1.axvline(x=row['x_start'], color='yellow', linestyle='--', alpha=0.5, linewidth=1)
            ax1.axvline(x=row['x_end'], color='yellow', linestyle='--', alpha=0.5, linewidth=1)

            # If using circular ROI, draw the circles
            if has_circular_roi:
                circle = plt.Circle((row['center_x'], row['center_y']),
                                   row['roi_radius'],
                                   color='cyan', fill=False, linewidth=2, alpha=0.8)
                ax1.add_patch(circle)
                # Mark center with a small dot
                ax1.plot(row['center_x'], row['center_y'], 'r+', markersize=8, markeredgewidth=2)
        
        # 2. Pixel Values
        ax2 = axes[1]
        x = np.arange(len(results_df))
        width = 0.25
        ax2.bar(x - width, results_df['red_pv'], width, label='Red', color='red', alpha=0.7)
        ax2.bar(x, results_df['green_pv'], width, label='Green', color='green', alpha=0.7)
        ax2.bar(x + width, results_df['blue_pv'], width, label='Blue', color='blue', alpha=0.7)
        ax2.set_title('Mean Pixel Values')
        ax2.legend()
        
        # 3. Optical Density
        ax3 = axes[2]
        ax3.bar(x - width, results_df['red_od'], width, label='Red', color='red', alpha=0.7)
        ax3.bar(x, results_df['green_od'], width, label='Green', color='green', alpha=0.7)
        ax3.bar(x + width, results_df['blue_od'], width, label='Blue', color='blue', alpha=0.7)
        ax3.set_title('Optical Density')
        ax3.legend()
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150)
        return fig

    def export_to_csv(self, results_df: pd.DataFrame, filepath: str = 'film_analysis_results.csv'):
        results_df.to_csv(filepath, index=False)
        print(f"Results exported to: {filepath}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze RGB channels from scanned radiochromic film')
    parser.add_argument('input_file', help='Path to TIFF scan file')
    parser.add_argument('--roi-radius', type=int, default=30,
                       help='Radius in pixels for circular ROI (default: 30)')
    parser.add_argument('--legacy-mode', action='store_true',
                       help='Use legacy full-height ROI method instead of circular ROI')
    parser.add_argument('--pv-unexposed', type=int, default=None,
                       help='Pixel value for unexposed film (default: 65536 for 16-bit)')

    args = parser.parse_args()

    input_path = Path(args.input_file)
    analyzer = FilmAnalyzer(args.input_file, pv_unexposed=args.pv_unexposed)

    # Create output folder based on input filename (without extension)
    output_dir = Path("outputs") / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze with configurable ROI method
    use_circular = not args.legacy_mode
    results = analyzer.analyze_all_regions(roi_radius=args.roi_radius, use_circular_roi=use_circular)

    if use_circular:
        print(f"\nUsing circular ROI with radius: {args.roi_radius} pixels")
    print("\n" + "="*50 + "\nRESULTS\n" + "="*50)
    print(results.to_string(index=False))

    # Save outputs to the dedicated folder
    plot_path = output_dir / "film_analysis_plot.png"
    csv_path = output_dir / "film_analysis_results.csv"

    analyzer.plot_analysis(results, save_path=str(plot_path))
    analyzer.export_to_csv(results, filepath=str(csv_path))
    print(f"\nOutputs saved to: {output_dir}")
    plt.show()

if __name__ == "__main__":
    main()
