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
    
    def extract_roi(self, x_start: int, x_end: int, 
                    y_start: Optional[int] = None, y_end: Optional[int] = None,
                    margin_percent: float = 10.0) -> np.ndarray:
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
    
    def analyze_roi(self, roi: np.ndarray) -> Dict:
        channels = ['red', 'green', 'blue']
        results = {}
        for i, ch in enumerate(channels):
            channel_data = roi[:, :, i].astype(np.float64)
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
                           margin_percent: float = 10.0) -> pd.DataFrame:
        if regions is None:
            regions = self.detect_film_regions()
        
        results_list = []
        for i, (x_start, x_end) in enumerate(regions):
            roi = self.extract_roi(x_start, x_end, margin_percent=margin_percent)
            stats = self.analyze_roi(roi)
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
        ax1.set_title('Detected Regions')
        for _, row in results_df.iterrows():
            ax1.axvline(x=row['x_start'], color='yellow', linestyle='--', alpha=0.7)
            ax1.axvline(x=row['x_end'], color='yellow', linestyle='--', alpha=0.7)
        
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
    import sys
    if len(sys.argv) < 2:
        print("Usage: python film_rgb_analysis.py <path_to_tiff_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    analyzer = FilmAnalyzer(sys.argv[1])

    # Create output folder based on input filename (without extension)
    output_dir = Path("outputs") / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    results = analyzer.analyze_all_regions()
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
