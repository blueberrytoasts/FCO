#!/usr/bin/env python3
"""
Film Dosimetry RGB Channel Analysis
====================================
This script analyzes 48-bit TIFF scans from flatbed scanners (like Epson Expression 10000XL)
to extract RGB channel data for radiochromic film dosimetry.

Author: [Your Name]
Date: January 2026
Project: IR-FCO Dosimetry Validation

Requirements:
    pip install tifffile numpy matplotlib pandas scipy --break-system-packages

Usage:
    python film_rgb_analysis.py <path_to_tiff_file>
    
    Or import as a module:
        from film_rgb_analysis import FilmAnalyzer
        analyzer = FilmAnalyzer('scan.tif')
        results = analyzer.analyze_all_regions()
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
    
    The scanner produces a 48-bit RGB image (16 bits per channel).
    For unexposed film, pixel values are near the maximum (65535).
    For exposed film, pixel values decrease as optical density increases.
    
    Optical Density (OD) = log10(I0 / I) = log10(PV_unexposed / PV_exposed)
    
    For a 16-bit scanner with unexposed film at max:
        OD = log10(65535 / PV_measured)
    """
    
    # Maximum pixel value for 16-bit scanner
    MAX_PV_16BIT = 65535
    
    def __init__(self, filepath: str, pv_unexposed: Optional[int] = None):
        """
        Initialize the analyzer with a TIFF file.
        
        Parameters:
        -----------
        filepath : str
            Path to the 48-bit TIFF scan file
        pv_unexposed : int, optional
            Pixel value for unexposed film (I0). If None, uses 65535 (max 16-bit value).
            You can measure this from an unexposed film region for better accuracy.
        """
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
        """
        Extract a single color channel from the image.
        
        Parameters:
        -----------
        channel : str
            'red', 'green', or 'blue' (or 'r', 'g', 'b')
        
        Returns:
        --------
        np.ndarray : 2D array of pixel values for that channel
        """
        channel_map = {'red': 0, 'r': 0, 'green': 1, 'g': 1, 'blue': 2, 'b': 2}
        idx = channel_map.get(channel.lower())
        if idx is None:
            raise ValueError(f"Unknown channel: {channel}. Use 'red', 'green', or 'blue'.")
        return self.img_array[:, :, idx]
    
    def pixel_value_to_od(self, pv: np.ndarray) -> np.ndarray:
        """
        Convert pixel values to optical density.
        
        OD = log10(I0 / I) = log10(PV_unexposed / PV_measured)
        
        Parameters:
        -----------
        pv : np.ndarray
            Pixel values (can be single value or array)
        
        Returns:
        --------
        np.ndarray : Optical density values
        """
        # Clip to avoid log(0) - minimum PV of 1
        pv_clipped = np.clip(pv, 1, self.pv_unexposed)
        return np.log10(self.pv_unexposed / pv_clipped)
    
    def od_to_pixel_value(self, od: np.ndarray) -> np.ndarray:
        """
        Convert optical density back to pixel values.
        
        PV = I0 / 10^OD
        
        Parameters:
        -----------
        od : np.ndarray
            Optical density values
        
        Returns:
        --------
        np.ndarray : Pixel values
        """
        return self.pv_unexposed / (10 ** od)
    
    def extract_roi(self, x_start: int, x_end: int, 
                    y_start: Optional[int] = None, y_end: Optional[int] = None,
                    margin_percent: float = 10.0) -> np.ndarray:
        """
        Extract a region of interest from the image.
        
        Parameters:
        -----------
        x_start, x_end : int
            Horizontal pixel range
        y_start, y_end : int, optional
            Vertical pixel range. If None, uses full height with margin.
        margin_percent : float
            Percentage of region to exclude from edges (default 10%)
        
        Returns:
        --------
        np.ndarray : ROI array with shape (height, width, 3)
        """
        # Apply margin to x
        x_margin = int((x_end - x_start) * margin_percent / 100)
        x_start_adj = x_start + x_margin
        x_end_adj = x_end - x_margin
        
        # Handle y coordinates
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
        """
        Calculate statistics for an ROI across all channels.
        
        Parameters:
        -----------
        roi : np.ndarray
            Region of interest array with shape (height, width, 3)
        
        Returns:
        --------
        dict : Statistics for each channel including PV and OD values
        """
        channels = ['red', 'green', 'blue']
        results = {}
        
        for i, ch in enumerate(channels):
            channel_data = roi[:, :, i].astype(np.float64)
            
            # Pixel value statistics
            pv_mean = channel_data.mean()
            pv_std = channel_data.std()
            pv_min = channel_data.min()
            pv_max = channel_data.max()
            
            # Optical density statistics
            od_mean = self.pixel_value_to_od(pv_mean)
            
            # OD uncertainty from error propagation:
            # SD(OD) = (1 / ln(10)) * sqrt((SD(I0)/I0)^2 + (SD(I)/I)^2)
            # Assuming SD(I0) ≈ 0 for calibrated scanner:
            # SD(OD) ≈ (1 / ln(10)) * (SD(PV) / PV)
            od_std = (1 / np.log(10)) * (pv_std / pv_mean) if pv_mean > 0 else 0
            
            results[ch] = {
                'pv_mean': pv_mean,
                'pv_std': pv_std,
                'pv_min': pv_min,
                'pv_max': pv_max,
                'od_mean': od_mean,
                'od_std': od_std,
                'n_pixels': channel_data.size
            }
        
        return results
    
    def detect_film_regions(self, threshold_factor: float = 0.9,
                            min_width: int = 20) -> List[Tuple[int, int]]:
        """
        Automatically detect film pieces in the scan.
        
        Uses the red channel to find regions where pixel values drop
        below a threshold, indicating exposed film.
        
        Parameters:
        -----------
        threshold_factor : float
            Fraction of max value to use as threshold (default 0.9)
        min_width : int
            Minimum width in pixels to count as a film region
        
        Returns:
        --------
        list : List of (start, end) x-coordinate tuples for each region
        """
        # Use red channel profile (averaged across height)
        red_profile = self.get_channel('red').mean(axis=0)
        
        # Set threshold
        threshold = self.pv_unexposed * threshold_factor
        
        # Find regions below threshold
        below_threshold = red_profile < threshold
        
        # Find transitions
        transitions = np.diff(below_threshold.astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        
        # Handle edge cases
        if below_threshold[0]:
            starts = np.concatenate([[0], starts])
        if below_threshold[-1]:
            ends = np.concatenate([ends, [len(red_profile) - 1]])
        
        # Filter by minimum width
        regions = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_width]
        
        print(f"Detected {len(regions)} film regions")
        return regions
    
    def analyze_all_regions(self, regions: Optional[List[Tuple[int, int]]] = None,
                           margin_percent: float = 10.0) -> pd.DataFrame:
        """
        Analyze all detected film regions.
        
        Parameters:
        -----------
        regions : list, optional
            List of (start, end) tuples. If None, auto-detects regions.
        margin_percent : float
            Percentage margin to exclude from ROI edges
        
        Returns:
        --------
        pd.DataFrame : Results table with PV and OD for each channel
        """
        if regions is None:
            regions = self.detect_film_regions()
        
        results_list = []
        
        for i, (x_start, x_end) in enumerate(regions):
            roi = self.extract_roi(x_start, x_end, margin_percent=margin_percent)
            stats = self.analyze_roi(roi)
            
            row = {
                'region': i + 1,
                'x_start': x_start,
                'x_end': x_end,
                'width_px': x_end - x_start,
            }
            
            for ch in ['red', 'green', 'blue']:
                row[f'{ch}_pv'] = stats[ch]['pv_mean']
                row[f'{ch}_pv_std'] = stats[ch]['pv_std']
                row[f'{ch}_od'] = stats[ch]['od_mean']
                row[f'{ch}_od_std'] = stats[ch]['od_std']
            
            results_list.append(row)
        
        return pd.DataFrame(results_list)
    
    def compare_with_densitometer(self, densitometer_od: np.ndarray,
                                  results_df: pd.DataFrame) -> Dict:
        """
        Compare scanner RGB channels with densitometer readings.
        
        Performs linear regression for each channel to find the best match.
        
        Parameters:
        -----------
        densitometer_od : np.ndarray
            OD readings from the handheld densitometer (same order as regions)
        results_df : pd.DataFrame
            Results from analyze_all_regions()
        
        Returns:
        --------
        dict : Regression results for each channel
        """
        from scipy import stats
        
        comparison = {}
        
        for ch in ['red', 'green', 'blue']:
            scanner_od = results_df[f'{ch}_od'].values
            
            # Linear regression: densitometer_od = slope * scanner_od + intercept
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                scanner_od, densitometer_od
            )
            
            # Calculate residuals
            predicted = slope * scanner_od + intercept
            residuals = densitometer_od - predicted
            rmse = np.sqrt(np.mean(residuals**2))
            
            comparison[ch] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err,
                'rmse': rmse
            }
            
            print(f"\n{ch.upper()} Channel:")
            print(f"  Slope: {slope:.4f}")
            print(f"  Intercept: {intercept:.4f}")
            print(f"  R²: {r_value**2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
        
        # Find best channel
        best_channel = max(comparison.keys(), key=lambda x: comparison[x]['r_squared'])
        print(f"\n>>> Best matching channel: {best_channel.upper()} (R² = {comparison[best_channel]['r_squared']:.4f})")
        
        return comparison
    
    def plot_analysis(self, results_df: pd.DataFrame, 
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of the analysis results.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from analyze_all_regions()
        save_path : str, optional
            Path to save the figure
        
        Returns:
        --------
        matplotlib.Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Full image with regions marked
        ax1 = axes[0, 0]
        rgb_display = (self.img_array / 256).astype(np.uint8)
        ax1.imshow(rgb_display)
        ax1.set_title('Scanned Film with Detected Regions', fontsize=12)
        
        for _, row in results_df.iterrows():
            ax1.axvline(x=row['x_start'], color='yellow', linestyle='--', alpha=0.7)
            ax1.axvline(x=row['x_end'], color='yellow', linestyle='--', alpha=0.7)
            ax1.text((row['x_start'] + row['x_end'])/2, 10, f"{int(row['region'])}", 
                    color='white', fontsize=9, ha='center',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # 2. Pixel value comparison
        ax2 = axes[0, 1]
        x = np.arange(len(results_df))
        width = 0.25
        
        ax2.bar(x - width, results_df['red_pv'], width, label='Red', color='red', alpha=0.7)
        ax2.bar(x, results_df['green_pv'], width, label='Green', color='green', alpha=0.7)
        ax2.bar(x + width, results_df['blue_pv'], width, label='Blue', color='blue', alpha=0.7)
        
        ax2.set_xlabel('Film Region')
        ax2.set_ylabel('Mean Pixel Value (16-bit)')
        ax2.set_title('Raw Pixel Values by Channel')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df['region'].astype(int))
        ax2.legend()
        ax2.axhline(y=65535, color='gray', linestyle='--', alpha=0.5)
        
        # 3. Optical density comparison
        ax3 = axes[1, 0]
        ax3.bar(x - width, results_df['red_od'], width, label='Red', color='red', alpha=0.7)
        ax3.bar(x, results_df['green_od'], width, label='Green', color='green', alpha=0.7)
        ax3.bar(x + width, results_df['blue_od'], width, label='Blue', color='blue', alpha=0.7)
        
        ax3.set_xlabel('Film Region')
        ax3.set_ylabel('Optical Density')
        ax3.set_title('Optical Density by Channel')
        ax3.set_xticks(x)
        ax3.set_xticklabels(results_df['region'].astype(int))
        ax3.legend()
        
        # 4. Channel correlation
        ax4 = axes[1, 1]
        ax4.scatter(results_df['red_od'], results_df['green_od'], 
                   label='Red vs Green', alpha=0.7, s=50)
        ax4.scatter(results_df['red_od'], results_df['blue_od'], 
                   label='Red vs Blue', alpha=0.7, s=50)
        
        # Add 1:1 line
        max_od = max(results_df['red_od'].max(), results_df['green_od'].max(), 
                    results_df['blue_od'].max())
        ax4.plot([0, max_od], [0, max_od], 'k--', alpha=0.5, label='1:1 line')
        
        ax4.set_xlabel('Red Channel OD')
        ax4.set_ylabel('Other Channel OD')
        ax4.set_title('Channel Correlation')
        ax4.legend()
        ax4.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def export_to_csv(self, results_df: pd.DataFrame, 
                      filepath: str = 'film_analysis_results.csv'):
        """
        Export results to CSV file for use in Google Sheets or Excel.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from analyze_all_regions()
        filepath : str
            Output file path
        """
        results_df.to_csv(filepath, index=False)
        print(f"Results exported to: {filepath}")


def main():
    """Main function for command-line usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python film_rgb_analysis.py <path_to_tiff_file>")
        print("\nExample:")
        print("  python film_rgb_analysis.py scan.tif")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Create analyzer
    analyzer = FilmAnalyzer(filepath)
    
    # Analyze all regions
    results = analyzer.analyze_all_regions()
    
    # Print results table
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    print(results.to_string(index=False))
    
    # Create visualization
    fig = analyzer.plot_analysis(results, save_path='film_analysis_plot.png')
    
    # Export to CSV
    analyzer.export_to_csv(results)
    
    plt.show()


if __name__ == "__main__":
    main()
