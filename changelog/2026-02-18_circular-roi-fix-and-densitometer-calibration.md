# Changelog - February 18, 2026

## Session Summary: Circular ROI Fix & RGB-to-Densitometer Calibration

---

## 🐛 Major Bug Fix: Incorrect PV Values from Full-Height ROI

### Problem Identified
- **Issue**: `film_rgb_analysis.py` was producing incorrect red channel PV values
  - Code produced: 54k-59k (almost background level)
  - Correct values: 22k-45k (actual film measurements)
  - Values were ~10k-30k PV off from advisor's algorithm and ImageJ verification

### Root Cause Analysis
1. Films are **small diamond-shaped pieces** arranged horizontally
2. Original code sampled **entire image height** (265 pixels) with 10% margins
3. ROI captured mostly **background** (white scanner bed) instead of just film
4. Example Region 1:
   - Mean PV: 54,435 (WRONG - mixed film + background)
   - Min PV: 9,748 (actual film pixels)
   - Max PV: 65,535 (background contamination)
   - Std Dev: 18,878 (huge variation = mixed sampling)

### Solution Implemented
**Center-based circular ROI extraction** matching advisor's approach:

1. **Added `find_film_center()` method** (lines 74-91)
   - Uses center-of-mass detection on thresholded film pixels
   - Finds 2D centroid of each film piece

2. **Added `extract_roi_circular()` method** (lines 93-109)
   - Extracts circular region around film center
   - Creates circular mask (pixels outside circle = NaN)
   - Configurable radius parameter

3. **Updated `analyze_roi()` method** (lines 136-163)
   - Added `masked` parameter for handling NaN values
   - Uses `np.nanmean()` and `np.nanstd()` for circular ROIs

4. **Updated `analyze_all_regions()` method** (lines 180-221)
   - Added `roi_radius` parameter (default: 30 pixels)
   - Added `use_circular_roi` flag for backward compatibility
   - Stores center coordinates in output

5. **Enhanced visualization** (lines 227-244)
   - Plots cyan circles showing exact ROI location
   - Adds red crosshairs at film centers
   - Shows both detection boundaries and sampling regions

6. **Improved CLI** (lines 276-301)
   - Added `--roi-radius` argument for configurable radius
   - Added `--legacy-mode` flag for old behavior
   - Uses argparse for better UX

### Results
**Optimal radius: 20-25 pixels**

| Radius | Region 3 PV | Region 10 PV | Quality |
|--------|-------------|--------------|---------|
| 20px   | 32,162.61   | 45,229.17    | ✅ Best match |
| 25px   | 32,179.36   | 45,230.96    | ✅ Very good |
| 30px   | 32,124.44   | 45,187.97    | ✅ Good |
| 35px   | 31,401.62   | 44,306.66    | ⚠️ Background contamination |
| 40px   | 31,249.19   | 42,793.66    | ❌ Too much background |

**Accuracy**: Values now match advisor's algorithm within ~20-80 PV (vs. 10,000+ PV error before)

---

## 🆕 New Feature: RGB-to-Densitometer Calibration

### Background
Based on 02/17/26 meeting notes with advisor:
- Goal: Create "pseudo-densitometer" from flatbed scanner RGB channels
- Approach: Find linear combination `OD_densitometer ≈ a*OD_red + b*OD_green + c*OD_blue`
- Purpose: Match analog densitometer response using digital scanner

### Files Created

#### 1. `rgb_densitometer_calibration.py` (276 lines)
**Purpose**: Finds optimal RGB channel weights to match densitometer

**Key Features**:
- `RGBDensitometerCalibrator` class using scikit-learn LinearRegression
- Multi-linear regression: `OD_densitometer = a*OD_red + b*OD_green + c*OD_blue + intercept`
- Calculates R² and RMSE metrics
- Comprehensive 3-panel visualization:
  1. Predicted vs. Measured (1:1 plot with R²)
  2. Residual plot (quality check)
  3. RGB coefficient weights (bar chart)
- Command-line interface with argparse
- Outputs:
  - Calibration equation as text
  - Results CSV with predictions and residuals
  - High-quality plot (150 DPI PNG)

**Usage**:
```bash
python rgb_densitometer_calibration.py \
    --rgb-data rgb_calibration_data.csv \
    --densitometer-data "regression data.csv"
```

#### 2. `calculate_net_od.py` (215 lines)
**Purpose**: Calculate NET OD from pre/post irradiation scan pairs

**Key Features**:
- Processes matched pre/post film pairs
- Calculates `NET_OD = OD_post - OD_pre` for each RGB channel
- Handles multiple film pieces per scan
- Aggregates replicates by dose level
- Outputs:
  - Individual pair NET OD values
  - Combined results from all pairs
  - Aggregated calibration-ready data (mean ± SD by dose)
  - Raw pre/post OD values for verification

**Pre/Post Pairs Identified**:
- Pre-001.tif ↔ Post-003.tif (10 films each)
- Pre-002.tif ↔ Post-004.tif (10 films each)

**Usage**:
```bash
python calculate_net_od.py --config pre_post_mapping.json --roi-radius 20
```

#### 3. `prepare_rgb_calibration_data.py` (147 lines)
**Purpose**: Extract and aggregate RGB OD from multiple scans

**Key Features**:
- JSON-based configuration for scan→dose mapping
- Processes multiple scans at different dose levels
- Aggregates by dose (mean or median)
- Handles film number filtering
- Outputs detailed and summary CSV files

#### 4. Configuration Files

**`pre_post_mapping.json`**:
```json
{
  "pairs": [
    {
      "pre": "02.03.26 15 MeV Scans/Pre-001.tif",
      "post": "02.03.26 15 MeV Scans/Post-003.tif",
      "doses": [...]  // To be filled by user
    },
    ...
  ]
}
```

**`dose_mapping_template.json`**: Template for single-scan dose mapping

---

## 📊 Data Structure Clarifications

### Existing Data
- **`regression data.csv`**: Contains densitometer NET OD values
  - Columns: `Dose (cGy)`, `Avg Net OD (Dektronics)`, `SD (Dektronics)`
  - 5 dose levels: 0, 10, 50, 200, 500 cGy

### Scans Available
- Pre-001.tif (1588×329, 10 films detected)
- Post-003.tif (1271×329, 10 films detected)
- Pre-002.tif (1588×329, 10 films detected)
- Post-004.tif (1271×329, 10 films detected)
- Test 1.tif

### Data Flow
```
Pre-001.tif + Post-003.tif
    → calculate_net_od.py
    → NET_OD_red, NET_OD_green, NET_OD_blue

Densitometer measurements (from spreadsheet)
    → OD_densitometer values

Both datasets matched by Dose
    → rgb_densitometer_calibration.py
    → Optimal weights: a, b, c
    → Calibration equation
```

---

## 🔧 Technical Improvements

### Code Quality
- ✅ Fixed typo: `pd.DataFgrame` → `pd.DataFrame` (line 194)
- ✅ Added comprehensive docstrings
- ✅ Type hints in function signatures
- ✅ Better error handling and validation
- ✅ Informative print statements for user feedback

### Backward Compatibility
- Legacy ROI method preserved with `--legacy-mode` flag
- Original `extract_roi()` method maintained
- `use_circular_roi` parameter allows switching modes

### Visualization Enhancements
- Cyan circles show exact ROI boundaries
- Red crosshairs mark film centers
- Yellow dashed lines show detection boundaries
- Updated plot title: "Detected Regions & ROIs"

---

## 📋 Next Steps (Pending User Input)

### Required Information
1. **Dose mapping**: Which of the 10 films got which dose?
   - Likely: 2 replicates per dose (10 films = 5 doses × 2)

2. **Densitometer data**: Export from spreadsheet with:
   - Dose (cGy)
   - OD_densitometer (NET OD)
   - Optional: SD

### Planned Workflow
```bash
# 1. Calculate NET OD from pre/post pairs
python calculate_net_od.py --config pre_post_mapping.json --roi-radius 20

# 2. Run RGB→Densitometer calibration
python rgb_densitometer_calibration.py \
    --rgb-data outputs/net_od/rgb_calibration_data.csv \
    --densitometer-data densitometer_measurements.csv

# 3. Get calibration equation
# Output: OD_densitometer = a*OD_red + b*OD_green + c*OD_blue
```

---

## 📁 Files Modified

### Modified
- `film_rgb_analysis.py`:
  - Added `find_film_center()` method
  - Added `extract_roi_circular()` method
  - Updated `analyze_roi()` with masked parameter
  - Updated `analyze_all_regions()` with circular ROI support
  - Enhanced `plot_analysis()` visualization
  - Improved `main()` with argparse CLI
  - Fixed DataFrame typo

### Created
- `rgb_densitometer_calibration.py`
- `calculate_net_od.py`
- `prepare_rgb_calibration_data.py`
- `pre_post_mapping.json`
- `dose_mapping_template.json`
- `changelog/2026-02-18_circular-roi-fix-and-densitometer-calibration.md` (this file)

---

## 🎯 Key Achievements

1. ✅ **Fixed critical measurement bug** - Now getting accurate PV values
2. ✅ **Implemented advisor's algorithm** - Center-based circular ROI
3. ✅ **Made analysis configurable** - Adjustable ROI radius
4. ✅ **Created calibration framework** - RGB→Densitometer matching
5. ✅ **Automated NET OD calculation** - Pre/post pair processing
6. ✅ **Improved code quality** - Better CLI, docs, and error handling

---

## 💡 Lessons Learned

- **Verify ROI boundaries visually**: Large std dev (18,878) was a red flag
- **Compare with ground truth early**: User's advisor values caught the bug
- **Use center-based sampling for irregular shapes**: Avoids edge effects
- **Make parameters configurable**: ROI radius tuning was crucial
- **Preserve backward compatibility**: Legacy mode helps debugging

---

## 🔬 Scientific Context

**Problem**: Film dosimetry requires accurate OD measurements for dose calibration

**Challenge**: Diamond-shaped films are difficult to sample automatically

**Solution**: Center-based circular ROI mimics manual measurement approach

**Impact**: Enables automated, reproducible film dosimetry analysis matching manual densitometer readings

---

*Session Date: February 18, 2026*
*Model: Claude Sonnet 4.5*
*Files Created: 5 | Files Modified: 1 | Lines Added: ~900*
