# RGB Linear Combination — Observations on Spectral Physics
*2026-02-26*

## Model
`OD_dektronics = a·OD_R + b·OD_G + c·OD_B`

Current coefficients: a = 0.681, b = 1.619, c = −1.368

## Why c (blue coefficient) is negative

### LED source: Warm White (WW)
The Dektronics densitometer uses a Warm White LED. In the blue range (400–500 nm),
the WW LED emits only ~10–15% of its peak intensity. It emits strongly in red (~100%)
and rises steeply through green.

### Sensor: TSL2585 (AMS) — Photopic channel
The TSL2585 photopic channel has the following normalized sensitivity:
- Blue (400–500 nm): ~1–5%
- Green (500–600 nm): peaks at 100% around 560 nm
- Red (600–700 nm): drops to ~5% by 650 nm

### Combined effective response of densitometer
| Wavelength   | LED output (WW) | Sensor sensitivity | Combined effect    |
|--------------|-----------------|--------------------|--------------------|
| Blue 400–500 | ~10–15%         | ~1–5%              | ~0.5–1% effective  |
| Green 500–600| ~15–80%         | ~100%              | moderate–high      |
| Red 600–700  | ~100%           | ~5%                | ~5% effective      |

The densitometer is effectively a **green-wavelength instrument** (~540–580 nm),
despite the WW LED peaking in red. The photopic sensor is nearly blind to red.

### Implication for negative c
The flatbed scanner's dedicated blue channel measures blue film absorption with full
sensitivity. The densitometer, however, has ~0.5–1% combined sensitivity in blue.
The regression assigns a negative weight to OD_B to cancel the blue contribution
that the flatbed measures but the densitometer cannot detect.

The negative c is **physically correct and expected** — it is not a fitting artifact.

## Why b (green) is the dominant coefficient
Green is where the LED output and sensor sensitivity coincide most strongly.
Even though the WW LED emits less green than red, the photopic sensor peaks at
560 nm, making green the channel the densitometer actually responds to most.

## Net OD vs Post OD
Using net OD (Post − Pre) was questioned by the advisor. The negative c persists
in both net OD and post OD fits, confirming it is not an artifact of the pre-subtraction.
The advisor's suggestion to use post OD avoids propagating pre-scan uncertainty
into the fit.

## Fit method
Currently uses ordinary least squares (`np.linalg.lstsq`) with no intercept.
Weighted least squares (using computed σ_OD uncertainties) is a candidate
improvement for TG-191 compliance, but with R² = 1.000 and RMS = 0.0009
the practical impact is expected to be small.
