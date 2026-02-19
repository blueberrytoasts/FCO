#!/usr/bin/env python3
"""
Testing a regression
====================================

This script attempts to do a regression on the Dektronics densitometer and the flatbed scanner.

Requirements:
    csv file of the data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load data
keep_col = ['Dose (cGy)', 'Avg Net OD (Dektronics)', 'Avg Net OD (Flatbed)']
data = pd.read_csv('regression data.csv', usecols=keep_col)

# 2. Setup Variables for Regression
# We want to plug in Dektronics (x) and get out a "Scanner-equivalent" (y)
x_data = data['Avg Net OD (Dektronics)']
y_data = data['Avg Net OD (Flatbed)']

# Perform 2nd degree polynomial fit: y = Ax² + Bx + C
coeffs = np.polyfit(x_data, y_data, 2)
poly_func = np.poly1d(coeffs)

# Calculate R-squared (goodness of fit)
y_pred = poly_func(x_data)
y_resid = y_data - y_pred
ss_res = np.sum(y_resid**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r_squared = 1 - (ss_res / ss_tot)

# 3. Create the Combined Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- Left Plot: Dose Comparison ---
ax1.plot(data['Dose (cGy)'], data['Avg Net OD (Dektronics)'], 'o-', label='Dektronics', color='blue')
ax1.plot(data['Dose (cGy)'], data['Avg Net OD (Flatbed)'], 's-', label='Flatbed (Standard)', color='red')
ax1.set_xlabel('Dose (cGy)')
ax1.set_ylabel('Average Net OD')
ax1.set_title('Raw Device Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Right Plot: Calibration Curve (Densitometer -> Scanner) ---
# Create smooth line for the curve
x_range = np.linspace(x_data.min(), x_data.max(), 100)
ax2.scatter(x_data, y_data, color='black', label='Measured Data')
ax2.plot(x_range, poly_func(x_range), 'r--', 
         label=f'Fit: {coeffs[0]:.3f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}\n$R^2 = {r_squared:.4f}$')

ax2.set_xlabel('Measured Dektronics OD')
ax2.set_ylabel('Target Flatbed OD')
ax2.set_title('Calibration: Matching Dektronics to Scanner')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dosimetry_analysis.png')

print(f"Match Equation: Flatbed_OD = {coeffs[0]:.4f}*(Dek_OD)^2 + {coeffs[1]:.4f}*(Dek_OD) + {coeffs[2]:.4f}")
print(f"R-squared: {r_squared:.4f}")