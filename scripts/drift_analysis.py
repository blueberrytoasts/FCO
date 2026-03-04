import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pymannkendall import original_test as mann_kendall

if len(sys.argv) < 2:
    print("Usage: python scripts/drift_analysis.py <path/to/normalized_data.csv>")
    print("Example: python scripts/drift_analysis.py data/normalized_data_500cGy.csv")
    sys.exit(1)

DATA_PATH = sys.argv[1]
label = os.path.splitext(os.path.basename(DATA_PATH))[0]

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
time_col = df.columns[0]
t = df[time_col].values
trials = df.columns[1:]

print("=" * 55)
print(f"DRIFT ANALYSIS — {label}")
print("=" * 55)

# ── 1. Linear regression slope per trial ──────────────────
print("\n── 1. Linear Regression Slope (is drift significant?) ──")
slopes = {}
for trial in trials:
    y = df[trial].dropna().values
    t_sub = t[:len(y)]
    slope, intercept, r, p, se = stats.linregress(t_sub, y)
    slopes[trial] = slope
    print(f"\n  {trial}:")
    print(f"    Slope     = {slope:.6f} OD/min")
    print(f"    Intercept = {intercept:.5f}")
    print(f"    R²        = {r**2:.4f}")
    print(f"    p-value   = {p:.4f}  {'** significant drift' if p < 0.05 else '(no significant drift)'}")

# ── 2. Mann-Kendall trend test per trial ──────────────────
print("\n── 2. Mann-Kendall Trend Test (monotonic trend?) ──")
for trial in trials:
    y = df[trial].dropna().values
    result = mann_kendall(y)
    print(f"\n  {trial}:")
    print(f"    Trend     = {result.trend}")
    print(f"    p-value   = {result.p:.4f}")
    print(f"    Tau       = {result.Tau:.4f}  (strength of monotonic trend)")
    print(f"    Sen slope = {result.slope:.6f} OD/min")

# ── 3. Slope comparison between trials ────────────────────
print("\n── 3. Slope Comparison Between Trials ──")
# Use overlapping timepoints only
overlap = df.dropna()
t_ov = overlap[time_col].values
y1 = overlap[trials[0]].values
y2 = overlap[trials[1]].values

s1, i1, *_ = stats.linregress(t_ov, y1)
s2, i2, *_ = stats.linregress(t_ov, y2)

# t-test on slopes via residual standard errors
n = len(t_ov)
res1 = y1 - (s1 * t_ov + i1)
res2 = y2 - (s2 * t_ov + i2)
ss_t = np.sum((t_ov - t_ov.mean()) ** 2)
se1 = np.sqrt(np.sum(res1**2) / (n - 2) / ss_t)
se2 = np.sqrt(np.sum(res2**2) / (n - 2) / ss_t)
se_diff = np.sqrt(se1**2 + se2**2)
t_stat = (s1 - s2) / se_diff
p_slope = 2 * stats.t.sf(abs(t_stat), df=2 * (n - 2))

print(f"\n  Slope {trials[0]}: {s1:.6f} OD/min")
print(f"  Slope {trials[1]}: {s2:.6f} OD/min")
print(f"  Difference:  {s1 - s2:.6f} OD/min")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value:     {p_slope:.4f}  {'** slopes differ' if p_slope < 0.05 else '(slopes not significantly different)'}")

# ── 4. Paired t-test / Wilcoxon on overlapping timepoints ─
print("\n── 4. Session Offset (paired comparison on overlap) ──")
t_stat_p, p_paired = stats.ttest_rel(y1, y2)
w_stat, p_wilcox = stats.wilcoxon(y1, y2)
mean_diff = np.mean(y1 - y2)

print(f"\n  Mean difference ({trials[0]} − {trials[1]}): {mean_diff:.5f} OD")
print(f"  Paired t-test:  t = {t_stat_p:.4f},  p = {p_paired:.4f}  {'** significant offset' if p_paired < 0.05 else '(no significant offset)'}")
print(f"  Wilcoxon:       W = {w_stat:.1f},     p = {p_wilcox:.4f}  {'** significant offset' if p_wilcox < 0.05 else '(no significant offset)'}")

# ── 5. Grubbs outlier test on each trial ──────────────────
print("\n── 5. Grubbs Outlier Test ──")
def grubbs(y, alpha=0.05):
    n = len(y)
    mean, std = np.mean(y), np.std(y, ddof=1)
    g_stat = np.max(np.abs(y - mean)) / std
    idx = np.argmax(np.abs(y - mean))
    t_crit = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
    g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
    return g_stat, g_crit, idx, y[idx]

for trial in trials:
    y = df[trial].dropna().values
    t_sub = t[:len(y)]
    g, g_crit, idx, val = grubbs(y)
    print(f"\n  {trial}:")
    print(f"    G statistic = {g:.4f},  critical = {g_crit:.4f}")
    print(f"    Suspected outlier: OD = {val:.5f} at t = {t_sub[idx]:.0f} min")
    print(f"    {'** Outlier detected' if g > g_crit else 'No outlier detected'}")

# ── Plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['steelblue', 'darkorange']
for trial, color in zip(trials, colors):
    y = df[trial].dropna().values
    t_sub = t[:len(y)]
    ax.scatter(t_sub, y, s=20, color=color, label=trial, alpha=0.8)
    slope, intercept, *_ = stats.linregress(t_sub, y)
    xfit = np.array([t_sub[0], t_sub[-1]])
    ax.plot(xfit, slope * xfit + intercept, color=color, lw=1.5,
            linestyle='--', label=f'{trial} fit  ({slope*1000:.3f}×10⁻³ OD/min)')

ax.axhline(1.0, color='black', lw=0.8, linestyle=':')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Normalized OD')
ax.set_title(f'Normalized OD vs Time — {label} — Drift Analysis')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out_path = f'outputs/drift_analysis_{label}.png'
plt.savefig(out_path, dpi=150)
plt.show()
print(f"\nSaved: {out_path}")
