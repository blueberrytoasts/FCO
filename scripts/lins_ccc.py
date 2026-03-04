import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATA_PATH = 'data/regression data.csv'

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

x = df['Avg Net OD (Dektronics)'].values
y = df['Avg Net OD (Flatbed)'].values
xe = df['SD (Dektronics)'].values
ye = df['SD (Flatbed)'].values
doses = df['Dose (cGy)'].values

# Lin's CCC
mean_x, mean_y = np.mean(x), np.mean(y)
var_x = np.var(x, ddof=1)
var_y = np.var(y, ddof=1)
cov_xy = np.cov(x, y, ddof=1)[0, 1]
ccc = (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y) ** 2)

r, _ = stats.pearsonr(x, y)
cb = ccc / r

# Bootstrap 95% CI (better for small n)
rng = np.random.default_rng(42)
boot = []
for _ in range(5000):
    idx = rng.integers(0, len(x), size=len(x))
    xb, yb = x[idx], y[idx]
    cov = np.cov(xb, yb, ddof=1)[0, 1]
    denom = np.var(xb, ddof=1) + np.var(yb, ddof=1) + (np.mean(xb) - np.mean(yb)) ** 2
    if denom > 0:
        boot.append(2 * cov / denom)
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

print(f"Lin's CCC (rho_c):  {ccc:.5f}")
print(f"95% CI (bootstrap): [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Pearson r:          {r:.5f}")
print(f"Bias correction Cb: {cb:.5f}")

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
lo = min(x.min(), y.min()) * 0.85
hi = max(x.max(), y.max()) * 1.1

ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='y = x (perfect concordance)')
slope, intercept, *_ = stats.linregress(x, y)
xfit = np.linspace(lo, hi, 200)
ax.plot(xfit, slope * xfit + intercept, 'r-', lw=1.2,
        label=f'OLS  y = {slope:.3f}x + {intercept:.4f}')

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(x)))
for xi, yi, xei, yei, d, c in zip(x, y, xe, ye, doses, colors):
    ax.errorbar(xi, yi, xerr=xei, yerr=yei, fmt='o', color=c,
                capsize=4, markersize=7, label=f'{int(d)} cGy')

ax.set_xlabel('Net OD — Dektronics')
ax.set_ylabel('Net OD — Flatbed')
ax.set_title(f"Lin's CCC = {ccc:.4f}  |  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect('equal')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/lins_ccc.png', dpi=150)
plt.show()
print("Saved: outputs/lins_ccc.png")
