import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/normalized_data_0cGy.csv'

# Extract dose label from filename, e.g. "normalized_data_0cGy.csv" -> "0cGy"
basename = os.path.splitext(os.path.basename(csv_path))[0]  # e.g. "normalized_data_0cGy"
dose_label = basename.split('_')[-1]                         # e.g. "0cGy"

df = pd.read_csv(csv_path)
time_col = df.columns[0]
od_cols = df.columns[1:]

plt.figure(figsize=(9, 5))
for col in od_cols:
    series = df[col].dropna()
    plt.scatter(df[time_col][:len(series)], series, label=col, s=20)

plt.xlabel(time_col)
plt.ylabel('Normalized OD')
plt.title(f'Normalized OD vs Time ({dose_label})')
plt.legend()
plt.tight_layout()

os.makedirs('outputs', exist_ok=True)
out_path = f'outputs/normalized_od_{dose_label}.png'
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.show()
