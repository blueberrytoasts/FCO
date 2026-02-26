import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'normalized_data.csv'

df = pd.read_csv(csv_path)
time_col = df.columns[0]
od_cols = df.columns[1:]

plt.figure(figsize=(9, 5))
for col in od_cols:
    series = df[col].dropna()
    plt.scatter(df[time_col][:len(series)], series, label=col, s=20)

plt.xlabel(time_col)
plt.ylabel('Normalized OD')
plt.title('Normalized OD vs Time')
plt.legend()
plt.tight_layout()
plt.savefig('normalized_od_plot.png', dpi=150)
plt.show()
