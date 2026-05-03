"""
Analysis of full dataset Jacobi results.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os

RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_results.csv')
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(RESULTS_CSV, sep=',', skipinitialspace=True)
df.columns = df.columns.str.strip()

print(f"Loaded {len(df)} buildings from {RESULTS_CSV}")
print(df.head())
print()

# Average mean temperature
avg_mean_temp = df['mean_temp'].mean()
print(f"Q2 — Average mean temperature across all buildings: {avg_mean_temp:.4f} °C")

# Average temperature standard deviation
avg_std_temp = df['std_temp'].mean()
print(f"Q3 — Average temperature std deviation: {avg_std_temp:.4f} °C")

# Buildings with >= 50% area above 18°C
above_18 = (df['pct_above_18'] >= 50).sum()
print(f"Q4 — Buildings with ≥50% area above 18°C: {above_18} / {len(df)}"
      f"  ({100*above_18/len(df):.1f}%)")


# Buildings with >= 50% area below 15°C
below_15 = (df['pct_below_15'] >= 50).sum()
print(f"Q5 — Buildings with ≥50% area below 15°C: {below_15} / {len(df)}"
      f"  ({100*below_15/len(df):.1f}%)")

# Histogram of mean temperatures
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# --- Mean temperature distribution ---
axes[0].hist(df['mean_temp'], bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
axes[0].axvline(avg_mean_temp, color='firebrick', linestyle='--', linewidth=1.5,
                label=f'Mean = {avg_mean_temp:.2f} °C')
axes[0].set_xlabel('Mean temperature (°C)', fontsize=11)
axes[0].set_ylabel('Number of buildings', fontsize=11)
axes[0].set_title('Distribution of mean indoor temperatures', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# --- Std deviation distribution ---
axes[1].hist(df['std_temp'], bins=40, color='darkorange', edgecolor='white', linewidth=0.5)
axes[1].axvline(avg_std_temp, color='firebrick', linestyle='--', linewidth=1.5,
                label=f'Mean std = {avg_std_temp:.2f} °C')
axes[1].set_xlabel('Temperature std deviation (°C)', fontsize=11)
axes[1].set_ylabel('Number of buildings', fontsize=11)
axes[1].set_title('Distribution of temperature standard deviations', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'temperature_distributions.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nHistogram saved to: {out_path}")

# Summary table 

print("\n=== Summary ===")
print(f"Total buildings analysed: {len(df)}")
print(f"Average mean temperature: {avg_mean_temp:.4f} °C")
print(f"Average temperature std deviation: {avg_std_temp:.4f} °C")
print(f"Buildings ≥50% area above 18°C: {above_18} ({100*above_18/len(df):.1f}%)")
print(f"Buildings ≥50% area below 15°C: {below_15} ({100*below_15/len(df):.1f}%)")