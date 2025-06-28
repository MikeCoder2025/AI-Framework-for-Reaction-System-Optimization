import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
data = {
    'Exp': [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    'Conversion_MBDoE': [42.9, 88.9, 40.3, 37, 46.4, None, None, None, None, None, None],
    'Uncertainty_MBDoE': [0.57, 0.37, 0.18, 0.16, 0.13, None, None, None, None, None, None],
    'Conversion_BO': [43.9, 87.45, 88.01, 87.71, 87.72, 87.72, 87.72, 88.01, None, None, None],
    'Uncertainty_Bo': [9.81, 5.78, 5.08, 4.98, 4.85, 4.78, 4.65, 4.08, None, None, None],
    'Conversion_Hybrid': [42.9, 88.9, 40.3, 37, 46.4, 94.6, 97.8, 95.6, 96.9, 93.4, 98.8],
    'Uncertainty_Hybrid': [0.57, 0.37, 0.18, 0.16, 0.13, 7.92, 10.79, 6.94, 2.75, 3.28, 0.89]
}
df = pd.DataFrame(data)

# Create a figure
plt.figure(figsize=(12, 6))

# Plot each series with error bars
series = ['MBDoE', 'B0', 'Hybrid']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

for idx, s in enumerate(series):
    x = df['Exp'] + (idx - 1) * 0.2  # Offset markers for clarity
    y = df[f'Conversion_{s}']
    yerr = df[f'Uncertainty_{s}']
    
    plt.errorbar(
        x, y, yerr=yerr,
        fmt='o',  # Marker style
        color=colors[idx],
        markersize=8,
        capsize=5,
        label=f'Series {s}'
    )

# Customize the plot
plt.xlabel('Experiment (Exp)', fontsize=12)
plt.ylabel('Conversion (%)', fontsize=12)
plt.title('Conversion vs. Experiment with Uncertainty Bars', fontsize=14)
plt.xticks(df['Exp'])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()