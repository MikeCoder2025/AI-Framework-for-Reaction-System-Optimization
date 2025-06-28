import numpy as np
import matplotlib.pyplot as plt

# Data arrays (experiments 18 to 33)
experiments = np.array([18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])

# Hybrid (MBDoE-BO) data: conversion and uncertainty
conv_hybrid = np.array([94.63084238, 97.76526117, 95.64716808, 96.97843953, 93.38910844, 98.83202487,
                          95.07684625, 95.59673162, 93.5759467, 93.99901339, 95.20710531, 94.51343868,
                          95.62751408, 91.59459719, 124.1182278, 96.67011836])
unc_hybrid = np.array([7.921117819, 10.79145855, 6.939475995, 2.752575382, 3.284778859, 0.89133478,
                         1.343870397, 2.249339749, 1.496443782, 1.680708377, 0.791429868, 0.865699359,
                         0.927649376, 0.865610741, 8.877025117, 1.16358073])

# Insilico data: conversion only (no uncertainty)
conv_insilico = np.array([93.63, 90.82, 94.63, 98.82, 86.87, 95.21, 85.02, 84.53, 91.24, 98.20,
                           90.20, 88.25, 90.38, 96.13, 97.55, 95.19])

# Define thresholds for highlighting
highlight_conv_min = 95.0
highlight_conv_max = 100.0  # strictly greater than 95 and less than 100
highlight_uncert_max = 1.5  # uncertainty must be <= 1.5%

# Create the plot
plt.figure(figsize=(10, 6))

# Plot Hybrid conversion with error bars (no jitter added)
plt.errorbar(experiments, conv_hybrid, yerr=unc_hybrid, fmt='o', capsize=5, 
             label='Hybrid (MBDoE-BO)', color='blue')

# Plot Insilico conversion as markers (no error bars)
plt.scatter(experiments, conv_insilico, marker='s', s=80, label='Insilico', color='red')

# Identify Hybrid points that meet the highlighting criteria:
# conversion greater than 95% but less than 100% and uncertainty <= 1.5%
highlight_idx = np.where((conv_hybrid > highlight_conv_min) & 
                         (conv_hybrid < highlight_conv_max) & 
                         (unc_hybrid <= highlight_uncert_max))[0]

# Plot highlighted points with a distinct marker (e.g., star) and edge color
plt.scatter(experiments[highlight_idx], conv_hybrid[highlight_idx], 
            marker='*', s=150, color='green', edgecolors='black', 
            label='Hybrid (Highlighted)')

# Set labels and title
plt.xlabel('Experiment')
plt.ylabel('Conversion (%)')
plt.title('Conversion Comparison: Hybrid (MBDoE-BO) vs Insilico')
plt.xticks(np.arange(18, 34, 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
