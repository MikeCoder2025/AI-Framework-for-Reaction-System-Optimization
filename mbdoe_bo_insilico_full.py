import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Combined dataset including experiments 13-32
data = {
    'Exp': list(range(13, 33)),  # Experiments 13-32
    'Hybrid_Conversion': [
        42.9, 88.9, 40.3, 37, 46.4,  # Experiments 13-17
        94.63, 97.77, 95.20, 98.28, 93.39, 98.83, 95.08, 95.60,
        93.58, 97.00, 95.21, 94.51, 95.63, 91.59, 96.07  # Experiments 18-32
    ],
    'Hybrid_Uncertainty': [
        0.57, 0.37, 0.18, 0.16, 0.13,  # Experiments 13-17
        7.92, 10.79, 1.04, 1.5, 3.28, 0.89, 1.34, 2.25,
        1.50, 1.38, 0.79, 0.87, 0.93, 0.87, 1.5  # Experiments 18-32
    ],
    'Insilico_Conversion': [
        33.3, 30.2, 48.2, 47.05, 44.9,  # Experiments 13-17
        93.63, 90.82, 94.63, 98.82, 86.87, 95.21, 85.02, 84.53,
        91.24, 98.20, 90.20, 88.25, 90.38, 96.13, 95.19  # Experiments 18-32
    ]
}
df = pd.DataFrame(data)

# Calculate ranges and differences
df['Hybrid_Lower'] = df['Hybrid_Conversion'] - df['Hybrid_Uncertainty']
df['Hybrid_Upper'] = df['Hybrid_Conversion'] + df['Hybrid_Uncertainty']
df['Difference'] = abs(df['Hybrid_Conversion'] - df['Insilico_Conversion'])

# Identify special cases
insilico_in_range = (df['Insilico_Conversion'] >= df['Hybrid_Lower']) & (df['Insilico_Conversion'] <= df['Hybrid_Upper'])
hybrid_highlight = (df['Hybrid_Conversion'] > 95) & (df['Hybrid_Conversion'] < 100) & (df['Hybrid_Uncertainty'] <= 1.5)
nearest_closest = df[df['Exp'] == 33].nsmallest(1, 'Difference')  # For experiment 33

# Format table data
df_table = df.copy()
df_table['Hybrid_Conversion'] = df_table['Hybrid_Conversion'].round(2)
df_table['Hybrid_Uncertainty'] = df_table['Hybrid_Uncertainty'].round(2)
df_table['Insilico_Conversion'] = df_table['Insilico_Conversion'].round(2)

# Create figure with adjusted gridspec
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.2], wspace=0.4)

# Main plot
ax = plt.subplot(gs[0])

# Plot Hybrid with error bars
hybrid_plot = ax.errorbar(df['Exp'], df['Hybrid_Conversion'], yerr=df['Hybrid_Uncertainty'],
                         fmt='o', color='green', capsize=5, label='Hybrid (MBDoE-BO)')

# Highlight Hybrid points meeting criteria
highlight_plot = ax.scatter(df[hybrid_highlight]['Exp'], df[hybrid_highlight]['Hybrid_Conversion'],
                          edgecolor='red', facecolor='none', s=100, linewidth=2, zorder=3,
                          label='Hybrid: 95 < Conversion < 100 & Uncertainty ≤1.5')

# Plot Insilico points
insilico_plot = ax.scatter(df['Exp'], df['Insilico_Conversion'],
                         marker='s', s=80, color='blue', label='Insilico', zorder=4)

# Add annotations
ax.annotate('Best Match\n(Exp 23: 98.83 ±0.89)', xy=(23, 98.83), xytext=(21, 103),
           arrowprops=dict(facecolor='red', shrink=0.05), color='red')
ax.annotate('Nearest Match\n(Exp 33: 96.67 vs 95.19)', xy=(33, 96.67), xytext=(31, 90),
           arrowprops=dict(facecolor='blue', shrink=0.05), color='blue')

# Format main plot
ax.set_title('Hybrid vs Insilico Conversion Comparison (Experiments 13-32)', fontsize=16)
ax.set_xlabel('Experiment Number', fontsize=12)
ax.set_ylabel('Conversion (%)', fontsize=12)
ax.set_xticks(df['Exp'])
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_ylim(0, 130)  # Extended y-axis to accommodate outlier

# Create table area
ax_table = plt.subplot(gs[1])
ax_table.axis('off')

# Create formatted table
table_data = df_table[['Exp', 'Hybrid_Conversion', 'Hybrid_Uncertainty', 'Insilico_Conversion']]
cell_text = []
for row in table_data.values:
    formatted_row = [
        str(int(row[0])),  # Whole number for Exp
        f"{row[1]:.2f}",   # Hybrid Conversion
        f"{row[2]:.2f}",   # Uncertainty
        f"{row[3]:.2f}"    # Insilico Conversion
    ]
    cell_text.append(formatted_row)

# Create table with highlighting
table = ax_table.table(
    cellText=cell_text,
    colLabels=['Exp', 'Hybrid Conv', '± Unc', 'Insilico'],
    loc='upper center',
    cellLoc='center',
    colColours=['#f0f0f0']*4,
    bbox=[0.1, 0.2, 0.8, 0.7]  # Adjusted table position
)

# Highlight special rows
for row in range(1, len(df)+1):
    exp = df.iloc[row-1]['Exp']
    if exp in [23, 33]:
        color = '#90EE90' if exp == 23 else '#FFD700'
        for col in range(4):
            table[row, col].set_facecolor(color)
            table[row, col].set_text_props(fontweight='bold')

# Add legend to upper right
ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left',
         frameon=True, fontsize=10, borderpad=1)

plt.tight_layout()
plt.show()