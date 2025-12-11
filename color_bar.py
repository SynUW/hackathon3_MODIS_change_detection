import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' if you only want to save figures
import matplotlib.pyplot as plt

# Class names (without numbers)
class_names = [
    "Temp-needleleaf",
    "Taiga-needleleaf",
    "Broadleaf-deciduous",
    "Mixed-forest",
    "Shrubland",
    "Polar-shrubland",
    "Wetland",
    "Cropland",
    "Barren",
    "Urban",
    "Water"
]

# Corresponding colors (hex)
hex_colors = [
    '#003d00', '#949c70', '#006300', '#1eab05', '#5c752b',
    '#b38a33', '#e1cf8a', '#e6ae66', '#a8abae', '#dc2126', '#4c70a3'
]

# Plot color bar
fig, ax = plt.subplots(figsize=(16, 2))
for i, (color, name) in enumerate(zip(hex_colors, class_names)):
    ax.bar(i, 1, color=color, edgecolor='none', width=1)

ax.set_xticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=10)
ax.set_yticks([])
ax.set_xlim(-0.5, len(class_names) - 0.5)
ax.set_title('Land Cover Color Bar', fontsize=14)
plt.tight_layout()
plt.show()