import matplotlib.pyplot as plt
import numpy as np


days = ['Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon', 'Tue']
vals  = [50, 48, 55, 40, 39, 37, 55]


fig, ax = plt.subplots()

ax.plot(days, vals, color='red', marker="o", linewidth=2, markerfacecolor='white',
            markeredgecolor='red', markeredgewidth=2)

ax.axvline("Tue", color="#ecececda", linewidth=2)

ax.set_ylim([0, 150])
ax.set_yticks([0, 50, 100, 150])

for spine in ax.spines.values():
    spine.set_linestyle('--')
    spine.set_linewidth(0.5) 
    spine.set_color('gray')

ax.set_yticklabels([0, 50, 100, 150], color='gray')
ax.set_xticklabels(days, color='gray')
ax.yaxis.set_ticks_position('right')

ax.tick_params(axis='x', length=8, color='gray')
ax.tick_params(axis='y', length=0)

ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.5)


text = "AVERAGE\n\n            ms\n9 Apr 2024"
ax.text(0.76, 0.98, text, transform=ax.transAxes, fontname="DejaVu Sans",
        fontsize=14, ha="left", va="top", color="gray",
        bbox=dict(boxstyle='round',edgecolor='none', facecolor="#ecececda"))

ax.text(0.76, 0.92, "53", transform=ax.transAxes, fontweight='bold',
        fontsize=35, ha="left", va="top", color="black", fontname="DejaVu Sans")

plt.show()