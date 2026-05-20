import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig, ax = plt.subplots(1, 2)
data = np.random.standard_normal(100)
ax[0].hist(data, bins=20, color="black", alpha=0.3)

def hist(ax, nums, bins, color="black", alpha=1, rwidth=1):
    intervals = pd.cut(nums, bins)
    val_counts = intervals.value_counts()

    cats_center = np.fromiter(((c.right + c.left) / 2 for c in val_counts.index.categories), dtype=float)

    width = np.diff(cats_center).mean() * rwidth
    ax.bar(cats_center, val_counts.array, color=color, alpha=alpha, width=width)

hist(ax[1], data, bins=20, color="black", alpha=0.3)
plt.show()