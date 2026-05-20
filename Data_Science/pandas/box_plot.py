from dataframe import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from matplotlib.patches import Rectangle


def box_plot(x, y, data: DataFrame):
    x_column = data[x]
    y_column = data[y]

    fig, ax = plt.subplots()
    
    x_min, x_max = x_column._data.min(), x_column._data.max()
    x_margin = (x_max - x_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_xlabel(x)
    
    unique_y = np.unique(y_column._data)
    ax.set_yticklabels(unique_y)

    y_ticks = np.arange(1, len(unique_y) + 1)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_ticks[0] * 0.5, y_ticks[-1] * 1.2)

    y_ticks = ax.get_yticks()
    print(y_ticks)
    bar_width = 0.8

    for y in y_ticks:
        val = unique_y[y - 1]
        inxs = np.where(y_column._data == val)[0]
        data = x_column._data[inxs]

        q1, median, q3 = np.quantile(data, [0.25, 0.5, 0.75])
        iqr = q3 - q1

        lower_whisker = max(np.min(data), q1 - 1.5 * iqr)
        upper_whisker = min(np.max(data), q3 + 1.5 * iqr)

        outliers = data[(data < lower_whisker) | (data > upper_whisker)]

        rect = Rectangle((q1, y - (bar_width / 2)), iqr, bar_width, 
                         edgecolor='black', linewidth=2, facecolor='gray')
        ax.add_patch(rect)

        ax.hlines(y, lower_whisker, q1, colors='black')
        ax.hlines(y, q3, upper_whisker, colors='black')

        ax.vlines(median, y - (bar_width / 2), y + (bar_width / 2), colors='black')
        ax.vlines([lower_whisker, upper_whisker], y - (bar_width / 4), y + (bar_width / 4), colors='black')

        outliers_len = len(outliers)
        if outliers_len:
            ax.scatter(outliers, [y] * outliers_len, c='black', marker='d')

    ax.grid(axis='x')
    return ax



tips = pd.read_csv("examples/tips.csv")
tips["tip_pct"] = tips["tip"] / (tips["total_bill"] - tips["tip"])
data=tips[tips.tip_pct < 0.5]

my_df = DataFrame(data=np.array([data["tip_pct"].array, data["day"].array]).T, columns=["tip_pct", "day"])
box_plot('tip_pct', 'day', data=my_df)
plt.show()

# data = [
#     [1, 'A'],
#     [2, 'A'],
#     [3, 'A'],
#     [10, 'A'],  # outlier
#     [2, 'B'],
#     [3, 'B'],
#     [4, 'B'],
#     [20, 'B'],  # outlier
#     [5, 'C'],
#     [6, 'C'],
#     [7, 'C']
# ]

# df = DataFrame(data, columns=['value', 'category'])
# box_plot('value', 'category', data=df)
# plt.show()




# data = [
#     [1, 10],
#     [2, 10],
#     [3, 10],
#     [10, 10],  # outlier
#     [2, 20],
#     [3, 20],
#     [4, 20],
#     [20, 20],  # outlier
#     [5, 30],
#     [6, 30],
#     [7, 30]
# ]

# df = DataFrame(data, columns=['value', 'category'])
# box_plot('value', 'category', data=df)
# plt.show()
