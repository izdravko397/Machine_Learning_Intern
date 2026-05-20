import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/housing.csv")

fig, axes = plt.subplots(3, 3)
fig.subplots_adjust(hspace=0.35)
data_iter = iter(data)

for i in range(3):
    for j in range(3):
        col_name = next(data_iter)
        axes[i, j].hist(data[col_name], bins=50)
        axes[i, j].set_title(col_name)
        axes[i, j].grid()


plt.show()