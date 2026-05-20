from dataframe import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = [[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1], [6.4, 3.2, 1], [5.9, 3.0, 2]]

df = pd.DataFrame(data, columns=['length', 'width', 'species'])
df.plot.scatter(x='length', y='width', c='DarkBlue')

df = DataFrame(data, columns=['length', 'width', 'species'])
df.plot_scatter(x='length', y='width', c='DarkBlue', hue='species')

plt.show()


# data = [
#     [1, 100, 0],
#     [2, 200, 1],
#     [3, 400, 2],
#     [4, 800, 1],
#     [5, 1600, 0]
# ]

# df = pd.DataFrame(data, columns=['length', 'width', 'species'])
# df.plot.scatter(x='length', y='width', c='DarkBlue', s=200)

# df_custom = DataFrame(data, columns=['length', 'width', 'species'])
# df_custom.plot_scatter(x='length', y='width', c='DarkBlue', s=200)

# plt.show()



# data = [
#     [1, 2, 0],
#     [2, 4, 1],
#     [3, 1, 2],
#     [4, 3, 0],
#     [5, 5, 1]
# ]

# df = pd.DataFrame(data, columns=['x', 'y', 'cat'])
# df.plot.scatter(x='x', y='y', s=200, c='red')

# df_custom = DataFrame(data, columns=['x', 'y', 'cat'])
# df_custom.plot_scatter(x='x', y='y', c='red', hue='cat')

# plt.show()