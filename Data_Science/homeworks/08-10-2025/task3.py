import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def equal_distances(start, stop, count):
    val = (stop - start) / (count - 1)
    res = np.fromiter((start + (val * i) for i in range(count)), dtype=float)
    return res


fig = plt.figure()
ax = fig.add_subplot()

# data1 = np.linspace(4, 10, 100)

data1 = equal_distances(4, 10, 100)
x = equal_distances(0, 2, 100)

ax.plot(x, data1, color="red", label="Predictions")

noise = np.random.standard_normal(100)
data2 = data1 + noise

ax.scatter(x, data2)

ax.set_xlabel('x1', fontstyle='italic')
ax.set_ylabel('y', rotation=0, fontstyle='italic')
ax.set_xlim([0, 2])
ax.set_ylim([0, 14])

plt.grid()
plt.legend(loc='upper left')
plt.show()