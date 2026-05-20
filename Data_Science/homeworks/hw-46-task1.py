from index import Index, RangeIndex
import numpy as np
from dataframe import DataFrame

data = DataFrame(np.arange(12).reshape((3, 4)),
index=["Ohio", "Colorado", "New York"],
columns=["one", "two", "three", "four"])

def transform(x):
    return x[:4].upper()

d = {"Ohio": 'O', "Colorado": 'C', "New York": 'NY'}

# print(repr(data.index.map(transform)))
# data.index = data.index.map(transform)
data.index = data.index.map(d)

print(data)