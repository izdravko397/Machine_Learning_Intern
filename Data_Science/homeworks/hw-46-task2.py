from index import Index, RangeIndex
import numpy as np
from dataframe import DataFrame
from series import Series

df = DataFrame([[1.4, np.nan], [7.1, -4.5],
[np.nan, np.nan], [0.75, -1.3]],
index=["a", "b", "c", "d"],
columns=["one", "two"])

# print(df)
# print()
# print(df['one'].describe())
# print(df.describe())

# obj = Series(["a", "a", "b", "c"] * 4)
# print(obj)
# print()

# print(obj.describe())

# data = DataFrame(np.random.standard_normal((1000, 4)))
# print(data.describe())

df = DataFrame([[1.4, 'a'], [7.1, 'b'],
[2, 'a'], [0.75, 'c']],
index=["a", "b", "c", "d"],
columns=["one", "two"])

# print(
a = df.describe()
print(a)
# print(a['one'])