from dataframe import DataFrame
from Series import Series
import numpy as np

# obj = Series(np.arange(5), index=["a", "a", "b", "b", "c"])
# print(obj)

# print(obj.index.is_unique)

# print(obj["a"])

# print(obj["c"])

df = DataFrame(np.random.standard_normal((5, 3)), index=["a", "a", "b", "b", "c"])
print(df)
print()
print(df.loc["b"])
print()
print(df.loc["c"])