from series import Series
from dataframe import DataFrame
import numpy as np
# import pandas as pd

s = Series([1, 2, 2, 3, 1])
print(s)
print(s.drop_duplicates())
print(s.drop_duplicates(keep='last'))
# print(s.duplicated())

data = DataFrame({"k1": ["one", "two"] * 3 + ["two"],
"k2": [1, 1, 2, 3, 3, 4, 4]})
# print(data)

# print(data.duplicated())
# print(data.drop_duplicates())

# data["v1"] = range(7)
# print(data)

# print(data.drop_duplicates(subset=["k1"]))

# print(data.drop_duplicates(["k1", "k2"], keep="last"))