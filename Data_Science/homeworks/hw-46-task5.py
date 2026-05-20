from index import Index, RangeIndex
import numpy as np
from dataframe import DataFrame
from series import Series
from get_dummies import get_dummies


# df = DataFrame({"key": ["b", "b", "a", "c", "a", "b"], "data1": list(range(6))})
# print(df)
# print()
# dummies = get_dummies(df["key"], prefix="key")
# print(dummies)

# df_with_dummy = df[["data1"]].join(dummies)
# print(df_with_dummy)

df1 = DataFrame({"A": [1, 2, 3]}, index=["x", "y", "z"])
df2 = DataFrame({"B": [10, 20, 30]}, index=["x", "y", "z"])

print(df1.join(df2))
# # 
# df1 = DataFrame({"A": [1, 2, 3]}, index=["x", "y", "z"])
# df2 = DataFrame({"B": [10, 20]}, index=["y", "z"])
# # 
# print(df1.join(df2))

# df1 = DataFrame({"A": [1, 2]}, index=["x", "y"])
# df2 = DataFrame({"B": [10, 20]}, index=["a", "b"])

# # print(df1.join(df2))

# df1 = DataFrame({"A": [1, 2]}, index=["x", "y"])
# df2 = DataFrame({"B": [10, 20, 30]}, index=["x", "y", "z"])

# print(df1.join(df2))

