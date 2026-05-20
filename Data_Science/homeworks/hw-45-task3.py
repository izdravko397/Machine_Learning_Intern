from series import Series
from dataframe import DataFrame
import numpy as np
import pandas as pd


df = DataFrame(np.random.standard_normal((7, 3)))
df.iloc[:4, 1] = np.nan
df.iloc[:2, 2] = np.nan
print(df)


df.fillna(method="bfill", limit=1)
print(df)

# df.fillna(0)
# print(df)

# df.fillna({1: 0.5, 2: 0})
# print(df)

df = DataFrame(np.random.standard_normal((6, 3)))
df.iloc[2:, 1] = np.nan
df.iloc[4:, 2] = np.nan
# print(df)

# df.fillna(method="ffill")
# print(df)

# df.fillna(method="bfill", limit=2)
# print(df)


# data = Series([1., np.nan, 3.5, np.nan, 7])
# print(data)
# data.fillna(data.mean())
# print(data)
