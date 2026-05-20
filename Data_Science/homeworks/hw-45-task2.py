from series import Series
from dataframe import DataFrame
import numpy as np

data = Series([1, np.nan, 3.5, np.nan, 7])
print(data)

print(data.dropna())

# data = DataFrame([[1., 6.5, 3.], [1., np.nan, np.nan],
# [np.nan, np.nan, np.nan], [np.nan, 6.5, 3.]])
# print(data)

# print(data.dropna())
# print(data.dropna(how="all"))

# data[4] = np.nan
# print(data)
# print(data.dropna(axis="columns", how="all"))

df = DataFrame(np.random.standard_normal((7, 3)))
df.iloc[:4, 1] = np.nan
df.iloc[:2, 2] = np.nan
# print(df)

# print(df.dropna())
# print(df.dropna(thresh=2))
# print(df.dropna(thresh=4, axis=1))
