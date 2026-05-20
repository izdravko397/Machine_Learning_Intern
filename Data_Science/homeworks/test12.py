from dataframe import DataFrame
from series import Series

df = DataFrame({
    "A": [1, 2],
    "B": [3, 4]
})

# print(df)

# df1 = df.reindex([1, 0, 2])
# print(df1)

# df1 = df.reindex([1, 0, 2], method='ffill')
# print(df1)

# df1 = df.reindex(columns=['B', 'A', 'D'])
# print(df1)

# df1 = df.reindex(columns=['B', 'A', 'D'], method='ffill')
# print(df1)

# df1 = df.reindex(columns=['B', 'A', 'D', 'R', 'E'])
# print(df1)

# df1 = df.reindex(columns=['B', 'A', 'D', 'R', 'E', 'B', 'K'], method='bfill')
# print(df1)


import pandas as pd
import numpy as np

# df1 = DataFrame([[1, 2], [3, 4]], columns=['A', 'B'], index=['a', 'b'])
# df2 = DataFrame([[1, 2], [3, 0]], columns=['A', 'B'], index=['a', 'b'])

# print(df1)
# print(df2)
# print(df1 == df2)


data = DataFrame(np.arange(16).reshape((4, 4)),
index=["Ohio", "Colorado", "Utah", "New York"],
columns=["one", "two", "three", "four"])

print(data)
# print(data["two"])
# print(data[["three", "one"]])
# print(data[:2])
# print(data[data["three"] > 5])
# print(data < 5)
data[data < 5] = 0
# print(data.loc["Colorado"])
# print(data.loc[["Colorado", "New York"]])
# print(data.loc["Colorado", ["two", "three"]])
# print(data.iloc[2])
# print(data.iloc[[2, 1]])
# print(data.iloc[2, [3, 0, 1]])
# print(data.iloc[[1, 2], [3, 0, 1]])
#!!!
# print(data.loc[:"Utah", "two"])
# print(data.iloc[:, :3][data.three > 5])
# print(data.loc[data.three >= 2])
#!!!
