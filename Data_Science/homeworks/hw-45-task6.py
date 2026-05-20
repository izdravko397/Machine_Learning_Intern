from series import Series
from dataframe import DataFrame
import numpy as np

data = Series([1., -999., 2., -999., -1000., 3.])
# print(data)

# print(data.replace(-999, np.nan))
# print(data.replace([-999, -1000], np.nan))

# print(data.replace([-999, -1000], [np.nan, 0]))
# print(data.replace({-999: np.nan, -1000: 0}))


df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]})
print(df)
# print(df.replace(2, 99))
# print(df.replace([2, 3], 0))
# print(df.replace([2, 3], [20, 30]))
# print(df.replace({"A": {1: 100}, "B": {4: 400}}))