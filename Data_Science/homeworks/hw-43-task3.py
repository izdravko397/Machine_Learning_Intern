from dataframe import DataFrame
from series import Series
import numpy as np

obj = Series([4, 1, np.nan, 3, 2], index=['d', 'a', 'e', 'c', 'b'], dtype=float)
print(obj)
# a = obj.sort_index(ascending=False, inplace=False)
# print(obj)
# print(a)

a = obj.sort_values(ascending=True, na_position='last', inplace=False)
print(obj)
print(a)