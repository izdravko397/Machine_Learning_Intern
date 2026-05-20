from series import Series
from dataframe import DataFrame
import numpy as np

# a = Series([np.nan, 2.5, 0.0, 3.5, 4.5, np.nan],
# index=["f", "e", "d", "c", "b", "a"])

# b = Series([0., np.nan, 2., np.nan, np.nan, 5.],
# index=["a", "b", "c", "d", "e", "f"])

# print(a)
# print()
# print(b)
# print()

# print(a.combine_first(b))


# df1 = DataFrame({"a": [1., np.nan, 5., np.nan], "b": [np.nan, 2., np.nan, 6.], "c": list(range(2, 18, 4))})
# df2 = DataFrame({"a": [5., 4., np.nan, 3., 7.], "b": [np.nan, 3., 4., 6., 8.]})

# print(df1)
# print()
# print(df2)
# print()

# print(df1.combine_first(df2))