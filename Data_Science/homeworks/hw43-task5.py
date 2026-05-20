from series import Series
import numpy as np

obj = Series([4, 1, np.nan, 3, 2], index=['d', 'a', 'e', 'c', 'b'])
# print(obj)

# print(obj.median())

# s = Series([1, 2, None, 4])

# print(s.median())     

# s = Series([1, 3, 4, 6, 7])
# print(s.quantile(0.25))
# print(s.quantile(0.75))

# s = Series([1, 2, 3, 4])
# print(s.quantile(0.25))

# s = Series([100, 120, 90, 150])
# pct = s.pct_change()
# print(pct)
# # 0         NaN
# # 1    0.200000
# # 2   -0.250000
# # 3    0.666667

# pct = s.pct_change(periods=2)
# print(pct)
# 0   nan
# 1   nan
# 2   -0.1
# 3   0.25


# s = Series([1, 2, 3, 4, 5])
# print(s.var())

# s = Series([100, 120, 90, 150])
# diff = s.diff()
# print(diff)

# diff2 = s.diff(periods=2)
# print(diff2)