from concat import concat
from series import Series
from dataframe import DataFrame
import numpy as np
from index import MultiIndex

s1 = Series([0, 1], index=["a", "b"])
s2 = Series([2, 3, 4], index=["c", "d", "e"])
s3 = Series([5, 6], index=["f", "g"])

# print(s1)
# print()
# print(s2)
# print()
# print(s3)
# print()

# print(concat([s1, s2, s3]))
# print(concat([s1, s2, s3], axis="columns"))

# s4 = concat([s1, s3])
# print(s4)

# print(concat([s1, s4], axis="columns"))
# print(concat([s1, s4], axis="columns", join="inner"))

# result = concat([s1, s1, s3], keys=["one", "two", "three"])
# print(result)

# print(concat([s1, s2, s3], axis="columns", keys=["one", "two", "three"]))

s1 = Series([0, 1, 2, 3], index=["a", "b", "c", "d"])
s2 = Series([4, 5, 6], index=["c", "d", "e"])

# data2 = concat([s1, s2], keys=["one", "two"])
# print(data2)
# print(data2.unstack())
# print(data2.unstack().stack())


df1 = DataFrame(np.arange(6).reshape(3, 2), index=["a", "b", "c"], columns=["one", "two"])
df2 = DataFrame(5 + np.arange(4).reshape(2, 2), index=["a", "c"], columns=["three", "four"])

# print(df1)
# print()
# print(df2)
# print()

# print(concat([df1, df2], axis="columns", keys=["level1", "level2"]))
# print(concat({"level1": df1, "level2": df2}, axis="columns"))

# print(concat([df1, df2], axis="columns", keys=["level1", "level2"], names=["upper", "lower"]))


# df1 = DataFrame(np.random.standard_normal((3, 4)),
# columns=["a", "b", "c", "d"])

# df2 = DataFrame(np.random.standard_normal((2, 3)),
# columns=["b", "d", "a"])

# print(df1)
# print()
# print(df2)
# print()
# print(concat([df1, df2], ignore_index=True))
# print(concat([df1, df2], ignore_index=False))



# s1 = Series([1, 2], index=["a", "b"])
# s2  = Series([3, 4], index=["a", "b"])

# res = concat([s1, s2], keys=["one", "two"])
# print(res)


# df5 = DataFrame([1], index=['a'])
# df6 = DataFrame([2], index=['a'])
# print(concat([df5, df6], verify_integrity=True))


# df1 = DataFrame({"A": [1, 2]}, index=[0, 1])
# df2 = DataFrame({"A": [3, 4]}, index=[1, 2])

# print(concat([df1, df2], verify_integrity=False))
# print(concat([df1, df2], verify_integrity=True))


# df1 = DataFrame({"A": [1, 2]})
# df2 = DataFrame({"A": [3, 4]})

# print(concat([df1, df2], axis=1, verify_integrity=True))#verify_integrity=True

# s1 = Series([1, 2], index=["a", "b"])
# s2 = Series([3, 4], index=["b", "c"])

# res = concat([s1, s2])
# print(res)

# res = concat([s1, s2], verify_integrity=True)
# print(res)



s1 = Series([1, 2])
s2 = Series([3, 4])

print(concat([s1, s2], keys=["one", "two"], levels=[["zero", "one"]]))# levels=[["zero", "one"]]