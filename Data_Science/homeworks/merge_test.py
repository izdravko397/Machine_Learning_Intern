from merge import merge
from dataframe import DataFrame
from series import Series
import numpy as np
import pandas as pd
from index import Index, RangeIndex, MultiIndex

df1 = DataFrame({"key": Series(["b", "b", "a", "c", "a", "a", "b"]),
"data1": Series(list(range(7)))})

df2 = DataFrame({"key": Series(["a", "b", "d"]),
"data2": Series(list(range(3)))})

# print(df1)
# print()
# print(df2)
# print()

# print(merge(df1, df2, how="outer"))
# print(merge(df1, df2))
# print(merge(df1, df2, on="key"))

# df3 = DataFrame({"lkey": Series(["b", "b", "a", "c", "a", "a", "b"]),
# "data1": Series(list(range(7)))})

# df4 = DataFrame({"rkey": Series(["a", "b", "d"]),
# "data2": Series(list(range(3)))})

# print(df3)
# print()
# print(df4)
# print()
# print(merge(df3, df4, left_on="lkey", right_on="rkey", how="outer"))
# print(merge(df3, df4, left_on="lkey", right_on="rkey"))


df1 = DataFrame({"key": Series(["b", "b", "a", "c", "a", "b"]),
"data1": Series(list(range(6)))})

df2 = DataFrame({"key": Series(["a", "b", "a", "b", "d"]),
"data2": Series(list(range(5)))})

# print(df1)
# print()
# print(df2)
# print()
# print(merge(df1, df2, on="key", how="left"))
# print(merge(df1, df2, how="inner"))
# print(merge(df1, df2, how="outer"))


left = DataFrame({"key1": ["foo", "foo", "bar"],
                "key2": ["one", "two", "one"],
                "lval": [1, 2, 3]})

right = DataFrame({"key1": ["foo", "foo", "bar", "bar"],
                    "key2": ["one", "one", "one", "two"],
                    "rval": [4, 5, 6, 7]})

# print(left)
# print()
# print(right)
# print()

# print(merge(left, right, on=["key1", "key2"], how="outer"))
# print(merge(left, right, on=["key1", "key2"], how="inner"))

# print(merge(left, right, on="key1"))
# print(merge(left, right, on="key1", suffixes=("_left", "_right")))

left1 = DataFrame({"key": Series(["a", "b", "a", "a", "b", "c"]),
"value": Series(list(range(6)))})

right1 = DataFrame({"group_val": [3.5, 7]}, index=["a", "b"])

# print(left1)
# print()
# print(right1)
# print()
# print(merge(left1, right1, left_on="key", right_index=True))
# print(merge(left1, right1, left_on="key", right_index=True, how="outer"))

left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]],
index=["a", "c", "e"],
columns=["Ohio", "Nevada"])

right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
index=["b", "c", "d", "e"],
columns=["Missouri", "Alabama"])

# print(left2)
# print()
# print(right2)
# print()

# print(merge(left2, right2, how="outer", left_index=True, right_index=True))



lefth = DataFrame({"key1": ["Ohio", "Ohio", "Ohio",
"Nevada", "Nevada"],
"key2": [2000, 2001, 2002, 2001, 2002],
"data": list(range(5))})

righth_index = MultiIndex.from_arrays([["Nevada", "Nevada", "Ohio", "Ohio", "Ohio", "Ohio"],
                                            [2001, 2000, 2000, 2000, 2001, 2002]])

righth = DataFrame({"event1": Series([0, 2, 4, 6, 8, 10], index=righth_index), 
                       "event2": Series([1, 3, 5, 7, 9, 11], index=righth_index)}, index=righth_index)


# print(lefth)
# print()
# print(righth)
# print()

# print(merge(lefth, righth, left_on=["key1", "key2"], right_index=True))
# print(merge(lefth, righth, left_on=["key1", "key2"], right_index=True, how="outer"))

# ---------------- joinn -------

# print(left2.join(right2, how="outer"))
# print(left1.join(right1, on="key"))

another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
index=["a", "c", "e", "f"],
columns=["New York", "Oregon"])

# print(left2.join([right2, another]))
# print(left2.join([right2, another], how="outer"))