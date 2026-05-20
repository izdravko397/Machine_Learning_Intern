from series import Series
from dataframe import DataFrame
import numpy as np
import pandas as pd
from index import Index

# obj = Series([4.5, 7.2, -5.3, 3.6], index=["d", "b", "a", "c"])
# print(obj)

# obj2 = obj.reindex(["a", "b", "c", "d", "e"])
# print(obj2)

# obj3 = Series(["blue", "purple", "yellow"], index=[0, 2, 4])
# print(obj3)
# print(obj3.reindex(np.arange(6), method="ffill"))

# frame = DataFrame(np.arange(9).reshape((3, 3)),
#                   index=["a", "c", "d"],
#                   columns=["Ohio", "Texas", "California"])

# print(frame)

# frame2 = frame.reindex(index=["a", "b", "c", "d"])
# print(frame2)

# states = ["Texas", "Utah", "California"]
# print(frame.reindex(columns=states))

# a = DataFrame([1, 2, 3], columns=['a', 's'], index=[2, 5, 6 ,7])
# print(a)
# print(a.loc[2])


# obj = Series(np.arange(5.), index=["a", "b", "c", "d", "e"])
# print(obj)
# new_obj = obj.drop("c")
# print(new_obj)
# print(obj.drop(["d", "c"]))

# data = DataFrame(np.arange(16).reshape((4, 4)),
# index=["Ohio", "Colorado", "Utah", "New York"],
# columns=["one", "two", "three", "four"])
# print(data)

# print(data["two"])
# print(data[["three", "one"]])
# print(data[:2])

# print(data < 5)
# data[data < 5] = 0
# print(data)

# print(data.loc["Colorado"])
# print(data.loc[["Colorado", "New York"]])
# print(data.loc["Colorado", ["two", "three"]])

# print(data.iloc[2])
# print(data.iloc[[2, 1]])
# print(data.iloc[2, [3, 0, 1]])
# print(data.iloc[[1, 2], [3, 0, 1]])
# print(data.iloc[[1, 2], 1])

# print(data[data["three"] > 9])
# masl = data["three"] > 9
# print()
# s = data["three"]
# print()
# print(s[masl])

# print(data)
# print(data.drop(["two", "four"], axis="columns"))
# print(data.drop(index=["Colorado", "Ohio"]))
# print(data.drop(columns=["two"]))
# print(data.drop("two", axis=1))

# i = Index(['a', 'c', 'd', 'b', 'a'])
# print(i.isin(['a']))

# data = {"Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
# "Nevada": {2001: 2.4, 2002: 2.9}}
# index = set(key for val in data.values() for key in val)
# print(index)

# data = {"Ohio": Series([1, 2, 3], index=['a', 'b', 'c']),
# "Nevada": Series([1, 5, 3, 6], index=['a', 'b', 'c', 'd'])}
# index = sorted(set(inx for s in data.values() for inx in s.index))
# print(index)

# frame = DataFrame([[1, 2, 3], [4, 5 ,6]], columns=['A', 'B', 'C'], index=np.array([1, 2, 3, 4]))
# print(frame)

# a = DataFrame(np.arange(9).reshape((3, 3)), index=list('aab'))
# b = DataFrame(np.arange(3).reshape((1, 3)), index=list('a'))

# # print(a)
# # print(b)
# print(a + b)