from dataframe import DataFrame
from series import Series
import numpy as np

obj = Series(np.arange(5.), index=["a", "b", "c", "d", "e"])
# print(obj)

# new_obj = obj.drop("c")
# print(new_obj)

# print(obj.drop(["d", "c"]))


data = DataFrame(np.arange(16).reshape((4, 4)),
index=["Ohio", "Colorado", "Utah", "New York"],
columns=["one", "two", "three", "four"])
# print(data)

# print(data.drop(index=["Colorado", "Ohio"]))
# print(data.drop(columns=["two"]))
# print(data.drop("two", axis=1))
# print(data.drop(["two", "four"], axis="columns"))

data = DataFrame(np.arange(25).reshape((5, 5)),
index=["Ohio", "Colorado", "Utah", "New York", "Ohio"],
columns=["one", "two", "three", "four", "one"])
print(data)

print(data.drop(index="Ohio", columns="one"))
# print(data.drop(columns="one"))
