from series import Series
from dataframe import DataFrame
from index import Index
import numpy as np

data = DataFrame(np.arange(6).reshape((2, 3)),
index=Index(["Ohio", "Colorado"], name="state"),
columns=Index(["one", "two", "three"],
name="number"))

# print(data)
result = data.stack()
# print(result)
# print()

df = DataFrame({"left": result, "right": result + 5},
columns=Index(["left", "right"], name="side"), index=result.index)

print(df)
# print(df.unstack(level="state"))
# print(df.unstack())
# print(df.unstack(level="state").stack(level="side"))
# print(df.unstack(level="state").stack(level="state"))

# print(df.unstack(level="state").stack())
# print(df.unstack())


data = Series(np.random.uniform(size=9),
index=[["a", "a", "a", "b", "b", "c", "c", "d", "d"],
[1, 2, 3, 1, 3, 1, 2, 2, 3]])

# print(data)
# print(data.unstack())
# print(data.unstack().stack())


data = DataFrame(np.arange(6).reshape((2, 3)),
index=Index(["Ohio", "Colorado"], name="state"),
columns=Index(["one", "two", "three"],name="number"))

# print(data)
# result = data.stack()
# print(result)



# s = Series(
#     [10, 20, 30, 40],
#     index=[["a", "a", "b", "b"], ["x", "y", "x", "y"]]
# )

# print(s)
# print(s.unstack())
# print(s.unstack(level=0))