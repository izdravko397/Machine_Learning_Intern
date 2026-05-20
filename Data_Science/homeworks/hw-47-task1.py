from index import Index, RangeIndex
import numpy as np
from dataframe import DataFrame
from series import Series


data = Series(np.random.uniform(size=9), 
    index=[["a", "a", "a", "b", "b", "c", "c", "d", "d"],
           [1, 2, 3, 1, 3, 1, 2, 2, 3]])

# print(data)

data.index.names = ['literal', 'num']
print(data)

# print(data.index)

# print(data["b"])

# print(data["b":"d"])
# print(data.loc[["b", "d", 'a']])
# print(data.loc[:, 2])

print(data.unstack())
print(data.unstack().stack())

frame = DataFrame({"a": list(range(7)), "b": list(range(7, 0, -1)),
"c": ["one", "one", "one", "two", "two", "two", "two"],
"d": [0, 1, 2, 0, 1, 2, 3]})

# print(frame)
# print()
# frame2 = frame.set_index(["c", "d"])
# print(frame2)

# frame2 = frame.set_index(["c"], False)
# print(frame2)

# print(frame.set_index(["c", "d"], drop=False))

# print()
# print(frame.reset_index())