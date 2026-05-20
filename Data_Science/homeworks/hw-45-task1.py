from series import Series
import numpy as np

obj = Series(np.arange(4.), index=["a", "b", "c", "d"])
# print(obj)

# print(obj["b"])
# print(obj[1])
# print(obj[2:4])

# print(obj[["b", "a", "d"]])
# print(obj[[1, 3]])
# print(obj[obj < 2])

# print(obj.loc[["b", "a", "d"]])

obj1 = Series([1, 2, 3], index=[2, 0, 1])
obj2 = Series([1, 2, 3], index=["a", "b", "c"])

# print(obj1)
# print(obj2)

# print(obj1[[0, 1, 2]])
# print(obj2[[0, 1, 2]])

# print(obj2.loc[[0, 1]]) #error
# print(obj1.iloc[[0, 1, 2]])
# print(obj2.iloc[[0, 1, 2]])
# print(obj2.loc["b":"c"])
# obj2.loc["b":"c"] = 5
# print(obj2)

s = Series([10, 20, 30, 40], index=["a", "b", "a", "c"])
# print(s)
# print(s.loc[["a", "c"]])
# print(s.loc['a'])
# print(s.loc['c'])


ser = Series(np.arange(3.))
print(ser)

ser2 = Series(np.arange(3.), index=["a", "b", "c"])
# print(ser2)

print(ser[-1])
# print(ser2[-1])
# print(ser.iloc[-1])

print(ser[:2])