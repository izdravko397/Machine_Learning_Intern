from dataframe import DataFrame
from series import Series

left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]],
index=["a", "c", "e"],
columns=["Ohio", "Nevada"])

right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
index=["b", "c", "d", "e"],
columns=["Missouri", "Alabama"])

# print(left2.join(right2, how="outer"))


# left1 = DataFrame({"key": ["a", "b", "a", "a", "b", "c"],
# "value": list(range(6))})

# right1 = DataFrame({"group_val": [3.5, 7]}, index=["a", "b"])

# print(left1)
# print(right1)
# print(left1.join(right1, on="key"))

another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
index=["a", "c", "e", "f"],
columns=["New York", "Oregon"])

print(left2.join([right2, another]))


# print(left2.join([right2, another], how="outer"))