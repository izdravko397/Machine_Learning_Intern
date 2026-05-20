from series import Series
import numpy as np

# s = Series([1, 2, 3, 4])
# print(s.map(lambda x: x ** 2))

# s = Series(['apple', 'banana', 'cherry'])
# s_upper = s.map(str.upper)
# print(s_upper)


# s = Series(['c', 'd', 'm'])
# f = s.map({'c': 'cat', 'd': 'dog'})
# print(f)

# s = Series([1, 1, 2, 3, 4])
# mapped = s.map({1: 'one', 2: 'two'})
# print(mapped)

s = Series(['c', 'd', 'm'])
f = Series({'c': 'cat', 'd': 'dog'})
mapped = s.map(f)
print(mapped)