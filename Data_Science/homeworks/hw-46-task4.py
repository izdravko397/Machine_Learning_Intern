from index import Index, RangeIndex
import numpy as np
from dataframe import DataFrame
from series import Series

# s = Series(np.arange(5))
# print(s)
# print()

# sampler = np.random.permutation(5)
# print(sampler)
# print()

# print(s.take(sampler))

df = DataFrame(np.arange(5 * 7).reshape((5, 7)))
# print(df)
# print()

# sampler = np.random.permutation(5)
# print(sampler)
# print()

# print(df.take(sampler))
# column_sampler = np.random.permutation(7)
# print(column_sampler)
# print()

# print(df.take(column_sampler, axis="columns"))

# print(df.sample(n=3))
# print(df.sample(n=10, replace=True))

choices = Series([5, 7, -1, 6, 4])
print(choices.sample(n=10, replace=True))
