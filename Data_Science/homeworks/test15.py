from dataframe import DataFrame
from series import Series
import numpy as np
from read_csv import read_csv
import pandas as pd

# ============= task 1 =============
# close_px = read_csv("examples/stock_px.csv", index_col=0)
# print(close_px.info())

# df = pd.DataFrame({
#     'A': list(range(1000)),
#     'B': [x**2 for x in range(1000)],
#     'C': ['text'] * 1000
# })
# print(df.info())
# print()
# df = DataFrame({
#     'A': list(range(1000)),
#     'B': [x**2 for x in range(1000)],
#     'C': ['text'] * 1000
# })
# print(df.info())
# print(df.head())


# ============= task 2 =============
# s = Series([10, 20, 30, 40, 50])
# key = ['a', 'b', 'a', 'b', 'a']
# grouped = s.groupby(key)


# print(grouped.sum())
# print(grouped.mean())
# print(grouped.min())
# print(grouped.max())
# print(grouped.count())
# print(grouped.size())


# s = Series([10, 20, 30, 40, 50])
# key = ['a', None, 'a', 'b', 'a']
# grouped = s.groupby(key)


# print(grouped.sum())
# print(grouped.mean())
# print(grouped.min())
# print(grouped.max())
# print(grouped.count())
# print(grouped.size())




# s = Series([10, None, 30, np.nan, 50])
key = ['a', 'b', 'a', 'b', 'a']
# grouped = s.groupby(key)

# print('sum')
# print(grouped.sum())
# print('mean')

# print(grouped.mean())
# print('min')

# print(grouped.min())
# print('max')

# print(grouped.max())
# print('count')

# print(grouped.count())
# print('size')

# print(grouped.size())


# df = DataFrame({'A': list(range(1, 6)), 'B': list(range(10, 60, 10)), 'C': ['txt'] * 5})
# print(df)

# grouped = df.groupby(key)

# print('sum')
# print(grouped.sum())
# print('mean')

# print(grouped.mean())
# print('min')

# print(grouped.min())
# print('max')

# print(grouped.max())
# print('count')

# print(grouped.count())
# print('size')

# print(grouped.size())



# ============= task 3 =============

#      --- Series ---
df = DataFrame({'key': np.array(['a', 'b', 'c'] * 4), 'value': np.arange(12.)})
# print(df)
g = df['value'].groupby(df['key'])
# print(g.mean())

def get_mean(group):
    return group.mean()

# print(g.transform(get_mean))
# print(g.transform('mean'))

def times_two(group):
    return group * 2

# print(g.transform(times_two))

def get_ranks(group):
    return group.rank()

# print(g.transform(get_ranks))

def normalize(x):
    return (x - x.mean()) / x.std()

def std(x):
    return x.std()

# print(g.transform(normalize))
# normalized = (df['value'] - g.transform('mean')) / g.transform(std)
# print(normalized)


#      --- DF ---

df = DataFrame({'State':['Texas', 'Texas', 'Florida', 'Florida'], 
                   'a':[4,5,1,3], 'b':[6,10,3,11]})
print(df)
print()

def rand_group_len(x):
    # print(x)
    res = np.random.rand(len(x))
    # print(res)
    return res


# print(df.groupby('State').transform(rand_group_len))


def group_sum(x):
    return x.sum()

print(df.groupby('State').transform(group_sum))




df = DataFrame({'A' : ['foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
                   'B' : ['one', 'one', 'two', 'three', 'two', 'two'],
                   'C' : [1, 5, 5, 2, 5, 5],
                   'D' : [2.0, 5., 8., 1., 2., 9.]})
# print(df)
# print()

# grouped = df[['C', 'D']].groupby(df['A'])
# print(grouped.transform(lambda x: (x - x.mean()) / x.std()))
# print(grouped.transform("mean"))
# print(grouped.transform(lambda x: x.astype(int).max()))




# df = pd.DataFrame({'A' : [1, 5, 5, 2, 5, 5],
#                    'B' : [2.0, 5., 8., 1., 2., 9.],
#                    'C' : [1, 5, 5, 2, 5, 5],
#                    'D' : [2.0, 5., 8., 1., 2., 9.]})
# print(df)


# grouped = df.groupby({'A': 'int', 'C': 'int', 'B': 'float', 'D': 'float'}, axis=1)
# print(grouped.transform('sum'))

# df = DataFrame({'A' : [1, 5, 5, 2, 5, 5],
#                    'B' : [2.0, 5., 8., 1., 2., 9.],
#                    'C' : [1, 5, 5, 2, 5, 5],
#                    'D' : [2.0, 5., 8., 1., 2., 9.]})

# grouped = df.groupby({'A': 'int', 'C': 'int', 'B': 'float', 'D': 'float'}, axis=1)
# print(grouped.sum())
