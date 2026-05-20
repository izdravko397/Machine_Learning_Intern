from series import Series
from dataframe import DataFrame
import pandas as pd
import numpy as np
from index import MultiIndex
from read_csv import read_csv


def peak_to_peak(arr):
    return arr.max() - arr.min()

# ======== Series ========
# s = Series([1, 2, 3, 4, 5, 6], index=['a','a','b','b','c','c'], name='qsha')
# grouped = s.groupby(s.index._data)
# print(s)

# result = grouped.agg(['min', 'max', ('SUM', 'sum')])
# print(result)

# result = grouped.agg('mean')
# print(result)

# result = grouped.agg(['mean', ('diff', peak_to_peak)])
# print(result)


# ======== DF ========


# df = DataFrame({"key1" : np.array(["a", "a", None, "b", "b", "a", None]),
# "key2" : Series([1, 2, 1, 2, 1, None, 1], dtype=float).array,
# "data1" : np.random.standard_normal(7),
# "data2" : np.random.standard_normal(7)})
# print(df)

# grouped = df.groupby("key1")
# print(grouped.agg(peak_to_peak))






# tips = read_csv("examples/tips.csv")
# tips["tip_pct"] = tips["tip"] / tips["total_bill"]
# print(tips.head())

# grouped = tips.groupby(["day", "smoker"])
# grouped_pct = grouped["tip_pct"]

# grouped_pct = tips['tip_pct'].groupby([tips["day"], tips["smoker"]])
# print(grouped_pct.agg("mean"))
# print(grouped_pct.agg(["mean", "sum", peak_to_peak]))
# print(grouped_pct.agg([("average", "mean"), ("peak", peak_to_peak)]))

# functions = ["count", "mean", "max"]
# result = tips[["tip_pct", "total_bill"]].groupby([tips["day"], tips["smoker"]])
# result = result.agg(functions)
# print(result)
# print(result["tip_pct"])
      
# ftuples = [("Average", "mean"), ("Variance", 'max')]
# print(result.agg(ftuples))

# print(grouped.agg({"tip" : 'max', "size" : "sum"}))






# df = pd.DataFrame({
#     "A": [1, 2, 3],
#     "B": [4, 5, 6],
#     "C": [7, 8, 9],
#     "D": [10, 11, 12]
# })
# print(df)
# groups = {'A': 'x', 'B': 'x', 'C': 'y', 'D': 'y'}
# # groups = {0: 'x', 1: 'x', 2: 'y'}
# g = df.groupby(groups, axis=1)
# # print(g.agg(peak_to_peak))
# print(g.agg(['min', 'max']))




# mlt = MultiIndex.from_arrays([list('AABB'), list('abab')], names=['inx1', 'inx2'])

# df = DataFrame(np.arange(1, 17).reshape((4, 4)), columns=list('abcd'))
# df.index.name = 'inx'
# df.columns.name = 'col'
# print(df)

# print(df.groupby(list('aabb')).agg('sum'))
# print(df.groupby(list('aabb')).agg(peak_to_peak))
# print(df.groupby(list('aabb')).agg([('peak', peak_to_peak), 'sum']))

# print(df.groupby([list('XXYY'), list('aabb')]).agg(['min', 'max']))
# print(df.groupby([list('XXYY'), list('aabb')]).agg('max'))


# print(df.groupby(list('aabb'), axis=1).agg('sum'))
# print(df.groupby(list('aabb'), axis=1).agg(peak_to_peak))
# print(df.groupby(list('aabb'), axis=1).agg([('peak', peak_to_peak), 'sum']))


# print(df.groupby([list('XXYY'), list('aabb')], axis=1).agg([('malko', 'min'), 'max', peak_to_peak]))
# print(df.groupby([list('XXYY'), list('aabb')], axis=1).agg('max'))



# print(df.groupby(list('aabb')).agg({'a': 'sum', 'b': 'sum'}))
# print(df.groupby(list('aabb')).agg({'a': 'sum', 'b': ['sum', 'max']}))
# print(df.groupby(list('aabb')).agg({'a': 'sum', 'b': [('sum2', 'sum'), 'max']}))
# print(df.groupby([list('XXYY'), list('aabb')]).agg({'a': 'sum', 'b': [('sum2', 'sum'), 'max']}))
