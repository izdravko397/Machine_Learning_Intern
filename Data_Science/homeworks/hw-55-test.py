from dataframe import DataFrame
from series import Series
import numpy as np
import pandas as pd
from index import MultiIndex

df = DataFrame({"key1" : np.array(["a", "a", None, "b", "b", "a", None]),
"key2" : Series([1, 2, 1, 2, 1, None, 1], dtype=float).array,
"data1" : np.random.standard_normal(7),
"data2" : np.random.standard_normal(7)})
print(df)

# === SeriesGroupBy ===
# grouped = df["data1"].groupby(df["key1"])
# print(grouped.mean())
# print(grouped.count())
# print(grouped.size())
# print(grouped.min())
# print(grouped.max())
# print(grouped.sum())

# for g_name, g_val in grouped:
#     print(g_name)
#     print(g_val)

groups2 = df["data1"].groupby([df["key1"], df["key2"]]).mean()
print(groups2)
# print(groups2.unstack())

# states = np.array(["OH", "CA", "CA", "OH", "OH", "CA", "OH"])
# years = [2005, 2005, 2006, 2005, 2006, 2005, 2006]
# print(df["data1"].groupby([states, years]).mean())



s = Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
# print(s)

# -- with dict --
# mapping = {'a': 'X', 'b': 'X', 'c': 'Y', 'f': 8}
# result = s.groupby(mapping).sum()
# print(result)


# -- with func --
def factor(i):
    if i in ('a', 'b'):
        return 'first'
    return 'last'

# print(s.groupby(factor).sum())


# === DFGroupBy ===
# grouped = df.groupby(['key1', 'key2'])
# grouped = df.groupby([df['key1'], df['key2']])
# grouped = df.groupby("key2")

# print(grouped.mean())
# print(grouped.count())
# print(grouped.size())
# print(grouped.min())
# print(grouped.max())
# print(grouped.sum())

# for g_name, g_val in grouped:
#     print(g_name)
#     print(g_val)

# print(df.groupby("key2").mean())
# print(df.groupby("key1").size())
# print(df.groupby("key1", dropna=False).size())
# print(df.groupby("key1").count())


# print(df.groupby(["key1", "key2"]).mean())
# print(df.groupby(["key1", "key2"]).size())
# print(df.groupby(["key1", "key2"], dropna=False).size())


# print(df.groupby([df["key1"], df["key2"]]).mean())
# print(df.groupby([df["key1"], df["key2"]]).max())
# print(df.groupby([df["key1"], df["key2"]]).size())
# print(df.groupby([df["key1"], df["key2"]]).count())





people = DataFrame(np.random.standard_normal((5, 5)),
columns=["a", "b", "c", "d", "e"],
index=["Joe", "Steve", "Wanda", "Jill", "Trey"])
people.iloc[2:3, 1: 3] = np.nan
# print(people)

mapping = {"Joe": "red", "Steve": "red", "Wanda": "blue",
"Jill": "blue", "Trey": "red", "f" : "orange"}
# print(people.groupby(mapping, axis="index").sum())
# print(people.groupby(mapping, axis="index").count())


mapping = {"a": "red", "b": "red", "c": "blue",
"d": "blue", "e": "red", "f" : "orange"}
# print(people.groupby(mapping, axis="columns").sum())
# print(people.groupby(mapping, axis="columns").mean())
# print(people.groupby(mapping, axis="columns").max())
# print(people.groupby(mapping, axis="columns").min())
# print(people.groupby(mapping, axis="columns").count())
# print(people.groupby(mapping, axis="columns").size())



# mapping = {"a": "red", "b": "red", "c": "blue",
# "d": "blue", "e": "red"}
# map_series = Series(mapping)
# print(map_series)
# print(people.groupby(map_series, axis="columns").sum())
# print(people.groupby(map_series, axis="columns").mean())
# print(people.groupby(map_series, axis="columns").max())
# print(people.groupby(map_series, axis="columns").min())
# print(people.groupby(map_series, axis="columns").count())
# print(people.groupby(map_series, axis="columns").size())



# grouped = df.groupby({"key1": "key", "key2": "key",
# "data1": "data", "data2": "data"}, axis="columns")
# for group_key, group_values in grouped:
#     print(group_key)
#     print(group_values)

# print(grouped.count())


# print(people.groupby(len).sum())
# print(people.groupby(factor, axis=1).sum())

key_list = ["one", "one", "one", "two", "two"]
# print(people.groupby([len, key_list]).min())
# print(people.groupby([len, key_list]).count())


key1 = ['a', 'a', 'a', 'b', 'b']
key2 = [1, 2, 1, 1, 2]

# print(people.groupby([key1, key2], axis='columns').mean())
# print(people.groupby([key1, key2], axis='columns').max())
# print(people.groupby([key1, key2], axis='columns').count())
# print(people.groupby([key1, key2], axis='columns').size())


# --- multiindex test ---
columns = MultiIndex.from_arrays([["US", "US", "US", "JP", "JP"],
                                    [1, 3, 5, 1, 3]],
                                    names=["cty", "tenor"])

hier_df = DataFrame(np.random.standard_normal((4, 5)), columns=columns)

# print(hier_df)
# print(hier_df.groupby(level="cty", axis="columns").count())
# print(hier_df.groupby(level="cty", axis="columns").sum())




# arrays = [
#     ['A', 'A', 'B', 'B'],
#     [1, 2, 1, 2]
# ]
# index = MultiIndex.from_arrays(arrays, names=['letter', 'number'])

# s = Series([10, 20, 30, 40], index=index)
# print("Series:")
# print(s)
# print("-" * 40)

# print("level='letter':")
# print(s.groupby(level='letter').sum())
# print("-" * 40)

# print("level=0:")
# print(s.groupby(level=0).sum())
# print("-" * 40)

# print("level='number':")
# print(s.groupby(level='number').sum())
# print("-" * 40)

# print("level=['letter', 'number']:")
# print(s.groupby(level=['letter', 'number']).sum())




# arrays = [
#     ['X', 'X', 'Y', 'Y'],
#     [1, 2, 1, 2]
# ]
# index = MultiIndex.from_arrays(arrays, names=['group', 'id'])
# df = DataFrame({'value1': [5, 10, 15, 20], 'value2': [50, 100, 150, 200]}, index=index)

# print("DataFrame:")
# print(df)
# print("-" * 40)

# print("level='group':")
# print(df.groupby(level='group').sum())
# print("-" * 40)

# print("level=1:")
# print(df.groupby(level=1).sum())
# print("-" * 40)

# print(df.groupby(level=[1, 0]).sum())




# columns = MultiIndex.from_tuples(
#     [('A', 'x'), ('A', 'y'), ('B', 'x'), ('B', 'y')],
#     names=['main', 'sub']
# )

# df2 = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=columns)
# print(df2)
# print("-" * 40)

# print("level='main', axis=1:")
# print(df2.groupby(level='main', axis=1).sum())
# print("-" * 40)

# print("level='sub', axis=1:")
# print(df2.groupby(level='sub', axis=1).sum())
# print("-" * 40)

# print("level='main', 1 ; axis=1:")
# print(df2.groupby(level=['main', 1], axis=1).sum())