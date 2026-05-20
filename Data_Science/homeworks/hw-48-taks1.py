from index import Index, MultiIndex

# tuples = [('bar', 'one'),
#  ('bar', 'two'),
#  ('baz', 'one'),
#  ('baz', 'two'),
#  ('foo', 'one'),
#  ('foo', 'two'),
#  ('qux', 'one'),
#  ('qux', 'two')]

# inx = MultiIndex.from_tuples(tuples)
# print(inx)


# mi = MultiIndex.from_arrays([list('abb'), list('def')])
# print(mi)
# print(mi.nlevels)

# values = mi._data
# print(type(values))
# print(values)

# unique_index = Index(list('abc'))
# print(unique_index.get_loc('b')) # 1


# monotonic_index = Index(list('abbc'))
# print(monotonic_index.get_loc('b')) # slice(1, 3, None)

# non_monotonic_index = Index(list('abcb'))
# print(non_monotonic_index.get_loc('b')) # array([1, 3])

# mi = MultiIndex.from_arrays([list('abb'), list('def')])
# print(mi.get_loc('b')) # slice(1, 3, None)
# print(mi.get_loc('a')) # slice(1, 3, None)
# print(mi.get_loc(('b', 'e'))) # 1