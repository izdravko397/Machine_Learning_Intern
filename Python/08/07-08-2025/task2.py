from functools import reduce
from collections import Counter
c = Counter({'red': 4, 'blue': 2}, red=4, blue=8)             
print(c)
# a = 'gallahad'
# d = reduce(lambda d, key: d.update({key: d.get(key, 0) + 1}) if d.update({key: d.get(key, 0) + 1}) else d, a, {})
# print(d)

b = {'red': 4, 'blue': 2}
v = reduce(lambda d, key: d.update({key[0]: d.get(key[0], 0) + key[1]}) or d, b.items(), {})
print(v)

# class Counter:
#     def __init__(self, iterable={}, **kwargs):
#         dictionary = {}

#         if isinstance(iterable, dict):
#             dictionary = iterable
#         else:
#             for i in iterable:
#                 dictionary[i] = dictionary.get(i, 0) + 1

#         for key, val in kwargs.items():
#             dictionary[key] = dictionary.get(key, 0) + val
        
#         self.dict = dictionary
    
#     def __getitem__(self, name):
#         return self.dict.get(name, 0)
    
#     def __str__(self):
#         return f'Counter({dict(sorted(self.dict.items(), key=lambda items: items[1], reverse=True)) if self.dict else ''})'
    
# c = Counter()                           # a new, empty counter
# print(c)
# c = Counter('gallahad')                 # a new counter from an iterable
# print(c)
# c = Counter({'red': 4, 'blue': 2})      # a new counter from a mapping
# print(c)
# c = Counter(cats=4, dogs=8)             # a new counter from keyword args
# print(c)
# c = Counter('gallahad', a=4, dogs=8)
# print(c)
# c = Counter({'red': 4, 'blue': 2}, red=4, blue=8)             
# print(c)

# Counter()
# Counter({'a': 3, 'l': 2, 'g': 1, 'h': 1, 'd': 1})
# Counter({'red': 4, 'blue': 2})
# Counter({'dogs': 8, 'cats': 4})