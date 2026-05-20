from functools import reduce
from collections import Counter

str = "hello world"
c = Counter(str)
print(c)

c = reduce(lambda acc, i: acc.update({i: acc.get(i, 0) + 1}) or acc, str, {})
print(c)