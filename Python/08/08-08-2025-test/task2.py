def map(func, *iter):
    for i in zip(*iter):
        yield func(*i)

# l = [1, 2, 3, 4]
# a = map(lambda a: a * a, l)

# print(a)
# for i in a:
#     print(i)

def filter(func, iter):
    for i in iter:
        if func(i):
            yield i

# l = [1, 2, 3, 4]
# a = filter(lambda a: a % 2 == 0, l)

# print(a)
# for i in a:
#     print(i)

def reduce(func, iterable, initializer='_default'):
    make_to_iterable = iter(iterable)

    if initializer == '_default':
        initializer = next(make_to_iterable)

    res = initializer
    for i in make_to_iterable:
        res = func(res, i)

    return res

# from functools import reduce
# l = [1, 2, 3, 4]
# print(reduce(lambda a,b: a + b, l))

# a = 'gallahad'
# d = reduce(lambda d, key: d.update({key: d.get(key, 0) + 1}) or d, a, {})
# print(d)
