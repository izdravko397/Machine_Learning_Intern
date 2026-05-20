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

def zip(*iterables):
    iterators = [iter(i) for i in iterables]

    while True:
        try:
            yield tuple(next(i) for i in iterators)
        except:
            break

a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
c = {9, 10, 11}

for y in zip(a, b, c):
    print(y)

def range(*start_stop_step):
    len_args = len(start_stop_step)

    start = 0 if len_args == 1 else start_stop_step[0]
    stop = start_stop_step[0] if len_args == 1 else start_stop_step[1]
    step = 1 if len_args < 3 else start_stop_step[2]

    if step > 0:
        while start < stop:
            yield start
            start += step

    elif step < 0:
        while start > stop:
            yield start
            start += step

    else:
        raise ValueError('Step cant be 0')

# for y in range(1, 4):
#     print(y)

def enumerate(iterable, start=0):
    for e in iterable:
        yield (start, e)
        start += 1

# c = [9, 10, 11]

# for y in enumerate(c, start=5):
#     print(y)