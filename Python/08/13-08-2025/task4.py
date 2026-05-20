def lzip(*iterables):
    iterators = [iter(i) for i in iterables]
    iterators_len = len(iterators)
    iterators_set = set()
    longest_iter_end = False

    while True:
        val = []

        for i in range(iterators_len):
            try:
                item = next(iterators[i])
            except:
                if len(iterators_set) + 1 == iterators_len:
                    longest_iter_end = True
                    break

                iterators[i] = iter(iterables[i])
                item = next(iterators[i])
                iterators_set.add(iterators[i])
            val.append(item)

        if longest_iter_end:
            break

        yield tuple(val)

def lzip(*iterables):
    iterators = [iter(i) for i in iterables]
    lengths = [len(i) for i in iterables]
    longest_iter = max(lengths)

    for _ in range(longest_iter):
        val = []

        for i in range(len(iterators)):
            try:
                item = next(iterators[i])
            except:
                iterators[i] = iter(iterables[i])
                item = next(iterators[i])
            val.append(item)

        yield tuple(val)

s1 = "ivan"
s2 = [1, 2]
s3 = {3, 4, 5}

for e in lzip(s1, s2, s3):
  print(e)
# ("i", 1)
# ("v", 2)
# ("a", 1)
# ("n", 2)