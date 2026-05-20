def union(set1: set, set2: set):
    res = set()

    for num in set1:
        res.add(num)

    for num in set2:
        res.add(num)

    return res

def union_2(set1: set, set2: set):
    return {*set1, *set2}

def intersect(set1: set, set2: set):
    return {num for num in set1 if num in set2}

def diff(set1: set, set2: set):
    return {num for num in set1 if num not in set2}

def symetrical_diff(set1: set, set2: set):
    return union(diff(set1, set2), diff(set2, set1))

set1 = {1, 2, 3}
set2 = {3, 4, 5}

print(union_2(set1, set2))
print(intersect(set1, set2))
print(diff(set1, set2))
print(diff(set2, set1))
print(symetrical_diff(set1, set2))