def union1(arr1, arr2):
    return sorted(arr1 + arr2)


def union2(arr1, arr2):
    res = []
    len_arr1 = len(arr1)
    len_arr2 = len(arr2)

    inx1 = 0
    inx2 = 0

    while True:
        element1 = arr1[inx1] if inx1 < len_arr1 else None
        element2 = arr2[inx2] if inx2 < len_arr2 else None

        if element1 is None or element2 is None:
            if element1 is not None:
                res.extend(arr1[inx1:])

            elif element2 is not None:
                res.extend(arr2[inx2:])
            break

        if element1 < element2:
            res.append(element1)
            inx1 += 1

            if inx1 < len_arr1 and element2 <= arr1[inx1]:
                res.append(element2)
                inx2 += 1

        else:
            res.append(element2)
            inx2 += 1

            if inx2 < len_arr2 and element1 <= arr2[inx2]:
                res.append(element1)
                inx1 += 1

    return res

def get_next_el(it):
    try:
        return next(it)
    except:
        return

def union3(arr1, arr2):
    iter1 = iter(arr1)
    iter2 = iter(arr2)
    
    res = []

    element1 = get_next_el(iter1)
    element2 = get_next_el(iter2)

    while True:

        if element1 is None or element2 is None:
            it = []
            if element1 is not None:
                it = iter1
                res.append(element1)

            elif element2 is not None:
                it = iter2
                res.append(element2)

            for i in it:
                res.append(i)

            break

        if element1 < element2:
            res.append(element1)
            element1 = get_next_el(iter1)

        elif element1 > element2:
            res.append(element2)
            element2 = get_next_el(iter2)

        else: # == 
            res.append(element1)
            res.append(element2)
            element1 = get_next_el(iter1)
            element2 = get_next_el(iter2)

    return res

list1 = [1, 3, 5, 7]
list2 = [2, 4, 6, 8]

# list1 = [1, 2, 3]
# list2 = [4, 5, 6]
list_all = union3(list1, list2); # [1, 2, 3, 4, 5, 6]
print(list_all)

arr1 = [1, 3, 5]
arr2 = [3, 4, 6]
print(union3(arr1, arr2))

arr1 = [1, 3, 3]
arr2 = [3, 4, 4]
print(union3(arr1, arr2))