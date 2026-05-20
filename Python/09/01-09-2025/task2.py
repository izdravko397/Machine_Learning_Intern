def remove_all(lst1, lst2):
    for item in lst2:
        i = 0

        while i < len(lst1):
            if item == lst1[i]:
                lst1[i], lst1[-1] = lst1[-1], lst1[i]
                lst1.pop()
            else:
                i += 1


a = [1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 10]
b = [2, 3, 4, 5, 10]

remove_all(a, b)
print(a)


def retain_all(lst1, lst2):
    i = 0

    while i < len(lst1):
        item1 = lst1[i]

        for item2 in lst2:
            if item1 == item2:
                i += 1
                break
        else:
            lst1[i], lst1[-1] = lst1[-1], lst1[i]
            lst1.pop()


a = [1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 1]
b = [2, 3, 4, 5, 10]

retain_all(a, b)
print(a)