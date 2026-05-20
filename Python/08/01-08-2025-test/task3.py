def remove_repeats(arr: list):
    arr_set = set()
    inx = 0

    while inx < len(arr):
        num = arr[inx]
        if num not in arr_set:
            arr_set.add(num)
            inx += 1
            continue
        
        arr[inx], arr[-1] = arr[-1], arr[inx]
        arr.pop()


def remove_repeats_2(arr: list):
    inx = 0

    while inx < len(arr):
        num = arr[inx]
        i = inx + 1
        uniqe_num = True

        while i < len(arr):
            if num == arr[i]:
                arr[inx], arr[-1] = arr[-1], arr[inx]
                arr.pop()
                uniqe_num = False
            i += 1

        if uniqe_num:
            inx += 1

    
a = [3, 12, 5, 12, 8, 5]
remove_repeats(a)
print(a)
a = [3, 12, 5, 12, 8, 5]
remove_repeats_2(a)
print(a)
print()

a = [1, 2, 3, 2, 1]
remove_repeats(a)
print(a)
a = [1, 2, 3, 2, 1]
remove_repeats_2(a)
print(a)
print()

a = [1, 2, 3, 2, 4, 3]
remove_repeats(a)
print(a)
a = [1, 2, 3, 2, 4, 3]
remove_repeats_2(a)
print(a)
print()

a = [1, 2, 3, 4, 5]
remove_repeats(a)
print(a)
a = [1, 2, 3, 4, 5, 4, 3, 2, 1]
remove_repeats_2(a)
print(a)
# 3, 12, 5, 12, 8, 5 => 3, 12, 5, 8
# 1, 2, 3, 2, 1 => 1, 2, 3