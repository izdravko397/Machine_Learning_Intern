def remove_first_element(arr):
    try:
        res = arr[0]
    except IndexError:
        return []
    
    for i in range(len(arr) - 1):
        arr[i] = arr[i + 1]

    del arr[-1]

    return res

arr = [1, 2, 3, 0, 12]
arr_2 = []

print(remove_first_element(arr))
print(arr)

print(remove_first_element(arr_2))
print(arr_2)