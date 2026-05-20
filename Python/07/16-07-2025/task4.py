def insert_element(arr: list, new_element):
    arr.append(None)

    for inx in range(len(arr) - 1, 0, -1):
        arr[inx] = arr[inx - 1]

    arr[0] = new_element

arr = [1, 2, 3, 0, 12]
insert_element(arr, 8)
print(arr)