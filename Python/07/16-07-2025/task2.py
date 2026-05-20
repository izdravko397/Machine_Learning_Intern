def element_search(arr, element, from_index=0):
    if from_index >= len(arr):
        raise ValueError('Index out of range')

    for inx in range(from_index, len(arr)):
        if arr[inx] is element:
            return True
    else:
        return False
    

# V2
def element_search_v2(arr, element, from_index):
    if from_index >= len(arr):
        raise ValueError('Index out of range')

    for _ in range(len(arr)):
        if from_index == len(arr):
            from_index = 0

        if arr[from_index] is element:
            return True
        
        from_index += 1
    else:
        return False
    

arr = [1, 2, 3, 0, 12]
arr_2 = [1, 2, False, 20, 'str']

print(element_search_v2(arr, 3, 3))
print(element_search(arr, 1))
print(element_search(arr, 4))
print(element_search(arr_2, 'str'))
print(element_search(arr_2, True))
print(element_search(arr, False))