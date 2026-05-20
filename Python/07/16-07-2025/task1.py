def element_index(arr, element, from_index=0):
    if from_index >= len(arr):
        raise ValueError('Index out of range')

    for inx in range(from_index, len(arr)):
        if arr[inx] is element:
            return inx
    else:
        return -1
    
# V2
def element_index_v2(arr, element, from_index):
    if from_index >= len(arr):
        raise ValueError('Index out of range')

    for _ in range(len(arr)):
        if from_index == len(arr):
            from_index = 0

        if arr[from_index] is element:
            return from_index
        
        from_index += 1
    else:
        return -1
    

arr = [1, 2, 4, 0, 12]
arr_2 = [1, 2, True, False, 20, 'str']

print(element_index_v2(arr, 3, 3))
print(element_index(arr, 1))
print(element_index(arr, 4))
print(element_index(arr_2, 'str'))
print(element_index(arr_2, True))