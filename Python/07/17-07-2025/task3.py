def subarray_1(arr, start):
    if start >= len(arr) or start < 0:
        raise IndexError('Index out of range')
    
    # return arr[start:]

    res = []

    for inx in range(start, len(arr)):
        res.append(arr[inx])

    return res

def subarray_2(arr, start, end):
    if start > end:
        raise IndexError('Index out of range')
    
    if start >= len(arr) or start < 0:
        raise IndexError('Index out of range')
    
    if end >= len(arr) or end < 0:
        raise IndexError('Index out of range')
    
    # return arr[start:end + 1]

    res = []

    for inx in range(start, end + 1):
        res.append(arr[inx])

    return res


arr1 = [1, 2, 3, 3, 3]
arr2 = [1, 2, 3, 4, 3]

print(subarray_1(arr1, 2))
print(subarray_2(arr2, 2, 3))