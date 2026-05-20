def last_index_of(arr, element, start_index=0):
    if start_index >= len(arr) or start_index < 0:
        raise ValueError('Index out of range')
    
    last_coincidence = ''
    counter = 0

    for inx in range(start_index, len(arr)):
        if arr[inx] == element:
            last_coincidence = counter
        counter += 1
            
    if last_coincidence != '':
        return last_coincidence
    
    return -1

# def last_index_of(arr, element, from_index=0):
#     if from_index >= len(arr) or from_index < 0:
#         raise ValueError('Index out of range')
    
#     last_index = ''

#     for _ in range(len(arr)):
#         if from_index == len(arr):
#             from_index = 0

#         if arr[from_index] is element:
#             last_index = from_index
        
#         from_index += 1

#     if last_index != '':
#         return last_index
    
#     return -1
    
arr = [1, 0, 3, 0, 12]
print(last_index_of(arr, 0))
print(last_index_of(arr, 0, 2))
print(last_index_of(arr, 4))

# [1, 0, 3, 0, 12], 0 => 3
# [1, 0, 3, 0, 12], 0, 2 => 1
# [1, 2, 3, 0, 12], 4 => -1