# def find(arr: list, n: int):
#     if not arr or arr[0] > n:
#         return [-1, -1]
    
#     res = [-1, -1]

#     for i in range(len(arr)):
#         if arr[i] != n:
#             if res[0] != -1:
#                 res[1] = i
#                 break
#             continue

#         res[0] = i

#     return res
import numpy as np

def find(arr: np.ndarray, n: int):
    if len(arr) == 0 or not (arr[0] <= n <= arr[-1]):
        return [-1, -1]
    
    left = 0
    right = len(arr) - 1
    first = -1
    while left <= right:
        mid = (right + left) // 2
        mid_num = arr[mid]

        if mid_num >= n:
            right = mid - 1
        else:
            left = mid + 1

        if mid_num == n:
            first = mid

    if first == -1:
        return [-1, -1]

    left = 0
    right = len(arr) - 1

    last = -1
    while left <= right:
        mid = (right + left) // 2
        mid_num = arr[mid]

        if mid_num <= n:
            left = mid + 1
        else:
            right = mid - 1

        if mid_num == n:
            last = mid

    return [first, last]

        


# a = [1, 2, 3, 5, 8]
# r = find(a, 3) 
# print(r)

# # [2, 4]
# a = [1, 1, 2, 3, 4, 5, 8]
# r = find(a, 3) 
# print(r)

# # [3, 3]
# a = [1, 1, 2, 3, 4, 5, 8]
# r = find(a, 6) 
# print(r)

# [-1, -1]