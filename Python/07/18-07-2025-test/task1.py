# def biggest_element(arr: list, k: int):
#     if k > len(arr):
#         raise IndexError
    
#     unique_elements = []  # set(arr)

#     for num in arr:
#         for i in unique_elements:  # if num in res:
#             if num == i:
#                 break
#         else:
#             unique_elements.append(num)

#     res = sorted(unique_elements, reverse=True)[:k]
    
#     if len(res) < k:
#         raise ValueError('Not enough unique elements')
#     return res


def biggest_element(arr: list, k: int):
    res = []
    arr_unique = set(arr)

    for _ in range(k):
        max_num = max(arr_unique)
        res.append(max_num)       
        arr_unique.remove(max_num)

    return res



arr = [1, 8, 8, 8, 0, 6, 3]

print(biggest_element(arr, 3))
# print(biggest_element(arr, 6))