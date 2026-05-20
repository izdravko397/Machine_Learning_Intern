# def remove_duplicates(arr: list):
#     res = []
#     # [res.append(num) for num in arr if num not in res]

#     for num in arr:
#         for i in res:
#             if num == i:
#                 break
#         else:
#             res.append(num)

#     return res

def remove_duplicates(arr: list):
    res = []
    unique = set()

    for num in arr:
        if num not in unique:
            res.append(num)
            unique.add(num)

    return res

arr = [1, 3, 8, 1, 16, 3, 4]
print(remove_duplicates(arr))