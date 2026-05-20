def remove_duplicates_array(arr):
    res = []
    [res.append(num) for num in arr if not res or num != res[-1]]

    # for num in arr:
    #     if not res or num != res[-1]:
    #         res.append(num)

    return res

arr = [1, 2, 3, 3, 4, 4, 5]
print(remove_duplicates_array(arr))
# [1, 2, 3, 3, 4, 4, 5] -> [1, 2, 3, 4, 5]