def biggest_element(arr: list, k: int):
    if k > len(arr):
        raise ValueError
    
    res = []
    last_max_num = float('inf')

    for _ in range(k):
        current_max_num = float('-inf')

        for num in arr:
            if current_max_num < num < last_max_num:
                current_max_num = num
        
        if current_max_num == float('-inf'):
            raise ValueError('Not enough unique elements')
        
        last_max_num = current_max_num
        res.append(last_max_num)

    return res

arr = [1, 8, 8, 8, 0, 6, 3]
# arr = [1, 8, 8, 8, 8, 8, 8]

print(biggest_element(arr, 3))
# print(biggest_element(arr, 6))