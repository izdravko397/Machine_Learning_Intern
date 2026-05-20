def same_numbers(arr1, arr2):
    # return True if set(arr1) == set(arr2) else False

    unique_num_1 = []
    [unique_num_1.append(num) for num in arr1 if num not in unique_num_1]

    unique_num_2 = []
    [unique_num_2.append(num) for num in arr2 if num not in unique_num_2]

    if unique_num_1 == unique_num_2:
        return True
    return False

arr1 = [1, 1, 2, 3]
arr2 = [1, 2, 2, 2, 3]

print(same_numbers(arr1, arr2))

arr3 = [1, 2, 3, 3, 3]
arr4 = [2, 2, 2, 4]

print(same_numbers(arr3, arr4))