def same_numbers(arr1, arr2):
    compared_nums = set()

    arr1_i = 0
    arr2_i = 0

    while True:
        arr1_e = arr1[arr1_i] if len(arr1) > arr1_i else None
        arr2_e = arr2[arr2_i] if len(arr2) > arr2_i else None

        if arr1_e == arr2_e:
            if arr1_e is None:
                return True
            compared_nums.add(arr1_e)
            arr1_i += 1
            arr2_i += 1
            continue
            
        if arr1_e in compared_nums:
            arr1_i += 1
        elif arr2_e in compared_nums:
            arr2_i += 1
        else:
            return False

arr1 = [1, 1, 2, 3]
arr2 = [1, 2, 2, 2, 3]
print(same_numbers(arr1, arr2))

arr3 = [1, 2, 3, 3, 3]
arr4 = [2, 2, 2, 4]
print(same_numbers(arr3, arr4))