def compare(arr1: list, arr2: list):
    dict_list = {num: False for num in arr1}
    
    for e in arr2:
        if e not in dict_list:
            return False
        dict_list[e] = True

    for value in dict_list.values():
        if value == False:
            return False
        
    return True

list1 = [1, 2, 6, 4, 4, 3, 4, 5]
list2 = [3, 3, 3, 1, 4, 5, 5, 4, 2, 2, 2, 2]

print(compare(list1, list2))