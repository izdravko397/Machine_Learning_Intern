def comparison(list1, list2):
    set_list = set(list2)

    for num in list1:
        up_value =  num + 1
        down_value = num - 1

        if up_value not in set_list and num not in set_list and down_value not in set_list:
            return False
    return True

def approximate_equality(list1: list, list2: list):
    if len(list1) != len(list2):
        return False
    
    return comparison(list1, list2) and comparison(list2, list1)


# print(approximate_equality([4, 6, 2], [5, 3, 1]))    
# print(approximate_equality([1, 5, 3], [2, 4, 6]))    
# print(approximate_equality([1, 3, 5], [2, 4, 6]))
# print(approximate_equality([1, 3, 5], [2, 4]))
# print(approximate_equality([1, 3, 5], [2]))
# print(approximate_equality([1, 3, 5], [2, 4, 7]))
# print(approximate_equality([1, 3, 2], [2]))

list1 = [1, 1, 1]
list2 = [2, 2, 3]
print(approximate_equality(list1, list2))

# [1, 3, 5], [2, 4, 6] => true
# [1, 3, 5], [2, 4] => true
# [1, 3, 5], [2] => false
# [1, 3, 5], [2, 4, 7] => false