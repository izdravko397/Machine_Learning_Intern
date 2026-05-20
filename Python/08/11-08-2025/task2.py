def compare(a, b):
    # return a == b

    if type(a) != type(b):
        return False

    if len(a) != len(b):
        return False
    
    for element_a, element_b in zip(a, b):
        if isinstance(element_a, list):
            if not isinstance(element_b, list):
                return False
            
            if not compare(element_a, element_b):
                return False
            continue
                
        elif isinstance(element_a, tuple) and not isinstance(a, set):
            if not isinstance(element_b, tuple):
                return False

            if not compare(element_a, element_b):
                return False
            continue
                
        elif isinstance(element_a, set):
            if not isinstance(element_b, set):
                return False

            if not compare(element_a, element_b):
                return False
            continue

        elif isinstance(element_a, dict):
            if not isinstance(element_b, dict):
                return False

            if not compare(element_a, element_b):
                return False
            continue
        
        if isinstance(a, (list, tuple)):
            if element_a != element_b:
                return False
            
        elif isinstance(a, set):
            if element_a not in b or element_b not in a:
                return False
            
        elif isinstance(a, dict):
            if element_a != element_b:
                return False
            
            if isinstance(a[element_a], (list, tuple, set, dict)):
                if not compare(a[element_a], b[element_b]):
                    return False
                continue

            if a[element_a] != b[element_b]:
                return False

    return True


a = {(1, 2), (3, 4)}
b = {(3, 4), (1, 2)}
# b = {(1, 2), (3, 4)}
print(compare(a, b))

a = [[1, 2], 3, 4, 5]
b = [[1, 2], 3, 4, 5]
print(compare(a, b)) # True

a = [[3, 2], 3, 4, 5]
b = [[1, 2], 3, 4, 5]
print(compare(a, b)) # False

a = [{12, 5}, 3, 4, 5]
b = [12, 5, 3, 4, 5]
print(compare(a, b)) # False 

a = [{12, 5}, 3, 4, 5]
b = [[12, 5], 3, 4, 5]
print(compare(a, b)) # False 

a = [{1: [12, 14]}, 3, 4, 5]
b = [{1: [12, 14]}, 3, 4, 5]
print(compare(a, b)) # True

a = [{1: [12, 14]}, 3, 4, 5]
b = [{1: [12]}, 3, 4, 5]
print(compare(a, b)) # False

a = [{1: [12, 14]}, 3, 4, 5]
b = [{1, 12, 14}, 3, 4, 5]
print(compare(a, b)) # False

a = [{1: [12, 14]}, 3, 4, 5]
b = [{1, 12, 14}, 3, 4, 5]
print(compare(a, b))#False

a = [(1, 2), 3, 4, 5]
b = [1, 2, 3, 4, 5]
print(compare(a, b)) # False

a = [(1, 2), 3, 4, 5]
b = [[1, 2], 3, 4, 5]
print(compare(a, b)) # False