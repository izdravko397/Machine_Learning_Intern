# def splice(arr: list, start, count=None, /, *args):
#     arr_len = len(arr)

#     if start < 0 or start > arr_len - 1:
#         raise Exception('Invalid start index')
    
#     count = arr_len - start if count == None or count > arr_len - start else count
#     res = []
#     arr_inx = start

#     while count > 0:
#         exchange_index = arr_inx + count if arr_inx + count < arr_len else arr_len - 1

#         if arr_inx < arr_len - 1:
#             arr[arr_inx], arr[exchange_index] = arr[exchange_index], arr[arr_inx]

#         while exchange_index < arr_len - 1:
#             arr[exchange_index], arr[exchange_index + 1] = arr[exchange_index + 1], arr[exchange_index]
#             exchange_index += 1
            
#         res.append(arr.pop())
#         arr_len -= 1
#         count -= 1
#         arr_inx += 1

#     if args:
#         # for i, num in enumerate(args, start=start):
#         #     arr.insert(i, num)

#         second_half = reversed([arr.pop() for _ in range(arr_len - start)])
        
#         arr.extend(args)
#         arr.extend(second_half)

#     return res

def splice_2(arr: list, start, count=None, /, *args):
    arr_len = len(arr)

    if start < 0 or start > arr_len - 1:
        raise Exception('Invalid start index')
    
    count = arr_len - start if count == None or count > arr_len - start else count
    res = arr[start : start + count]
    args_len = len(args)

    if args_len == count:
        for i in range(count):
            arr[start + i] = args[i]

    elif args_len < count:
        for i in range(args_len):
            arr[start + i] = args[i]

        x = 0
        for i in range(start + count, arr_len):
            arr[start + args_len + x] = arr[i]
            x += 1

        for _ in range(count - args_len):
            arr.pop()

    else:
        for i in range(count):
            arr[start + i] = args[i]

        end_elements_arr = reversed([arr.pop() for _ in range(arr_len - (start + count))])
        
        for i in range(count, args_len):
            arr.append(args[i])

        arr.extend(end_elements_arr)

    return res

a = [1, 2, 3, 4, 5]
b = splice_2(a, 1, 3, 7)
print(a)
print(b)
print()

# a = [1, 2, 3,  4]
# b = splice(a, 1, 1)
# print(a) # a: [1, 3, 4]
# print(b) # b: [2]
# print()

# a = [1, 2, 3,  4]
# b = splice(a, 1, 2, 5, 6) 
# print(a)# a: [1, 5, 6, 4]
# print(b)# b: [2, 3]
# print()

# a = [1, 2, 3,  4]  
# b = splice(a, 1) 
# print(a) # a: [1]
# print(b) # b: [2, 3, 4]
# print()

# a = [1, 2, 3,  4]
# b = splice(a, 2, 2, 7, 8, 9); 
# print(a) # a: [1, 2, 7, 8, 9]
# print(b) # b: [3, 4]