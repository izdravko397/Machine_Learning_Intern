# да се напише функция rotate, която получава като параметри масив и число. 
# функцията да завърти надясно елементите в масива толкова пъти колкото е числото. 
# при завъртане първия елемент отива на втора позиция, втория на трета и т.н., 
# като последния отива на първа позиция.

def rotate(arr: list, n: int):
    rotated_num_counter = 0
    num_for_rotate = arr[0]
    num_for_rotate_inx = 0
    rotated_pos = set()
    len_arr = len(arr)

    while rotated_num_counter < len_arr:
        new_inx = num_for_rotate_inx + n if num_for_rotate_inx + n < len_arr else (num_for_rotate_inx + n) % len_arr
        rotated_pos.add(num_for_rotate_inx)
        
        temp = arr[new_inx]
        arr[new_inx] = num_for_rotate
        
        if new_inx in rotated_pos:
            try:
                num_for_rotate = arr[new_inx + 1]
            except IndexError:
                pass
            num_for_rotate_inx = new_inx + 1
        else:
            num_for_rotate = temp
            num_for_rotate_inx = new_inx

        rotated_num_counter += 1

# def rotate(arr: list, n: int):
#     while n > 0:
#         temp = arr[0]

#         for i in range(len(arr)):
#             inx = i + 1 if i + 1 < len(arr) else 0
#             t = arr[inx]
#             arr[inx] = temp
#             temp = t
#         n -= 1

a = [1, 2, 3]
rotate(a, 1)
print(a)

a = [1, 2, 3, 4]
rotate(a, 6)
print(a)

a = [1, 2, 3, 4, 5, 6, 7, 8]
rotate(a, 3)
print(a)

# [1, 2, 3], 1 => [3, 1, 2]
# [1, 2, 3, 4], 2 => [3, 4, 1, 2]
# [1, 2, 3, 4, 5, 6, 7, 8], 3 => [6, 7, 8, 1, 2, 3, 4, 5]