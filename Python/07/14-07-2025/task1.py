def revers_list(arr):
    # arr = arr[::-1]

    for i in range(len(arr)):
        arr.insert(i, arr.pop(-1))

    # return arr
 

arr = [1, 2, 3, 4, 5]

revers_list(arr)

print(arr)