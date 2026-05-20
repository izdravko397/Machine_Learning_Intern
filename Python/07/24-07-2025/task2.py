def reverse_list(arr: list):
    gen = (arr[inx] for inx in range(len(arr) - 1, -1, -1))

    for i in gen:
        print(i)

list1 = [1, 2, 3, 4, 5]
list2 = ['s', 't', 'r']

reverse_list(list1)
reverse_list(list2)