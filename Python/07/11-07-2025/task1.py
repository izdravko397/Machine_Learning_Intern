def num_1_to_100():
    arr = []
    if_counter = 0

    for i in range(1, 16):
        line_res = ''
        if_counter += 3

        if i % 3 == 0:
            line_res += 'git'

        if i % 5 == 0:
            line_res += 'hub'
        
        arr.append(line_res if line_res else i)

    for i in range(1, 101):
        index = (i - 1) % 15
        if_counter += 1

        if isinstance(arr[index], int):
            print(arr[index] + (15 * ((i - 1) // 15)))
        else:
            print(arr[index])
        
    print(if_counter) # 145

# def num_1_to_100():
#     if_counter = 0

#     for i in range(1, 101):
#         line_res = ''
#         if_counter += 3

#         if i % 3 == 0:
#             line_res += 'git'

#         if i % 5 == 0:
#             line_res += 'hub'
        
#         print(line_res if line_res else i)
    
#     print(if_counter) # 300

num_1_to_100()