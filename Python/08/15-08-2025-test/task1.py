def longest_num_sequence(arr):
    if not arr:
        return []
    
    longest_sequence = [arr[0]]
    last_longest_sequence = []

    for i in range(1, len(arr)):
        num = arr[i]

        if num - 1 == longest_sequence[-1]:
            longest_sequence.append(num)
            continue

        if len(longest_sequence) > len(last_longest_sequence):
            last_longest_sequence = longest_sequence.copy()

        longest_sequence = [num]

    return longest_sequence if len(longest_sequence) > len(last_longest_sequence) else last_longest_sequence

def longest_num_sequence_2(arr):
    if not arr:
        return []
    
    longest_sequence_inx = [0, 0]
    last_longest_sequence_inx = [0, 0]

    for i in range(1, len(arr)):
        if arr[i] - 1 == arr[longest_sequence_inx[1]]:
            longest_sequence_inx[1] = i
            continue

        if longest_sequence_inx[1] - longest_sequence_inx[0] > last_longest_sequence_inx[1] - last_longest_sequence_inx[0]:
            last_longest_sequence_inx = longest_sequence_inx.copy()

        longest_sequence_inx = [i, i]

    return arr[longest_sequence_inx[0]:longest_sequence_inx[1] + 1] if longest_sequence_inx[1] - longest_sequence_inx[0] > last_longest_sequence_inx[1] - last_longest_sequence_inx[0] else arr[last_longest_sequence_inx[0]:last_longest_sequence_inx[1] + 1]

a = [0, 5, 1, 2, 3, 4, 5, 2, 8, 9, 10]
print(longest_num_sequence(a))
# 0, 5, 1, 2, 3, 4, 5, 2, 8, 9, 10 -> 1, 2, 3, 4, 5

a = [0, 5, 1, 2, 3, 4, 5, 2, 8, 9, 10, 11, 12, 13]
print(longest_num_sequence(a))

a = [0, 5, 1, 2, 3, 4, 5, 2, 8, 9, 10]
print(longest_num_sequence_2(a))
# 0, 5, 1, 2, 3, 4, 5, 2, 8, 9, 10 -> 1, 2, 3, 4, 5

a = [0, 5, 1, 2, 3, 4, 5, 2, 8, 9, 10, 11, 12, 13]
print(longest_num_sequence_2(a))