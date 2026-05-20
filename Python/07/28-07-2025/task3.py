# def merge_ranges(arr: list):
#     i = 0
#     while i < len(arr) - 1:
#         start = arr[i][0]
#         end = arr[i][1]

#         for j in range(i + 1, len(arr)):
#             next_start = arr[j][0]
#             next_end = arr[j][1]

#             if start <= next_end and end >= next_start:
#                 arr[j] = [min(start, next_start), max(end, next_end)]
#                 del arr[i]
#                 break
#         else:
#             i += 1

#     return arr

# def merge_ranges(arr: list):
#     sorted_arr = sorted(arr, key=lambda item: item[0])
#     i = 0

#     while i < len(sorted_arr) - 1:
#         start = sorted_arr[i][0]
#         end = sorted_arr[i][1]

#         next_start = sorted_arr[i + 1][0]
#         next_end = sorted_arr[i + 1][1]

#         if start <= next_end and end >= next_start:
#             sorted_arr[i + 1] = [min(start, next_start), max(end, next_end)]
#             del sorted_arr[i]
#         else:
#             i += 1

#     return sorted_arr
from range import Range

def merge_ranges(arr: list):
    sorted_arr = sorted(arr, key=lambda item: item[0])
    res = [Range(sorted_arr[0][0], sorted_arr[0][1])]
    res_i = 0

    for i in range(1, len(sorted_arr)):
        range_ob = Range(sorted_arr[i][0], sorted_arr[i][1])
        if not res[res_i].merge(range_ob):
            res.append(range_ob)
            res_i += 1

    return [[obj.start, obj.end] for obj in res]


print(merge_ranges([[1, 5], [2, 4]]))# => [[1, 5]]
print(merge_ranges([[1, 5], [6, 8]]))# => [[1, 5], [6, 8]]
print(merge_ranges([[1, 5], [5, 6]]))# => [[1, 6]]
print(merge_ranges([[1, 5], [4, 7], [6, 8]])) #=> [[1, 8]]

print(merge_ranges([[1, 5], [6, 8], [4, 7]])) #=> [[1, 8]]
print(merge_ranges([[6, 8], [1, 5], [4, 7]])) #=> [[1, 8]]
print(merge_ranges([[1, 3], [2, 4], [3, 5]]))
print(merge_ranges([[1, 3], [4, 5], [6, 8]]))
print(merge_ranges([[1, 3], [4, 6], [7, 10], [2, 8]]))
                
