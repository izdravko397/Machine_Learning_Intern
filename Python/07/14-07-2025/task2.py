def current_sum(arr):
    result = []

    for inx in range(len(arr)):
        if inx == 0:
            result.append(arr[inx])
        else:
            result.append(result[inx - 1] + arr[inx])

    return result

print(current_sum([1, 2, 3, 4, 5]))