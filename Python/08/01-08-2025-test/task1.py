def sort_element_frequency(arr: list):
    counter = {}  
    for num in arr:
        if num not in counter:
            counter[num] = 0
        counter[num] += 1

    res = []
    res_set = set()
    last_bigges_num = None
    last_bigges_num_frequency = float('inf')

    for _ in range(len(counter)):
        min_num = None
        min_num_frequency = float('-inf')

        for k, v in counter.items():
            if min_num_frequency <= v <= last_bigges_num_frequency and k not in res_set:
                min_num = k
                min_num_frequency = v

        last_bigges_num = min_num
        last_bigges_num_frequency = min_num_frequency

        index = len(res)
        for i in range(len(res)-1, -1, -1):
            n = res[i]
            if counter[n] == last_bigges_num_frequency and n > last_bigges_num:
                index = i

        # if index == None:
        #     index = len(res)

        res.insert(index, last_bigges_num)
        res_set.add(last_bigges_num)
    
    final_res = []
    for num in res:
        final_res.extend([num] * counter[num])

    [final_res.extend([num] * counter[num]) for num in res]

    return final_res
    # return [num * counter[num] for num in res]

print(sort_element_frequency([2,3,5,3,7,9,5,3,7]))
print(sort_element_frequency([2, 1, 2]))
print(sort_element_frequency([2, 1, 2, 1]))

# [2,3,5,3,7,9,5,3,7] => [3,3,3,5,5,7,7,2,9]
# [2, 1, 2] => [2, 2, 1]
# [2, 1, 2, 1] => [1, 1, 2, 2]