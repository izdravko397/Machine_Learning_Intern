def rank1(arr: list):
    if len(arr) != len(set(arr)):
        raise ValueError('list must be unique')
    
    sorted_arr = sorted(arr)
    return [sorted_arr.index(num) + 1 for num in arr]

print(rank1([4,2,7,1]))

def rank2(arr: list):
    if len(arr) != len(set(arr)):
        raise ValueError('list must be unique')
    
    sorted_arr = sorted(arr)
    def rank_pos(n):
        left = 0
        right = len(sorted_arr) - 1
        pos = -1
        while left <= right:
            mid = (right + left) // 2
            mid_num = sorted_arr[mid]

            if mid_num == n:
                pos = mid
                break

            if mid_num > n:
                right = mid - 1
            else:
                left = mid + 1

        return pos + 1
    
    return [rank_pos(num) for num in arr]

print(rank2([4,2,7,1]))

# [4,2,7,1] = > [3,2,4,1]
# sorted: [1, 2, 4, 7]
# Ranks: 
# 1 -> 1
# 2 -> 2
# 4 -> 3
# 7 -> 4