def second_largest_digit_in_string(str: str):
    nums = sorted(list(set(int(char) for char in str if char.isdigit())))
    return nums[1] if len(nums) > 1 else -1
    
def second_largest_digit_in_string(str: str):
    min_num = None
    second_largest_num = float('inf')

    for char in str:
        if char.isdigit():
            num = int(char)

            if min_num is None:
                min_num = num
            elif num < min_num:
                min_num, second_largest_num = num, min_num
            elif min_num < num < second_largest_num:
                second_largest_num = num

    return second_largest_num if second_largest_num != float('inf') else -1

print(second_largest_digit_in_string("3dm1o24"))
print(second_largest_digit_in_string("alabala"))
print(second_largest_digit_in_string("1git1hub"))

# "3dm1o24" => 2
# "alabala" => -1
# "1git1hub" => -1