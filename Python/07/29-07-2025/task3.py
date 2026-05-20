# def repeated_pattern(str: str):
#     str_len = len(str)

#     if str_len % 2 == 0:
#         first_half = str[:str_len // 2]
#         second_holf = str[str_len // 2:]

#         if first_half == second_holf:
#             return True
    
#     for i in range(1, str_len // 2):
#         possible_pattern = str[:i]
#         possible_pattern_len = len(possible_pattern)

#         if str_len % possible_pattern_len != 0:
#             continue

#         if possible_pattern != str[i:i + possible_pattern_len]:
#             continue

#         test_str = possible_pattern * (str_len // possible_pattern_len)
#         if test_str == str:
#             return True
#     return False

def repeated_pattern(str: str):
    str_len = len(str)

    for i in range(1, (str_len // 2) + 1):
        pattern = str[:i]
        pattern_len = len(pattern)

        if str_len % pattern_len != 0:
            continue

        for j in range(i, str_len - pattern_len + 1, i):
            next_pattern = str[j:j + pattern_len]

            if pattern != next_pattern:
                break
        else:
            return True
    return False
            
print(repeated_pattern('abcdabcdabcd')) #True
print(repeated_pattern("abc")) #=> false
print(repeated_pattern("1212")) #=> true
print(repeated_pattern("alaala")) #=> true
print(repeated_pattern("alaal")) #=> false
print(repeated_pattern("zzzzz")) #=> true
print(repeated_pattern("acacac")) #=> true
print(repeated_pattern("acaca")) #=> false