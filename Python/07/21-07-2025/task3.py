def common_chars(str1, str2):
    # return ''.join(s for s in set(str1) & set(str2))

    result = []
    unique_str1 = set(str1)

    for char in str2:
        if char in unique_str1:
            result.append(char)
            unique_str1.remove(char)

    return ''.join(result)

print(common_chars('abc', 'def'))
print(common_chars('abc', 'cde'))
print(common_chars('abc', 'dafc'))
print(common_chars('abca', 'deaf'))

# "abc", "def" => ""
# "abc", "cde" => "c"
# "abc", "dafc" => "ac"
# "abca", "deaf" => "a"