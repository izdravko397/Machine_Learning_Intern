def common_chars(str1, str2):
    result = []
    unique_chars_str1 = ''
    
    for char in str1:
        if char not in unique_chars_str1:
            unique_chars_str1 += char

    for char in unique_chars_str1:
        if char in str2:
            result.append(char)

    return ''.join(result)

print(common_chars('hello', 'world'))

print(common_chars('abc', 'def'))

print(common_chars('abca', 'deaf'))