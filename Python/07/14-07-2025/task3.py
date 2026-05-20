def string_encryption(str, key):
    result = ''
    uppercase_max_value = ord('Z')
    lowercase_max_value = ord('z')

    for char in str:
        if not char.isalpha():
            result += char
            continue

        new_ascii_value = ord(char) + key

        if new_ascii_value > uppercase_max_value and char.isupper():
            new_ascii_value = ord('A') + (new_ascii_value - uppercase_max_value - 1)

        elif new_ascii_value > lowercase_max_value and char.islower():
            new_ascii_value = ord('a') + (new_ascii_value - lowercase_max_value - 1)
        
        result += chr(new_ascii_value)

    return result


print(string_encryption("abc", 1))
print(string_encryption("abc", 3))
print(string_encryption("zab", 2))
print(string_encryption("Z AB", 2))
