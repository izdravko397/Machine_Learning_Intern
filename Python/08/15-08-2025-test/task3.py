def unique_char(str: str):
    str_len = len(str)
    not_unique = set()

    for i in range(str_len):
        char = str[i]
        if char in not_unique:
            continue

        for j in range(i + 1, str_len):
            if char == str[j]:
                not_unique.add(char)
                break
        else:
            return i
    return -1

from collections import Counter

def unique_char_2(str: str):
    char_counts = Counter(str)

    for i in range(len(str)):
        if char_counts[str[i]] == 1:
            return i
    return -1

# "alabala" => 3
# "github" => 0
# "alabalab" => -1

print(unique_char("alabala"))
print(unique_char("github"))
print(unique_char("alabalab"))

print(unique_char_2("alabala"))
print(unique_char_2("github"))
print(unique_char_2("alabalab"))

