# def remove_previous_char(str: str):
#     SEARCH_CHAR = '#'

#     for char in str:
#         if char == SEARCH_CHAR:
#             char_inx = str.index(char)
#             previous_char_inx = char_inx - 1 if char_inx - 1 >= 0 else 0
#             str = str.replace(str[previous_char_inx: char_inx + 1], '', 1)

#     return str

def remove_previous_char(str: str):
    SEARCH_CHAR = '#'
    res = []

    for i in range(len(str)):
        if str[i] == SEARCH_CHAR:
            if res:
                res.pop()
        else:
            res.append(str[i])

    return ''.join(res)

print(remove_previous_char("abc#"))
print(remove_previous_char("a#bc"))
print(remove_previous_char("abc##"))
print(remove_previous_char("a##bc"))
print(remove_previous_char("ab##b#b#"))

# "abc#" => "ab"
# "a#bc" => "bc"
# "abc##" => "a"
# "a##bc" => "bc"