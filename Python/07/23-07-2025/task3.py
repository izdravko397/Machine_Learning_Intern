def remove_char(str: str, *chars):
    res = []
    chars = set(chars)

    for char in str:
        if char not in chars:
            res.append(char)

    return ''.join(res)

print(remove_char("alabala", 'a', 'b'))


# "alabala", 'a', 'b' => "ll"