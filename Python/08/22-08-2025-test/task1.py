def split(str: str, separator: str, maxsplit=-1):
    sep_len = len(separator)
    start = 0
    i = 0

    while i < len(str):
        if str[i:i + sep_len] == separator and maxsplit != 0:
            yield str[start:i]
            start = i + sep_len
            i = i + sep_len
            maxsplit -= 1
            continue
        i += 1
    yield str[start:]

d = "12-05-2005"
for date_part in split(d, "-", 1):
    print(date_part)

# 12
# 05
# 2005

# print(d.split('-', 0))