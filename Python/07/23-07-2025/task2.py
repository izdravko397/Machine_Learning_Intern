def remove_element(str: str, inx: int):
    if inx >= len(str):
        raise IndexError
    
    # return str[:inx] + str[inx + 1:]

    res = []

    for i in range(len(str)):
        if i == inx:
            continue
        res.append(str[i])

    return ''.join(res)
    

print(remove_element("alabala", 2))
# "alabala", 2 => "albala"