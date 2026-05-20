def hexadecimal_num(num):
    # return hex(num)[2:]
    
    dec_to_hex = {
        10 : 'A',
        11 : 'B',
        12 : 'C',
        13 : 'D',
        14 : 'E',
        15 : 'F'
    }
    res = []

    while num > 0:
        num, remainder = divmod(num, 16)

        if remainder > 9:
            res.insert(0, dec_to_hex[remainder])
        else:
            res.insert(0, str(remainder))

    return ''.join(res)

print(hexadecimal_num(5))
print(hexadecimal_num(10))
print(hexadecimal_num(16))
print(hexadecimal_num(255))
print(hexadecimal_num(4294967295))



# 5 => 5
# 10 => a
# 16 => 10
# 255 => ff