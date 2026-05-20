def set_bit(num, inx):
    return num | 1 << inx
    
    # binary = bin(num)[2:]
    # len_binary = len(binary)

    # if len_binary < inx + 1:
    #     binary = ('0' * (inx + 1 - len_binary)) + binary

    # pos = len(binary) - (inx + 1)
    # part1 = binary[:pos]
    # part2 = binary[pos + 1:]

    # return int(part1 + '1' + part2, 2) 

# print(set_bit(1, 1)); # 3
# print(set_bit(0, 2)); # 4

def clear_bit(num, inx):
    return num & ~(1 << inx)

    # binary = bin(num)[2:]
    # len_binary = len(binary)

    # if inx >= len_binary:
    #     raise IndexError('index is outside the binary length')

    # pos = len_binary - (inx + 1)
    # part1 = binary[:pos]
    # part2 = binary[pos + 1:]

    # return int(part1 + '0' + part2, 2)

# print(clear_bit(3, 0)); # 2
# print(clear_bit(7, 1)) # 5

def toggle_bit(num, inx):
    return num ^ (1 << inx)
 
    # binary = bin(num)[2:]
    # len_binary = len(binary)

    # if len_binary < inx + 1:
    #     binary = ('0' * (inx + 1 - len_binary)) + binary

    # pos = len(binary) - (inx + 1)
    # pos_val = 0 if int(binary[pos]) else 1
    # part1 = binary[:pos]
    # part2 = binary[pos + 1:]

    # return int(part1 + str(pos_val) + part2, 2)

print(toggle_bit(0, 1)); # 2
print(toggle_bit(5, 0)); # 4

def get_bit(num, inx):
    return (num >> inx) & 1 == 1

    # binary = bin(num)[2:]
    # len_binary = len(binary)

    # if inx >= len_binary:
    #     raise IndexError('index is outside the binary length')
    
    # pos = len_binary - (inx + 1)
    # return True if int(binary[pos]) else False

# print(get_bit(5, 0)); # True
# print(get_bit(4, 0)); # False