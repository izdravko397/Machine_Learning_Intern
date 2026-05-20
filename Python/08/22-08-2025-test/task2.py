def is_num(str: str):
    val_0 = ord('0')
    val_9 = ord('9')
    dot_val = ord('.')
    e_val = ord('e')
    minus_val = ord('-')
    plus_val = ord('+')

    dot = False
    e = False

    for i, char in enumerate(str):
        ascii_val = ord(char)

        if not (val_0 <= ascii_val <= val_9):
            if ascii_val == dot_val and not dot and not e and (0 < i < len(str) - 1):
                dot = True
            elif ascii_val == e_val and not e and (0 < i < len(str) - 1):
                e = True
            elif (ascii_val == minus_val or ascii_val == plus_val) and (i == 0 or str[i - 1] == 'e'):
                continue
            else:
                return False
            
    return True
    
print(is_num('0'))
print(is_num("0.1"))
print(is_num("abc"))
print(is_num("1 a"))
print(is_num("2e10"))
print(is_num("1.234e2"))
print(is_num("1.2e2.5"))
print(is_num("."))
print(is_num("e"))
print(is_num("+1.234e+2"))

# "0" => true 
# "0.1" => true 
# "abc" => false 
# "1 a" => false 
# "2e10" => true
# "1.234e2" => true
# "1.2e2.5" => false