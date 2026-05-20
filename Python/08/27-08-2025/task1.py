format_map = {
    'str': {
        's': lambda x: x,
    },
    'int': {
        'd': lambda x: str(x),
        'b': lambda x: bin(x)[2:],
        'o': lambda x: oct(x)[2:],
        'x': lambda x: hex(x)[2:],
    },
    'float': {
        'f': lambda x: str(float(x)),
        'e': lambda x: f"{float(x):e}",
        'g': lambda x: f"{float(x):g}", 
        '%': lambda x: str(float(x) * 100),     
    }
}

def my_format(val, fromatter=None):
    if not fromatter:
        return str(val)
    
    val_type = type(val).__name__
    format_char = fromatter[-1]
    special_chars = ''

    if format_char.isalpha() or format_char == '%':
        val_type_form_chars = format_map[val_type]

        if format_char not in val_type_form_chars and (val_type == 'int' and format_char not in format_map['float']):
            raise TypeError(f'Invalid format for: {val_type}')
        
        format_func = val_type_form_chars[format_char]
        res = format_func(val)

        if format_char == 'e':
            split_e = res.split('e')
            special_chars = 'e' + split_e[1]
            res = split_e[0]
        elif format_char == '%':
            special_chars = '%'
            res = res[:-1]
    else:
        format_char = None
        res = str(val)

    align_symbols = {'<', '^', '>'}
    signs = {'+', '-'}

    fill = ' '
    align = '>'
    sign = None
    width = ''
    precision = ''

    get_width = True
    get_precision = False
    
    for i in range(len(fromatter) - 1 if format_char else len(fromatter)):
        char = fromatter[i]

        if get_width and char.isdigit():
            width += char

        elif char in align_symbols:
            align = char

        elif char in signs:
            sign = char

        elif char == '.':
            get_width = False
            get_precision = True

        elif get_precision:
            precision += char

        elif i == 0:
            fill = char

    if len(width) > 1 and width[0] == '0':
        fill = '0'
        width = int(width[1:])
    else:
        width = int(width) if width else 0

    # print(f'fill: {fill}, align: {align}, sing: {sign}, width: {width}, precision: {precision}, format: {format}')
    if precision:
        precision = int(precision)
        num_arr = res.split('.')
        if len(num_arr) == 1:
            num_arr.append('0' * precision)
        else:
            decimals = num_arr[1]
            num_arr_len = len(decimals)

            if precision < num_arr_len:
                num_arr[1] = decimals[:precision]
            else:
                num_arr[1] += '0' * (precision - num_arr_len)

        res = '.'.join(num_arr)

    if sign == '+' and res[0] != '-':
        res = '+' + res

    if special_chars:
        res += special_chars

    fill_count = width - len(res)

    if fill_count < 1:
        return res

    if res[0] in signs and fill == '0':
        sign = res[0]
        res = res[1:]
    else:
        sign = ''
    
    if align == '>':
        res = sign + (fill * fill_count) + res
    elif align == '<':
        res += fill * fill_count
    else:   # ^
        left_fill = fill_count // 2
        right_fill = fill_count - left_fill
        res = sign + (fill * left_fill) + res + (fill * right_fill)

    return res

# print(my_format(-98765.4, ".2e"))
# print(my_format(123, '0<10d'))
# print(my_format(3.14159, '.10f'))
# print(my_format(12345.6789, '.2f'))
# print(my_format(3.14159, '+010.3f'))

# x = 123.456
# print(my_format(x, '0.2f'))     # '123.46'
# print(my_format(x, '10.4f'))    # '    123.4560'
# print(my_format(x, '*<10.2f'))  # '123.46****'

# name = 'Elwood'
# print(my_format(name, '<10'))   # 'Elwood    '
# print(my_format(name, '>10'))   # '    Elwood'
# print(my_format(name, '^10'))   # '  Elwood  '
# print(my_format(name, '*^10'))  # '**Elwood**'

# x = 42
# print(my_format(x, '10d'))  # '        42'
# print(my_format(x, '10x'))  # '        2a'
# print('-----' + my_format(x, '10b'))  # '    101010'
# print(my_format(x, '010b')) # '0000101010'

# y = 3.1415926
# print(my_format(y, '10.2f'))   # '      3.14'
# print(my_format(y, '.2f'))     # '3.14'
# print(my_format(y, '10.2e'))   # '  3.14e+00'
# print(my_format(y, '+10.2f'))  # '     +3.14'
# print(my_format(y, '+010.2f')) # '+000003.14'
# print(my_format(y, '+10.2%'))  # '  +314.16%'

# from dataclasses import dataclass
# @dataclass
# class A:
#     name: str
#     age: int

#     def __str__(self):
#         return 'a'

# person = A('ivan', 12)
# # x = 123.456
# print("'" + format(person, "s") + "'")
# # [[fill]align][sign][#][0][width][grouping_option][.precision][type]