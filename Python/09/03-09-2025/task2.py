import re

def fdecompose(literal):
    # pattern = r'([-]?[0-9]*)?\.?([0-9]*)?e?([-]?[0-9]*)?'
    pattern = r'^([-+]?\d*\.\d+|[-+]?\d+[^\.]?)([eE][+-]?\d*)?'
    match = re.search(pattern, literal)
    
    if not match:
        raise ValueError(f'invalid float literal: {literal}')

    split_num = match.group(1).split('.')
    num = split_num[0] or '0'
    decimals = split_num[1] if len(split_num) > 1 else '0'
    exponent = int(match.group(2)[1:]) if match.group(2) else 0

    negative = ''
    if num.startswith('-'):
        num = num[1:]
        negative = '-'

    if num not in {'1', '2', '3', '4', '5', '6', '7', '8', '9'}:
        if len(num) > 1:
            d = num[1:]
            num, decimals = num[:1], d + decimals
            exponent += len(d) 

        else:
            for i in range(len(decimals)):
                n = decimals[i]
                exponent -= 1
                if n != '0':
                    num = n
                    decimals = decimals[i + 1:]
                    break

    decimals_len = len(decimals)
    if decimals_len < 2:
        decimals += '0' * (2 - decimals_len)
    else:
        decimals = decimals[:2]
    
    return (negative + num + '.' + decimals, str(exponent))

values = [".1", "1.5", "3", "-12", "25", "1000", "12.3e-2"]

for v in values:
    em = fdecompose(v)
    print(em)
# (1.0, -1)
# (1.5, 0)
# (3.0, 0)
# (-1.2, 1)
# (2.5, 1)
# (1.0, 3)
# (1.23, -1)