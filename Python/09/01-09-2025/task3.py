def get_float_parts(x):
    positive_exponent = True if x >= 1 else False
    positive_mantissa = True if x > 0 else False
    x = abs(x)
    e = 0

    while not (1 <= x < 10):
        if x > 1:
            x /= 10
        else:
            x *= 10
        e += 1
        # print(x)
    e = e if positive_exponent else e * -1
    x = x if positive_mantissa else x * -1

    return (f'{x:.2f}', f'{e:+03}')


x = 0.00031345
mantissa, exponent = get_float_parts(x)
print(mantissa, exponent)
# 3.13 -04

x = 3145
mantissa, exponent = get_float_parts(x)
print(mantissa, exponent)

x = 0.00031345
mantissa, exponent = get_float_parts(x)
print(mantissa, exponent)

x = -0.00031345
mantissa, exponent = get_float_parts(x)
print(mantissa, exponent)

print('.3'.split('.'))