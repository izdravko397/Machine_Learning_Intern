from task1 import my_format

def strformat(format_str, *args, **kwargs):
    res = ''
    formatter = ''
    arg_inx = 0
    
    get_next_chars = False
    for char in format_str:
        if char == '{':
            if get_next_chars:
                raise ValueError("unmatched '{' in format spec")
            
            get_next_chars = True

        elif char == '}':
            if not get_next_chars:
                raise ValueError("Single '}' encountered in format string")
            
            var_form = formatter.split(':')
            var = var_form[0]
            form = var_form[1] if len(var_form) > 1 else ''

            if var.isdigit():
                value = args[int(var)]
            elif var.isalpha():
                value = kwargs[var]
            else:
                value = args[arg_inx]
                arg_inx += 1

            res += my_format(value, form)
            get_next_chars = False
            formatter = ''

        elif get_next_chars:
            formatter += char

        else:
            res += char

    if formatter:
        raise ValueError("unmatched '{' in format spec")

    return res

x = 123.456
print(strformat('Va}lue is {:0.2f}', x))           # 'Value is 123.46'
print(strformat('Value is {0:.4f}', x))         # 'Value is 123.4560'
print(strformat('Value is {val:*<10.2f}', val=x)) # 'Value is 123.46****'

name = 'IBM'
shares = 50
price = 490.1

print(strformat('{:>10s} {:10d} {:10.2f}', name, shares, price)) # '     IBM     50      490.10'

tag = 'p'
text = 'hello world'

print(strformat('<{0}>{1}</{0}>', tag, text))  # '<p>hello world</p>'
print(strformat('<{tag}>{text}</{tag}>', tag='p', text='hello world')) # '<p>hello world</p>'



print('Val}ue is {:0.2f}'.format(x))