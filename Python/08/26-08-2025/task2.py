# да се имплементира функцията format_exc която получава като параметър Exception и работи 
# по същия начин като едноименната функция от този клас.
# това означава ако има вложени Exception-и да изпечата и техния traceback.
import traceback
import sys

file_lines = None

def format_exc(error: Exception):
    trace = ['Traceback (most recent call last):']
    exc_format = '  File "{fname}", line {line_num}, in {func_name}\n    {error_line}'
    tb = error.__traceback__
    global file_lines

    while tb:
        fname = tb.tb_frame.f_code.co_filename
        line_num = tb.tb_lineno
        func_name = tb.tb_frame.f_code.co_name

        if file_lines is None:
            with open(fname) as file:
                file_lines = file.readlines()
        error_line = file_lines[line_num - 1].strip()

        trace.append(exc_format.format(fname=fname, line_num=line_num, func_name=func_name, error_line=error_line))
        tb = tb.tb_next

    trace.append(f'{type(error).__name__}: {error}')
    return '\n'.join(trace)

# Traceback (most recent call last):
#   File "/home/zdravko/Machine_Learning_Intern/08/26-08-2025/task2.py", line 18, in <module>
#     b()
#     ~^^
#   File "/home/zdravko/Machine_Learning_Intern/08/26-08-2025/task2.py", line 15, in b
#     a()
#     ~^^
#   File "/home/zdravko/Machine_Learning_Intern/08/26-08-2025/task2.py", line 12, in a
#     raise TypeError("Oups!")
# TypeError: Oups!

def a():
    raise TypeError("Oups!")

def b():
    a()

try:
    b()
except Exception as err:
    # traceback_msg = traceback.format_exc()
    traceback_msg = format_exc(err)
    print(traceback_msg)
    