import re
from datetime import datetime

MONTHS = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

SHORT_MONTHS = [m[:3] for m in MONTHS]

format_codes = {
    "%Y": lambda d, v: d.update(year=int(v)), # 1900
    "%y": lambda d, v: d.update(year=1900 + int(v)), # 99
    "%m": lambda d, v: d.update(month=int(v)), # 01
    "%d": lambda d, v: d.update(day=int(v)), # 01
    "%H": lambda d, v: d.update(hour=int(v)), # 24
    "%I": lambda d, v: d.update(hour=int(v)), # 12
    "%M": lambda d, v: d.update(minute=int(v)), # 01
    "%S": lambda d, v: d.update(second=int(v)), # 00
    "%f": lambda d, v: d.update(microsecond=int(v)), # 00
    "%b": lambda d, v: d.update(month=SHORT_MONTHS.index(v)+1), # "Jan"
    "%B": lambda d, v: d.update(month=MONTHS.index(v)+1), # "January"
    "%Z": lambda d, v: d.update(tzinfo=v), # UTC
}

# def strptime(string, format_str):
#     str_sep = re.findall(r'[^\w]', string)
#     fmt_sep = re.findall(r'[^\w%]', format_str)

#     if str_sep != fmt_sep:
#         raise ValueError(f'time data {string} does not match format {format_str}')
    
#     str_values = re.findall(r'\w+', string)
#     fmt_codes = re.findall(r'%\w', format_str)

#     default_vals = {
#         "year": 1900,
#         "month": 1,
#         "day": 1,
#         "hour": 0,
#         "minute": 0,
#         "second": 0,
#         "microsecond": 0,
#         "tzinfo": None
#     }

#     for code, val in zip(fmt_codes, str_values):
#         format_codes[code](default_vals, val)

#     return datetime(**default_vals)

def strptime(string, format_str):
    str_values = re.finditer(r'\w+', string)
    fmt_codes = re.finditer(r'%\w', format_str)

    # for 



# print(datetime.strptime("2020-01-01 14:00", "%Y-%m-%d %H:%M"))
# print(datetime(2020, 1, 1, 14, 0))
print(strptime("2020-01-01 14:00", "%Y-%m-%d %H:%M"))
print(strptime("2020-01-Feb 14:00", "%Y-%m-%b %H:%M"))
print(strptime("99-01-March 14:00", "%y-%m-%B %H:%M"))
# print(strptime("99-01-March 14:00", "%y-%m-%m %H:%M"))

# print(type(strptime("2020-01-01 14:00", "%Y-%m-%d %H:%M")))



