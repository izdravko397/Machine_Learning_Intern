from datetime import datetime, date

WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

def format_date(dt, format: str):
    if not isinstance(dt, (datetime, date)):
        raise TypeError('invalid date object')

    year = dt.year
    month = dt.month
    str_month = str(month) if month > 9 else f'0{month}'
    month_name = MONTHS[month - 1]

    day = dt.day
    str_day = str(day) if day > 9 else f'0{day}'
    weekday = dt.weekday()
    weekday_name = WEEK[weekday]

    format_time_codes = {}
    if isinstance(dt, datetime):
        hour = dt.hour
        h_24_format = str(hour) if hour > 9 else f'0{hour}'
        h_12 = hour % 12
        h_12_format = '12' if h_12 == 0 else str(h_12) if h_12 < 10 else f'0{h_12}'

        minute = dt.minute
        str_minute = minute if minute > 9 else f'0{minute}'
        second = dt.second
        str_second = second if second > 9 else f'0{second}'
        microsec = dt.microsecond

        format_time_codes = {
            "H": h_24_format,
            "I": h_12_format,
            "p": "AM" if hour < 12 else "PM",
            "M": str_minute,
            "S": str_second,
            "f": str(microsec),
            "z": dt.tzname() or '',
            "c": f"{weekday_name[:3]} {month_name[:3]} {h_24_format}:{str_minute}:{str_second} {year}",
            "X": f"{h_24_format}:{str_minute}:{str_second}"
        }

    format_date_codes = {
        "a": weekday_name[:3],
        "A": weekday_name,
        "w": str((weekday + 1) % 7),
        "d": str_day,
        "b": month_name[:3],
        "B": month_name, 
        "m": str_month,
        "y": str(year)[-2:],
        "Y": str(year),
        "x": f"{str_month}/{str_day}/{str(year)[-2:]}",  
    }

    res = ''
    get_next_char = False

    for char in format:
        if get_next_char:
            res += format_date_codes.get(char) or format_time_codes.get(char, '')
            get_next_char = False
        elif char == '%':
            get_next_char = True
        else:
            res += char

    return res

from datetime import timezone, timedelta
v = date(2022, 3, 2)
z = datetime(2012, 9, 23, 21, 37, 4, 177393) # tzinfo=timezone(timedelta(hours=3))
s = datetime.strftime(z, '%A-%w %B %d, %Y   %H   %X')
print(s)# Sunday September 23, 2012

print(format_date(z, '%A-%w %B %d, %Y   %H   %X: %z'))
print(format_date(v, '%A-%w %B %d, %Y   %H   %X: %z'))
