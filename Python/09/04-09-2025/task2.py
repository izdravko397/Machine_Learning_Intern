import re

def convert_date_format_eu(date):
    pattern = r'(\d{2})/(\d{2})/(\d{4})'
    match = re.search(pattern, date)

    day = match.group(2)
    month = match.group(1)
    year = match.group(3)
    return f'{day}.{month}.{year}'

ads = '05/01/2024'

eds = convert_date_format_eu(ads)
print(eds)
# 01.05.2024