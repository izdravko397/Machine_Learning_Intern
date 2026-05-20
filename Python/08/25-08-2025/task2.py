# да се извършат следните изчисления с дати
from datetime import datetime, timezone, timedelta, date

# ----да се покаже текущото време в париж----
paris_offset = timedelta(hours=2)
paris_tz = timezone(offset=paris_offset)
# print(datetime.now(tz=paris_tz))

# ----от зададена дата и час местно време в софия да се покаже местното време и час в берлин----
berlin_offset = timedelta(hours=2)
berlin_tz = timezone(offset=berlin_offset)

sofia_offset = timedelta(hours=3)
sofia_tz = timezone(offset=sofia_offset)

sofia_date = datetime(2025, 8, 24, 20, 30, 12, 12223, tzinfo=sofia_tz)
# print(sofia_date)

berlin = sofia_date.astimezone(tz=berlin_tz)
# print(berlin)

# ----при зададена дата да покаже следващата----
time_delta = timedelta(days=1)
date1 = date(year=2025, month=8, day=24)

date2 = date1 + time_delta
# print(date2)

# ----при зададена дата да даде датата на понеделник от същата седмица----
date1 = date(year=2025, month=8, day=24)
time_delta = timedelta(days=date1.weekday())

date2 = date1 - time_delta
# print(date2)

# ----при зададена дата да върне датата на следващия петък----
date1 = date(year=2025, month=8, day=23)
weekday = date1.weekday()
time_delta = timedelta(days=(4 - weekday) % 7)

date2 = date1 + time_delta
# print(date2)

# ----при дадена дата да въне последния петък от месеца----
date1 = date(year=2025, month=8, day=20)

next_month = date1.month + 1
year = date1.year

if next_month > 12:
    next_month = 1
    year = date1.year + 1

last_day_of_month = date(year, next_month, 1)
day_1 = timedelta(days=1)
last_day_of_month -= day_1

weekday = last_day_of_month.weekday()
last_friday = timedelta(days=(weekday - 4) % 7)

# print(last_day_of_month - last_friday)

# ----date_range(year, month) функция която връща итератор за всички дати от съответния месец като стрингове----
def date_range(year, month):
    dt = date(year, month, 1)
    time_delta = timedelta(days=1)

    while dt.month == month:
        yield dt.strftime('%Y-%m-%d')
        dt += time_delta

# for i in date_range(2025, 8):
#     print(i)