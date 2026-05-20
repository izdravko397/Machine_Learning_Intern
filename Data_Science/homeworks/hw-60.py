from index import DatetimeIndex, to_datetime, date_range, Hour, Minute, Day, MonthEnd, WeekOfMonth, YearEnd
from datetime import datetime
from series import Series
import numpy as np

# print(to_datetime('2005-11-30'))
# print(type(to_datetime('2005-11-30')))

# print(to_datetime(datetime(2005, 11, 30)))
# print(type(to_datetime(datetime(2005, 11, 30))))


datestrs = ["2011-07-06 12:00:00", "2011-08-06 00:00:00"]
# print(to_datetime(datestrs))
# print(repr(to_datetime(datestrs)))

# datestrs = [datetime(2005, 11, 30), datetime(2005, 12, 10)]
# print(to_datetime(datestrs))
# print(repr(to_datetime(datestrs)))

# print(to_datetime(datestrs + [None]))

# print(date_range("2000-01-01", periods=3))
# print(date_range("2000-01-01", end="2000-01-06", periods=3))
# print(date_range("2000-01-01", end="2000-01-06"))

# print(date_range("2012-04-01", "2012-06-01"))



dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
    datetime(2011, 1, 7), datetime(2011, 1, 8),
    datetime(2011, 1, 10), datetime(2011, 1, 12)]

# inx = DatetimeIndex(dates)
# # print(inx)
# print(inx.index('2011-01-15', False))

# dates = [datetime(2011, 1, 2), datetime(2011, 1, 2),
#     datetime(2011, 1, 7), datetime(2011, 1, 8),
#     datetime(2011, 1, 10), datetime(2011, 1, 12)]

# inx = DatetimeIndex(dates)
# # print(inx.is_monotonic)
# print(inx.index('2011-01-2', False))
# print(inx.all_indexes('2011-01'))
import pandas as pd




# print(np.datetime64('2000-10-10'))


dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
        datetime(2011, 1, 7), datetime(2011, 1, 8),
        datetime(2011, 1, 10), datetime(2011, 1, 12), datetime(2011, 2, 12)]

ts = Series(np.random.standard_normal(7), index=dates)

# print(ts)
# print(ts['2011-01-02'])
# print(ts['2011-01-01':'2011-01-09'])
# print(ts['2011-01'])
# print(ts['2011'])

# print(ts)
# print(ts.index)
# print(ts.index.is_monotonic)
# print(type(ts.index))


# stamp = ts.index[0]
# print(type(stamp))
# print(stamp.dtype)



# shift
data = np.random.standard_normal(4)
ts = Series(data,
index=date_range("2000-01-01", periods=4, freq="M"))
# print('========== My ==========')
# print(ts)
# print(ts.shift(2))
# print(ts.shift(-2))
# print(ts.shift(2, freq="M"))
# print(ts.shift(3, freq="D"))
# print(ts.shift(1, freq="90T"))

ts = pd.Series(data,
index=pd.date_range("2000-01-01", periods=4, freq="M"))
# print('========== Pandas ==========')
# print(ts)
# print(ts.shift(2))
# print(ts.shift(-2))
# print(ts.shift(2, freq="M"))
# print(ts.shift(3, freq="D"))
# print(ts.shift(1, freq="90T"))



date = datetime(2025, 10, 1)
# print(date + WeekOfMonth(1, week_num=0, weekday=0))
# print(date - WeekOfMonth(1, week_num=0, weekday=0))



# date = datetime(2020, 10, 22)
# print(date)
# h2 = Hour(2)
# d1 = Day()
# print(h2)
# print(date + h2 + d1 - Minute(10))
# print(type(date + h2))
# print(h2 + date)
# print(date + MonthEnd(4))
# print(date - MonthEnd(3))

from pandas.tseries.offsets import Hour, Minute, Day, MonthEnd, WeekOfMonth, YearEnd
# print(date + WeekOfMonth(1, week=0, weekday=0))
# print(date - WeekOfMonth(1, week=0, weekday=0))

h = Hour(2)
h2 = h * 3
# print(h)
# print(h2)

# print(date + MonthEnd(4))
# print(date - MonthEnd(3))
# print(type(date + Day()))


# print(date_range('2020-03-10', '2020-03-15'))
# print(date_range('2020-03-4', end='2020-03-10', periods=5))
# print(pd.date_range('2020-03-4', end='2020-03-10', periods=5))


# print(date_range('2020-03-5', periods=5, freq='3D'))
# print(date_range('2020-03-5', periods=5, freq='3M'))
# print(date_range('2020-03-5', periods=5, freq='3H'))



# print(date_range('2020-01-01', periods=5, freq="WOM-0MON"))
# print(pd.date_range('2020-01-01', periods=5, freq="WOM-1MON"))

