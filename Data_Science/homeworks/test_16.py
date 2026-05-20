from series import Series
from index import *
from periods import *


# dates = date_range("2000-01-01", periods=100)
# ts = Series(np.random.standard_normal(len(dates)), index=dates)
# print(ts)

# print(ts.resample("M").mean())
# print(ts.resample("M", kind="period").mean())



# dates = date_range("2000-01-01", periods=12, freq="T")
# ts = Series(np.arange(len(dates)), index=dates)
# print(ts)
# print(ts.resample("5min").sum())
# print(ts.resample("5min", closed="right").sum())
# print(ts.resample("5min", closed="right", label="right").sum())
# print(ts.resample("5min").ohlc())




# dates = date_range("2000-01-01", periods=2, freq="W-WED")
# ts = Series(np.arange(1, len(dates) + 1), index=dates)
# print(ts)

# print(ts.resample("D").asfreq())
# print(ts.resample("D").ffill())
# print(ts.resample("W-THU").asfreq())
# print(ts.resample("W-THU").ffill())




# data = np.random.standard_normal(24),
# index = period_range("1-2000", "12-2001", freq="M")
# ts = Series(data, index)
# # print(ts)

# annual_frame = ts.resample("A-DEC").mean()
# print(annual_frame)

# print(annual_frame.resample("Q-DEC").ffill())
# print(annual_frame.resample("Q-DEC", closed='left').asfreq())

# print(annual_frame.resample("Q-MAR").ffill())
# print(annual_frame.resample("Q-MAR").asfreq())


# N = 15
# times = date_range("2017-05-20 00:00", freq="1min", periods=N)
# ts = Series(np.arange(N), times)
# print(ts)
# print(ts.resample("5min").count())


