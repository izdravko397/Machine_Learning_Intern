from series import Series
from index import *


# s = Series([10, 20, 30, 40, 50])
# print(s)
# print(s.rolling(window=3).mean())
# print(s.rolling(window=3, min_periods=1).mean())
# print(s.rolling(window=3, min_periods=2).mean())

# s = Series([10, np.nan, 30, 40, 50])
# print(s)
# print(s.rolling(window=3, min_periods=1).mean())
# print(s.rolling(window=3, min_periods=3).mean())


# --- corr ---

# s2 = Series(s._data, [2, 3, 4, 1, 0])
# print(s.rolling(3).corr(s2))





# -------- task2 --------------------
import matplotlib.pyplot as plt

#                    === 1 ===
close_px_all = pd.read_csv("examples/stock_px.csv", parse_dates=True, index_col=0)
close_px = close_px_all["AAPL"]
# close_px = close_px.resample("B").ffill()

inx = to_datetime(close_px.index.to_numpy())
inx.freq = 'B'
s = Series(close_px.to_numpy(), inx)

# print(s)
# print(s.rolling("20D").mean())


# fig, ax = plt.subplots()
# s.plot(ax=ax)
# s.rolling(250).mean().plot(ax=ax)

# plt.show()


#                    === 2 ===

# std250 = s.pct_change().rolling(250, min_periods=10).std()
# std250.plot()
# plt.show()

#                    === 3 ===
# s.rolling(60).mean().plot(logy=True)
# plt.show()

#                    === 4 ===
# close_px = close_px["2006":"2007"]

# aapl_px = Series(close_px.to_numpy(), to_datetime(close_px.index.to_numpy()))
# ma30 = aapl_px.rolling(30, min_periods=20).mean()
# ewma30 = aapl_px.ewm(span=10).mean()

# fig, ax = plt.subplots()
# aapl_px.plot(style="k-", label="Price", ax=ax)
# ma30.plot(style="k--", label="Simple Moving Avg", ax=ax)
# ewma30.plot(style="k-", label="EW MA", ax=ax)
# plt.legend()
# plt.show()


#                    === 5 ===
# spx_px = close_px_all["SPX"]
# s2 = Series(spx_px.to_numpy(), spx_px.index.to_numpy())
# spx_rets = s2.pct_change()
# returns = s.pct_change()
# corr = returns.rolling(125, min_periods=100).corr(spx_rets)

# corr.plot()
# plt.show()



# ======= emw ======= 

s = pd.Series([10, 20, 30, 40, 50])
print(s.ewm(span=3, adjust=False).mean())

s = Series([10, 20, 30, 40, 50])
print(s.ewm(span=3).mean())
