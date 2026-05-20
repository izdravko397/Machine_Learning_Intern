from periods import *
from series import Series

periods = period_range("2006", "2009", freq="A-DEC")
# print(periods)
# print(periods._data)

# ts = Series(np.random.standard_normal(len(periods)), index=periods)
# print(ts)
# print()
# print(ts.asfreq("M", how="start"))


# dates = date_range("2000-01-01", periods=3, freq="M")
# ts = Series(np.random.standard_normal(3), index=dates)
# print(ts)
# print(ts.to_period())


dates = date_range("2000-01-29", periods=6)
ts2 = Series(np.random.standard_normal(6), index=dates)

# print(ts2)
# print(ts2.to_period('M'))
# print(ts2.to_period())


# print(pd.Timestamp("2012Q4"))
# print(pd.offsets.QuarterEnd(2, month=12))

# qe = QuarterEnd(month=12)

# print(qe.rollforward("2024-02-15"))  
# print(qe.rollback("2024-02-15"))     

# print(qe.rollforward("2024-07-20"))  
# print(qe.rollback("2024-07-20")) 

date = pd.Timestamp("2012Q4")
# print(date + QuarterEnd(3))
# print(date + MonthEnd(3))

# p = Period("2012Q2", freq="Q-JAN")
# print(p)
# print(p.start)
# print(p.end)
# print(p.asfreq("D", how="start"))
