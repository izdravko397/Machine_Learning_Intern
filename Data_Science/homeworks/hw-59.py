from series import Series
from dataframe import DataFrame
import numpy as np
import pandas as pd
from index import MultiIndex
from read_csv import read_csv


tips = read_csv("examples/tips.csv")
tips["tip_pct"] = tips["tip"] / tips["total_bill"]
print(tips.head())
print()


print(tips.pivot_table(index=["day", "smoker"]))

# print(tips.pivot_table(index="day", columns="smoker",
# values=["tip_pct", "size"]))

# print(tips.pivot_table(index="day", columns="smoker"))

# print(tips.pivot_table(index="time", columns="smoker",
# values=["tip_pct", "size"]))


# print(tips.pivot_table(index=["time", "day"], columns="smoker",
# values=["tip_pct", "size"]))

# print(tips.pivot_table(index=["time", "day"], columns="smoker",
# values=["tip_pct", "size"], margins=True))


# print(tips.pivot_table(index=["time", "smoker"], columns="day",
# values="tip_pct", aggfunc=len, margins=True))

