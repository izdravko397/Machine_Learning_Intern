import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from read_csv import read_csv
from concat import concat

# names1880 = pd.read_csv("examples/babynames/yob1880.txt",
# names=["name", "sex", "births"])
# print(names1880)

# print(names1880.groupby("sex")["births"].sum())

pieces = []
for year in range(1880, 2011):
    path = f"examples/babynames/yob{year}.txt"
    frame = pd.read_csv(path, names=["name", "sex", "births"])
    # Add a column for the year
    frame["year"] = year
    pieces.append(frame)

# Concatenate everything into a single DataFrame
names = pd.concat(pieces, ignore_index=True)
# print(names)

total_births = names.pivot_table("births", index="year", columns="sex", aggfunc=sum)
# print(total_births.tail())
# total_births.plot(title="Total births by sex and year")


def add_prop(group):
    group["prop"] = group["births"] / group["births"].sum()
    return group

names = names.groupby(["year", "sex"], group_keys=False).apply(add_prop)
# print(names)
# print(names.groupby(["year", "sex"])["prop"].sum())


def get_top1000(group):
    return group.sort_values("births", ascending=False)[:1000]

grouped = names.groupby(["year", "sex"])
top1000 = grouped.apply(get_top1000)
top1000 = top1000.reset_index(drop=True)
# print(top1000.head())


boys = top1000[top1000["sex"] == "M"]
# girls = top1000[top1000["sex"] == "F"]
# print(boys.head())
# print(girls.head())


# total_births = top1000.pivot_table("births", index="year",
#                                     columns="name",
#                                     aggfunc=sum)
# print(total_births.info())

# subset = total_births[["John", "Harry", "Mary", "Marilyn"]]
# subset.plot(subplots=True, figsize=(12, 10),
#             title="Number of births per year")



# table = top1000.pivot_table("prop", index="year", columns="sex", aggfunc=sum)
# table.plot(title="Sum of table1000.prop by year and sex",
#             yticks=np.linspace(0, 1.2, 13))


df = boys[boys["year"] == 2010]
# print(df)

prop_cumsum = df["prop"].sort_values(ascending=False).cumsum()
# print(prop_cumsum[:10])
print(prop_cumsum.searchsorted(0.5))

df = boys[boys.year == 1900]
in1900 = df.sort_values("prop", ascending=False).prop.cumsum()
print(in1900.searchsorted(0.5) + 1)



def get_quantile_count(group, q=0.5):
    group = group.sort_values("prop", ascending=False)
    return group.prop.cumsum().searchsorted(q) + 1

# diversity = top1000.groupby(["year", "sex"]).apply(get_quantile_count)
# diversity = diversity.unstack()
# print(diversity.head())
# diversity.plot(title="Number of popular names in top 50%")




def get_last_letter(x):
    return x[-1]

# last_letters = names["name"].map(get_last_letter)
# last_letters.name = "last_letter"
# table = names.pivot_table("births", index=last_letters,
#                     columns=["sex", "year"], aggfunc=sum)

# subtable = table.reindex(columns=[1910, 1960, 2010], level="year")
# print(subtable.head())
# print(subtable.sum())
# letter_prop = subtable / subtable.sum()
# print(letter_prop.head())


# fig, axes = plt.subplots(2, 1, figsize=(10, 8))
# letter_prop["M"].plot(kind="bar", rot=0, ax=axes[0], title="Male")
# letter_prop["F"].plot(kind="bar", rot=0, ax=axes[1], title="Female",legend=False)




# letter_prop = table / table.sum()
# dny_ts = letter_prop.loc[["d", "n", "y"], "M"].T
# print(dny_ts.head())
# dny_ts.plot()


# all_names = pd.Series(top1000["name"].unique())
# lesley_like = all_names[all_names.str.contains("Lesl")]
# print(lesley_like)

# filtered = top1000[top1000["name"].isin(lesley_like)]
# print(filtered.groupby("name")["births"].sum())


# table = filtered.pivot_table("births", index="year", columns="sex", aggfunc="sum")
# table = table.div(table.sum(axis="columns"), axis="index")
# print(table.tail())

# table.plot(style={"M": "k-", "F": "k--"})


print('_____________________ MY _____________________')

# names1880 = read_csv("examples/babynames/yob1880.txt",
# names=["name", "sex", "births"])
# print(names1880.head())
# print(names1880.groupby("sex")["births"].sum())

pieces = []
for year in range(1880, 2011):
    path = f"examples/babynames/yob{year}.txt"
    frame = read_csv(path, names=["name", "sex", "births"])
    # Add a column for the year
    frame["year"] = year
    pieces.append(frame)

print('Ready File Readig!!!')

# Concatenate everything into a single DataFrame
names = concat(pieces, ignore_index=True)
print('Ready DFs Concat!!!')
# print(names.head())

# total_births = names.pivot_table("births", index="year", columns="sex", aggfunc=sum)
# print(total_births.tail())
# total_births.plot(title="Total births by sex and year")


names = names.groupby(["year", "sex"], group_keys=False).apply(add_prop)
# print(names.head())
# print(names.tail())

# print(names.groupby(["year", "sex"])["prop"].sum().head())

grouped = names.groupby(["year", "sex"], group_keys=False)
top1000 = grouped.apply(get_top1000)
top1000 = top1000.reset_index(drop=True)
# print(top1000.head())


boys = top1000[top1000["sex"] == "M"]
# girls = top1000[top1000["sex"] == "F"]
# print(boys.head())
# print(girls.head())


# total_births = top1000.pivot_table("births", index="year",
#                                     columns="name",
#                                     aggfunc=sum)
# print(total_births.info())



# subset = total_births[["John", "Harry", "Mary", "Marilyn"]]
# subset.plot(subplots=True, figsize=(12, 10),
#             title="Number of births per year")


# table = top1000.pivot_table("prop", index="year", columns="sex", aggfunc=sum)
# table.plot(title="Sum of table1000.prop by year and sex",
#             yticks=np.linspace(0, 1.2, 13))


df = boys[boys["year"] == 2010]
# print(df.head())
# print(df.tail())


prop_cumsum = df["prop"].sort_values(ascending=False).cumsum()
# print(prop_cumsum[:10])
print(prop_cumsum.searchsorted(0.5))


df = boys[boys.year == 1900]
in1900 = df.sort_values("prop", ascending=False).prop.cumsum()
print(in1900.searchsorted(0.5) + 1)



# diversity = top1000.groupby(["year", "sex"]).apply(get_quantile_count)
# diversity = diversity.unstack()
# print(diversity.head())
# diversity.plot(title="Number of popular names in top 50%")



# last_letters = names["name"].map(get_last_letter)
# last_letters.name = "last_letter"
# table = names.pivot_table("births", index=last_letters,
#                     columns=["sex", "year"], aggfunc=sum)

# subtable = table.reindex(columns=[1910, 1960, 2010], level="year")
# print(subtable.head())
# print(subtable.sum())
# letter_prop = subtable / subtable.sum()
# print(letter_prop.head())


# fig, axes = plt.subplots(2, 1, figsize=(10, 8))
# letter_prop["M"].plot(kind="bar", rot=0, ax=axes[0], title="Male")
# letter_prop["F"].plot(kind="bar", rot=0, ax=axes[1], title="Female")




# letter_prop = table / table.sum()
# dny_ts = letter_prop.loc[["d", "n", "y"], "M"].T
# print(dny_ts.head())
# dny_ts.plot()

from series import Series
# all_names = Series(top1000["name"].unique())
# lesley_like = all_names[all_names.str.contains("Lesl")]
# # print(lesley_like)

# filtered = top1000[top1000["name"].isin(lesley_like)]
# print(filtered.groupby("name")["births"].sum())

# table = filtered.pivot_table("births", index="year", columns="sex", aggfunc="sum")
# table = table.div(table.sum(axis="columns"), axis="index")
# print(table.tail())

# table.plot(style={"M": "k-", "F": "k--"})

# plt.show()
