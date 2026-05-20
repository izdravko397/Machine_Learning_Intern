from series import Series
from dataframe import DataFrame
import numpy as np
import pandas as pd
from index import MultiIndex
from read_csv import read_csv

#  ============ Series ============
states = ["Ohio", "New York", "Vermont", "Florida",
"Oregon", "Nevada", "California", "Idaho"]

group_key = ["East", "East", "East", "East",
"West", "West", "West", "West"]

data = Series(np.random.standard_normal(8), index=states)
data[["Vermont", "Nevada", "Idaho"]] = np.nan
print(data)

# print(data.groupby(group_key).size())
# print(data.groupby(group_key).count())
# print(data.groupby(group_key).mean())

def fill_mean(group):
    return group.fillna(group.mean())

print(data.groupby(group_key).apply(fill_mean))


def describe(group):
    return Series({'mean': group.mean(), 'sum': group.sum()})

# print(data.groupby(group_key).apply(describe))


def func(x):
    # print(x)
    res = DataFrame({"orig": x, "sq": x**2})
    # print('func')
    # print(res)
    return res

# print(data.groupby(group_key, group_keys=False).apply(func))








data = pd.Series(np.random.standard_normal(8), index=states)
data[["Vermont", "Nevada", "Idaho"]] = np.nan
# print(data)
# print(data.groupby(group_key).size())
# print(data.groupby(group_key).count())
# print(data.groupby(group_key).mean())

def fill_mean(group):
    return group.fillna(group.mean())
# print(data.groupby(group_key).apply(fill_mean))
def describe(group):
    return Series({'mean': group.mean(), 'sum': group.sum()})
# print(data.groupby(group_key).apply(describe))
def func(x):
    return pd.DataFrame({"orig": x, "sq": x**2})

# print(data.groupby(group_key).apply(func))




#  ============ DF ============


df = DataFrame({
    "team": ["A","A","B","B","B"],
    "score": [10, 20, 30, 40, 50],
    "assists": [1, 2, 3, 4, 5]
})
# print(df)

g = df.groupby("team")

def group_stats(group):
    return Series({
        "sum_score": group["score"].sum(),
        "mean_assists": group["assists"].mean()
    })

a = 1
# def group_stats(group):
#     global a
#     if a:
#         a = 0
#         return Series({
#             "sum_score": group["score"].sum(),
#             "mean_assists": group["assists"].mean()
#         })
    
#     return Series({
#             "sum_score": group["score"].sum(),
#         })
    

# result = g.apply(group_stats)
# print(result)



def expand(group):
    return DataFrame({
        "score": group["score"],
        "score_x2": group["score"] * 2
    })

# result = g.apply(expand)
# print(result)

def variable_cols(grp):
    global a
    if a:
        a = 0
        return DataFrame({"sum": [grp["score"].sum()]})
    else:
        return DataFrame({"avg": [grp["score"].mean()]})

# print(g.apply(variable_cols))




key = ['X', 'X', 'Y', 'Y']
df = DataFrame(np.arange(1, 13).reshape((3, 4)), columns=['A', 'B', 'A', 'B'], index=['a', 'b', 'c'])
# print(df)
g = df.groupby(key, axis=1)

# result = g.apply(lambda subdf: subdf['A'].sum())
# print(result)

def group_sum(subdf):
    return subdf['A'] + subdf['B']

# result = g.apply(group_sum)
# print(result)


def expand(subdf):
    return DataFrame({
        "sum": subdf['A'] + subdf['B'],
        "diff": subdf['A'] - subdf['B']
    })

# result = g.apply(expand)
# print(result)

def f(subdf, n=0):
    global a
    if n:
        a = 0
        return DataFrame({
        "sum": subdf['A'] + subdf['B'],
        "diff": subdf['A'] - subdf['B']
    })
    else:
        d = subdf['A'] + subdf['B']
        return DataFrame({
        "sum": d
    })

# res = g.apply(f, n=1)
# print(res)



# ============ task2 ============
suits = ["H", "S", "C", "D"] # Hearts, Spades, Clubs, Diamonds
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ["A"] + list(range(2, 11)) + ["J", "K", "Q"]
cards = []
for suit in suits:
    cards.extend(str(num) + suit for num in base_names)
deck = Series(card_val, index=cards) 

# print(deck)

def get_suit(card):
    return card[-1]

def draw(deck, n=5):
    return deck.sample(n)

# print(deck.groupby(get_suit).apply(draw, n=2))
# print(deck.groupby(get_suit, group_keys=False).apply(draw, n=2))


def top(df, n=5, column="tip_pct"):
    return df.nsmallest(n=n, columns=column)

tips = read_csv("examples/tips.csv")
tips["tip_pct"] = tips["tip"] / tips["total_bill"]
# print(tips.head())
# print()

# print(tips.groupby("smoker").apply(top))
def f(group):
    return group.describe()

grouped = tips.groupby("smoker")
# print(grouped.apply(f))



# print(tips.groupby("smoker").apply(top))
# print(tips.groupby("smoker", group_keys=False).apply(top, n=3))


# print(tips.groupby(["day", "smoker"]).mean())
# print(tips.groupby(["day", "smoker"], as_index=False).mean())


# print(tips.groupby(["smoker", "day"], group_keys=False).apply(top, n=1, column="total_bill"))


def describe(gp):
    return gp.describe()

# print(tips["tip_pct"].groupby(tips["smoker"]).apply(describe))

result = tips["tip_pct"].groupby(tips["smoker"]).describe()
# print(result)


# ============
from cut_and_qcut import cut, qcut

frame = DataFrame({"data1": np.random.standard_normal(1000),
"data2": np.random.standard_normal(1000)})
# print(frame.head())

quartiles = cut(frame["data1"], 4).to_series()
# print(type(quartiles.iloc[0]))

def get_stats(group):
    return DataFrame(
        {"min": group.min(), "max": group.max(),
        "count": group.count(), "mean": group.mean()})

grouped = frame.groupby(quartiles)

# print(grouped.apply(get_stats))
# print(grouped.agg(["min", "max", "count", "mean"]))