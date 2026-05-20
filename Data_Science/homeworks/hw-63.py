path = "examples/example.txt"

# with open(path) as f:
#     print(f.readline())

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
with open(path) as f:
    records = [json.loads(line) for line in f]

# print(records[0])

# time_zones = [rec["tz"] for rec in records] # error

time_zones = [rec["tz"] for rec in records if "tz" in rec]
# print(time_zones[:10])


from collections import defaultdict
def get_counts2(sequence):
    counts = defaultdict(int) # values will initialize to 0
    for x in sequence:
        counts[x] += 1  
    return counts

# counts = get_counts2(time_zones)
# print(counts["America/New_York"])
# print(len(time_zones))

from collections import Counter
counts = Counter(time_zones)
# print(counts.most_common(10))

import pandas as pd
from dataframe import DataFrame
from series import Series


frame = pd.DataFrame(records)
# print(frame.head())
# print(frame.info())
# print(frame["tz"].head())

# tz_counts = frame["tz"].value_counts()
# print(tz_counts.head())

# clean_tz = frame["tz"].fillna("Missing")
# clean_tz[clean_tz == ""] = "Unknown"
# tz_counts = clean_tz.value_counts()
# # print(tz_counts.head())
# subset = tz_counts.head()

# sns.barplot(y=subset.index, x=subset.to_numpy())

# print(frame["a"][1])
# print(frame["a"][50])
# print(frame["a"][51][:50])

# results = pd.Series([x.split()[0] for x in frame["a"].dropna()])
# print(results.head(5))
# print(results.value_counts().head(8))

cframe = frame[frame["a"].notna()].copy()
cframe["os"] = np.where(cframe["a"].str.contains("Windows"), "Windows", "Not Windows")
# print(cframe["os"].head(5))

by_tz_os = cframe.groupby(["tz", "os"])
agg_counts = by_tz_os.size().unstack().fillna(0)
# print(agg_counts.head())

indexer = agg_counts.sum("columns").argsort()
# print(indexer.values[:10])

count_subset = agg_counts.take(indexer[-10:])
# print(count_subset)

print(agg_counts.sum(axis="columns").nlargest(10))

count_subset = count_subset.stack()
count_subset.name = "total"
count_subset = count_subset.reset_index()
# print(count_subset.head(10))

# sns.barplot(x="total", y="tz", hue="os", data=count_subset)

def norm_total(group):
    group["normed_total"] = group["total"] / group["total"].sum()
    return group

# results = count_subset.groupby("tz").apply(norm_total)
# sns.barplot(x="normed_total", y="tz", hue="os", data=results)


print('______________ MY ______________')
frame = DataFrame(records)
# print(frame.head())
# print(frame.info())
# print(frame["tz"].head())

# tz_counts = frame["tz"].value_counts()
# print(tz_counts.head())

# clean_tz = frame["tz"].fillna("Missing")
# clean_tz[clean_tz == ""] = "Unknown"
# tz_counts = clean_tz.value_counts()
# print(tz_counts.head())

# subset = tz_counts.head()
# # subset.barplot()
# subset.plot(kind='barh')

# print(frame["a"][1])
# print(frame["a"][50])
# print(frame["a"][51][:50])

# results = Series([x.split()[0] for x in frame["a"].dropna()])
# print(results.head(5))
# print(results.value_counts().head(8))

cframe = frame[frame["a"].notna()].copy()
cframe["os"] = np.where(cframe["a"].str.contains("Windows"), "Windows", "Not Windows")
# print(cframe["os"].head(5))

by_tz_os = cframe.groupby(["tz", "os"])
agg_counts = by_tz_os.size().unstack()
agg_counts.fillna(0)
# print(agg_counts.head())

indexer = agg_counts.sum("columns").argsort()
# print(indexer.values[:10])

count_subset = agg_counts.take(indexer[-10:])
# print(count_subset)

# print(agg_counts.sum(axis="columns").nlargest(10))

count_subset = count_subset.stack()
count_subset.name = "total"
count_subset = count_subset.reset_index()
# print(count_subset.head(10))


# sns.barplot(x="total", y="tz", hue="os", data=count_subset)
# count_subset.plot(kind='barh', y='tz', x='total', hue="os")

def norm_total(group):
    group["normed_total"] = group["total"] / group["total"].sum()
    return group

# results = count_subset.groupby("tz").apply(norm_total).reset_index()
# print(results.head())
# sns.barplot(x="normed_total", y="tz", hue="os", data=results)
# results.plot(kind='barh', y='tz', x='normed_total', hue="os")


# g = count_subset["total"].groupby(count_subset["tz"])
# results2 = count_subset["total"] / g.transform("sum")
# print(results2.head())

# plt.show()