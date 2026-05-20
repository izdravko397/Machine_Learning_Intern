import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataframe import DataFrame
from merge import merge 
from concat import concat
from read_csv import read_csv
from cut_and_qcut import cut

parties = {"Bachmann, Michelle": "Republican",
            "Cain, Herman": "Republican",
            "Gingrich, Newt": "Republican",
            "Huntsman, Jon": "Republican",
            "Johnson, Gary Earl": "Republican",
            "McCotter, Thaddeus G": "Republican",
            "Obama, Barack": "Democrat",
            "Paul, Ron": "Republican",
            "Pawlenty, Timothy": "Republican",
            "Perry, Rick": "Republican",
            "Roemer, Charles E. 'Buddy' III": "Republican",
            "Romney, Mitt": "Republican",
            "Santorum, Rick": "Republican"
            }

occ_mapping = {
    "INFORMATION REQUESTED PER BEST EFFORTS" : "NOT PROVIDED",
    "INFORMATION REQUESTED" : "NOT PROVIDED",
    "INFORMATION REQUESTED (BEST EFFORTS)" : "NOT PROVIDED",
    "C.E.O.": "CEO"
}

emp_mapping = {
    "INFORMATION REQUESTED PER BEST EFFORTS" : "NOT PROVIDED",
    "INFORMATION REQUESTED" : "NOT PROVIDED",
    "SELF" : "SELF-EMPLOYED",
    "SELF EMPLOYED" : "SELF-EMPLOYED",
}

print('___________________ PANDAS ___________________')
fec = pd.read_csv("examples/P00000001-ALL.csv", low_memory=False)
fec.info()
# print(fec.iloc[123456])

# unique_cands = fec["cand_nm"].unique()
# print(unique_cands)
# print(unique_cands[2])
# print(fec["cand_nm"][123456:123461])
# print(fec["cand_nm"][123456:123461].map(parties))

fec["party"] = fec["cand_nm"].map(parties)
# print(fec["party"].value_counts())
# print((fec["contb_receipt_amt"] > 0).value_counts())

fec = fec[fec["contb_receipt_amt"] > 0]
# fec.info()

fec_mrbo = fec[fec["cand_nm"].isin(["Obama, Barack", "Romney, Mitt"])]
# fec_mrbo.info()

#----------------
# print(fec["contbr_occupation"].value_counts()[:10])

def get_occ(x):
    return occ_mapping.get(x, x)
# fec["contbr_occupation"] = fec["contbr_occupation"].map(get_occ)

def get_emp(x):
    return emp_mapping.get(x, x)
# fec["contbr_employer"] = fec["contbr_employer"].map(get_emp)

# by_occupation = fec.pivot_table("contb_receipt_amt",
#             index="contbr_occupation",
#             columns="party", aggfunc="sum")

# over_2mm = by_occupation[by_occupation.sum(axis="columns") > 2000000]
# print(over_2mm)
# over_2mm.plot(kind="barh")


def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)["contb_receipt_amt"].sum()
    res = totals.nlargest(n)
    return res

# grouped = fec_mrbo.groupby("cand_nm")
# print(grouped.apply(get_top_amounts, "contbr_occupation", n=7))
# print(grouped.apply(get_top_amounts, "contbr_employer", n=10))

#--------------------------
bins = np.array([0, 1, 10, 100, 1000, 10000,
                100_000, 1_000_000, 10_000_000])

labels = pd.cut(fec_mrbo["contb_receipt_amt"], bins)
# # print(labels)

grouped = fec_mrbo.groupby(["cand_nm", labels])
# # print(grouped.size().unstack(level=0))

bucket_sums = grouped["contb_receipt_amt"].sum().unstack(level=0)
normed_sums = bucket_sums.div(bucket_sums.sum(axis="columns"), axis="index")

print(normed_sums)
normed_sums[:-2].plot(kind="barh")

#----------------------------------
# grouped = fec_mrbo.groupby(["cand_nm", "contbr_st"])
# totals = grouped["contb_receipt_amt"].sum().unstack(level=0).fillna(0)
# totals = totals[totals.sum(axis="columns") > 100000]
# print(totals.head(10))

# percent = totals.div(totals.sum(axis="columns"), axis="index")
# print(percent.head(10))



print('___________________ MY ___________________')
fec = read_csv("examples/P00000001-ALL.csv")
print('Ready DF')
print(fec.info())
# print(fec.iloc[123456])

# unique_cands = fec["cand_nm"].unique()
# print(unique_cands)
# print(unique_cands[2])
# print(fec["cand_nm"][123456:123461])
# print(fec["cand_nm"][123456:123461].map(parties))

fec["party"] = fec["cand_nm"].map(parties)
# print(fec["party"].value_counts())
# print((fec["contb_receipt_amt"] > 0).value_counts())

fec = fec[fec["contb_receipt_amt"] > 0]
# print(fec.info())

mask = fec["cand_nm"].isin(["Obama, Barack", "Romney, Mitt"])
fec_mrbo = fec[mask]
# print(fec_mrbo.info())

#----------------
# print(fec["contbr_occupation"].value_counts()[:10])

def get_occ(x):
    return occ_mapping.get(x, x)
# fec["contbr_occupation"] = fec["contbr_occupation"].map(get_occ)

def get_emp(x):
    return emp_mapping.get(x, x)
# fec["contbr_employer"] = fec["contbr_employer"].map(get_emp)

# by_occupation = fec.pivot_table("contb_receipt_amt",
#             index="contbr_occupation",
#             columns="party", aggfunc="sum")

# over_2mm = by_occupation[by_occupation.sum(axis="columns") > 2000000]
# print(over_2mm)
# over_2mm.plot(kind="barh")


def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)["contb_receipt_amt"].sum()
    res = totals.nlargest(n)
    # print(res.index.name)
    return res

# grouped = fec_mrbo.groupby("cand_nm")
# print(grouped.apply(get_top_amounts, "contbr_occupation", n=7))
# print(grouped.apply(get_top_amounts, "contbr_employer", n=10))

#-----------------------------
bins = np.array([0, 1, 10, 100, 1000, 10000,
                100_000, 1_000_000, 10_000_000])
labels = cut(fec_mrbo["contb_receipt_amt"], bins)
# # print(labels.head())
# # print(labels.tail())

grouped = fec_mrbo.groupby(["cand_nm", labels])
# # print(grouped.size().unstack(level=0))

bucket_sums = grouped["contb_receipt_amt"].sum().unstack(level=0)
normed_sums = bucket_sums.div(bucket_sums.sum(axis="columns"), axis="index")

print(normed_sums)
normed_sums[:-2].plot(kind="barh")

#-----------------------------
# grouped = fec_mrbo.groupby(["cand_nm", "contbr_st"])
# totals = grouped["contb_receipt_amt"].sum().unstack(level=0).fillna(0)
# totals = totals[totals.sum(axis="columns") > 100000]
# print(totals.head(10))


# percent = totals.div(totals.sum(axis="columns"), axis="index")
# print(percent.head(10))



plt.show()