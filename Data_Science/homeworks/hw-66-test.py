import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataframe import DataFrame
from merge import merge 
from concat import concat


db = json.load(open("examples/usda_food/database.json"))
# print(len(db))
# print(db[0].keys())
# print(db[0]["nutrients"][0])
print('___________________ PANDAS ___________________')
# nutrients = pd.DataFrame(db[0]["nutrients"])
# print(nutrients.head(7))

info_keys = ["description", "group", "id", "manufacturer"]
info = pd.DataFrame(db, columns=info_keys)
# # print(info.head())
# # info.info()

# print(pd.value_counts(info["group"])[:10])

#-------------------------
nutrients = []
for rec in db:
    fnuts = pd.DataFrame(rec["nutrients"])
    fnuts["id"] = rec["id"]
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)
# print(nutrients)

# print(nutrients.duplicated().sum())
nutrients = nutrients.drop_duplicates()
# print(nutrients)

col_mapping = {"description" : "food", "group": "fgroup"}
info = info.rename(columns=col_mapping, copy=False)
# print(info.info())



col_mapping = {"description" : "nutrient", "group" : "nutgroup"}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
# print(nutrients)


#---------------------------
ndata = pd.merge(nutrients, info, on="id")
# print(ndata)
ndata.info()

# print(ndata.iloc[30000])


# result = ndata.groupby(["nutrient", "fgroup"])["value"].quantile(0.5)
# result["Zinc, Zn"].sort_values().plot(kind="barh")
#------------------------

# by_nutrient = ndata.groupby(["nutgroup", "nutrient"])
# def get_maximum(x):
#     return x.loc[x.value.idxmax()]

# max_foods = by_nutrient.apply(get_maximum)[["value", "food"]]
# # # make the food a little smaller
# max_foods["food"] = max_foods["food"].str[:50]

# print(max_foods.loc["Amino Acids"]["food"]) 


print('___________________ MY ___________________')

# nutrients = DataFrame(db[0]["nutrients"])
# print(nutrients.head(7))

info = DataFrame(db, columns=info_keys) 
# print(info.head())
# print(info.info())

# print(pd.value_counts(info["group"])[:10]) 
# print(info["group"].value_counts()[:10]) 



#-------------------------
nutrients = []
for rec in db:
    fnuts = DataFrame(rec["nutrients"])
    fnuts["id"] = rec["id"]
    nutrients.append(fnuts)

nutrients = concat(nutrients, ignore_index=True)
# print(nutrients.head())
# print(nutrients.tail())

# print(nutrients.duplicated().sum())
nutrients = nutrients.drop_duplicates()
# print(nutrients.head())
# print(nutrients.tail())
# print(len(nutrients.index))

# print('-------------------------')
# nutrients['group'] = nutrients['group'].astype('category')
# print(nutrients['group'].head())
# print('-------------------------')
# print(nutrients.head())
# print(nutrients.tail())



col_mapping = {"description" : "food", "group": "fgroup"}
info = info.rename(columns=col_mapping)
# print(info.info())



col_mapping = {"description" : "nutrient", "group" : "nutgroup"}
nutrients = nutrients.rename(columns=col_mapping)
# print(nutrients.head())
# print(nutrients.tail())

#---------------------------
ndata = merge(nutrients, info, on="id")
# print(ndata.info())
# print(ndata.head())
# print(ndata.tail())

# print(ndata.iloc[30000])

# result = ndata.groupby(["nutrient", "fgroup"])["value"].quantile(0.5)
# result["Zinc, Zn"].sort_values().plot(kind="barh")

#------------------------
# by_nutrient = ndata.groupby(["nutgroup", "nutrient"])
# def get_maximum(x):
#     return x.loc[x.value.idxmax()]

# max_foods = by_nutrient.apply(get_maximum)[["value", "food"]]
# # # make the food a little smaller
# max_foods["food"] = max_foods["food"].str[:50]
# print(max_foods.loc["Amino Acids"]["food"])


# plt.show()