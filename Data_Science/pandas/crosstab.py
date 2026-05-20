from series import Series
from dataframe import DataFrame
import numpy as np
import pandas as pd
from index import MultiIndex, Index
from read_csv import read_csv

def check_params(inx_or_col):
    if isinstance(inx_or_col, list):
        if any(not isinstance(item, Series) for item in inx_or_col) or len(inx_or_col) > 2:
            raise ValueError('Invalid "index" or "columns"')
    elif isinstance(inx_or_col, Series):
        inx_or_col = [inx_or_col]
    else:
        raise TypeError('Invalid type for "index" or "columns"')
    
    return inx_or_col


def crosstab(index, columns, margins=False):
    index = check_params(index)
    columns = check_params(columns)
    
    all_series = index + columns
    
    groups = {}
    for row_vals in zip(*(s._data for s in all_series)):
        groups[row_vals] = groups.get(row_vals, 0) + 1

    if len(all_series) == 2:
        names = [all_series[0].name, all_series[1].name]
        inx = MultiIndex.from_tuples(list(groups.keys()), names)

        res_ser = Series(list(groups.values()), inx)
        res = res_ser.unstack()
    
    elif len(all_series) == 3:
        names = [(all_series[0].name, all_series[1].name), all_series[2].name]
        inx_data = [((gr[0], gr[1]), gr[2]) for gr in groups]
        inx = MultiIndex.from_tuples(inx_data, names)

        res_ser = Series(list(groups.values()), inx)
        res = res_ser.unstack()

        new_inx = MultiIndex.from_tuples(res.index, list(res.index.name))
        res.index = new_inx

        if len(columns) == 2:
            res = res.unstack()

    res.fillna(0)

    if margins:
        all_col = res.sum()
        res['All'] = all_col

        all_row = res.sum(axis=1)
        for i, ser in enumerate(res):
            ser._data = np.concatenate([ser._data, [all_row._data[i]]])

        if isinstance(res.index, MultiIndex):
            new_row = MultiIndex.from_tuples([('All', 'All')])
        else:
            new_row = Index(['All'])

        inx = res.index.append(new_row)

        res.index = inx

    return res
        

data = read_csv("examples/hw-59.csv")
print(data)
print()

print(crosstab(data["Nationality"], data["Handedness"], margins=True))

tips = read_csv("examples/tips.csv")
# print(crosstab([tips["time"], tips["day"]], tips["smoker"], margins=True))
# print()
# print(crosstab(tips["smoker"], [tips["time"], tips["day"]]))

# print(crosstab(tips["day"], tips["size"]).reindex(index=["Thur", "Fri", "Sat", "Sun"]))