from index import Index, RangeIndex
import numpy as np
from dataframe import DataFrame
from series import Series
from cut_and_qcut import Categorical, cut

def get_dummies(data, prefix=None): # Series, Categorical
    if isinstance(data, Categorical):
        data = data.to_series()
    
    if data._dtype == 'category':
        cols = np.unique(data._data.codes)
        cols_labels = [data.cat.categories[c] for c in cols]
    else:
        cols = np.unique(data._data)
        cols_labels = cols

    series = []
    for col in cols:
        s_data = np.fromiter((1 if v == col else 0 for v in data), dtype=int)

        if data._dtype == 'category':
            s = Series(s_data, data.index, cols_labels[col])
        else:
            s = Series(s_data, data.index, col)
            
        series.append(s)

    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = str(prefix)

        cols_labels = np.array([prefix + '_' + str(c) for c in cols_labels])

        for i, s in enumerate(series):
            s.name = cols_labels[i]

    return DataFrame(series, data.index, cols_labels)


# df = DataFrame({"key": ["b", "b", "a", "c", "a", "b"], "data1": list(range(6))})
# print(df)
# print()
# print(get_dummies(df["key"]))
# print(get_dummies(df["key"], prefix="key"))

# values = np.random.uniform(size=10)

# bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
# print(get_dummies(cut(values, bins)))

# cat_s = Series(['a', 'b', 'c', 'd'] * 2)
# print(get_dummies(cat_s))