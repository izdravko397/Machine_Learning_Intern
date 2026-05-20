from dataframe import DataFrame
from index import RangeIndex, Index
import numpy as np
import re
import pandas as pd

def read_csv(fname, sep=None, header=0, names=None, index_col=None, skiprows=[], 
             na_values=[], keep_default_na=True, nrows=-1):
    
    if sep is None:
        pattern = r'"[^"]*"|[^,]+'
    else:
        if not isinstance(sep, (str, re.Pattern)):
            raise TypeError(f'Invalid separator type: {type(sep).__name__}')

        if not sep:
            raise ValueError('separator cannot be empty')
    
    if header is not None and not isinstance(header, int):
        raise TypeError(f'Invalid header type: {type(header).__name__}')
    
    if names is not None and not isinstance(names, (list, Index)):
        raise TypeError(f'Invalid names type: {type(names).__name__}')
    
    if not isinstance(skiprows, (int, list, tuple)):
        raise TypeError(f'Invalid skiprows type: {type(skiprows).__name__}')
    
    skiprows = {skiprows} if isinstance(skiprows, int) else set(skiprows)

    if not isinstance(na_values, (list, tuple, set, dict)):
        raise TypeError(f'Invalid na_values type: {type(na_values).__name__}')
    
    if not isinstance(keep_default_na, bool):
        raise TypeError(f'Invalid keep_default_na type: {type(keep_default_na).__name__}')

    def gen(nrows):
        with open(fname) as file:
            for i, line in enumerate(file):
                if i in skiprows:
                    continue
                
                if sep is not None:
                    val = np.array(re.split(sep, line.strip()))
                else:
                    val = np.array(re.findall(pattern, line.strip()))
                    mask = np.char.startswith(val, '"')
                    val[mask] = np.char.strip(val[mask], '"')

                yield val
                nrows -= 1
                if nrows == 0:
                    break

    normalized_data = np.fromiter(gen(nrows), dtype=np.object_)

    if names is not None:
        cols = Index(names)
    else:
        if header is None:
            cols = RangeIndex(len(normalized_data[0]))
        else:
            cols = Index(normalized_data[header])
            normalized_data = np.delete(normalized_data, header)

    if index_col is None:
        if len(cols) + 1 == len(normalized_data[0]):
            inx_data = next(zip(*normalized_data))
            normalized_data = np.vectorize(lambda x: x[1:])(normalized_data)
            inx = Index(inx_data)
        else:
            inx = RangeIndex(len(normalized_data))

    else:
        target_i = index_col
        if not isinstance(index_col, int):
            target_i = cols.index(index_col)
        cols = cols.delete(target_i)

        def inx_gen():
            for i, line in enumerate(normalized_data):
                yield line[target_i]
                normalized_data[i] = np.delete(line, target_i)

        inx = Index(np.fromiter(inx_gen(), dtype=normalized_data[0].dtype), name=index_col)

    df = DataFrame(normalized_data, columns=cols, index=inx)

    make_mask = None
    if keep_default_na and na_values and not isinstance(na_values, dict):
        make_mask = lambda x: (x == '') | (x == 'NA') | (np.isin(x, na_values))
    elif keep_default_na:
        make_mask = lambda x: (x == '') | (x == 'NA')
    elif na_values and not isinstance(na_values, dict):
        make_mask = lambda x: np.isin(x, na_values)

    if isinstance(na_values, dict):
        for label, val in na_values.items():
            if (i := cols.index(label, False)) >= 0:
                mask = df._series[i]._data == val
                df._series[i]._data[mask] = np.nan

    for col in df:
        if make_mask is not None:
            mask = make_mask(col._data)
            col._data[mask] = np.nan

        col._data = pd.to_numeric(col._data, errors='ignore')
        col._dtype = col._data.dtype
        
    return df


# df = read_csv("examples/ex1.csv")
# print(df)
# print(type(df['d'][2]))

# print(read_csv("examples/ex2.csv", header=None))
# print(read_csv("examples/ex2.csv", names=["a", "b", "c", "d", "message"]))

# names = ["a", "b", "c", "d", "message"]
# print(read_csv("examples/ex2.csv", names=names, index_col="message"))

# df = read_csv("examples/ex3.txt", sep="\s+")
# print(df)
# print(type(df['A']['aaa']))

# print(read_csv("examples/ex4.csv", skiprows=[0, 2, 3]))

# result = read_csv("examples/ex5.csv")
# print(result)
# print()
# print(read_csv("examples/ex5.csv", na_values=["NULL"]))
# print()
# print(read_csv("examples/ex5.csv", keep_default_na=False))
# print()
# print(read_csv("examples/ex5.csv", keep_default_na=False, na_values=["NA"]))

# sentinels = {"message": ["foo", "NA"], "something": ["two"]}
# print(read_csv("examples/ex5.csv", na_values=sentinels, keep_default_na=False))













# def read_csv(fname, sep=',', header=0, names=None, index_col=None, skiprows=[], 
#              na_values=[], keep_default_na=True, nrows=-1, chunksize=None):
    
#     def data_to_dataframe(gen):
#         data = np.fromiter(gen(), dtype=np.object_)
#         print(data)

#         for i in range(len(data)):
#             line = data[i]
#             if keep_default_na:
#                 mask = (line == '') | (line == 'NA') | np.isin(line, na_values)
#                 line[mask] = np.nan

#             elif na_values and isinstance(na_values, (list, tuple, set)):
#                 mask = np.isin(line, na_values)
#                 line[mask] = np.nan

#         if names is not None:
#             cols = Index(names)
#         else:
#             if header is None:
#                 cols = RangeIndex(len(data[0]))
#             else:
#                 cols = Index(data[header])
#                 data = np.delete(data, header)

#         if index_col is None:
#             if len(cols) + 1 == len(data[0]):
#                 inx_data = next(zip(*data))
#                 for i in range(len(data)):
#                     data[i] = data[i][1:]

#                 inx = Index(inx_data)
#             else:
#                 inx = RangeIndex(len(data))
#         else:
#             target_i = cols.index(index_col[0])
#             cols = cols.delete(target_i)

#             def inx_gen():
#                 for i in range(len(data)):
#                     yield data[i][target_i]
#                     data[i] = np.delete(data[i], target_i)

#             inx = Index(np.fromiter(inx_gen(), dtype=data[0].dtype), name=index_col[0])

#         if isinstance(na_values, dict):
#             for label in na_values:
#                 if (i := cols.index(label, False)) >= 0:
#                     for line in data:
#                         if line[i] in na_values[label]:
#                             line[i] = np.nan

#         return DataFrame(data, columns=cols, index=inx)
    
#     if chunksize is not None:
#         with open(fname) as file:
#             r_inx = 0
#             def chunkgen():
#                 data = []
#                 data_len = 0

#                 for line in file:

#                     if r_inx in skiprows:
#                         continue

                    
#         return chunkgen()

#     def gen():
#         nonlocal nrows
#         with open(fname) as file:
#             for i, line in enumerate(file):
#                 if i in skiprows:
#                     continue
#                 yield np.array(re.split(sep, line.strip()))

#                 nrows -= 1
#                 if nrows == 0:
#                     break

#     return data_to_dataframe(gen)
    # def gen():
    #     nonlocal nrows
    #     with open(fname) as file:
    #         for i, line in enumerate(file):
    #             if i in skiprows:
    #                 continue

    #             yield np.array(re.split(sep, line.strip()))
    #             nrows -= 1
    #             if nrows == 0:
    #                 break

    # normalized_data = np.fromiter(gen(), dtype=np.object_)


# def read_csv(fname, sep=',', header=0, names=None, index_col=None, skiprows=[], 
#              na_values=[], keep_default_na=True, nrows=-1):
#     rows = []
#     with open(fname) as file:
#         for i, line in enumerate(file):
#             if i in skiprows:
#                 continue
#             rows.append(re.split(sep, line.strip()))
#             nrows -= 1
#             if nrows == 0:
#                 break

#     normalized_data = np.array(rows)

#     if keep_default_na:
#         mask = (normalized_data == '') | (normalized_data == 'NA')
#         if na_values:
#             mask |= np.isin(normalized_data, na_values)
#         normalized_data[mask] = np.nan

#     elif na_values and isinstance(na_values, (list, tuple, set)):
#         mask = np.isin(normalized_data, na_values)
#         normalized_data[mask] = np.nan

#     if names is not None:
#         cols = Index(names)
#     else:
#         if header is None:
#             cols = RangeIndex(normalized_data.shape[1])
#         else:
#             cols = Index(normalized_data[header])
#             normalized_data = np.delete(normalized_data, header, axis=0)

#     if index_col is None:
#         if len(cols) + 1 == normalized_data.shape[1]:
#             inx_data = normalized_data[:, 0]
#             normalized_data = normalized_data[:, 1:]
#             inx = Index(inx_data)
#         else:
#             inx = RangeIndex(normalized_data.shape[0])
#     else:
#         target_i = cols.index(index_col)
#         cols = cols.delete(target_i)
#         inx_data = normalized_data[:, target_i]
#         normalized_data = np.delete(normalized_data, target_i, axis=1)
#         inx = Index(inx_data, name=index_col)

#     if isinstance(na_values, dict):
#         for label, values in na_values.items():
#             if label in cols:
#                 i = cols.index(label)
#                 mask = np.isin(normalized_data[:, i], values)
#                 normalized_data[mask, i] = np.nan

#     return DataFrame(normalized_data, columns=cols, index=inx)