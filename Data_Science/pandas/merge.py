from dataframe import DataFrame
from series import Series
import numpy as np
import pandas as pd
from index import Index, RangeIndex, MultiIndex
from collections import defaultdict

def _inner_optimized(left_key, right_key):
    right_map = defaultdict(list)
    for j, val in enumerate(right_key):
        right_map[val].append(j)

    left_res_inx = []
    right_res_inx = []

    for i, val in enumerate(left_key):
        if val in right_map:
            matches = right_map[val]
            left_res_inx.extend([i] * len(matches))
            right_res_inx.extend(matches)

    return left_res_inx, right_res_inx


def _inner(left_key, right_key):
    inner = np.where(np.isin(left_key, right_key))[0]
    left_res_inx = []
    right_res_inx = []

    for i in inner:
        matches = np.where(right_key == left_key[i])[0]
        left_res_inx.extend([i] * len(matches))
        right_res_inx.extend(matches)

    return left_res_inx, right_res_inx

def tuple_gen(df, inx):
    for row_i in range(len(df.index)):
        row = [df._series[ser_i]._data[row_i] for ser_i in inx]
        yield tuple(row)

def merge(left: DataFrame, right: DataFrame, on=None, left_on=None, right_on=None, left_index=False, right_index=False, how='inner', suffixes=("_x", "_y")):
    # valdate
    if on is not None and (left_on is not None or right_on is not None):
        raise ValueError("Cannot pass both 'on' and 'left_on'/'right_on'.")
    
    if on is not None and (left_index or right_index):
        raise ValueError("Cannot pass both 'on' and 'left_index'/'right_index'.")

    if (left_on is not None and left_index) or \
       (right_on is not None and right_index):
        raise ValueError('Can only pass argument "left_on/right_on" OR "left_index/right_index", not both.')
    
    if not isinstance(suffixes, tuple) or len(suffixes) != 2:
        raise ValueError("'suffixes' must be a tuple with length-2")
    
    # get keys
    common_cols_f = False
    if on is not None:
        if isinstance(on, list):
            left_i = [left.columns.index(l) for l in on]
            right_i = [right.columns.index(l) for l in on]
                
            left_key = np.fromiter(tuple_gen(left, left_i), dtype=np.object_)
            right_key = np.fromiter(tuple_gen(right, right_i), dtype=np.object_)

        else:
            left_i = left.columns.index(on)
            right_i = right.columns.index(on)

            left_key = left._series[left_i]._data
            right_key = right._series[right_i]._data

    elif left_on is not None and right_on is not None:
        left_i = left.columns.index(left_on)
        right_i = right.columns.index(right_on)

        left_key = left._series[left_i]._data
        right_key = right._series[right_i]._data

    elif left_index and right_index:
        left_key = left.index._data
        right_key = right.index._data

    elif left_on is not None and right_index:
        if isinstance(left_on, list):
            left_i = [left.columns.index(l) for l in left_on]
            left_key = np.fromiter(tuple_gen(left, left_i), dtype=np.object_)

            if not isinstance(right.index, MultiIndex) or right.index.nlevels != len(left_key[0]):
                raise ValueError("'left_on' and 'right_index' require right_index to be a MultiIndex with correct nlevels")
        else:
            left_i = left.columns.index(left_on)
            left_key = left._series[left_i]._data

        right_key = right.index._data

    elif right_on is not None and left_index:
        right_i = right.columns.index(right_on)
        right_key = right._series[right_i]._data
        left_key = left.index._data

    else:
        common_cols_f = True
        mask = np.isin(left.columns._data, right.columns._data)
        all_common_col_label = left.columns[mask]

        if not len(all_common_col_label):
            raise ValueError('No common columns to perform merge on.')
        
        left_i = [left.columns.index(l) for l in all_common_col_label]
        right_i = [right.columns.index(l) for l in all_common_col_label]
            
        left_key = np.fromiter(tuple_gen(left, left_i), dtype=np.object_)
        right_key = np.fromiter(tuple_gen(right, right_i), dtype=np.object_)

    # comparison
    if how == 'inner':
        left_res_inx, right_res_inx = _inner_optimized(left_key, right_key)

    elif how == 'outer':
        left_res_inx, right_res_inx = _inner(left_key, right_key)

        left_outer = np.where(~np.isin(left_key, right_key))[0]
        left_res_inx.extend(left_outer)
        right_res_inx.extend([-1] * len(left_outer))

        right_outer = np.where(~np.isin(right_key, left_key))[0]
        left_res_inx.extend([-1] * len(right_outer))
        right_res_inx.extend(right_outer)

    elif how == 'left':
        left_res_inx, right_res_inx = _inner(left_key, right_key)

        left_outer = np.where(~np.isin(left_key, right_key))[0]
        left_res_inx.extend(left_outer)
        right_res_inx.extend([-1] * len(left_outer))

    elif how == 'right':
        left_res_inx, right_res_inx = _inner(left_key, right_key)

        right_outer = np.where(~np.isin(left_key, right_key))[0]
        left_res_inx.extend([-1] * len(left_outer))
        right_res_inx.extend(right_outer)

    else:
        raise ValueError('invalid how')

    # set index
    if left_index and right_index:
        if how == 'outer':
            new_index = left.index.union(right.index)

        elif how == 'inner':
            new_index = np.intersect1d(left_key, right_key)

        elif how == 'left':
            new_index = np.sort(left_key)

        elif how == 'right':
            new_index = np.sort(right_key)

        left_res_inx = [left.index.index(l, False) for l in new_index]
        right_res_inx = [right.index.index(l, False) for l in new_index]
    else:
        new_index = RangeIndex(len(left_res_inx))

    #set columns
    if on is not None or common_cols_f:
        new_columns = left.columns.append(right.columns.delete(right_i))

    else:
        new_columns = left.columns.append(right.columns)

    # get series with correct values
    on_flag = False
    if on is not None:
        on_flag = True
        on = set(on) if isinstance(on, list) else {on}

    elif common_cols_f:
        on_flag = True
        on = set(all_common_col_label)

    elif left_on:
        on_flag = True
        on = set(left_on) if isinstance(left_on, list) else {left_on}

    if not left_index:
        left_i = set(left_i) if isinstance(left_i, list) else {left_i}

    if not right_index:
        right_i = set(right_i) if isinstance(right_i, list) else {right_i}

    key_col_counter = 0
    new_data_l = []
    mask_l = np.array(left_res_inx) == -1
    for i, ser in enumerate(left._series):
        s_data = ser._data[left_res_inx]
        if not np.issubdtype(s_data.dtype, np.floating):
            s_data = s_data.astype(np.object_)
        s_data[mask_l] = np.nan

        if on_flag and i in left_i:
            nan_inx = np.where(mask_l)[0]
            for ind in nan_inx:
                val = right_key[right_res_inx[ind]]
                if isinstance(val, tuple):
                    val = val[key_col_counter]
                s_data[ind] = val

            if np.all(isinstance(x, int) for x in s_data):
                s_data = s_data.astype(int)

            key_col_counter += 1

        s_name = ser.name
        if s_name in right.columns and i not in left_i:
            s_name = s_name + suffixes[0]
            col_inx = new_columns.all_indexes(ser.name)[0]

            new_col_data = new_columns._data
            new_col_data[col_inx] = s_name
            new_columns = Index(new_col_data, new_columns.name)

        new_s = Series(s_data, new_index, s_name)
        new_data_l.append(new_s)


    new_data_r = []
    mask_r = np.array(right_res_inx) == -1
    for ser in right._series:
        if on_flag and ser.name in on:
            continue

        s_data = ser._data[right_res_inx]
        if not np.issubdtype(s_data.dtype, np.floating):
            s_data = s_data.astype(np.object_)

        s_data[mask_r] = np.nan

        s_name = ser.name
        if s_name in left.columns and i not in right_i:
            s_name = s_name + suffixes[1]
            col_inx = new_columns.all_indexes(ser.name)[0]

            new_col_data = new_columns._data
            new_col_data[col_inx] = s_name
            new_columns = Index(new_col_data, new_columns.name)

        new_s = Series(s_data, new_index, s_name)
        new_data_r.append(new_s)

    new_common_data = new_data_l + new_data_r
    return DataFrame(new_common_data, new_index, new_columns)





# def merge_2(left: DataFrame, right: DataFrame, on=None, left_on=None, right_on=None, left_index=False, right_index=False, how='inner'):  # "outer", "left", "right"
#     if on is not None and (left_on is not None or right_on is not None):
#         raise ValueError("Cannot pass both 'on' and 'left_on'/'right_on'.")
    
#     if how == 'inner':
#         if left_on is not None and right_on is not None:
#             left_inx = left.columns.index(left_on)
#             right_inx = right.columns.index(right_on)

#             left_col = left._series[left_inx]
#             right_col = right._series[right_inx]

#             left_inner = np.where(np.isin(left_col._data, right_col._data))[0]

#             new_right_res_index = []
#             new_left_res_index = []

#             for i in left_inner:
#                 matches = np.where(right_col._data == left_col._data[i])[0]
#                 for m in matches:
#                     new_left_res_index.append(i)
#                     new_right_res_index.append(m)

#             new_index_labels = Index(left.index._data[new_left_res_index], left.index.name)
#             new_cols = left.columns.append(right.columns)

#             new_data_l = [Series(s._data[new_left_res_index], new_index_labels, s.name) for s in left._series]
#             new_data_r = [Series(s._data[new_right_res_index], new_index_labels, s.name) for s in right._series]

#             new_common_data = new_data_l + new_data_r

#             return DataFrame(new_common_data, new_index_labels, new_cols)

#         if on is not None:
#             common = on

#         else: # all params is None
#             mask = np.isin(left.columns._data, right.columns._data)
#             all_common_col_label = left.columns[mask]

#             if not len(all_common_col_label):
#                 raise ValueError('No common columns to perform merge on.')
            
#             common = all_common_col_label[0]

#         left_inx = left.columns.index(common)
#         right_inx = right.columns.index(common)

#         left_col = left._series[left_inx]
#         right_col = right._series[right_inx]

#         left_inner = np.where(np.isin(left_col._data, right_col._data))[0]

#         new_right_res_index = []
#         new_left_res_index = []

#         for i in left_inner:
#             matches = np.where(right_col._data == left_col._data[i])[0]
#             for m in matches:
#                 new_left_res_index.append(i)
#                 new_right_res_index.append(m)

#         new_index_labels = Index(left.index._data[new_left_res_index], left.index.name)
#         new_cols = left.columns.append(right.columns.delete(right_inx))

#         new_data_l = [Series(s._data[new_left_res_index], new_index_labels, s.name) for s in left._series]
#         new_data_r = [Series(s._data[new_right_res_index], new_index_labels, s.name) 
#                             for s in right._series if s is not right_col]

#         new_common_data = new_data_l + new_data_r

#         return DataFrame(new_common_data, new_index_labels, new_cols)

#     if how == 'outer': # ====================================== outer ==================
#         if left_on is not None and right_on is not None:
#             left_inx = left.columns.index(left_on)
#             right_inx = right.columns.index(right_on)

#             left_col = left._series[left_inx]
#             right_col = right._series[right_inx]

#             mask = np.isin(left_col._data, right_col._data)
#             target_col_l = left_col._data[mask]
#             target_col_r = left_col._data[mask]
#             left_inner = np.where(mask)[0]

#             new_right_res_index = []
#             new_left_res_index = []

#             for i in left_inner:
#                 matches = np.where(right_col._data == left_col._data[i])[0]
#                 for m in matches:
#                     new_left_res_index.append(i)
#                     new_right_res_index.append(m)

            
#             mask = ~np.isin(left_col._data, right_col._data)
#             l_outer = left_col._data[mask]
#             target_col_l = np.append(target_col_l, l_outer)
#             target_col_r = np.append(target_col_r, [np.nan] * len(l_outer))

#             left_outer = np.where(mask)[0]
#             new_left_res_index.extend(left_outer)
#             new_right_res_index.extend([-1] * len(left_outer))


#             mask = ~np.isin(right_col._data, left_col._data)
#             r_outer = right_col._data[mask]
#             target_col_r = np.append(target_col_r, r_outer)
#             target_col_l = np.append(target_col_l, [np.nan] * len(r_outer))

#             right_outer = np.where(mask)[0]
#             new_left_res_index.extend([-1] * len(left_outer))
#             new_right_res_index.extend(right_outer)


#             new_index_labels = RangeIndex(len(new_left_res_index))
#             new_cols = left.columns.append(right.columns)

#             new_data_l = []
#             mask_l = np.array(new_left_res_index) == -1
#             for s in left._series:
#                 if s is left_col:
#                     new_data_l.append(Series(target_col_l, new_index_labels, left_col.name))
#                     continue

#                 s_data = s._data[new_left_res_index]
#                 if not np.issubdtype(s_data.dtype, np.floating):
#                     s_data = s_data.astype(np.object_)

#                 s_data[mask_l] = np.nan
#                 new_s = Series(s_data, new_index_labels, s.name)
#                 new_data_l.append(new_s)

#             new_data_r = []
#             mask_r = np.array(new_right_res_index) == -1
#             for s in right._series:
#                 if s is right_col:
#                     new_data_r.append(Series(target_col_r, new_index_labels, right_col.name))
#                     continue

#                 s_data = s._data[new_right_res_index]
#                 if not np.issubdtype(s_data.dtype, np.floating):
#                     s_data = s_data.astype(np.object_)

#                 s_data[mask_r] = np.nan
#                 new_s = Series(s_data, new_index_labels, s.name)
#                 new_data_r.append(new_s)

#             new_common_data = new_data_l + new_data_r

#             return DataFrame(new_common_data, new_index_labels, new_cols)

#         if on is not None and isinstance(on, list):
#             left_cinx = [left.columns.index(l) for l in on]
#             right_cinx = [right.columns.index(l) for l in on]

#             left_cols = [left._series[i] for i in left_cinx]
#             right_cols = [right._series[i] for i in right_cinx]

#             def gen(rng, cols):
#                 for i in range(len(rng)):
#                     row = [s[i] for s in cols]
#                     yield tuple(row)

#             left_tuples = np.fromiter(gen(left.index, left_cols), dtype=np.object_)
#             right_tuples = np.fromiter(gen(right.index, right_cols), dtype=np.object_)

#             mask = np.isin(left_tuples, right_tuples)
#             target_cols = list(left_tuples[mask])
#             left_inner = np.where(mask)[0]

#             new_right_res_index = []
#             new_left_res_index = []

#             for i in left_inner:
#                 matches = [j for j, rt in enumerate(right_tuples) if rt == left_tuples[i]]
#                 for m in matches:
#                     new_left_res_index.append(i)
#                     new_right_res_index.append(m)

#                 target_cols.extend([right_tuples[i]] * (len(matches) - 1))

#             mask = ~np.isin(left_tuples, right_tuples)
#             target_cols.extend(left_tuples[mask])
#             left_outer = np.where(mask)[0]
#             new_left_res_index.extend(left_outer)
#             new_right_res_index.extend([-1] * len(left_outer))


#             mask = ~np.isin(right_tuples, left_tuples)
#             target_cols.extend(right_tuples[mask])
#             right_outer = np.where(mask)[0]
#             new_left_res_index.extend([-1] * len(right_outer))
#             new_right_res_index.extend(right_outer)


#             new_index_labels = RangeIndex(len(target_cols))
#             new_cols = left.columns.append(right.columns.delete(right_cinx))

#             new_data_targers = []
#             for i, data in enumerate(zip(*target_cols)):
#                 t_ser = Series(data, new_index_labels, on[i])
#                 new_data_targers.append(t_ser)

#             new_data_l = []
#             mask_l = np.array(new_left_res_index) == -1
#             for i, s in enumerate(left._series):
#                 if i in left_cinx:
#                     continue

#                 s_data = s._data[new_left_res_index]
#                 if not np.issubdtype(s_data.dtype, np.floating):
#                     s_data = s_data.astype(np.object_)

#                 s_data[mask_l] = np.nan
#                 new_s = Series(s_data, new_index_labels, s.name)
#                 new_data_l.append(new_s)

#             new_data_r = []
#             mask_r = np.array(new_right_res_index) == -1
#             for i, s in enumerate(right._series):
#                 if i in right_cinx:
#                     continue

#                 s_data = s._data[new_right_res_index]
#                 if not np.issubdtype(s_data.dtype, np.floating):
#                     s_data = s_data.astype(np.object_)

#                 s_data[mask_r] = np.nan
#                 new_s = Series(s_data, new_index_labels, s.name)
#                 new_data_r.append(new_s)

#             new_common_data = new_data_targers + new_data_l + new_data_r

#             return DataFrame(new_common_data, new_index_labels, new_cols)

#         if on is not None and not isinstance(on, list):
#             common = on

#         else: # all params is None
#             mask = np.isin(left.columns._data, right.columns._data)
#             all_common_col_label = left.columns[mask]

#             if not len(all_common_col_label):
#                 raise ValueError('No common columns to perform merge on.')
            
#             common = all_common_col_label[0]

#         left_inx = left.columns.index(common)
#         right_inx = right.columns.index(common)

#         left_col = left._series[left_inx]
#         right_col = right._series[right_inx]

#         mask = np.isin(left_col._data, right_col._data)
#         target_col = left_col._data[mask]
#         left_inner = np.where(mask)[0]

#         new_right_res_index = []
#         new_left_res_index = []

#         for i in left_inner:
#             matches = np.where(right_col._data == left_col._data[i])[0]
#             for m in matches:
#                 new_left_res_index.append(i)
#                 new_right_res_index.append(m)

        
#         mask = ~np.isin(left_col._data, right_col._data)
#         target_col = np.append(target_col, left_col._data[mask])
#         left_outer = np.where(mask)[0]
#         new_left_res_index.extend(left_outer)
#         new_right_res_index.extend([-1] * len(left_outer))


#         mask = ~np.isin(right_col._data, left_col._data)
#         target_col = np.append(target_col, right_col._data[mask])
#         right_outer = np.where(mask)[0]
#         new_left_res_index.extend([-1] * len(left_outer))
#         new_right_res_index.extend(right_outer)


#         new_index_labels = RangeIndex(len(new_left_res_index))
#         new_cols = left.columns.append(right.columns.delete(right_inx))

#         new_data_l = []
#         mask_l = np.array(new_left_res_index) == -1
#         for s in left._series:
#             if s is left_col:
#                 new_data_l.append(Series(target_col, new_index_labels, left_col.name))
#                 continue

#             s_data = s._data[new_left_res_index]
#             if not np.issubdtype(s_data.dtype, np.floating):
#                 s_data = s_data.astype(np.object_)

#             s_data[mdef _inner(left_key, right_key):
#     inner = np.where(np.isin(left_key, right_key))[0]
#     left_res_inx = []
#     right_res_inx = []
    
#     for i in inner:
#         # matches = np.where(right_key == left_key[i])[0]
#         matches = [j for j, rval in enumerate(right_key) if rval == left_key[i]]
#         left_res_inx.extend([i] * len(matches))
#         right_res_inx.extend(matches)

#     return left_res_inx, right_res_inx

# def merge(left: DataFrame, right: DataFrame, on=None, left_on=None, right_on=None, left_index=False, right_index=False, how='inner'):
#     # valdate
#     if on is not None and (left_on is not None or right_on is not None):
#         raise ValueError()
    
#     if on is not None and (left_index or right_index):
#         raise ValueError()

#     if (left_on is not None and left_index) or \
#        (right_on is not None and right_index):
#         raise ValueError()
    
#     # get keys
#     common_cols_f = False
#     if on is not None:
#         if isinstance(on, list):
#             left_i = [left.columns.index(l) for l in on]
#             right_i = [right.columns.index(l) for l in on]

#             def tuple_gen(df, inx):
#                 for row_i in range(len(df.index)):
#                     row = [df._series[ser_i]._data[row_i] for ser_i in inx]
#                     yield tuple(row)
                
#             left_key = np.fromiter(tuple_gen(left, left_i), dtype=np.object_)
#             right_key = np.fromiter(tuple_gen(right, right_i), dtype=np.object_)

#         else:
#             left_i = left.columns.index(on)
#             right_i = right.columns.index(on)

#             left_key = left._series[left_i]._data
#             right_key = right._series[right_i]._data

#     elif left_on is not None and right_on is not None:
#         left_i = left.columns.index(left_on)
#         right_i = right.columns.index(right_on)

#         left_key = left._series[left_i]._data
#         right_key = right._series[right_i]._data

#     elif left_index and right_index:
#         left_key = left.index._data
#         right_key = right.index._data

#     elif left_on is not None and right_index:
#         left_i = left.columns.index(left_on)
#         left_key = left._series[left_i]._data
#         right_key = right.index._data

#     elif right_on is not None and left_index:
#         right_i = right.columns.index(right_on)
#         right_key = right._series[right_i]._data
#         left_key = left.index._data

#     else:
#         common_cols_f = True
#         mask = np.isin(left.columns._data, right.columns._data)
#         all_common_col_label = left.columns[mask]

#         if not len(all_common_col_label):
#             raise ValueError('No common columns to perform merge on.')
        
#         left_i = [left.columns.index(l) for l in all_common_col_label]
#         right_i = [right.columns.index(l) for l in all_common_col_label]

#         def tuple_gen(df, inx):
#             for row_i in range(len(df.index)):
#                 row = [df._series[ser_i]._data[row_i] for ser_i in inx]
#                 yield tuple(row)
            
#         left_key = np.fromiter(tuple_gen(left, left_i), dtype=np.object_)
#         right_key = np.fromiter(tuple_gen(right, right_i), dtype=np.object_)

#     # print(left_key, right_key)
#     # comparison
#     if how == 'inner':
#         left_res_inx, right_res_inx = _inner(left_key, right_key)

#     elif how == 'outer':
#         left_res_inx, right_res_inx = _inner(left_key, right_key)

#         left_outer = np.where(~np.isin(left_key, right_key))[0]
#         left_res_inx.extend(left_outer)
#         right_res_inx.extend([-1] * len(left_outer))

#         right_outer = np.where(~np.isin(right_key, left_key))[0]
#         left_res_inx.extend([-1] * len(right_outer))
#         right_res_inx.extend(right_outer)

#     elif how == 'left':
#         left_res_inx, right_res_inx = _inner(left_key, right_key)

#         left_outer = np.where(~np.isin(left_key, right_key))[0]
#         left_res_inx.extend(left_outer)
#         right_res_inx.extend([-1] * len(left_outer))

#     elif how == 'right':
#         left_res_inx, right_res_inx = _inner(left_key, right_key)

#         right_outer = np.where(~np.isin(left_key, right_key))[0]
#         left_res_inx.extend([-1] * len(left_outer))
#         right_res_inx.extend(right_outer)

#     else:
#         raise ValueError('invalid how')

#     # print(left_res_inx, right_res_inx)
#     # set index
#     if left_index and right_index:
#         if how == 'outer':
#             new_index = left.index.union(right.index)

#         elif how == 'inner':
#             new_index = np.intersect1d(left_key, right_key)

#         elif how == 'left':
#             new_index = np.sort(left_key)

#         elif how == 'right':
#             new_index = np.sort(right_key)

#         left_res_inx = [left.index.index(l, False) for l in new_index]
#         right_res_inx = [right.index.index(l, False) for l in new_index]
#     else:
#         new_index = RangeIndex(len(left_res_inx))

#     #set columns
#     if on is not None or common_cols_f:
#         new_columns = left.columns.append(right.columns.delete(right_i))

#     else:
#         new_columns = left.columns.append(right.columns)

#     # get series with corect values
#     on_flag = False
#     if on is not None:
#         on_flag = True
#         on = set(on) if isinstance(on, list) else {on}

#     elif common_cols_f:
#         on_flag = True
#         on = set(all_common_col_label)

#     if not left_index:
#         left_i = set(left_i) if isinstance(left_i, list) else {left_i}

#     if not right_index:
#         right_i = set(right_i) if isinstance(right_i, list) else {right_i}


#     key_col_counter = 0
#     new_data_l = []
#     mask_l = np.array(left_res_inx) == -1
#     for i, ser in enumerate(left._series):
#         s_data = ser._data[left_res_inx]
#         if not np.issubdtype(s_data.dtype, np.floating):
#             s_data = s_data.astype(np.object_)

#         s_data[mask_l] = np.nan

#         if on_flag and i in left_i:
#             nan_inx = np.where(mask_l)[0]
#             for ind in nan_inx:
#                 val = right_key[right_res_inx[ind]]
#                 if isinstance(val, tuple):
#                     val = val[key_col_counter]

#                 s_data[ind] = val

#             key_col_counter += 1

#         new_s = Series(s_data, new_index, ser.name)
#         new_data_l.append(new_s)


#     new_data_r = []
#     mask_r = np.array(right_res_inx) == -1
#     for ser in right._series:
#         if on_flag and ser.name in on:
#             continue

#         s_data = ser._data[right_res_inx]
#         if not np.issubdtype(s_data.dtype, np.floating):
#             s_data = s_data.astype(np.object_)

#         s_data[mask_r] = np.nan
#         new_s = Series(s_data, new_index, ser.name)
#         new_data_r.append(new_s)

#     new_common_data = new_data_l + new_data_r
#     return DataFrame(new_common_data, new_index, new_columns)ask_l] = np.nan
#             new_s = Series(s_data, new_index_labels, s.name)
#             new_data_l.append(new_s)

#         new_data_r = []
#         mask_r = np.array(new_right_res_index) == -1
#         for s in right._series:
#             if s is right_col:
#                 continue

#             s_data = s._data[new_right_res_index]
#             if not np.issubdtype(s_data.dtype, np.floating):
#                 s_data = s_data.astype(np.object_)

#             s_data[mask_r] = np.nan
#             new_s = Series(s_data, new_index_labels, s.name)
#             new_data_r.append(new_s)

#         new_common_data = new_data_l + new_data_r

#         return DataFrame(new_common_data, new_index_labels, new_cols)

#     if how == 'left': # ====================================== left ==================
#         if on is not None:
#             common = on

#         else: # all params is None
#             mask = np.isin(left.columns._data, right.columns._data)
#             all_common_col_label = left.columns[mask]

#             if not len(all_common_col_label):
#                 raise ValueError('No common columns to perform merge on.')
            
#             common = all_common_col_label[0]

#         left_inx = left.columns.index(common)
#         right_inx = right.columns.index(common)

#         left_col = left._series[left_inx]
#         right_col = right._series[right_inx]

#         mask = np.isin(left_col._data, right_col._data)
#         target_col = left_col._data[mask]
#         left_inner = np.where(mask)[0]

#         new_right_res_index = []
#         new_left_res_index = []

#         for i in left_inner:
#             matches = np.where(right_col._data == left_col._data[i])[0]
#             for m in matches:
#                 new_left_res_index.append(i)
#                 new_right_res_index.append(m)

#             target_col = np.append(target_col, [left_col._data[i]] * (len(matches) - 1))

#         print(target_col)

#         mask = ~np.isin(left_col._data, right_col._data)
#         target_col = np.append(target_col, left_col._data[mask])
#         left_outer = np.where(mask)[0]
#         new_left_res_index.extend(left_outer)
#         new_right_res_index.extend([-1] * len(left_outer))

#         new_index_labels = RangeIndex(len(new_left_res_index))
#         new_cols = left.columns.append(right.columns.delete(right_inx))

#         new_data_l = []
#         mask_l = np.array(new_left_res_index) == -1
#         for s in left._series:
#             if s is left_col:
#                 new_data_l.append(Series(target_col, new_index_labels, left_col.name))
#                 continue

#             s_data = s._data[new_left_res_index]
#             if not np.issubdtype(s_data.dtype, np.floating):
#                 s_data = s_data.astype(np.object_)

#             s_data[mask_l] = np.nan
#             new_s = Series(s_data, new_index_labels, s.name)
#             new_data_l.append(new_s)

#         new_data_r = []
#         mask_r = np.array(new_right_res_index) == -1
#         for s in right._series:
#             if s is right_col:
#                 continue

#             s_data = s._data[new_right_res_index]
#             if not np.issubdtype(s_data.dtype, np.floating):
#                 s_data = s_data.astype(np.object_)

#             s_data[mask_r] = np.nan
#             new_s = Series(s_data, new_index_labels, s.name)
#             new_data_r.append(new_s)

#         new_common_data = new_data_l + new_data_r

#         return DataFrame(new_common_data, new_index_labels, new_cols)

#     if how == 'right':
#         pass

#     raise ValueError(f'Invalid how must be one of "outer", "left", "right", "inner"; not: {how}')
