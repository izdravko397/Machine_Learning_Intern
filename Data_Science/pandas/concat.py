from series import Series
from dataframe import DataFrame
from index import Index, RangeIndex, MultiIndex
import numpy as np
from functools import reduce

def concat(objs, *, axis=0, join='outer', ignore_index=False, keys=None, names=None, verify_integrity=False, levels=None):
    if not isinstance(objs, (list, tuple, dict)) or len(objs) < 2:
        raise ValueError('"concat" accepts a sequence of at least 2 objects')
    
    if keys is not None and len(keys) != len(objs):
        raise ValueError('Length of keys must match number of objects')
    
    if names is not None and len(names) != 2:
        raise ValueError('Invalid length for names')
    
    if keys is not None and levels is not None and np.any(~np.isin(keys, levels)):
        raise ValueError('"keys" need to be present in "levels"')
    
    if isinstance(objs, dict):
        keys = []
        dframes = []
        for key, df in objs.items():
            keys.append(key)
            dframes.append(df)

        return concat(dframes, keys=keys, axis=axis, join=join) 
    
    if all(isinstance(obj, Series) for obj in objs):
        if axis == 0 or axis == 'index':
            new_data = objs[0]._data
            new_index = objs[0].index

            for i in range(1, len(objs)):
                obj = objs[i]

                if not ignore_index:
                    new_index = new_index.append(obj.index)

                new_data = np.concatenate([new_data, obj._data])

            if ignore_index:
                new_index = RangeIndex(len(new_data))

            elif verify_integrity and not new_index.is_unique:
                raise ValueError('Indexes have overlapping values')

            if keys is not None:
                multi_lev_1 = []
                for i, key in enumerate(keys):
                    multi_lev_1.extend([key] * len(objs[i]))

                new_index = MultiIndex.from_arrays([multi_lev_1, new_index._data])
            
            return Series(new_data, new_index)
        
        # axis -> 1 or columns
        if keys is not None:
            new_columns = Index(keys)
        else:
            new_columns = RangeIndex(len(objs))

        if join == 'outer':
            new_index = objs[0].index
            for i in range(1, len(objs)):
                new_index = new_index.append(objs[i].index)
            new_index = new_index.unique()

            new_data = {c: {i: np.nan for i in new_index} for c in new_columns}
            for c, s in zip(new_columns, objs):
                for s_inx, s_val in zip(s.index, s):
                    new_data[c][s_inx] = s_val

            return DataFrame(new_data, new_index, new_columns)
        
        # join -> inner
        common_index = reduce(np.intersect1d, [s.index._data for s in objs])
        ser_data_inx = []
        for s in objs:
            inx = s.index.get_indexer(common_index)
            ser_data_inx.append(inx)

        new_data = {i: s._data[ser_data_inx[i]] for i, s in enumerate(objs)}
        return DataFrame(new_data, common_index, new_columns)
        
    if all(isinstance(obj, DataFrame) for obj in objs):
        if axis == 0 or axis == 'index':
            new_col_data = np.concatenate(np.fromiter((df.columns._data for df in objs), np.object_))
            new_columns = Index(new_col_data).unique()

            if not ignore_index:
                all_objs_inxs = np.fromiter((df.index._data for df in objs), np.object_, count=len(objs))
                inx_data = np.concatenate(all_objs_inxs)
                new_index = Index(inx_data)

            else: # ignore_index == True
                inx_len = 0
                for df in objs:
                    inx_len += len(df)

                new_index = RangeIndex(inx_len)

                nested_dict = {i: np.nan for i in new_index}
                new_data = {c: nested_dict.copy() for c in new_columns}

                index_iters = [iter(new_index) for _ in range(len(new_columns))]
                ser_counter = 0
                for df in objs:
                    for s in df:
                        inx_iter = index_iters[ser_counter]
                        ser_counter += 1

                        for ser_val in s._data:
                            new_data[s.name][next(inx_iter)] = ser_val

                    ser_counter = 0

                return DataFrame(new_data, new_index, new_columns)
            
            if verify_integrity and not new_index.is_unique:
                raise ValueError('Indexes have overlapping values')

            df_start_row = {}
            for i, _ in enumerate(objs):
                if i == 0:
                    df_start_row[i] = 0
                    continue
                df_start_row[i] = df_start_row[i - 1] + len(objs[i - 1])

            new_data = [{} for _ in range(len(new_index))]    
            for i, df in enumerate(objs):
                for s, col in zip(df, df.columns):
                    for j, (ser_inx, ser_val) in enumerate(zip(s.index, s._data), start=df_start_row[i]):
                        new_data[j]['inx'] = ser_inx
                        new_data[j][str(col)] = ser_val

            res = DataFrame(new_data).set_index('inx')
            res.index.name = ''
            return res
        
        # axis -> 1 or columns
        new_index = objs[0].index
        new_columns = objs[0].columns

        for i in range(1, len(objs)):
            obj = objs[i]
            new_index = new_index.append(obj.index)
            new_columns = new_columns.append(obj.columns)
        new_index = new_index.unique()

        if ignore_index:
            new_columns = RangeIndex(len(new_columns))

        elif verify_integrity and not new_columns.is_unique:
            raise ValueError('Indexes have overlapping values')

        if keys is not None:
            lv1 = []
            for i, key in enumerate(keys):
                lv1.extend([key] * len(objs[i]._series))

            new_columns = MultiIndex.from_arrays([lv1, new_columns._data], names=names)

        col_freq = {}
        new_data = np.empty((len(new_index), len(new_columns)))
        for df in objs:
            for s in df:
                inxs = new_columns.all_indexes(s.name)
                col_freq[s.name] = col_freq.get(s.name, -1) + 1

                if not len(inxs):
                    continue

                j = inxs[col_freq[s.name]]
                for s_inx, s_val in zip(s.index, s._data):
                    i = new_index.index(s_inx, False)
                    if i == -1:
                        continue

                    new_data[i, j] = s_val

        return DataFrame(new_data, new_index, new_columns)

