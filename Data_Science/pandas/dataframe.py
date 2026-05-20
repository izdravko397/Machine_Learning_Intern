from index import RangeIndex, Index, MultiIndex
from series import Series, SeriesGroupBy
import numpy as np
import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class DataFrameGroupBy:
    funcs = {
        'min':   lambda x: x.min(),
        'max':   lambda x: x.max(),
        'count': lambda x: x.count(),
        'sum':   lambda x: x.sum(),
        'mean':  lambda x: x.mean(),
        'size':  lambda x: x.size(),
    }

    def __init__(self, df, groups: dict, key_name: str, axis=0, is_multi=False, group_keys=True, as_index=True):
        self.df = df
        self.groups = groups
        self._key_name = key_name
        self._axis = axis
        self._is_multi = is_multi
        self.group_keys = group_keys
        self.as_index = as_index

    def __getitem__(self, key):
        if isinstance(key, str):
            return SeriesGroupBy(self.df[key], self.groups, self._key_name,
                                  self._is_multi, self.group_keys)

        raise TypeError(f'Invalid GroupBy indexing {type(key).__name__}')

    def _group_reduce(self, kind):
        df_data = []

        if self._axis == 0:
            for s in self.df._series:
                # if not np.issubdtype(s._dtype, np.number) and kind != 'count':
                #     continue
                
                if self._is_multi:
                    if s.name in self._key_name:
                        continue
                else:
                    if s.name == self._key_name:
                        continue

                s_gby = SeriesGroupBy(s, self.groups, self._key_name, self._is_multi)
                
                vals = self.funcs[kind](s_gby)
                df_data.append(vals)

            res = DataFrame(df_data)
            if not self.as_index:
                res = res.reset_index()

            return res
        
        # axis == 1
        for i in range(len(self.df)):
            row_ser = self.df.iloc[i]
            s_gby_obj = SeriesGroupBy(row_ser, self.groups, self._key_name, self._is_multi)
            agg_ser = self.funcs[kind](s_gby_obj)
            df_data.append(agg_ser._data)

        df_cols = list(self.groups.keys())
        return DataFrame(df_data, self.df.index, df_cols)

    def mean(self):
        return self._group_reduce('mean')
    
    def min(self):
        return self._group_reduce('min')

    def max(self):
        return self._group_reduce('max')
    
    def sum(self):
        return self._group_reduce('sum')
    
    def count(self):
        return self._group_reduce('count')
    
    def size(self):
        ser_data = []
        inx_data = []
        
        for key, inxs in self.groups.items():
            ser_data.append(len(inxs))
            inx_data.append(key)

        if self._is_multi:
            ser_inx = MultiIndex.from_tuples(inx_data, names=self._key_name)
        else:
            ser_inx = Index(inx_data, name=self._key_name)
        
        return Series(ser_data, index=ser_inx)
    
    def transform(self, func):
        df_data = []
        if self._axis == 0:
            for ser in self.df._series:
                if self._is_multi:
                    if ser.name in self._key_name:
                        continue
                else:
                    if ser.name == self._key_name:
                        continue

                s_gby_obj = SeriesGroupBy(ser, self.groups, self._key_name, self._is_multi)
                new_s = s_gby_obj.transform(func)
                df_data.append(new_s)

            return DataFrame(df_data)
        
        # axis == 1
        for i in range(len(self.df)):
            ser_row = self.df.iloc[i]
            s_gby_obj = SeriesGroupBy(ser_row, self.groups, self._key_name, self._is_multi)
            new_s = s_gby_obj.transform(func)
            df_data.append(new_s._data)

        return DataFrame(df_data, self.df.index, self.df.columns)

    def _agg_single_func(self, func):
        if isinstance(func, str):
            return self.funcs[func](self)
        
        if not callable(func):
            raise TypeError('Invalid type for "func"')

        if self._axis == 0:
            df_data = []
            for s in self.df._series:
                if self._is_multi:
                    if s.name in self._key_name:
                        continue
                else:
                    if s.name == self._key_name:
                        continue

                s_gby_obj = SeriesGroupBy(s, self.groups, 
                            self._key_name, self._is_multi)
                
                df_data.append(s_gby_obj.agg(func))

            return DataFrame(df_data)

        # callable axis == 1
        df_col_data = []
        df_data = []

        for g_name, inxs in self.groups.items():
            df_col_data.append(g_name)

            group_ser_data = []
            func_param_inx = self.df.columns._data[inxs]
            for row_val in zip(*(self.df._series[i]._data for i in inxs)):
                func_param = Series(row_val, func_param_inx)
                func_res = func(func_param)

                if not np.isscalar(func_res):
                    raise ValueError('Function did not reduce')
            
                group_ser_data.append(func_res)
            df_data.append(Series(group_ser_data, self.df.index, name=g_name))

        
        if self._is_multi:
            df_col = MultiIndex.from_tuples(df_col_data, names=self._key_name)
        else:
            df_col = Index(df_col_data, name=self._key_name)

        return DataFrame(df_data, index=self.df.index, columns=df_col)

    def _agg_col_funcs(self, ser, func, make_multi=True):
        s_gby_obj = SeriesGroupBy(ser, self.groups, self._key_name, self._is_multi)
        res_df = s_gby_obj.agg(func)

        if make_multi:
            level1 = [ser.name for _ in range(len(func))]
            level2 = res_df.columns._data
            cols = MultiIndex.from_arrays([level1, level2], names=[self.df.columns.name, ''])
        else:
            cols = Index([ser.name])

        res_df.columns = cols
        return res_df

    def agg(self, func):
        if not isinstance(func, (list, str, dict)) and not callable(func):
            raise TypeError('Invalid type for "func"')

        if not isinstance(func, (list, dict)):
            return self._agg_single_func(func)

        from concat import concat
        if isinstance(func, dict):
            if self._axis == 1:
                raise ValueError('"agg" with func param dict is posible only in groupby(axis -> 1/"index")')
            
            multi_cols = False
            if any(isinstance(v, list) for v in func.values()):
                multi_cols = True
            
            dfs = []
            for c_name, func in func.items():
                ser = self.df[c_name]
                funcs_list = func if isinstance(func, list) else [func]
                dfs.append(self._agg_col_funcs(ser, funcs_list, multi_cols))

            return concat(dfs, axis=1)
        
        # list -> axis == 0
        if self._axis == 0:
            dfs = [self._agg_col_funcs(ser, func) for ser in self.df._series]
            return concat(dfs, axis=1)

        # list -> axis == 1
        sers_data = [[] for _ in range(len(self.groups))]
        df_inx_data = []
        for i in range(len(self.df)):
            for fn in func:
                if isinstance(fn, tuple):
                    f_name = fn[0]
                    fn = fn[1]
                elif isinstance(fn, str):
                    f_name = fn
                elif callable(fn):
                    f_name = fn.__name__
                else:
                    raise TypeError('Invalid func type')
                
                df_inx_data.append((self.df.index._data[i], f_name))

                ser = self.df.iloc[i]
                s_gby_obj = SeriesGroupBy(ser, self.groups, self._key_name, self._is_multi)
                res_ser = s_gby_obj.agg(fn)

                for j, group_val in enumerate(res_ser):
                    sers_data[j].append(group_val)

        df_inx = MultiIndex.from_tuples(df_inx_data, names=[self.df.index.name, ''])
        df_col_data = list(self.groups)

        if self._is_multi:
            df_col = MultiIndex.from_tuples(df_col_data, names=self._key_name)
        else:
            df_col = Index(df_col_data, name=self._key_name)

        df_data = [Series(data, df_inx, df_col_data[i]) for i, data in enumerate(sers_data)]
        return DataFrame(df_data, df_inx, df_col)

    def _apply_make_res_inx(self):
        if self._is_multi:
            names = self._key_name if any(self._key_name) else None
            inx = MultiIndex.from_tuples(list(self.groups.keys()), names)
        else:
            inx = Index(list(self.groups.keys()), self._key_name)

        return inx

    def apply(self, func, *args, **kwargs):
        if not isinstance(func, str) and not callable(func):
            raise TypeError('Invalid type for "func"')

        if isinstance(func, str):
            return self.funcs[func](self)
        
        # func is callable
        if self._axis == 0:
            scalar_results = False
            series_result = False
            dfs_results = False

            new_data = []
            df_multi_inx = [[], []]
            for g_name, inxs in self.groups.items():
                df_func_param = self.df.iloc[inxs]

                if self.group_keys:
                    if self._is_multi:
                        for col_name in self._key_name:
                            if col_name in df_func_param.columns:
                                df_func_param = df_func_param.drop(columns=col_name)
                    else:
                        if self._key_name in df_func_param.columns:
                            df_func_param = df_func_param.drop(columns=self._key_name)

                func_res = func(df_func_param, *args, **kwargs)

                if np.isscalar(func_res):
                    if not scalar_results:
                        scalar_results = True

                    if any([series_result, dfs_results]):
                        raise ValueError('Function returned different types for different groups')
                    
                elif isinstance(func_res, Series):
                    if not series_result:
                        series_result = True

                    if any([scalar_results, dfs_results]):
                        raise ValueError('Function returned different types for different groups')
                    
                elif isinstance(func_res, DataFrame):
                    if not dfs_results:
                        dfs_results = True

                    if any([series_result, scalar_results]):
                        raise ValueError('Function returned different types for different groups')
                    
                    if self.group_keys:
                        df_multi_inx[0].extend([g_name] * len(func_res))
                    df_multi_inx[1].extend(func_res.index._data)

                else:
                    raise TypeError(f'Invalid type from "func" result: {type(func_res)}')
                
                new_data.append(func_res)

            from concat import concat
            if scalar_results:
                inx = self._apply_make_res_inx()
                return Series(new_data, inx)
            
            if series_result:
                first_ser_inx = new_data[0].index._data
                if all(np.array_equal(first_ser_inx, new_data[i].index._data)
                            for i in range(1, len(new_data))):
                    res_df = concat(new_data, axis=1)
                    res_df = res_df.T
                    inx = self._apply_make_res_inx()
                    res_df.index = inx

                    if isinstance(inx, MultiIndex):
                        inxs = np.argsort(inx._data)
                        res_df = res_df.iloc[inxs]

                    return res_df
                
                # new_data is Series with different indexes
                res_ser = concat(new_data)

                lv1_multi = []
                for gname, s in zip(self.groups, new_data):
                    lv1_multi.extend([gname] * len(s.index))

                new_inx = MultiIndex.from_arrays([lv1_multi, res_ser.index._data], 
                                                 [self._key_name, res_ser.index.name])
                
                res_ser.index = new_inx
                res_ser.name = new_data[0].name
                return res_ser
            
            if dfs_results:
                df = concat(new_data)
                
                if self.group_keys:
                    names = None
                    if self._key_name:
                        names = [self._key_name, df.index.name]
                    
                    inx = MultiIndex.from_arrays(df_multi_inx, names)
                else:
                    inx = Index(df_multi_inx[1], df.index.name)

                df.index = inx
                return df
            
        # axis == 1
        scalar_results = False
        series_result = False
        dfs_results = False

        new_data = []
        df_multi_inx = [[], []]
        for g_name, inxs in self.groups.items():
            df_func_param = DataFrame([self.df._series[i] for i in inxs])
            func_res = func(df_func_param, **kwargs)

            if np.isscalar(func_res):
                if not scalar_results:
                    scalar_results = True

                if any([series_result, dfs_results]):
                    raise ValueError('Function returned different types for different groups')
                
            elif isinstance(func_res, Series):
                if not series_result:
                    series_result = True

                if any([scalar_results, dfs_results]):
                    raise ValueError('Function returned different types for different groups')
                
            elif isinstance(func_res, DataFrame):
                if not dfs_results:
                    dfs_results = True

                if any([series_result, scalar_results]):
                    raise ValueError('Function returned different types for different groups')

                df_multi_inx[0].extend([g_name] * len(func_res.columns))
                df_multi_inx[1].extend(func_res.columns._data)

            else:
                raise TypeError(f'Invalid type from "func" result: {type(func_res)}')
            
            new_data.append(func_res)

        from concat import concat
        if scalar_results:
            inx = self._apply_make_res_inx()
            return Series(new_data, inx)
        
        if series_result:
            res_df = concat(new_data, axis=1)
            inx = self._apply_make_res_inx()
            res_df.columns = inx
            return res_df
        
        if dfs_results:
            df = concat(new_data, axis=1)

            names = None
            if self._key_name:
                names = [self._key_name, df.index.name]
            
            df_multi_inx = MultiIndex.from_arrays(df_multi_inx, names)
            df.columns = df_multi_inx
            return df

    def __iter__(self):
        if self._axis == 0:
            for key, inxs in self.groups.items():
                yield key, self.df.iloc[inxs]

        for key, inxs in self.groups.items():
            yield key, DataFrame([self.df._series[i] for i in inxs])



class Loc:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, key):
        if isinstance(key, Index):
            inxs = self.dataframe.index.get_indexer(key._data)
            return self.dataframe.iloc[inxs]

        if isinstance(key, slice):
            if isinstance(self.dataframe.index, MultiIndex):
                inx = MultiIndex.from_tuples(self.dataframe.index._data[key], self.dataframe.index.names)
            else:
                inx = Index(self.dataframe.index._data[key], self.dataframe.index.name)
            
            return DataFrame({s.name: s._data[key] for s in self.dataframe}, 
                             index=inx, 
                             columns=self.dataframe.columns)
        
        if isinstance(key, list):
            return DataFrame([s[key] for s in self.dataframe._series], index=key, columns=self.dataframe.columns)
        
        if isinstance(key, tuple):
            key1, key2 = key

            if not hasattr(key1, '__iter__') or isinstance(key1, str):
                series = self[key1]
                if isinstance(key2, list):
                    return series[key2]

            if isinstance(key1, (list, np.ndarray)) and np.isscalar(key2):
                df = self.dataframe[key2]
                return DataFrame([s[key1] for s in df._series], key1, df.columns)

        if isinstance(self.dataframe.index, MultiIndex):
            mask = np.isin(self.dataframe.index._level_1, [key])
            new_inx = Index(self.dataframe.index._level_2[mask], 
                                             self.dataframe.index.names[1])
            new_data = [Series(s._data[mask], new_inx, s.name) 
                                for s in self.dataframe._series]
            return DataFrame(new_data, new_inx, self.dataframe.columns)
        
        label_inx = self.dataframe.index.all_indexes(key)
        len_label_inx = len(label_inx)

        if not len_label_inx:
            raise IndexError(f'invalid label: {key}')
        
        elif len_label_inx == 1:
            return Series([s._data[label_inx[0]] for s in self.dataframe._series], index=self.dataframe.columns, name=key)
        
        return DataFrame([s[key] for s in self.dataframe._series], index=[key] * len_label_inx, columns=self.dataframe.columns)
    
class Iloc(Loc):
    def __init__(self, dataframe):
        super().__init__(dataframe)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return super().__getitem__(key)
        
        if isinstance(key, tuple):
            key1, key2 = key

            if isinstance(key1, int):
                s = self[key1]

                if isinstance(key2, int):
                    key2 = [key2]

                return Series([s._data[i] for i in key2], index=[s.index[i] for i in key2], name=s.name)
            
            if isinstance(key1, list):
                dframe = self[key1]

                if isinstance(key2, int):
                    return dframe.iloc[key2]

                return DataFrame([dframe._series[i] for i in key2], columns=[dframe.columns[i] for i in key2])
        
        if isinstance(key, (list, np.ndarray)):
            # if len(key) == 1:
            #     return self[key[0]]
            if isinstance(self.dataframe.index, MultiIndex):
                inx = MultiIndex.from_tuples(self.dataframe.index._data[key], self.dataframe.index.names)
            else:
                inx = Index(self.dataframe.index._data[key], self.dataframe.index.name)

            new_data = [Series(s._data[key], inx, s.name) for s in self.dataframe._series]
            return DataFrame(new_data, index=inx, columns=self.dataframe.columns)
        
        if not (0 <= key < len(self.dataframe.index)) or not isinstance(key, int):
            raise IndexError(f'invalid index: {key}')
        
        return Series([s._data[key] for s in self.dataframe._series], index=self.dataframe.columns, 
                      name=self.dataframe.index[key])
    
    def __setitem__(self, key, val):
        if isinstance(key, tuple):
            inx = key[0]
            col = key[1]

            series = self.dataframe._series[col]

            if isinstance(series, Series):
                series.iloc[inx] = val
                return
            
            for s in series:
                s.iloc[inx] = val
            return




class DataFrame:
    def __init__(self, data, index=None, columns=None):
        self._series = []

        if not hasattr(data, '__iter__') or isinstance(data, str):
            raise TypeError('invalid type for data')
        
        if len(data) == 0:
                raise ValueError('Not expected empty sequence')

        if isinstance(data, dict): # list, dict, Series
            if columns is not None and not isinstance(columns, Index):
                columns = Index(columns)
                
            if index is not None and not isinstance(index, Index):
                index = Index(index)
                
            all_vals = iter(data.values())
            first_val = next(all_vals)
            first_val_len = len(first_val)

            if not all(isinstance(val, type(first_val)) for val in all_vals):
                raise TypeError('dict cannot store values of different types')
            
            if isinstance(first_val, (list, np.ndarray)):
                if not all(len(first_val) == len(val) for val in data.values()):
                    raise ValueError('dict values must have same length')
                
                if index is not None:
                    if len(index) < first_val_len:
                        raise ValueError('there cannot be fewer indices than values')
                else:
                    index = RangeIndex(first_val_len)

                if columns is not None:
                    self._series = [Series(data=data.get(name, [None] * first_val_len), index=index, name=name) for name in columns]
                    self._index = index
                    self._columns = columns
                    return
                
                self._series = [Series(data=v, index=index, name=k) for k, v in data.items()]
                self._columns = Index([s.name for s in self._series])
                self._index = index
                return
            
            if isinstance(first_val, dict):
                if index is None:
                    index = Index(sorted(set(key for val in data.values() for key in val)))
                
                if columns is not None:
                    self._series = [Series(data=[data.get(name, {}).get(i, np.nan) for i in index], index=index, name=name) for name in columns]
                    self._index = index
                    self._columns = columns
                    return
                
                self._series = [Series(data=[val.get(i) for i in index], index=index, name=name) for name, val in data.items()]
                self._index = index
                self._columns = Index([s.name for s in self._series])
                return
                    
            if isinstance(first_val, Series):
                if index is None:
                    inx_data = {inx: None for s in data.values() for inx in s.index}
                    index = Index(list(inx_data.keys()))
                    
                if columns is not None:
                    for col in columns:
                        series = data.get(col) or Series(data=[None] * len(index))
                        series.name = col
                        # series.reindex(index)
                        series.index = index
                        self._series.append(series)

                    self._index = index
                    self._columns = columns
                    return
                    
                columns = []
                for name, s in data.items():
                    s.name = name
                    columns.append(name)
                    s.index = index
                    self._series.append(s)

                self._index = index
                self._columns = Index(columns)
                return

        if isinstance(data, (list, np.ndarray, tuple)): # 1D, 2D, Series, dict
            if isinstance(data[0], Series):
                if not all(isinstance(col, Series) for col in data):
                    raise TypeError('the types in the sequence must be the same')
                
                if index is None and columns is None:
                    self._series = data
                    self._index = data[0].index
                    self._columns = Index([s.name for s in data])
                    return
                
                if index is not None:
                    if not isinstance(index, Index):
                        index = Index(index)

                    if len(index) < len(data[0]):
                        raise ValueError(f'{len(index)} index passed, passed data had {len(data[0])} index')
                    
                    # self._series = [s.reindex(index) for s in data]
                    self._series = data
                    for s in self._series:
                        s.index = index

                    self._index = index
                else:
                    self._series = data
                    self._index = data[0].index

                if columns is not None:
                    if not isinstance(columns, Index):
                        columns = Index(columns)

                    if len(columns) < len(data):
                        raise ValueError(f'{len(columns)} columns passed, passed data had {len(data)} columns')
                    
                    self._series = []
                    for col in columns:
                        series = None

                        for s in data:
                            if s.name == col:
                                series = s
                                break

                        if series is None:
                            series = Series(data=[], index=index, name=col)

                        self._series.append(series)
                    self._columns = columns
                    return
                
                self._columns = Index([s.name for s in self._series])
                return

            if isinstance(data[0], dict): # for read json
                if len(data) == 1:
                    dictionary = data[0]
                    first_val = next(iter(dictionary.values()))
                    is_scalar = True

                    if hasattr(first_val, '__iter__') and not isinstance(first_val, str):
                        is_scalar = False
                        first_val_len = len(first_val)

                        if not np.all(isinstance(val, type(first_val)) and first_val_len == len(val) 
                                    for val in dictionary.values()):
                            raise ValueError('Invalid data for 1 dict')
                        
                        if index is None:
                            index = RangeIndex(len(first_val))

                    if index is None:
                        index = RangeIndex(1)

                    columns = []
                    for key, val in dictionary.items():
                        if is_scalar:
                            val = [val]

                        self._series.append(Series(val, index, key))
                        columns.append(key)

                    self._columns = Index(columns)

                    if not isinstance(index, Index):
                        index = Index(index)

                    self._index = index
                    return

                if columns is None:
                    columns = list({key: None for d in data for key in d})

                columns = Index(columns)
                index = RangeIndex(len(data))

                for col in columns:
                    s_data = []
                    for d in data:
                        val = d.get(col, np.nan)
                        if hasattr(val, '__iter__'):
                            val = str(val)

                        s_data.append(val)

                    s_data = np.array(s_data)
                    s = Series(s_data, index, col)
                    self._series.append(s)
                
                self._columns = columns 
                self._index = index 
                return

            if hasattr(data[0], '__iter__') and not isinstance(data[0], str): # 2D
                first_row_len = len(data[0])
                first_row_type = type(data[0])
                for row in data:
                    if not isinstance(row, first_row_type):
                        raise TypeError('the types in the sequence must be the same')
                    
                    if first_row_len != len(row):
                        raise ValueError('the length in the sequence must be the same')

                if index is not None:
                    if not isinstance(index, Index):
                        index = Index(index)
                    if len(index) < len(data):
                        raise ValueError(f'{len(index)} index passed, passed data had {len(data)} index')
                    
                    self._index = index
                else:
                    self._index = RangeIndex(len(data))

                self._series = [Series(data=col, index=self._index) for col in zip(*data)]

                if columns is not None:
                    if not isinstance(columns, Index):
                        columns = Index(columns)
                    
                    if len(columns) < len(self._series):
                        raise ValueError(f'{len(columns)} columns passed, passed data had {len(data[0])} columns')
                    
                    if len(columns) > len(self._series):
                        for _ in range(len(columns) - len(self._series)):
                            self._series.append(Series(data=[], index=self._index))
                else:
                    columns = RangeIndex(len(data[0]))

                for s, name in zip(self._series, columns):
                    s.name = name
                self._columns = columns
                return
            
            else: # 1D
                if index is not None:
                    if not isinstance(index, Index):
                        index = Index(index)

                    if len(index) < 1:
                        raise ValueError(f'{len(index)} index passed, passed data had {len(data)} index')
                    
                    self._series = [Series(data=data, index=index)]
                    self._index = index
                    
                else:
                    self._series = [Series(data=data)]
                    self._index = RangeIndex(len(data))

                if columns is not None:
                    if not isinstance(columns, Index):
                        columns = Index(columns)
                    
                    if len(columns) == 0:
                        raise ValueError(f'0 columns passed, passed data had 1 columns')
                    
                    if len(columns) > len(self._series):
                        for _ in range(len(columns) - len(self._series)):
                            self._series.append(Series(data=[], index=self._index))
                else:
                    columns = RangeIndex(1)

                for s, name in zip(self._series, columns):
                    s.name = name
                self._columns = columns
                return

        raise TypeError('invalid type for data')
    
    @property
    def columns(self):
        return self._columns
    
    @columns.setter
    def columns(self, val):
        if not isinstance(val, Index):
            val = Index(val)
        self._columns = val

        for s, c_name in zip(self._series, self._columns):
            s.name = c_name

    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, val):
        if len(val) != len(self._series[0]._data):
            raise IndexError('invalid indexes')
        
        if not isinstance(val, Index):
            val = Index(val)

        self._index = val

        for s in self._series:
            s.index = self._index

    def copy(self):
        new_inx = self.index.copy()
        new_data = [Series(s._data.copy(), new_inx, s.name) for s in self._series]
        return DataFrame(new_data, new_inx, self.columns.copy())

    def __getitem__(self, key):
        if isinstance(key, np.ndarray) and np.issubdtype(key.dtype, np.bool):
            if len(key) != len(self):
                raise ValueError('Invalid bool indexing')
            
            new_data = [Series(s._data[key], s.index._data[key], s.name)
                         for s in self._series]
            return DataFrame(new_data, self.index._data[key], self.columns)

        if isinstance(key, list):
            new_data = [self._series[self.columns.index(label)] for label in key]
            return DataFrame(new_data, index=self.index)
        
        if isinstance(key, slice):
            new_data = [s[key] for s in self._series]
            new_index = self.index[key]
            return DataFrame(new_data, index=new_index, columns=self.columns)
        
        if isinstance(key, Series):
            new_data = [s[key] for s in self._series]

            inx_data = self.index._data[key._data.astype(bool)]
            if isinstance(self.index, MultiIndex):
                new_index = MultiIndex.from_tuples(inx_data, self.index.names)
            else:
                new_index = Index(inx_data, self.index.name)

            return DataFrame(new_data, index=new_index, columns=self.columns)
        
        res = self._series[self.columns.get_loc(key)]

        if isinstance(res, Series):
            return res

        if isinstance(self.columns, MultiIndex):
            name = ''
            if self.columns.names is not None:
                name = self.columns.names[1]

            col_data = []
            for s in res:
                n = s.name[1]
                col_data.append(n)
                s.name = n

            columns = Index(col_data, name=name)

        else:
            columns = Index([s.name for s in res], name=self.columns.name)

        return DataFrame(res, self.index, columns)

    def __setitem__(self, key, val):
        if isinstance(key, DataFrame):
            for s in key._series:
                s_name = s.name
                c_inx = self.columns.index(s_name, False)

                if c_inx == -1:
                    continue

                for label in key.index:
                    i_inx = self.index.index(label, False)
                    if i_inx == -1 or not s[label]:
                        continue

                    self._series[c_inx]._data[i_inx] = val
            return

        if not np.any(self.columns.isin(key)):
            if isinstance(val, Series):
                val.index = self.index
                val.name = key
                # self.columns = self.columns.append(key)
                self._series.append(val)
            else:
                if isinstance(val, (list, tuple, np.ndarray)):
                    self._series.append(Series(val, self.index, key, type(val[0])))
                elif hasattr(val, '__iter__') and not isinstance(val, str):
                    self._series.append(Series(np.fromiter(val, float), self.index, key))
                else:
                    self._series.append(Series([val] * len(self.index), self.index, key, type(val)))
            self.columns = self.columns.append(key)

        elif hasattr(val, '__iter__') and not isinstance(val, str):
            if len(val) != len(self.index):
                raise ValueError('invalid iterator len')
            
            if isinstance(val, np.ndarray):
                self._series[self.columns.index(key)]._data = val
            else:
                self._series[self.columns.index(key)]._data = np.array(val)

        elif isinstance(val, Series):
            series = self[key]
            for inx in val.index:
                if np.any(series.index.isin(inx)):
                    series[inx] = val[inx]
        else:
            self._series[self.columns.index(key)]._data.fill(val)
    
    def __getattr__(self, key):
        return self._series[self.columns.index(key)]
    
    def _list_of_Tseries(self):
        res = []

        for inx in self.index:
            data = [s[inx] for s in self._series]
            res.append(Series(data, self.columns, inx))

        return res
    
    @property
    def loc(self):
        return Loc(self)

    @property
    def iloc(self):
        return Iloc(self)
    
    @property
    def T(self):
        return DataFrame(self._list_of_Tseries(), index=[s.name for s in self._series])
    
    def pop(self, column):
        col_inx = self.columns.index(column)
        col = self._series[col_inx]

        self._series.pop(col_inx)
        self.columns = self.columns.delete(col_inx)

        return col
    
    def explode(self, column):
        col = self[column]

        rep_per_element_count = np.fromiter((len(v) for v in col._data), int, count=len(col._data))
        inx_data = np.repeat(self.index._data, rep_per_element_count)
        new_data = []
        for s in self._series:
            if s.name == column:
                ser_data = np.concatenate(col._data)
            else:
                ser_data = np.repeat(s._data, rep_per_element_count)

            ser = Series(ser_data, inx_data, s.name)
            new_data.append(ser)

        if isinstance(self.index, MultiIndex):
            new_inx = MultiIndex.from_tuples(inx_data, self.index.names)    
        else:
            new_inx = Index(inx_data, self.index.name)

        return DataFrame(new_data, new_inx, self.columns)

    def reindex(self, labels=None, axis=0, index=None, columns=None, fill_value=np.nan, method=None, level=0):# method 'ffill'/'bfill'
        if labels is not None and (index is not None or columns is not None):
            raise ValueError("Cannot specify both 'labels' and 'columns'/'index'")
        
        if labels is not None:
            if axis == 0 or axis == 'index':
                return DataFrame([s.reindex(labels, fill_value, method) for s in self._series], index=labels)
            
            if axis == 1 or axis == 'columns':
                if isinstance(self.columns, MultiIndex):
                    lv_index = level
                    if not isinstance(lv_index, int):
                        names = self.columns.names
                        if not isinstance(names, list):
                            names = list(names)

                        lv_index = names.index(lv_index) 

                    target_lv_vals = self.columns._level_1
                    if lv_index == 1:
                        target_lv_vals = self.columns._level_2

                    mask = np.isin(target_lv_vals, labels)
                    data = self.columns._data[mask]
                    labels = MultiIndex.from_tuples(data, self.columns.names)
                else:
                    labels = Index(labels)

                new_data = []
                bfill_counter = 1
                labels_len = len(labels) - 1

                for i, label in enumerate(labels):
                    inx = self.columns.index(label, False)

                    if inx != -1:
                        if bfill_counter == 1:
                            new_data.append(self._series[inx])
                        else:
                            for i in range(i - bfill_counter, i):
                                new_data.append(Series(self._series[inx]._data, self.index, labels[i]))
                            bfill_counter = 1

                    elif method is None:
                        new_data.append(Series([fill_value] * len(self.index), index=self.index, name=label))

                    elif method == 'ffill':
                        if i == 0:
                            new_data.append(Series([fill_value] * len(self.index), index=self.index, name=label))
                        else:
                            new_data.append(Series(new_data[-1].array, index=self.index, name=label))

                    elif method == 'bfill':
                        if i == labels_len:
                            new_data.extend([Series([fill_value] * len(self.index), index=self.index, name=labels[i]) 
                                             for i in range(i - bfill_counter, i)])
                            
                        bfill_counter += 1
                    else:
                        raise ValueError('Invalid method')

                return DataFrame(new_data, index=self.index, columns=labels)
        
        if index is not None:
            return self.reindex(index, method=method, fill_value=fill_value, level=level)
        
        if columns is not None:
            return self.reindex(columns, 1, method=method, fill_value=fill_value, level=level)
        
        raise ValueError('0 parameters')

    def drop(self, labels=None, axis=0, index=None, columns=None):
        if labels is not None and (index is not None or columns is not None):
            raise ValueError("Cannot specify both 'labels' and 'columns'/'index'")
        
        if labels is not None:
            if not isinstance(labels, (list, np.ndarray)):
                labels = [labels]

            if axis == 0 or axis == 'index':
                new_index = self.index.drop(labels)
                new_data = [s.drop(labels) for s in self._series]
                return DataFrame(new_data, index=new_index, columns=self.columns)
            
            if axis == 1 or axis == 'columns':
                new_columns = self.columns.drop(labels)
                labels_set = set(labels)
                new_data = [s for s in self._series if s.name not in labels_set]

                return DataFrame(new_data, index=self.index, columns=new_columns)
            
            raise ValueError(f'axis can be only (0 or "index") or (1 or "columns") not: {axis}')
            
        if columns is not None:
            return self.drop(columns, axis='columns')

        if index is not None:
            return self.drop(index, axis='index')
        
        raise ValueError('you must pass one of the parameters: labels, index, columns')
    
    def dropna(self, axis=0, how='any', thresh=None):
        if thresh is not None and thresh < 1:
            raise ValueError('thresh must be >= 1')
        
        if axis == 0 or axis == 'index':
            if thresh is not None:
                valid_rows_inx = []

                for i, row in enumerate(zip(*self._series)):
                    np_row = np.array(row)
                    mask = ~pd.isna(np_row)
                    if np.sum(mask) >= thresh:
                        valid_rows_inx.append(i)

                inx = self.index.to_numpy() if isinstance(self.index, RangeIndex) else self.index._data
                new_index = inx[valid_rows_inx]
                new_data = [Series(s._data[valid_rows_inx], new_index, s.name, s._dtype) for s in self._series]
                return DataFrame(new_data, new_index, self.columns)

            skip_rows = set()
            all_indexes = set(range(len(self.index)))

            if how == 'any':
                for s in self._series:
                    mask = pd.isna(s._data)
                    inx = np.where(mask)[0]
                    skip_rows.update(inx)

            elif how == 'all':
                columns_count = len(self._series)

                for i in all_indexes:
                    non_counter = 0
                    for s in self._series:
                        if np.all(pd.isna(s.iloc[i])):
                            non_counter += 1

                    if non_counter == columns_count:
                        skip_rows.add(i)
            else:
                raise ValueError(f'Invalid "how" parameter: {how}')
            
            valid_rows = list(all_indexes.difference(skip_rows))
            inx = self.index.to_numpy() if isinstance(self.index, RangeIndex) else self.index._data
            new_index = inx[valid_rows]
            new_data = [Series(s._data[valid_rows], new_index, s.name, s._dtype) for s in self._series]
            return DataFrame(new_data, new_index, self.columns)

        if axis == 1 or axis == 'columns':
            if thresh is not None:
                new_data = []
                new_cols = []

                for s in self._series:
                    mask = ~pd.isna(s._data)
                    if np.sum(mask) >= thresh:
                        new_data.append(s)
                        new_cols.append(s.name)

                return DataFrame(new_data, self.index, new_cols)

            new_data = []
            new_cols = []

            for i, s in enumerate(self._series):
                mask = pd.isna(s._data)

                if how == 'any':
                    if not np.any(mask):
                        new_data.append(s)
                        new_cols.append(s.name)

                elif how == 'all':
                    if not np.all(mask):
                        new_data.append(s)
                        new_cols.append(s.name)
                
                else:
                    raise ValueError(f'Invalid "how" parameter: {how}')
                
            return DataFrame(new_data, self.index, new_cols)

        raise ValueError(f'Invalid axis: {axis}')
    
    def fillna(self, val=None, method=None, limit=None): # method 'ffill'/'bfill'
        if not isinstance(val, dict):
            for s in self._series:
                s.fillna(val, method, limit)
            return self
        
        for label, v in val.items():
            indexes = self.index.all_indexes(label)
            if len(indexes) == 0:
                raise KeyError(f'Invalid label: {label}')

            for i in indexes:
                self._series[i].fillna(v, limit=limit)
        return self

    def duplicated(self, keep="first"):
        if keep == 'last':
            def gen():
                rows = set()
                for i in range(len(self.index) -1, -1, -1):
                    row = tuple([s._data[i] for s in self._series])
                    yield row in rows
                    rows.add(row)

            return Series(np.fromiter(gen(), bool)[::-1], self.index, dtype=bool)

        if keep == 'first':
            def gen():
                rows = set()
                for row in zip(*self._series):
                    yield row in rows
                    rows.add(row)

            return Series(np.fromiter(gen(), bool), self.index, dtype=bool)
        
        raise ValueError(f'Invalid keep: {keep}')
    
    def drop_duplicates(self, subset=None, keep="first"):
        if subset is not None:
            if not isinstance(subset, list):
                subset = [subset]
            
            mask = self[subset].duplicated(keep=keep)
            indexes = np.where(mask)[0]
            inx_labels = self.index._data[indexes]
            return self.drop(inx_labels)

        bl_series = self.duplicated(keep=keep)
        mask = ~(bl_series._data.astype(bool))

        new_data = []
        new_inx = Index(self.index._data[mask], self.index.name)
        for ser in self._series:
            new_s = Series(ser._data[mask], new_inx, ser.name)
            new_data.append(new_s)
        
        return DataFrame(new_data, new_inx, self.columns)

    def map(self, func, na_action=None):
        if na_action not in {None, "ignore"}:
            raise ValueError(f'Invalid na_action: {na_action}')
        
        if not callable(func):
            raise ValueError('func param must be callable')
        
        new_data = []
        for s in self._series:
            def gen():
                for el in s._data:
                    if pd.isna(el) and na_action == 'ignore':
                        yield np.nan
                        continue
                    yield func(el)

            new_data.append(Series(np.fromiter(gen(), s._dtype), s.index, s.name, s._dtype))

        return DataFrame(new_data, self.index, self.columns)
    
    def replace(self, old_val, new_val=None):
        if isinstance(old_val, dict):
            new_data = []
            for s in self._series:
                if d := old_val.get(s.name):
                    new_data.append(s.replace(d))
                    continue
                new_data.append(s)

            return DataFrame(new_data, self.index, self.columns)
        
        if not isinstance(old_val, list):
            old_val = [old_val]

        if not isinstance(new_val, list): 
            new_data = []
            for s in self._series:
                data = s._data.copy()
                data[np.isin(data, old_val)] = new_val
                new_data.append(Series(data, s.index, s.name, s._dtype))

            return DataFrame(new_data, self.index, self.columns)
        
        if len(old_val) != len(new_val):
            raise ValueError('Replacement lists must be of same length.')
        
        d = {old: new for old, new in zip(old_val, new_val)}
        new_data = []
        for s in self._series:
            new_ser = np.fromiter((d.get(el, el) for el in s._data), s._dtype)
            new_data.append(Series(new_ser, s.index, s.name, s._dtype))

        return DataFrame(new_data, self.index, self.columns)
    
    def describe(self):
        if np.all([np.issubdtype(s._data.dtype, np.number) for s in self._series]):
            describe_inx = Index(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
            return DataFrame([s.describe() for s in self._series], describe_inx, self.columns)
        
        if np.all([not np.issubdtype(s._data.dtype, np.number) for s in self._series]):
            describe_inx = Index(['count', 'unique', 'top', 'freq'])
            return DataFrame([s.describe() for s in self._series], describe_inx, self.columns)
        
        inx = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'count', 'unique', 'top', 'freq']
        data = []
        for s in self._series:
            if np.issubdtype(s._data.dtype, np.number):
                describe_data = [sum(~pd.isna(s._data)), s.mean(),
                              round(np.nanstd(s._data), 3), np.nanmin(s._data),
                                s.quantile(0.25), s.median(), s.quantile(0.75), 
                                np.nanmax(s._data)] + [np.nan] * 4
            else:
                most_common, freq = Counter(s._data).most_common(1)[0]
                describe_data = [np.nan] * 8 + [len(s._data), len(np.unique(s._data)), most_common, freq]
            data.append(Series(describe_data, inx, s.name))

        return DataFrame(data, inx, self.columns)
    
    def take(self, inx, axis=0):
        if not hasattr(inx, '__iter__'):
            raise TypeError(f'index must be a iterable not: {type(inx).__name__}')

        if axis == 0 or axis == 'index':
            if not isinstance(inx, list):
                inx = list(inx)

            return self.iloc[inx]
        
        if axis == 1 or axis == 'columns':
            new_data = [self._series[i] for i in inx]
            cols = [s.name for s in new_data]
            return DataFrame(new_data, self.index, cols)
        
        raise ValueError('Invalid axis')
    
    def sample(self, n, replace=False, axis=0):
        if not isinstance(n, int):
            raise TypeError(f'Invalid type for n: {type(n).__name__}')

        if axis == 0 or axis == 'index':
            if n > len(self.index) and not replace:
                raise ValueError("Cannot take a larger sample than population when 'replace=False'")
            
            if replace:
                inx = np.random.choice(len(self.index), size=n, replace=True)
            else:
                inx = np.random.permutation(n)
        
        elif axis == 1 or axis == 'columns':
            if n > len(self.columns) and not replace:
                raise ValueError("Cannot take a larger sample than population when 'replace=False'")
            
            if replace:
                inx = np.random.choice(len(self.columns), size=n, replace=True)
            else:
                inx = np.random.permutation(n)
            
        else:
            raise ValueError('Invalid axis')
        
        return self.take(inx, axis=axis)
    
    def _stack_correct_level(self, level_to_inx, level_to_col, level_val):
        unique_level1 = np.unique(level_to_inx)
        new_lv1 = [i for i in self.index for _ in range(len(unique_level1))]
        new_lv2 = [i for _ in range(len(self.index)) for i in unique_level1]

        new_index = MultiIndex.from_arrays([new_lv1, new_lv2], 
                                names=[self.index.name, self.columns.names[level_val]])
        
        name_inx = 0 if level_val else 1
        new_columns = Index(np.unique(level_to_col), 
                                name=self.columns.names[name_inx])

        new_data = {c: {i: np.nan for i in new_index} for c in new_columns}
        # print(new_data)
        for ser in self._series:
            for s_inx, s_val in zip(ser.index, ser._data):
                # return ser.name, s_inx, s_val
                if level_val == 0:
                    key2, key1 = ser.name
                else:
                    key1, key2 = ser.name

                key3 = (s_inx, key2)
                new_data[key1][key3] = s_val

        return DataFrame(new_data, new_index, new_columns)

    def stack(self, level=1):
        if not isinstance(self.columns, MultiIndex):
            inx_lv1 = []
            inx_lv2 = []
            values = []

            for i, row_label in enumerate(self.index):
                for s in self._series:
                    val = s[i]
                    if pd.isna(val):
                        continue

                    inx_lv1.append(row_label)
                    inx_lv2.append(s.name)
                    values.append(val)

            names = None
            if self.index.name and self.columns.name:
                names = [self.index.name, self.columns.name]

            multi_index = MultiIndex.from_arrays([inx_lv1, inx_lv2], names)

            return Series(values, multi_index)
        
        if not isinstance(level, int):
            level = self.columns.names.index(level)

        if level == 0: 
            return self._stack_correct_level(self.columns._level_1, self.columns._level_2, level)
        
        if level == 1:
            return self._stack_correct_level(self.columns._level_2, self.columns._level_1, level)
        
        raise ValueError("Invalid level")

    def set_index(self, columns, drop=True):
        if not isinstance(columns, list):
            col_inx = self.columns.index(columns)
            new_index = Index(self._series[col_inx]._data, self._series[col_inx].name)
            
            new_data = self._series.copy()
            if drop:
                new_cols = self.columns.delete(col_inx)
                new_data.pop(col_inx)
            else:
                new_cols = self.columns

            return DataFrame(new_data, new_index, new_cols)
        
        cols_len = len(columns)
        if cols_len == 1:
            return self.set_index(columns[0], drop)
        
        if cols_len > 2:
            raise ValueError('Invalid length for list with columns')
        
        inx_lv1 = self.columns.index(columns[0])
        inx_lv2 = self.columns.index(columns[1])

        if drop:
            new_data = [s for i, s in enumerate(self._series) if i not in {inx_lv1, inx_lv2}]
            new_cols = self.columns.delete([inx_lv1, inx_lv2])
        else:
            new_cols = self.columns
            new_data = self._series.copy()

        names = None
        if self._series[inx_lv1].name and self._series[inx_lv2].name:
            names = [self._series[inx_lv1].name, self._series[inx_lv2].name]

        inx_data = [self._series[inx_lv1]._data, self._series[inx_lv2]._data]
        new_index = MultiIndex.from_arrays(inx_data, names)

        return DataFrame(new_data, new_index, new_cols)
    
    def reset_index(self, drop=False):
        new_index = RangeIndex(len(self.index))

        if not isinstance(self.index, MultiIndex):
            new_ser = Series(self.index._data, new_index, self.index.name or 'index')

            data = self._series
            new_cols = self.columns
            if not drop:
                new_cols = self.columns.insert(0, new_ser.name)
                data = [new_ser] + data

            return DataFrame(data, new_index, new_cols)
        

        lv1_name = 0
        lv2_name = 1
        if self.index.names is not None:
            lv1_name = self.index.names[0]
            lv2_name = self.index.names[1]

        ser_lv1 = Series(self.index._level_1, new_index, lv1_name)
        ser_lv2 = Series(self.index._level_2, new_index, lv2_name)

        new_cols = self.columns.insert(0, [ser_lv1.name, ser_lv2.name])
        new_data = [ser_lv1, ser_lv2] + self._series

        return DataFrame(new_data, new_index, new_cols)

        # lv1_data = [self.index.levels[0][code] for code in self.index.codes[0]]
        # new_ser_lv1 = Series(lv1_data, new_index, lv1_name)

        # lv2_data = [self.index.levels[1][code] for code in self.index.codes[1]]
        # new_ser_lv2 = Series(lv2_data, new_index, lv2_name)

        # new_cols = self.columns.insert(0, [new_ser_lv1.name, new_ser_lv2.name])
        # new_data = [new_ser_lv1, new_ser_lv2] + self._series

        # return DataFrame(new_data, new_index, new_cols)

    def rename(self, columns=None):
        if isinstance(columns, dict):
            new_cols = self.columns

            for label, new_name in columns.items():
                inx = self.columns.index(label)
                new_cols_data = new_cols._data
                new_cols_data[inx] = new_name

                new_cols = Index(new_cols_data, new_cols.name)
                self._series[inx].name = new_name

            return DataFrame(self._series, self.index, new_cols)

    def join(self, other, on=None, how='left'):
        if isinstance(other, list):
            new_df = self
            for df in other:
                new_df = new_df.join(df, on=on, how=how)

            return new_df

        from merge import merge

        if on is None:
            return merge(self, other, left_index=True, right_index=True, how=how)
        else:
            return merge(self, other, left_on=on, right_index=True, how=how)        

    def combine_first(self, other):
        new_columns = self.columns.union(other.columns)
        new_index = self.index.union(other.index)

        new_data = {c: {i: np.nan for i in new_index} for c in new_columns}

        for s in self._series:
            for s_inx, s_val in zip(s.index, s._data):
                new_data[s.name][s_inx] = s_val

        for s in other._series:
            for s_inx, s_val in zip(s.index, s._data):
                if pd.isna(new_data[s.name][s_inx]):
                    new_data[s.name][s_inx] = s_val

        return DataFrame(new_data)

    def _unstack_correct_level(self, level_to_inx, level_to_col, level_val):
        unique_level1 = np.unique(level_to_col)
        new_lv1 = [c for c in self.columns for _ in range(len(unique_level1))]
        new_lv2 = [c for _ in range(len(self.columns)) for c in unique_level1]

        new_columns = MultiIndex.from_arrays([new_lv1, new_lv2],
                                names=[self.columns.name, self.index.names[level_val]])
        
        name_inx = 0 if level_val else 1
        new_index = Index(np.unique(level_to_inx), name=self.index.names[name_inx])

        new_data = {c: {i: np.nan for i in new_index} for c in new_columns}
        for ser in self._series:
            for s_inx, s_val in zip(ser.index, ser._data):
                if level_val:
                    key2, key1 = s_inx
                else:
                    key1, key2 = s_inx

                key3 = (ser.name, key1)

                new_data[key3][key2] = s_val

        return DataFrame(new_data, new_index, new_columns)

    def unstack(self, level=1):
        if not isinstance(self.index, MultiIndex):
            raise ValueError('Index must be a MultiIndex')
        
        if not isinstance(level, int):
            level = self.index.names.index(level)

        if level == 0:
            return self._unstack_correct_level(self.index._level_2, self.index._level_1, level)

        if level == 1:
            if self.index._level_1.ndim == 2:
                self.index._level_1 = np.fromiter((tuple(row) for row in self.index._level_1), np.object_)
            return self._unstack_correct_level(self.index._level_1, self.index._level_2, level)
        
        raise ValueError("Invalid level")
    
    def pivot(self, *, columns, index=None, values=None):
        target_ser_cols = self[columns]
        target_ser_inx = self[index]

        if values is not None:
            target_ser_val = [self[values]]

        else:
            target_ser_val = []
            for i, col in enumerate(self.columns):
                if col in (columns, index):
                    continue

                target_ser_val.append(self._series[i])
        
        new_cols_data = np.unique(target_ser_cols._data)
        if len(target_ser_val) > 1:
            new_lv1 = [s.name for s in target_ser_val for _ in range(len(new_cols_data))]
            new_lv2 = [c for _ in range(len(target_ser_val)) for c in new_cols_data]

            new_cols = MultiIndex.from_arrays([new_lv1, new_lv2], names=['', columns])
        else:
            new_cols = Index(new_cols_data, name=columns)

        # new_index = Index(np.unique(target_ser_inx._data), target_ser_inx.name)
        inx_data = {val: None for val in target_ser_inx._data}
        new_index = Index(list(inx_data.keys()), target_ser_inx.name)

        counter = 0
        tg_ser_inx = 0
        new_cols_data_len = len(new_cols_data)
        new_data = []
        for col in new_cols:
            compare_label = col[1] if len(target_ser_val) > 1 else col
            col_val_i = np.where(target_ser_cols._data == compare_label)[0]

            new_s_data = target_ser_val[tg_ser_inx]._data[col_val_i]
            new_s = Series(new_s_data, new_index, col)
            new_data.append(new_s)

            counter += 1
            if counter == new_cols_data_len:
                counter = 0
                tg_ser_inx += 1

        return DataFrame(new_data, new_index, new_cols)

    def _plot_subplots(self, ax, kind, alpha, title, grid, rot, xlim, ylim, logy, style, bins, fig):
        if not isinstance(ax, np.ndarray):
            raise TypeError(f'Invalid type for ax: {type(ax)}')
    
        ax_arr_flat = ax.flatten()
        if len(ax_arr_flat) < len(self.columns):
            raise ValueError('The number of passed axes must be >= the number of columns.')
        
        for ax, ser in zip(ax_arr_flat, self._series):
            ser.plot(ax=ax, kind=kind, style=style, bins=bins, alpha=alpha, label=ser.name,
                    grid=grid, xlim=xlim, ylim=ylim, logy=logy, rot=rot)

        if title is not None and fig is not None:
            fig.suptitle(title)

        return ax
    
    def _plot_single_scatter(self, ax, x, y, alpha):
        if x is None or y is None:
            raise ValueError('scatter requires an x and y column')

        ax.scatter(self[x], self[y], alpha=alpha)
        ax.set_ylabel(y)
        ax.set_xlabel(x)

    def _plot_single_bar(self, ax,  use_index):
        total_width = 0.8
        bar_width = total_width / len(self.columns)
        x = np.arange(len(self.index))

        for i, s in enumerate(self._series):
            ax.bar(x + i * bar_width, s._data, width=bar_width, label=s.name)

        ax.set_xticks(x + bar_width)
        if use_index:
            x_ticklabels = self.index._data
        else:
            x_ticklabels = x

        ax.set_xticklabels(x_ticklabels, rotation=90)
        ax.legend()

    def _plot_single_barh(self, ax, x, y, hue, use_index):
        total_width = 0.7

        if hue is not None and x is not None and y is not None:
            x_data = self[x]
            y_data = self[y]
            hue_data = self[hue]

            hue_levels = np.unique(hue_data._data)
            bar_width = total_width / len(hue_levels)

            y_categories = list({val: None for val in y_data._data})[::-1]
            y_pos = np.arange(len(y_categories))

            for i, level in enumerate(hue_levels):
                mask = hue_data._data == level
                values = x_data._data[mask]
                lv_y_pos = np.array([y_categories.index(v) for v in y_data._data[mask]])

                ax.barh(lv_y_pos + i * bar_width - total_width / 2 + bar_width / 2,
                    values, height=bar_width, label=str(level))

            ax.set_yticks(y_pos)
            ax.set_yticklabels(y_categories)
            ax.legend(title=hue)

        else:
            bar_width = total_width / len(self.columns)
            y_pos = np.arange(len(self.index))

            if x is not None:
                ser = self[x]
                ax.barh(y_pos, ser._data, height=bar_width)
            else:
                for i, s in enumerate(self._series):
                    ax.barh(y_pos + i * bar_width, s._data, height=bar_width, label=s.name)

            ax.set_yticks(y_pos + bar_width)

            if use_index:
                y_ticklabels = self.index._data
            else:
                y_ticklabels = y_pos

            ax.set_yticklabels(y_ticklabels)
            ax.legend()

    def _apply_plot_decorations(self, ax, title, grid, rot, xlim, ylim, logy, xticks, yticks):
        if title is not None:
            ax.set_title(title)
        if grid:
            ax.grid(True)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        if rot is not None:
            pass
            # ax.set_xticklabels(ax.get_xticks(), rotation=rot)
        if logy:
            ax.set_yscale('log')

    def _plot_single(self, ax, x, y, kind, use_index, alpha, title, grid, rot,
                    xticks, yticks, xlim, ylim, logy, style, bins, hue):
        
        if kind == 'scatter':
            self._plot_single_scatter(ax, x, y, alpha)
        elif kind == 'bar':
            self._plot_single_bar(ax, use_index)
        elif kind == 'barh':
            self._plot_single_barh(ax, x, y, hue, use_index)
        else:
            each_col_bins = None
            if kind == 'hist':
                ax.set_ylabel('Frequency')
                if bins is None:
                    bins = 10

                # each_col_bins = int(np.ceil(bins / len(self.columns)))
                each_col_bins = bins // len(self.columns)
            
            for s in self._series:
                sty = style
                if isinstance(sty, dict):
                    sty = style.get(s.name)

                s.plot(ax=ax, label=s.name, style=sty, alpha=alpha, bins=each_col_bins, kind=kind, use_index=use_index)

        self._apply_plot_decorations(ax, title, grid, rot,
                                xlim, ylim, logy, xticks, yticks)        

        return ax

    def plot(self, ax=None, subplots=False, layout=None, sharex=False, sharey=False, x=None, y=None, style=None, alpha=None,
            bins=None, kind='line', figsize=None, logy=False, title=None, use_index=True, 
            rot=None, xticks=None, yticks=None, xlim=None, ylim=None, grid=False, hue=None):

        if ax is None:
            if subplots:
                lout = layout or (len(self.columns), 1)
            else:
                lout = (1, 1) 

            if figsize is not None:
                fig, ax = plt.subplots(*lout, sharex=sharex, sharey=sharey, figsize=figsize)
            else:
                fig, ax = plt.subplots(*lout, sharex=sharex, sharey=sharey)

        if subplots:
            return self._plot_subplots(ax, kind, alpha, title, grid, rot, 
                                    xlim, ylim, logy, style, bins, fig)
        
        elif not isinstance(ax, Axes):
            raise TypeError(f'Invalid type for ax: {type(ax)}')
        
        return self._plot_single(ax, x, y, kind, use_index, alpha, title, 
                                grid, rot, xticks, yticks, xlim, ylim, logy, style, bins, hue)
    
    def barplot(self, ax=None, figsize=None, stacked=False, whiskers=False, **kwargs):
        default_colors = ["#3e90ca", "#1be900", "#d9ff00", "#d62728", "#9467bd"]  
        from matplotlib.patches import Patch

        if ax is None:
            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()
        elif not isinstance(ax, Axes):
            raise TypeError(f'Invalid type for ax: {type(ax)}')
        
        total_width = 0.8
        bar_width = total_width / len(self.columns)

        if stacked:
            bottom_positive = np.zeros(len(self.index))
            bottom_negative = np.zeros(len(self.index))
        else:
            y_min = float('inf')
            y_max = float('-inf')

        for i, s in enumerate(self._series):
            if stacked:
                bottom_positive, bottom_negative = s.barplot(ax=ax, bottom_p=bottom_positive, bottom_n=bottom_negative, 
                                                            color=default_colors[i], stacked=stacked, **kwargs)
            else:
                s_min_val = np.min(s._data)
                y_min = s_min_val if s_min_val < y_min else y_min

                s_max_val = np.max(s._data)
                y_max = s_max_val if s_max_val > y_max else y_max

                s.barplot(ax=ax, color=default_colors[i], bar_width=bar_width, x_offset=i * bar_width, 
                          bar_count=len(self.columns), **kwargs)
                
        if stacked:
            bottom_min = np.min(bottom_negative)
            y_start = bottom_min * 1.1 if bottom_min < 0 else 0
            ax.set_ylim(y_start, np.max(bottom_positive) * 1.1) 

        else:
            ax.set_ylim(y_min * 1.1 if y_min < 0 else 0, y_max * 1.1)

        if whiskers:
            for i in range(len(self.index)):
                group_values = [s._data[i] for s in self._series]
                mean = np.mean(group_values)
                std = np.std(group_values)
                line_start = mean - std
                line_end = mean + std

                group_start = i - total_width / 2
                group_center = group_start + total_width / 2
                
                ax.vlines(group_center, line_start, line_end, color="black", lw=1.5)
                ax.hlines([line_start, line_end, mean], group_center-bar_width/2, group_center+bar_width/2, color="black", lw=1.5)


        legend_patches = [Patch(color=default_colors[i], label=s.name) for i, s in enumerate(self._series)]
        ax.legend(handles=legend_patches)
        return ax

    def plot_scatter(self, x=None, y=None, s=50, c=None, ax=None, figsize=None, hue=None, **kwargs):
        if x is None or y is None:
            raise ValueError('scatter requires x and y column names')
        
        if ax is None:
            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()
        elif not isinstance(ax, Axes):
            raise TypeError(f'Invalid type for ax: {type(ax)}')
        
        from matplotlib.patches import Ellipse, Patch

        x_column = self[x]
        y_column = self[y]

        ax.set_xlabel(x)
        ax.set_ylabel(y)

        x_min, x_max = x_column._data.min(), x_column._data.max()
        y_min, y_max = y_column._data.min(), y_column._data.max()

        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1

        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        fig.canvas.draw()

        p0 = ax.transData.transform((x_column._data[0], y_column._data[0]))
        p1 = ax.transData.transform((x_column._data[0]+1, y_column._data[0]+1))
        pixels_per_x = p1[0] - p0[0]
        pixels_per_y = p1[1] - p0[1]
        
        radius_in_pixels = np.sqrt(s / np.pi)

        radius_in_data_x = radius_in_pixels / pixels_per_x
        radius_in_data_y = radius_in_pixels / pixels_per_y

        if hue is not None:
            hue_column = self[hue]
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            unique_vals = np.unique(hue_column._data)
            cat_color = {cat: color for cat, color in zip(unique_vals, colors)}

            for x, y, h_val in zip(x_column._data, y_column._data, hue_column._data):
                c = cat_color[h_val]

                ellipse = Ellipse((x, y), width=2*radius_in_data_x, height=2*radius_in_data_y, color=c)
                ax.add_patch(ellipse)

            legend_patches = [Patch(color=col, label=cat) for cat, col in cat_color.items()]
            ax.legend(handles=legend_patches, title=hue)
            return ax         

        for x, y in zip(x_column._data, y_column._data):
            ellipse = Ellipse((x, y), width=2*radius_in_data_x, height=2*radius_in_data_y, color=c)
            ax.add_patch(ellipse)

        return ax

    def _groupby_multikey(self, key, current_axis):
        if isinstance(key, str):
            return self[key]._data, key

        if isinstance(key, Series):
            return key._data, key.name

        if isinstance(key, (list, np.ndarray)):
            return key, ''
        
        if isinstance(key, dict):
            key = [key.get(i, np.nan) for i in current_axis]
            return key, ''

        if callable(key):
            key = [key(i) for i in current_axis]
            return key, ''
        
        raise TypeError('Invalid type for multi key')

    def _get_multiindex_level(self, key, mlti_obj):
        if not isinstance(key, int):
            inx = mlti_obj.names.index(key)
            return self._get_multiindex_level(inx, mlti_obj)

        names = mlti_obj.names or ['', '']
        if key == 0:
            return mlti_obj._level_1, names[0]
        if key == 1:
            return mlti_obj._level_2, names[1]
        
        raise ValueError('Invalid index level for MultiIndex')
    
    def _get_key_and_keynames_from_mlti(self, level, mlti_obj):
        if not isinstance(level, list):
            return self._get_multiindex_level(level, mlti_obj)

        if len(level) != 2:
            raise ValueError('MultiIndex supports 2 levels')
        
        key1, name1 = self._get_multiindex_level(level[0], mlti_obj)
        key2, name2 = self._get_multiindex_level(level[1], mlti_obj)
        return [key1, key2], [name1, name2]

    def groupby(self, key=None, dropna=True, axis=0, level=None, group_keys=True, as_index=True):
        if key is not None and level is not None:
            raise TypeError("Cannot specify both 'key' and 'level'")
        
        if key is not None and not callable(key) \
            and not isinstance(key, (np.ndarray, list, Series, str, dict)):
            raise TypeError(f'"key" type must be "Series"/"list"/"ndarray"/"str"/"dict" or callable, not: {type(key).__name__}')
        
        key_name = ''
        if isinstance(key, Series):
            key_name = key.name
            key = key._data

        if isinstance(key, str):
            key_name = key
            key = self[key]._data

        is_multi = False
        if axis == 0 or axis == 'index':
            axis = 0

            if isinstance(key, dict):
                key = [key.get(i, np.nan) for i in self.index]

            if callable(key):
                key = [key(i) for i in self.index]

            if level is not None:
                if not isinstance(self.index, MultiIndex):
                    raise TypeError('DF index is not MultiIndex!')
                
                key, key_name = self._get_key_and_keynames_from_mlti(level, self.index)

            if len(key) != len(self.index):
                if not (isinstance(key, list) and len(key) == 2):
                    raise ValueError('Invlaid key length')
                is_multi = True
            
        elif axis == 1 or axis == 'columns':
            axis = 1

            if isinstance(key, dict):
                key = [key.get(i, np.nan) for i in self.columns]

            if callable(key):
                key = [key(i) for i in self.columns]

            if level is not None:
                if not isinstance(self.columns, MultiIndex):
                    raise TypeError('DF columns is not MultiIndex!')
                
                key, key_name = self._get_key_and_keynames_from_mlti(level, self.columns)

            if len(key) != len(self.columns):
                if not (isinstance(key, list) and len(key) == 2):
                    raise ValueError('Invlaid key length')
                is_multi = True
           
        else:    
            raise ValueError('Invalid axis')
        
        if is_multi:
            if axis == 0:
                current_axis = self.index
            else:
                current_axis = self.columns

            key1, name1 = self._groupby_multikey(key[0], current_axis)
            key2, name2 = self._groupby_multikey(key[1], current_axis)
            
            axis_len = len(current_axis)
            if len(key1) != axis_len or len(key2) != axis_len:
                raise ValueError('Invlaid key length')
            
            if level is None:
                key_name = [name1, name2]

            key = zip(key1, key2)

        groups = {}
        for i, val in enumerate(key):
            if dropna:
                if is_multi:
                    is_nan = any(pd.isna(v) for v in val)
                else:
                    is_nan = pd.isna(val)

                if is_nan:
                    continue

            group_inxs = groups.setdefault(val, [])
            group_inxs.append(i)

        return DataFrameGroupBy(self, groups, key_name, axis=axis, is_multi=is_multi, group_keys=group_keys, as_index=as_index)
        
    def pivot_table(self, values=None, index=None, columns=None, margins=False, aggfunc=None):
        if values is None:
            values = [ser for ser in self._series if ser.name != columns]
        else:
            if isinstance(values, str):
                values = [values]
            values = [self[col_name] for col_name in values]
        res = DataFrame(values)

        if index is not None:
            if isinstance(index, (str, Series)):
                index = [index]

            inx_items = []
            for i_name in index:
                if isinstance(i_name, str):
                    ser = self[i_name]
                elif isinstance(i_name, Series):
                    ser = i_name
                else:
                    raise TypeError('Invalid type for "index"')
                inx_items.append(ser)
        else:
            inx_items = []
        
        if columns is not None:
            if isinstance(columns, (str, Series)):
                columns = [columns]

            col_items = []
            for c_name in columns:
                if isinstance(c_name, str):
                    ser = self[c_name]
                elif isinstance(c_name, Series):
                    ser = c_name
                else:
                    raise TypeError('Invalid type for "index"')
                col_items.append(ser)
        else:
            col_items = []

        groups_key = inx_items + col_items
        if len(groups_key) <= 2:
            if len(res._series) == 1:
                res = res._series[0]

            if aggfunc is None:
                res = res.groupby(groups_key).mean()
            else:
                res = res.groupby(groups_key).agg(aggfunc)

            if col_items:
                col_name = col_items[0].name
                res = res.unstack(level=col_name)
            return res

        groups = {}
        for i, row in enumerate(zip(*(s._data for s in groups_key))):
            group_inxs = groups.setdefault(row, [])
            group_inxs.append(i)

        if len(col_items) == 1:
            temp_groups = {(key[:-1], key[-1]): inxs for key, inxs in groups.items()}
            key_names = [(groups_key[0].name, groups_key[1].name), groups_key[2].name]
        else:
            temp_groups = {(key[0], key[1:]): inxs for key, inxs in groups.items()}
            key_names = [groups_key[0].name, (groups_key[1].name, groups_key[2].name)]

        df_data = []
        for ser in res._series:
            s_gby = SeriesGroupBy(ser, temp_groups, key_names, is_multi=True)
            
            if aggfunc is None:
                val = s_gby.mean()
            else:
                val = s_gby.agg(aggfunc)
            df_data.append(val)

        df = DataFrame(df_data)

        if len(col_items) == 2:
            df = df.unstack()
            df.columns = MultiIndex.from_tuples(df.columns._level_2, df.columns.names[1])

        else:
            df = df.unstack(level=groups_key[2].name)
            df.index = MultiIndex.from_tuples(df.index, df.index.name)

        if margins:
            different_tables = list({k: None for k in df.columns._level_1})
            if len(different_tables) == 1:
                name = ''
                if df.columns.names is not None:
                    name = df.columns.names[1]
                df.columns = Index(df.columns._level_2, name)

            if isinstance(df.columns, MultiIndex):
                from concat import concat
                
                new_df_data = []
                for t_name in different_tables:
                    df_part = df[t_name]
                    if aggfunc is None:
                        val = df_part.mean()
                    else:
                        val = df_part.sum()
                    df_part['All'] = val
                    new_df_data.append(df_part)

                cols_names = df.columns.names
                df = concat(new_df_data, axis=1)

                multi_lv1 = []
                different_tables_iter = iter(different_tables)
                col = next(different_tables_iter)
                for col_name in df.columns:
                    multi_lv1.append(col)
                    if col_name == 'All':
                        try:
                            col = next(different_tables_iter)
                        except StopIteration:
                            break

                new_multi = MultiIndex.from_arrays([multi_lv1, df.columns._data], cols_names)
                df.columns = new_multi
                # print(df)

            else:
                if aggfunc is None:
                    new_col = df.mean()
                else:
                    new_col = df.sum()
                df['All'] = new_col

            for s in df._series:
                if aggfunc is None:
                    val = s.mean()
                else:
                    val = s.sum()

                s._data = np.concatenate([s._data, [val]])

            if isinstance(df.index, MultiIndex):
                new_row_inx = MultiIndex.from_tuples([('All', 'All')])
            else:
                new_row_inx = Index(['All'])

            inx = df.index.append(new_row_inx)
            df.index = inx

        return df

    @staticmethod
    def _nsmallest_alg(arr, n, keep, positions):
        mid = arr[len(arr) // 2]

        small_pos, small = [], []
        equal_pos, equal = [], []
        big_pos, big = [], []
        for i, x in zip(positions, arr):
            if x < mid:
                small_pos.append(i)
                small.append(x)
            elif x > mid:
                big_pos.append(i)
                big.append(x)
            else:
                equal_pos.append(i)
                equal.append(x)

        if len(small) > n:
            return Series._nsmallest_alg(small, n, keep, small_pos)
        
        if len(small) + len(equal) >= n:
            remainder = n - len(small)

            if keep == 'first':
                data = small + equal[:remainder]
                pos = small_pos + equal_pos[:remainder]
                return data, pos 
            
            if keep == 'last':
                data = small + equal[-remainder:]
                pos = small_pos + equal_pos[-remainder:]
                return data, pos
            
            if keep == 'all':
                data = small + equal
                pos = small_pos + equal_pos
                return data, pos
            
            raise ValueError("keep must be 'first', 'last', or 'all'")

        n = n - (len(small) + len(equal))
        data_bigger, pos_bigger = Series._nsmallest_alg(big, n, keep, big_pos)

        data = small + equal + data_bigger
        pos = small_pos + equal_pos + pos_bigger
        return data, pos 

    def nsmallest(self, n, columns, keep='first'):
        if n > len(self.index):
            n = len(self.index)

        if isinstance(columns, str):
            arr = self[columns]._data
        elif isinstance(columns, list):
            key_series_data = [self[c_name]._data for c_name in columns]
            arr = list(zip(*key_series_data))
        else:
            raise TypeError('Invalid type for "columns"')
        
        positions = np.arange(len(self.index))
        data, pos = self._nsmallest_alg(arr, n, keep, positions)

        if isinstance(columns, list):
            data = np.fromiter((d for d in data), dtype=np.object_)

        sort_inx = np.argsort(data)
        pos = np.array(pos) 
        sort_pos = pos[sort_inx]
        return self.iloc[sort_pos]

    def sort_values(self, column, ascending=True):
        col = self[column]

        inxs = np.argsort(col._data)
        if not ascending:
            inxs = inxs[::-1]

        new_inx = self.index._data[inxs]
        new_data = [Series(s._data[inxs], new_inx, s.name) for s in self._series]
        return DataFrame(new_data, new_inx, self.columns)
    

    def min(self):
        return Series([s.min() for s in self._series], self.columns)
    
    def max(self):
        return Series([s.max() for s in self._series], self.columns)
    
    def count(self, axis=0):
        if axis == 0:
            return Series([self.iloc[i].count() for i in range(len(self))], self.index)
        if axis == 1:
            return Series([s.count() for s in self._series], self.columns)

    def sum(self, axis=0):
        if axis in (0, 'index'):
            return Series([s.sum() for s in self._series], self.columns)

        if axis in (1, 'columns'):
            return Series([self.iloc[i].sum() for i in range(len(self))], self.index)

        raise ValueError('Invalid axis')

    def mean(self, axis=0):
        if axis == 0:
            return Series([self.iloc[i].mean() for i in range(len(self))], self.index)
        if axis == 1:
            return Series([s.mean() for s in self._series], self.columns)
        
        raise ValueError('Invalid axis')

    def div(self, other, axis=0):
        if np.isscalar(other):
            return self / other
        
        if not isinstance(other, Series):
            raise TypeError('Invalid type for other')
        
        if axis in (0, 'index'):
            new_inx = self.index.union(other.index)
            o = other.reindex(new_inx)
            df_data = []
            for ser in self._series:
                s = ser.reindex(new_inx)
                df_data.append(s / o)

            return DataFrame(df_data, new_inx, columns=self.columns)
                
        if axis in (1, 'columns'):
            new_columns = self.columns.union(other.index)
            new_data = [self[inx] / other[inx] for inx in new_columns]
            return DataFrame(new_data, self.index, new_columns)
        
        raise ValueError('Invalid axis')

    def corr(self):
        new_data = np.corrcoef(self.to_numpy(), rowvar=False)
        new_data = np.round(new_data, 4)
        return DataFrame(new_data, self.columns, self.columns)

    def to_numpy(self):
        return np.array([s._data for s in self.iloc])

    def to_json(self):
        def convert(val):
            if hasattr(val, "item"):
                return val.item()
            return val
        
        data = {s.name: {inx: convert(val) for inx, val in zip(s.index, s._data)} for s in self._series}
        return json.dumps(data)

    def __delitem__(self, label):
        i = self.columns.index(label)
        self.columns = self.columns.drop(label)
        self._series.pop(i)

    def head(self, n=5):
        return self.iloc[:n]
    
    def tail(self, n=5):
        return self.iloc[-n:]
    
    def _mat_ops(self, other, func):
        if isinstance(other, DataFrame):
            new_index = self.index.union(other.index)
            new_cols = self.columns.union(other.columns)
            series = []

            for col in new_cols:
                if col in other.columns and col in self.columns:
                    self_series = self[col]
                    other_series = other[col]
                    data = []
                    for label in new_index:
                        if label in self.index and label in other.index:
                            data.append(func(self_series._data[self_series.get_loc(label)], 
                                             other_series._data[other_series.get_loc(label)]))
                        else:
                            data.append(None)

                    s = Series(data, index=new_index, name=col)
                else:
                    s = Series([], index=new_index, name=col)
    
                series.append(s)
            return DataFrame(series, index=new_index, columns=new_cols)
        
        if isinstance(other, Series):
            if np.all(self.columns._data == other.index._data):
                data = [func(s, v) for s, v in zip(self._series, other._data)]
                return DataFrame(data, self.index, self.columns)
            
            raise ValueError('Invalid series index matching')
        
        return DataFrame([func(s, other) for s in self._series], index=self.index, 
                         columns=self.columns)
    
    def __add__(self, other):
        return self._mat_ops(other, lambda x, y: x + y)
    
    def __sub__(self, other):
        return self._mat_ops(other, lambda x, y: x - y)
        
    def __mul__(self, other):
        return self._mat_ops(other, lambda x, y: x * y)
    
    def __truediv__(self, other):
        return self._mat_ops(other, lambda x, y: x / y)
    

    def _op(self, other, func):
        if isinstance(other, DataFrame):
            if np.any(self.columns._data != other.columns._data) or np.any(self.index._data != other.index._data):
                raise ValueError('not a same index or columns')
            
            return DataFrame([Series(func(s._data, o._data), self.index, s.name) for s, o in 
                              zip(self._series, other._series)], index=self.index, columns=self.columns)

        return DataFrame([func(s, other) for s in self._series], index=self.index, columns=self.columns)

    def __eq__(self, other):
        return self._op(other, lambda x, y: x == y)

    def __ne__(self, other):
        return self._op(other, lambda x, y: x != y)

    def __lt__(self, other):
        return self._op(other, lambda x, y: x < y)

    def __le__(self, other):
        return self._op(other, lambda x, y: x <= y)

    def __gt__(self, other):
        return self._op(other, lambda x, y: x > y)

    def __ge__(self, other):
        return self._op(other, lambda x, y: x >= y)

    def __iter__(self):
        return iter(self._series)
    
    def __len__(self):
        return len(self.index)
    
    def _memory_usage(self, deep=True):
        import sys

        total = self.index._data.nbytes + self.columns._data.nbytes
        for ser in self._series:
            total += sys.getsizeof(ser.name)
            total += ser._data.nbytes

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if total < 1024:
                return f"{total:.2f} {unit}"
            total /= 1024  

        return f"{total:.2f} PB"
    
    def info(self):
        res = [repr(self)]
        
        inx = self.index._data
        inx_info = f"{self.index.__class__.__name__}: {len(inx)} entries, {inx[0]} to {inx[-1]}"
        res.append(inx_info)

        ser_dtypes = None
        if len(self.columns) <= 20:
            cols_info = []
            first_row = f"Data columns (total {len(self.columns)} columns):"
            cols_info.append(first_row)

            non_null = 'non-null'
            col_names = ['#', 'Column', 'Non-Null Count', 'Dtype']
            
            ser_names = []
            ser_non_null_counts = []
            ser_dtypes = []

            for s in self._series:
                ser_names.append(str(s.name))
                non_null_count = np.sum(~((pd.isna(s._data)) | (s._data == 'nan') | (s._data == '')))
                ser_non_null_counts.append(f'{non_null_count} {non_null}')
                ser_dtypes.append(str(s._dtype))


            col_width = [len(str(len(self.columns))) + 2]
            for i in range(1, len(col_names)):
                if i == 1:
                    val = max(len(s_name) for s_name in ser_names)
                    if val < len(col_names[i]):
                        val = len(col_names[i])
                
                elif i == 2:
                    val = max(len(count) for count in ser_non_null_counts)
                    if val < len(col_names[i]):
                        val = len(col_names[i])

                elif i == 3:
                    val = max(len(dtp) for dtp in ser_dtypes)
                    if val < len(col_names[i]):
                        val = len(col_names[i])
                col_width.append(val + 2)
            
            str_col_names = ''.join(f"{c:<{col_width[i]}}" for i, c in enumerate(col_names))
            dashes = ''.join(f"{'-' * len(col):<{col_width[i]}}" for i, col in enumerate(col_names))

            cols_info.append(str_col_names)
            cols_info.append(dashes)

            for i, s_name in enumerate(ser_names):
                j = 0
                row = f'{i:<{col_width[j]}}'
                j += 1
                row += f"{s_name:<{col_width[j]}}"
                j += 1
                row += f'{ser_non_null_counts[i]:<{col_width[j]}}'
                j += 1
                row += f'{ser_dtypes[i]:<{col_width[j]}}'
                cols_info.append(row)

            res.extend(cols_info)
        else:
            col = self.columns._data
            col_info = f"{self.columns.__class__.__name__}: {len(col)} entries, {col[0]} to {col[-1]}"
            res.append(col_info)

        if ser_dtypes is None:
            ser_dtypes = np.fromiter((s._dtype for s in self._series), np.object_, count=len(self._series))

        dtypes = 'dtypes: '
        counts = Counter(ser_dtypes)
        dtypes += ', '.join(f'{tp}({count})' for tp, count in counts.items())

        mem = f'memory usage: {self._memory_usage()}'

        res.append(dtypes)
        res.append(mem)
        
        return '\n'.join(res)

    def __repr__(self):
        return f"<class '{type(self).__module__}.{type(self).__name__}'>"

    def __str__(self):
        if isinstance(self.index, MultiIndex):
            lv1_width = max(len(str(x)) for x in self.index._level_1)
            lv2_width = max(len(str(x)) for x in self.index._level_2)
            
            if self.index.names is not None:
                lv1_width = lv1_width if lv1_width > len(self.index.names[0]) else len(self.index.names[0])
                lv2_width = lv2_width if lv2_width > len(self.index.names[1]) else len(self.index.names[1])

            index_width = [lv1_width + 1, lv2_width + 1]

        else:
            index_width = max(len(str(x)) for x in self.index)

            if isinstance(self.columns, MultiIndex) and self.columns.names is not None:
                index_width = max(index_width, len(self.index.name),
                                len(self.columns.names[0]), len(self.columns.names[1]))
                
            elif not isinstance(self.columns, MultiIndex):
                index_width = max(index_width, len(self.index.name), len(self.columns.name))

            else:
                index_width = max(index_width, len(self.index.name))

            index_width = [index_width + 1]

        res = []
        if isinstance(self.columns, MultiIndex):
            cols_width = []
            for col, s in zip(self.columns, self._series):
                from_data = max(len(str(x)) for x in s._data)
                col_lv1 = len(str(col[0]))
                col_lv2 = len(str(col[1]))
                cols_width.append(max(from_data, col_lv1, col_lv2))

            if isinstance(self.index, MultiIndex):
                count = sum(index_width) + 1
            else:
                count = sum(index_width)

            if self.columns.names is not None:
                first_row = f'{self.columns.names[0]:<{count}}'
                second_row = f'{self.columns.names[1]:<{count}}'
            else:
                first_row = ' ' * (count)
                second_row = ' ' * (count)

            col_names_lv1 = []
            col_names_lv2 = []

            for i, (lv1, lv2) in enumerate(self.columns):
                if i == 0 or lv1 != self.columns._data[i - 1][0]:
                    col_names_lv1.append(f'{lv1:<{cols_width[i]}}')
                else:
                    col_names_lv1.append(' ' * cols_width[i])

                col_names_lv2.append(f'{lv2:^{cols_width[i]}}')
            
            first_row += ' '.join(col_names_lv1)
            second_row += ' '.join(col_names_lv2)
            res.append(first_row + '\n' + second_row)

        else:
            cols_width = []
            for s in self._series:
                from_data = max(len(str(x)) for x in s._data)
                from_name = len(str(s.name))
                cols_width.append(from_name if from_name > from_data else from_data)

            first_row = f'{self.columns.name:<{sum(index_width)}}'

            col_names = []
            for i in range(len(self._series)):
                col_names.append(f'{self._series[i].name:>{cols_width[i]}}')

            first_row += ' '.join(col_names)
            res.append(first_row)

        
        if isinstance(self.index, MultiIndex):
            if self.index.names:
                res[0] += f'\n{self.index.names[0]:<{index_width[0]}}{self.index.names[1]:<{index_width[1]}}'

            for i, inx in enumerate(self.index):
                if inx[0] is None:
                    inx = (str(inx[0]), inx[1])

                if inx[1] is None:
                    inx = (inx[0], str(inx[1]))

                if i == 0 or self.index._data[i - 1][0] != inx[0]:
                    row = f'{inx[0]:<{index_width[0]}}'
                else:
                    row = ' ' * index_width[0]

                row +=  f'{inx[1]:<{index_width[1]}}'

                for j, s in enumerate(self._series):
                    row += f'{str(s._data[i]):>{cols_width[j]}} ' 
                res.append(row)

        else:
            if self.index.name:
                res[0] += f'\n{self.index.name:<{index_width[0]}}' 

            for i, inx in enumerate(self.index):
                row = f'{inx:<{index_width[0]}}'

                for j, s in enumerate(self._series):
                    row += f'{str(s._data[i]):>{cols_width[j]}} '
                res.append(row)

        return '\n'.join(res)
    

        # inx_width_levels = [max(len(str(x)) for x in level) for level in self.index.levels]

        # res = [f'{self.columns.name:<{sum(inx_width_levels) + 2}} ' + f'{' '.join(f'{self._series[i].name:>{cols_width[i]}}'
        #                     for i in range(len(self._series)))}' + 
        #                     (f'\n{self.index.names[0]:<{inx_width_levels[0]}}  {self.index.names[1]:<{inx_width_levels[1]}}' if self.index.names is not None else '')]
        
        # for i, inx in enumerate(self.index):
        #     row = f'{self.index.levels[0][inx[0]] if i == 0 or self.index.codes[0][i - 1] != inx[0] else '':<{inx_width_levels[0]}}  {self.index.levels[1][inx[1]]:<{inx_width_levels[1]}} '

        #     for counter, s in enumerate(self._series):
        #         row += f'{str(s._data[i]):>{cols_width[counter]}} ' 
        #     res.append(row)

        # return '\n'.join(res)