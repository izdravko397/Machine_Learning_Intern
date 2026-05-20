import numpy as np
import pandas as pd
from index import RangeIndex, Index, MultiIndex, DatetimeIndex, to_datetime, date_range, get_offset, Minute, Day, Hour
import math
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from datetime import datetime
from periods import *

class SeriesGroupBy:
    funcs = {
        'min':   lambda x: x.min(),
        'max':   lambda x: x.max(),
        'count': lambda x: x.count(),
        'sum':   lambda x: x.sum(),
        'mean':  lambda x: x.mean(),
        'size':  lambda x: x.size(),
        'std':   lambda x: x.std(),
    }

    def __init__(self, series, groups: dict, key_name='', is_multi=False, group_keys=True):
        self.series = series
        self.groups = groups
        self._is_multi = is_multi
        self._key_name = key_name
        self.group_keys = group_keys
        self._optimized_groups = None

    def _make_optimized_groups(self):
        if self._optimized_groups is not None:
            return self._optimized_groups

        unique_keys = np.fromiter(self.groups.keys(), np.object_)
        group_codes = np.full(len(self.series._data), np.nan)

        for i, (_, inxs) in enumerate(self.groups.items()):
            group_codes[inxs] = i

        data_mask = ~pd.isna(self.series._data)
        group_codes = group_codes[data_mask]
        values = self.series._data[data_mask]

        group_codes_mask = ~pd.isna(group_codes)
        group_codes = group_codes[group_codes_mask]
        values = values[group_codes_mask]

        if not np.issubdtype(group_codes.dtype, np.integer):
            group_codes = group_codes.astype(int)

        if not np.issubdtype(values.dtype, np.number):
            values = pd.to_numeric(values, errors='ignore')

        self._optimized_groups = (unique_keys, group_codes, values)
        return unique_keys, group_codes, values

    def _group_reduce(self, kind):
        unique_keys, codes, values = self._make_optimized_groups()

        if kind in ('sum', 'mean', 'std'):
            sums = np.bincount(codes, weights=values)

        if kind in ('mean', 'count', 'std'):
            counts = np.bincount(codes)

        match kind:
            case 'sum':
                ser_data = sums
            case 'mean':
                ser_data = sums / counts
            case 'count':
                ser_data = counts
            case 'std':
                means = sums / counts
                diffs = values - means[codes]
                sqdiffs = diffs ** 2
                sqdiff_sums = np.bincount(codes, weights=sqdiffs)
                ser_data = np.sqrt(sqdiff_sums / (counts - 1))
            case 'min' | 'max':
                is_min = False
                is_max = False
                if kind == 'min':
                    fill_val = np.inf
                    is_min = True
                elif kind == 'max':
                    fill_val = -np.inf
                    is_max = True

                vals = np.full(len(unique_keys), fill_val)

                for i, val in enumerate(values):
                    group = codes[i]
                    if (is_min and val < vals[group]) or (is_max and val > vals[group]):
                        vals[group] = val

                ser_data = np.where((vals == np.inf) | (vals == -np.inf), np.nan, vals)

        inx_data = np.fromiter(self.groups.keys(), dtype=np.object_)

        if self._is_multi:
            ser_inx = MultiIndex.from_tuples(inx_data, names=self._key_name)
        else:
            ser_inx = Index(inx_data, name=self._key_name)

        return Series(ser_data, index=ser_inx, name=self.series.name)


    def mean(self):
        return self._group_reduce('mean')
    
    def count(self):
        return self._group_reduce('count')
    
    def min(self):
        return self._group_reduce('min')
    
    def max(self):
        return self._group_reduce('max')
    
    def sum(self):
        return self._group_reduce('sum')
    
    def std(self):
        return self._group_reduce('std')

    def quantile(self, q=0.5):
        new_data = []
        for _, inxs in self.groups.items():
            val = np.quantile(self.series._data[inxs], q)
            new_data.append(val)
        
        new_inx_data = list(self.groups.keys())
        if self._is_multi:
            new_inx = MultiIndex.from_tuples(new_inx_data, self._key_name)
        else:
            new_inx = Index(new_inx_data, self._key_name)

        return Series(new_data, new_inx, self.series.name)
    
    def size(self):
        inx_data = []
        data = []
        for g, inxs in self.groups.items():
            inx_data.append(g)
            data.append(len(inxs))

        if self._is_multi:
            names = self._key_name if any(self._key_name) else None
            inx = MultiIndex.from_tuples(inx_data, names)
        else:
            inx = Index(inx_data, self._key_name)
    
        return Series(data, inx, self.series.name)

    def describe(self):
        new_df_data = []
        df_inx = []
        for g_name, inxs in self.groups.items():
            df_inx.append(g_name)
            group = self.series.iloc[inxs]

            if np.issubdtype(self.series._dtype, np.number):
                describe_data = [np.sum(~pd.isna(group._data)), group.mean(),
                                 group.std(), group.min(),
                                 group.quantile(0.25), group.median(), 
                                 group.quantile(0.75), group.max()]
            else:
                most_common, freq = Counter(self._data).most_common(1)[0]
                describe_data = [len(group), len(np.unique(group._data)), most_common, freq]
            
            new_df_data.append(describe_data)

        if self._is_multi:
            names = self._key_name if any(self._key_name) else None
            inx = MultiIndex.from_tuples(df_inx, names=names)
        else:
            inx = Index(df_inx, name=self._key_name)

        if np.issubdtype(self.series._dtype, np.number):
            cols = Index(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        else:
            cols = Index(['count', 'unique', 'top', 'freq'])
        
        from dataframe import DataFrame
        return DataFrame(new_df_data, inx, cols)

    def transform(self, func):
        if isinstance(func, str):
            aggregated = self.funcs[func](self)
            new_ser_data = np.empty_like(self.series._data)
            for g_name, inxs in self.groups.items():
                new_ser_data[inxs] = aggregated[g_name]

            return Series(new_ser_data, self.series.index, self.series.name)
        
        if callable(func):
            new_ser_data = np.empty_like(self.series._data, dtype=np.object_)
            for g_name, inxs in self.groups.items():
                func_res = func(self.series.iloc[inxs])

                if isinstance(func_res, (Series, np.ndarray)):
                    if len(func_res) != len(inxs):
                        raise ValueError('Invalid Series length from func result')

                    if isinstance(func_res, Series):
                        func_res = func_res._data

                elif not np.isscalar(func_res):
                    raise TypeError('Invalid type from func result')
                
                new_ser_data[inxs] = func_res
                
            return Series(new_ser_data, self.series.index, self.series.name)
        
        raise TypeError('Invalid type "func"')

    def _agg_per_func(self, func, _s_name=True):
        col_name = None
        if isinstance(func, tuple):
            col_name = func[0]
            func = func[1]
            
        if isinstance(func, str):
            res = self.funcs[func](self)
            if not _s_name:
                res.name = func

            if col_name is not None:
                res.name = col_name

            return res
        
        if not callable(func):
            raise TypeError('Invalid type for "func"')
        
        inx_data = []
        ser_data = []

        for g_name, inxs in self.groups.items():
            inx_data.append(g_name)

            func_res = func(self.series.iloc[inxs])
            if not np.isscalar(func_res):
                raise ValueError('Function did not reduce')
            
            ser_data.append(func_res)

        if self._is_multi:
            ser_inx = MultiIndex.from_tuples(inx_data, names=self._key_name)
        else:
            ser_inx = Index(inx_data, name=self._key_name)

        res = Series(ser_data, index=ser_inx, name=self.series.name)

        if not _s_name:
            res.name = func.__name__

        if col_name is not None:
            res.name = col_name

        return res

    def agg(self, func):
        if not isinstance(func, (list, str)) and not callable(func):
            raise TypeError('Invalid type for "func"')

        if not isinstance(func, list):
            return self._agg_per_func(func)
        
        from dataframe import DataFrame
        df_data = [self._agg_per_func(f, _s_name=False) for f in func]
        return DataFrame(df_data)

    def apply(self, func, **kwargs):
        if not isinstance(func, str) and not callable(func):
            raise TypeError('Invalid type for "func"')

        if isinstance(func, str):
            return self.funcs[func](self)
        
        # func is callable
        from dataframe import DataFrame
        scalar_results = False
        series_result = False
        dfs_results = False

        new_data = []
        df_multi_inx = [[], []]
        for g_name, inxs in self.groups.items():
            func_res = func(Series(self.series._data[inxs], 
                                self.series.index._data[inxs], g_name), **kwargs)

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


        if scalar_results:
            return Series(new_data, list(self.groups.keys()), self.series.name)

        if series_result:
            first_ser_inx = new_data[0].index._data
            if all(np.array_equal(first_ser_inx, new_data[i].index._data)
                        for i in range(1, len(new_data))):
                
                df_data = [s._data for s in new_data]
                return DataFrame(df_data, list(self.groups.keys()), first_ser_inx)
            
            # new_data is Series with different indexes
            ser_data = []
            multi_lv1 = [] 
            multi_lv2 = [] 

            for s, g_name in zip(new_data, self.groups):
                ser_data.extend(s._data)

                if self.group_keys:
                    multi_lv1.extend([g_name] * len(s))
                multi_lv2.extend(s.index._data)

            if self.group_keys:
                names = None
                if self._key_name:
                    names = [self._key_name, '']
                inx = MultiIndex.from_arrays([multi_lv1, multi_lv2], names)
            else:
                inx = Index(multi_lv2)

            return Series(ser_data, inx, self.series.name)
        
        if dfs_results:
            from concat import concat
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

    def __iter__(self):
        for key, indices in self.groups.items():
            yield key, self.series.iloc[indices]

class Resampler:
    funcs = {
        'sum': lambda x: x.sum(),
        'mean': lambda x: x.mean(),
        'count': lambda x: x.count(),
    }

    def __init__(self, series, groups, periods, rule, kind, label='right'):
        self.series = series
        self.groups = groups
        self.periods = periods
        self.rule = rule
        self.kind = kind
        self.label = label

    def _make_index(self):
        if self.kind == "period":
            return self.periods

        new_inx_data = []
        for period in self.periods._data:
            if self.label == 'left':
                val = period.start
            else:
                val = period.end

            new_inx_data.append(val)

        new_inx = DatetimeIndex(new_inx_data, self.rule)
        return new_inx

    def _agg(self, func):
        ser_gby = SeriesGroupBy(self.series, self.groups)
        res_ser = func(ser_gby)
        inx = self._make_index()
        res_ser.index = inx
        return res_ser


    def sum(self):
        return self._agg(self.funcs['sum'])
    
    def mean(self):
        return self._agg(self.funcs['mean'])
    
    def count(self):
        return self._agg(self.funcs['count'])


    def asfreq(self):
        new_inx = self._make_index()
        new_ser_data = np.full(len(new_inx), np.nan)

        for pos, inx in self.groups.items():
            if inx:
                new_ser_data[pos] = self.series._data[inx[0]]

        return Series(new_ser_data, new_inx, self.series.name)

    def ffill(self):
        res = self.asfreq()
        res.fillna(method='ffill')
        return res
    
    def ohlc(self):
        from dataframe import DataFrame
        open_data = []
        high_data = []
        low_data = []
        close_data = []
        data = self.series._data

        for inxs in self.groups.values():
            open_data.append(data[inxs[0]])
            close_data.append(data[inxs[-1]])
            high_data.append(np.max(data[inxs]))
            low_data.append(np.min(data[inxs]))

        return DataFrame({'open': open_data, 'high': high_data, 
                          'low': low_data, 'close': close_data})




class Rolling:
    funcs = {
        'mean': np.nanmean,
        'std': np.nanstd,
    }

    def __init__(self, series, window, min_periods, win_offset):
        self.series = series
        self.window = window
        self.min_periods = min_periods
        self.win_offset = win_offset

    def _agg(self, func):
        if not np.issubdtype(self.series._dtype, np.number):
            raise TypeError('No numeric types to aggregate')
        
        series_len = len(self.series)
        min_per = self.min_periods

        if isinstance(self.window, int):
            new_data = np.full(series_len, np.nan)

            while self.window > min_per:
                data = func(self.series._data[:min_per])
                new_data[min_per - 1] = data
                min_per += 1

            start = 0
            end = self.window

            while end <= series_len:
                data = func(self.series._data[start:end])
                new_data[end - 1] = data
                start += 1
                end += 1

            return Series(new_data, self.series.index, self.series.name)

        # window is freq
        ser_offset = get_offset(self.series.index.freq)
        new_data = Series(np.full(series_len, np.nan), self.series.index, self.series.name)

        first_date = self.series.index[0]
        while self.win_offset.n > min_per:
            curr_offset = self.win_offset.__class__(min_per)
            end_date = first_date + curr_offset
            if end_date not in new_data.index:
                end_date += ser_offset

            data = func(self.series._data[:end_date])
            new_data[end_date] = data
            min_per += 1

        start = to_datetime(first_date)
        end = first_date + self.win_offset

        if end not in new_data.index:
            end += ser_offset

        last_date = self.series.index._data[-1]
        while end <= last_date:
            data = func(self.series[start:end])
            new_data[end] = data
            start += ser_offset
            end += ser_offset

        return new_data
        

    def mean(self):
        return self._agg(self.funcs['mean'])

    def std(self):
        return self._agg(self.funcs['std'])
    
    def _calc_window_corr(self, data1, data2):
        var_1 = data1 - np.nanmean(data1)
        var_2 = data2 - np.nanmean(data2)

        non_nan_mask = ~np.isnan(var_1) & ~np.isnan(var_2)
        n = np.sum(non_nan_mask)

        if n < 2:
            return np.nan

        cov = np.nansum(var_1[non_nan_mask] * var_2[non_nan_mask]) / (n - 1)
        std1 = np.nanstd(data1[non_nan_mask])
        std2 = np.nanstd(data2[non_nan_mask])

        if std1 == 0 or std2 == 0:
            return np.nan

        return cov / (std1 * std2)

    def corr(self, series):
        if not isinstance(series, Series):
            raise TypeError("'corr' required series")
        
        align_s = series.reindex(self.series.index._data)
        series_len = len(self.series)
        new_data = np.full(series_len, np.nan)

        min_per = self.min_periods
        while self.window > min_per:
            data1 = self.series._data[:min_per]
            data2 = align_s._data[:min_per]
            data = self._calc_window_corr(data1, data2)

            new_data[min_per - 1] = data
            min_per += 1

        start = 0
        end = self.window

        while end <= series_len:
            data1 = self.series._data[start:end]
            data2 = align_s._data[start:end]

            data = self._calc_window_corr(data1, data2)
            new_data[end - 1] = data

            start += 1
            end += 1
        
        return Series(new_data, self.series.index, self.series.name)

class ExponentialMovingWindow:
    def __init__(self, series, span):
        self.series = series
        self.span = span
        self.alpha = 2 / (span + 1)

    def mean(self):
        if not np.issubdtype(self.series._dtype, np.number):
            raise TypeError('No numeric types to aggregate')

        new_data = [self.series._data[0]]
        for i in range(1, len(self.series)):
            num = self.series._data[i]
            prev = new_data[i - 1]
            new_data.append(self.alpha*num + (1 - self.alpha)*prev)

        return Series(new_data, self.series.index, self.series.name)


class StringMethods:
    def __init__(self, series):
        self.series = series

    def contains(self, substr):
        return np.fromiter((substr in val for val in self.series._data), bool)
    
    def split(self, sep=' '):
        data = np.fromiter((val.split(sep) for val in self.series._data), np.object_)
        return Series(data, self.series.index, self.series.name)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            stop = key.stop
            def gen():
                for data in self.series._data:
                    if stop < len(data):
                        new_data = data[slice(key.start, stop - 3, key.step)] + '...'
                    else:
                        new_data = data[key]
                    yield new_data 

            new_data = np.fromiter(gen(), np.object_, count=len(self.series._data))
        else:
            new_data = [s[key] for s in self.series._data]

        return Series(new_data, self.series.index, self.series.name)

class Loc:
    def __init__(self, series):
        self.series = series

    def __getitem__(self, key):
        if isinstance(key, Index):
            inxs = self.series.index.get_indexer(key._data)
            return self.series.iloc[inxs]

        if isinstance(key, list): #fancy
            if not isinstance(self.series.index, MultiIndex):
                if not key:
                    raise KeyError('Invalid empty list')

                inx = []
                def gen():
                    for label in key:
                        indexes = self.series.index.all_indexes(label)
                        indexes_len = len(indexes)

                        if not indexes_len:
                            raise IndexError(f'invalid label: {label}')
                        
                        if indexes_len == 1:
                            yield self.series._data[indexes[0]]
                            inx.append(label)
                        
                        else:
                            for i in indexes:
                                yield self.series._data[i]
                                inx.append(label)

                data = np.fromiter(gen(), self.series._dtype)
                return Series(data, index=inx, name=self.series.name, dtype=self.series._dtype)

            all_labels_inx = self.series.index.get_loc(key)
            labels_data = [self.series.index.levels[1][self.series.index.codes[1]._data[i]] for i in all_labels_inx]
            new_labels = Index(labels_data, self.series.index.levels[1].name)

            return Series(self.series._data[all_labels_inx], new_labels, self.series.name)

        if isinstance(key, tuple):
            key1, key2 = key

            if isinstance(key1, slice) and isinstance(key2, int):
                inx_slice = self.series.index.get_loc(key1)
                valid_codes = self.series.index.codes[1]._data[inx_slice]
                
                key2_code = self.series.index._find_inx_code(self.series.index.levels[1], key2)
                res_indexes = np.where(valid_codes == key2_code)[0]

                label_data = [self.series.index.levels[0][code] for code in self.series.index.codes[0]._data[res_indexes]]
                new_labels = Index(label_data, self.series.index.levels[0].name)

                return Series(self.series._data[res_indexes], new_labels, self.series.name)
        
        if isinstance(key, slice): # labes slice
            start = self.series.get_loc(key.start) if key.start else 0
            end = self.series.get_loc(key.stop) + 1 if key.stop else len(self.series.index)

            return Series(self.series._data[start:end], self.series.index._data[start:end], 
                          self.series.name, self.series._dtype)
        
        indexes = self.series.index.all_indexes(key)
        indexes_len = len(indexes)

        if not indexes_len:
            raise IndexError(f'invalid label: {key}')
        
        if indexes_len == 1:
            return self.series._data[indexes[0]]
        
        new_index = [key] * indexes_len
        return Series(self.series._data[indexes], new_index, self.series.name, self.series._dtype)
        
    def __setitem__(self, key, val):
        if isinstance(key, slice): # labes slice
            start = self.series.get_loc(key.start) if key.start else 0
            end = self.series.get_loc(key.stop) + 1 if key.stop else len(self.series.index)

            for i in range(start, end):
                self.series._data[i] = val
        
class Iloc(Loc):
    def __getitem__(self, key):
        if isinstance(key, (list, np.ndarray)):
            if not len(key):
                raise KeyError('Invalid empty key')
            
            if any(not isinstance(x, (int, np.integer)) for x in key):
                raise KeyError('key must contains only indexes')
            
            return Series(self.series._data[key], self.series.index._data[key],
                           self.series.name)
        
        if isinstance(key, int):
            return self.series._data[key]
        
        if isinstance(key, slice):
            return Series(self.series._data[key], self.series.index._data[key], self.series.name)
        
        raise KeyError(f'Invalid key {key}')
    
    def __setitem__(self, key, val):
        self.series._data[key] = val



class Series:
    def __init__(self, data, index=None, name='', dtype=None):
        if isinstance(data, np.ndarray):
            self._data = data
        elif isinstance(data, dict):
            if index:
                vals = [data.get(k) for k in index]
            else:
                vals = []
                index = []
                for k, v in data.items():
                    index.append(k)
                    vals.append(v)

            if dtype is not None and dtype != 'category':
                self._data = np.array(vals, dtype=dtype)
            else:
                self._data = np.array(vals)

            if all(isinstance(v, tuple) for v in index):
                self._index = MultiIndex.from_tuples(index)
            elif all(isinstance(v, datetime) for v in index):
                self._index = DatetimeIndex(index)
            else:
                self._index = Index(index)
        else:
            if dtype is not None and dtype != 'category':
                self._data = np.array(data, dtype=dtype)
            else:
                self._data = np.array(data)

        if index is not None:
            if isinstance(index, Index):
                self._index = index

            elif isinstance(index, (list, np.ndarray)):
                if all(isinstance(x, (list, np.ndarray)) for x in index):
                    self._index = MultiIndex.from_arrays(index)

                elif all(isinstance(x, tuple) for x in index):
                    self._index = MultiIndex.from_tuples(index)

                elif all(isinstance(x, datetime) for x in index):
                    self._index = DatetimeIndex(index)

                else:
                    if len(index) < len(data):
                        raise IndexError('invalid indexes')
                    
                    self._index = Index(index)

            else:
                if len(index) < len(data):
                    raise IndexError('invalid indexes')
                
                self._index = Index(index)

            if len(self._index) > len(self._data):
                self._data = np.append(self._data, np.full(len(self._index) - len(self._data), np.nan))
        else:
            self._index = RangeIndex(len(data))

        self.name = name

        if not np.issubdtype(self._data.dtype, np.number) and dtype != 'category':
            self._data = self._data.astype(np.object_)

        if dtype is None:
            self._dtype = self._data.dtype

        elif dtype == 'category':
            from cut_and_qcut import CategoricalAccessor, Categorical

            categorical = Categorical(self._data)
            self._data = CategoricalAccessor(categorical, self)
            self._dtype = dtype
        else:
            self._dtype = dtype

    def astype(self, tp):
        if tp == 'category':
            return Series(self._data, self.index, self.name, dtype='category')
        
        return Series(self._data.astype(tp), self.index, self.name)

    def __getattr__(self, name):
        if name == 'cat':
            if self._dtype != 'category':
                raise AssertionError("Can only use .cat accessor with a 'category' dtype")

            return self._data
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    def value_counts(self):
        if self._dtype == 'category':
            return self.cat.value_counts()
        
        counts = Counter(self._data)
        sorted_counts = counts.most_common(len(counts))
        ser_data = {str(val): count for val, count in sorted_counts}

        return Series(ser_data, name=self.name)
        
    def argsort(self):
        inxs = self._data.argsort()
        return Series(inxs, self.index._data[inxs], self.name)

    def to_dict(self):
        return {i: v for i, v in zip(self.index, self._data)}

    def get_loc(self, label, _error=True):
        return self.index.index(label, _error)

    @property
    def array(self):
        return self._data
    
    @property
    def values(self):
        return self._data
    
    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, val):
        if len(self._data) != len(val):
            raise IndexError('invalid indexes')
        
        if not isinstance(val, Index):
            val = Index(val)

        self._index = val

    @property
    def loc(self):
        return Loc(self)
    
    @property
    def iloc(self):
        return Iloc(self)

    @property
    def str(self):
        if not np.issubdtype(self._dtype, np.object_):
            raise AttributeError('Can only use .str accessor with string values!')
        
        return StringMethods(self)

    def __getitem__(self, key):
        return self.index._ser_get_item(self, key)

    def __setitem__(self, key, val):
        if isinstance(self.index, DatetimeIndex):
            if self.index.freq is None:
                inx = self.index.index(key)
            else:
                inx = self.index.index(key, False)
            
            offset = get_offset(self.index.freq)
            while inx == -1:
                key += offset
                inx = self.index.index(key, False)

            self._data[inx] = val
            return
        
        if isinstance(key, Series):
            if len(key) != len(self):
                raise ValueError("Boolean index must be the same length as Series")

            self._data[key._data.astype(bool)] = val
            return

        if isinstance(key, list):
            for k in key:
                self._data[self.get_loc(k)] = val
            return

        self._data[self.get_loc(key)] = val
        
    def shift(self, periods, freq=None):
        if not periods:
            return self
        
        if freq is None:
            empty_part = np.full(abs(periods), np.nan)
            if periods > 0:
                other_vals = self._data[:len(self._data) - periods]
                new_data = np.concatenate([empty_part, other_vals])
            else:
                other_vals = self._data[len(self._data) + periods:]
                new_data = np.concatenate([other_vals, empty_part])

            return Series(new_data, self.index, self.name)
        
        offset = get_offset(freq)
        if not isinstance(offset, (Minute, Day, Hour)):
            periods += 1

        offset *= periods

        if periods > 0:
            new_inx = [offset + date for date in self.index]
        else:
            new_inx = [offset - date for date in self.index._data]

        return Series(self._data, new_inx, self.name)
    
    def asfreq(self, freq, how='end'):
        if not isinstance(self.index, PeriodIndex):
            raise TypeError('Index is not PeriodInex')
        
        new_data = []
        for period in self.index._data:
            new_data.append(period.asfreq(freq, how))

        new_inx = PeriodIndex(new_data, freq, self.index.name)
        return Series(self._data, new_inx, self.name)
    
    def _resample_periodinx_groups(self, new_per, inx_per, closed):
        add_item = False
        if closed == 'left' and \
            new_per.start <= inx_per.start and inx_per.end < new_per.end:
            add_item = True
        elif closed == 'right' and \
            new_per.start < inx_per.start and inx_per.end <= new_per.end:
            add_item = True

        return add_item

    def _set_closed(self, closed, sampling, sides, label):
        if closed is not None:
            if closed not in sides:
                raise ValueError('Invalid closed')
        else:
            if sampling == 'upsampling':
                closed = 'right'
                if isinstance(self.index, DatetimeIndex):
                    label = 'right'
            else: # downsampling
                closed = 'left'
                if isinstance(self.index, DatetimeIndex):
                    label = 'right'

        return closed, label

    def _get_sampling(self, old_freq, new_freq):
        old_freq = 'T' if old_freq == 'min' else old_freq
        new_freq = 'T' if new_freq == 'min' else new_freq

        old_freq_val = frequencies.index(old_freq)
        new_freq_val = frequencies.index(new_freq)

        return 'downsampling' if old_freq_val <= new_freq_val else 'upsampling'

    def resample(self, rule: str, kind=None, closed=None, label='left'):
        if not isinstance(self.index, (PeriodIndex, DatetimeIndex)):
            raise TypeError('Only valid with DatetimeIndex or PeriodIndex')
        
        if not self.index.is_monotonic:
            raise ValueError('Index must be monotonic')
        
        if not isinstance(rule, str):
            raise TypeError('Invalid type for "rule"')

        kind_is_none = False
        if kind is not None:
            if kind not in ("timestamp", "period"):
                raise ValueError('Invalid kind')
        else:
            kind_is_none = True
        
        sides = ('left', 'right')
        if label not in sides:
            raise ValueError('Invalid label') 
        

        _, old_freq_type, *_ =   parse_freq(self.index.freq)
        _, rule_type, ex1, ex2 = parse_freq(rule)

        sampling = self._get_sampling(old_freq_type, rule_type)
        
        if isinstance(self.index, DatetimeIndex):
            if kind_is_none:
                kind = "timestamp"

            closed, label = self._set_closed(closed, sampling, sides, label)
            
            start = Period.str_repr[rule_type](pd.Timestamp(self.index._data[0]))
            end =   Period.str_repr[rule_type](pd.Timestamp(self.index._data[-1]))

            if closed == 'right':
                offset = get_offset(rule)
                start = Period.str_repr[rule_type](pd.Timestamp(start - offset))
                
            period_inx = period_range(start, end, freq=rule)

            groups = {i: [] for i in range(len(period_inx))}
            for i, date in enumerate(self.index._data):
                for j, period in enumerate(period_inx._data):
                    add_item = False
                    if closed == 'left' and \
                        period.start <= date < period.end:
                        add_item = True

                    elif closed == 'right' and \
                        period.start < date <= period.end:
                        add_item = True
                    
                    if add_item:
                        groups[j].append(i)
                        break

            return Resampler(self, groups, period_inx, rule, kind, label)

        # index is PeriodIndex
        if kind_is_none:
            kind = "period"

        closed, label = self._set_closed(closed, sampling, sides, label)

        start = Period.str_repr[rule_type](pd.Timestamp(self.index._data[0].start), ex1, ex2)
        end =   Period.str_repr[rule_type](pd.Timestamp(self.index._data[-1].end), ex1, ex2)
        period_inx = period_range(start, end, freq=rule)

        groups = {i: [] for i in range(len(period_inx))}
        for i, inx_per in enumerate(self.index._data):
            for j, new_per in enumerate(period_inx._data):
                if sampling == 'downsampling':
                    add_item = self._resample_periodinx_groups(new_per, inx_per, closed)
                else: # upsampling
                    add_item = self._resample_periodinx_groups(inx_per, new_per, closed)
                
                if add_item:
                    groups[j].append(i)
                    break

        return Resampler(self, groups, period_inx, rule, kind, label)        

    def to_period(self, freq=None):
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError('"to_period" required "DatetimeIndex"')
        
        if freq is None and self.index.freq is None:
            raise ValueError('You must supply a freq argument as the DatetimeIndex does not have one.')

        freq = freq or self.index.freq
        _, freq_type, *_ = parse_freq(freq)

        new_inx_data = []
        for date in self.index._data:
            date = pd.Timestamp(date)
            p_date_str = Period.str_repr[freq_type](date)
            new_inx_data.append(Period(p_date_str, freq))
        
        new_inx = PeriodIndex(new_inx_data, freq, self.index.name)
        return Series(self._data, new_inx, self.name)

    def _check_min_periods(self, min_periods, win_val):
        if min_periods is None:
            return win_val
        
        if isinstance(min_periods, int) and min_periods <= win_val:
            return min_periods
        
        raise ValueError('min_periods must be int <= window')

    def rolling(self, window, min_periods=None):
        win_offset = None
        if isinstance(window, str):
            if not isinstance(self.index, DatetimeIndex):
                raise ValueError('window must be an integer 0 or greater')
            win_offset = get_offset(window)
            min_periods = self._check_min_periods(min_periods, win_offset.n)
        elif isinstance(window, int):
            if window <= 0:
                raise ValueError('window must be an integer 0 or greater')
            min_periods = self._check_min_periods(min_periods, window)
        else:
            raise TypeError('window must be str->ferq or int')

        return Rolling(self, window, min_periods, win_offset)

    def ewm(self, span):
        if not isinstance(span, int) or span <= 0:
            raise ValueError('span must be an integer 0 or greater')
        
        return ExponentialMovingWindow(self, span)

    def reindex(self, index, fill_value=np.nan, method=None): # method 'ffill'/'bfill'
        if not hasattr(index, '__iter__') or isinstance(index, str):
            raise ValueError(f'index must be iterable, not: {type(index).__name__}')
        
        if not isinstance(index, Index):
            index = Index(index)

        data = []
        counter = 1
        for i in range(len(index)):
            inx = self.get_loc(index[i], False)

            if inx != -1:
                data.extend([self._data[inx]] * counter)
                counter = 1
                
            elif method == 'bfill':
                if i == len(index) -1:
                    data.extend([fill_value] * (len(index) - len(data)))
                counter += 1

            elif method == 'ffill':
                if data:
                    val = data [-1]
                else:
                    val = fill_value
                
                data.append(val)
            else:
                data.append(fill_value)

        return Series(data, index, self.name)
    
    def reset_index(self):
        from dataframe import DataFrame
        new_index = RangeIndex(len(self.index))

        if not isinstance(self.index, MultiIndex):
            new_cols = []
            val_i = self.index.name or 0
            new_cols.append(val_i)
            
            val_s = self.name or 1
            self.name = val_s
            new_cols.append(val_s)

            new_s = Series(self.index._data, val_i)
            new_data = [new_s, self]

            return DataFrame(new_data, new_index, new_cols)

        new_cols = []
        if self.index.names is not None:
            new_cols.extend(self.index.names + [self.name or '0'])
            self.name = self.name or '0'
            s1_name = self.index.names[0]
            s2_name = self.index.names[1]
        else:
            new_cols.extend([0, 1] + [self.names or 2])
            self.name = self.name or 2
            s1_name = 0
            s2_name = 1

        new_s1 = Series(self.index._level_1, new_index, s1_name)
        new_s2 = Series(self.index._level_2, new_index, s2_name)

        new_data = [new_s1, new_s2, self]
        return DataFrame(new_data, new_index, new_cols)

    def drop(self, label):
        if isinstance(label, (list, np.ndarray)):
            new_data = self._data
            new_index = self.index

            for l in label:
                inx = new_index.all_indexes(l)
                for i in sorted(inx, reverse=True):
                    new_index = new_index.delete(i)
                    new_data = np.delete(new_data, i)

            return Series(new_data, new_index, name=self.name)
        
        inx = self.get_loc(label)
        return Series(np.delete(self._data, inx), self.index.delete(inx), name=self.name)
    
    def notna(self):
        return (~pd.isna(self._data)) & (self._data != 'nan')

    def dropna(self):
        mask = self.notna()
        return Series(self._data[mask], self.index._data[mask], self.name)
    
    def fillna(self, val=None, method=None, limit=None): # method 'ffill'/'bfill'
        if val is not None and method is not None:
            raise ValueError("Cannot specify both 'value' and 'method'")
        
        if val is not None:
            mask = pd.isna(self._data)

            if limit is None:
                self._data[mask] = val
                return self
            
            for i, bl in enumerate(mask):
                if bl:
                    self._data[i] = val
                    limit -= 1

                    if limit == 0:
                        break
            return self
        
        if method == 'ffill':
            mask = pd.isna(self._data)

            for i, bl in enumerate(mask):
                if bl and i != 0:
                    self._data[i] = self._data[i - 1]

                    if limit is not None:
                        limit -= 1

                        if limit == 0:
                            break
            return self

        if method == 'bfill':
            mask = pd.isna(self._data)
            mask_len = len(mask)

            for i in range(mask_len -1, -1, -1):
                if mask[i] and i != mask_len - 1:
                    self._data[i] = self._data[i + 1]

                    if limit is not None:
                        limit -= 1

                        if limit == 0:
                            break
            return self

        raise TypeError("Must specify a fill 'value' or 'method'.")
    
    def duplicated(self, keep='first'):
        if keep == 'last':
            def gen():
                vals = set()
                for i in range(len(self._data) -1, -1, -1):
                    val = self._data[i]
                    yield val in vals
                    vals.add(val)

            return Series(np.fromiter(gen(), bool)[::-1], self.index, self.name, bool)

        if keep == 'first':
            def gen():
                vals = set()
                for val in self._data:
                    yield val in vals
                    vals.add(val)

            return Series(np.fromiter(gen(), bool), self.index, self.name, bool)
        
        raise ValueError(f'Invalid keep: {keep}')
    
    def drop_duplicates(self, keep="first"):
        mask = self.duplicated(keep)
        indexes = np.where(mask)[0]
        inx_labels = self.index.to_numpy()[indexes] if isinstance(self.index, RangeIndex) else self.index._data[indexes]
        return self.drop(inx_labels)
    
    def replace(self, old_val, new_val=None):
        if isinstance(old_val, dict):
            new_data = np.fromiter((old_val.get(el, el) for el in self._data), self._dtype)
            return Series(new_data, self.index, self.name, self._dtype)

        if not isinstance(old_val, list):
            old_val = [old_val]

        if not isinstance(new_val, list): 
            new_data = self._data.copy()
            new_data[np.isin(self._data, old_val)] = new_val
            return Series(new_data, self.index, self.name, self._dtype)
        
        if len(old_val) != len(new_val):
            raise ValueError('Replacement lists must be of same length.')
        
        d = {old: new for old, new in zip(old_val, new_val)}
        new_data = np.fromiter((d.get(el, el) for el in self._data), self._dtype)
        return Series(new_data, self.index, self.name, self._dtype)
    
    def describe(self):
        if np.issubdtype(self._data.dtype, np.number):
            describe_data = [sum(~pd.isna(self._data)), self.mean(),
                              round(np.nanstd(self._data), 3), np.nanmin(self._data),
                                self.quantile(0.25), self.median(), self.quantile(0.75), 
                                np.nanmax(self._data)]
            describe_inx = Index(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

            return Series(describe_data, describe_inx, self.name, float)
        
        most_common, freq = Counter(self._data).most_common(1)[0]

        describe_data = [len(self._data), len(np.unique(self._data)), most_common, freq]
        describe_inx = Index(['count', 'unique', 'top', 'freq'])

        return Series(describe_data, describe_inx, self.name)
    
    def take(self, inx):
        if not hasattr(inx, '__iter__'):
            raise TypeError(f'index must be a iterable not: {type(inx).__name__}')
        
        if not isinstance(inx, list):
            inx = list(inx)
        
        return self.iloc[inx]
    
    def sample(self, n, replace=False):
        if not isinstance(n, int):
            raise TypeError(f'Invalid type for n: {type(n).__name__}')
        
        if n > len(self.index) and not replace:
            raise ValueError("Cannot take a larger sample than population when 'replace=False'")
        
        if replace:
            inx = np.random.choice(len(self.index), size=n, replace=True)
        else:
            inx = np.random.permutation(n)

        return self.take(inx)

    def unstack(self, level=1):
        if not isinstance(self.index, MultiIndex):
            raise ValueError("index must be a MultiIndex to unstack")
        
        if not isinstance(level, int):
            level = self.index.names.index(level)

        if level == 0:
            data = {col: {inx: np.nan for inx in np.unique(self.index._level_2)} 
                        for col in np.unique(self.index._level_1)}

            for i, (lv1, lv2) in enumerate(self.index):
                data[lv1][lv2] = self._data[i]

            from dataframe import DataFrame
            df = DataFrame(data)

            if self.index.names is not None:
                df.index.name = self.index.names[1]
                df.columns.name = self.index.names[0]

            return df.fillna(0)

        if level == 1:
            nested_dict = {inx: np.nan for inx in np.unique(self.index._level_1)}
            data = {col: nested_dict.copy() for col in np.unique(self.index._level_2)}

            for i, (lv1, lv2) in enumerate(self.index):
                data[lv2][lv1] = self._data[i]

            from dataframe import DataFrame
            df = DataFrame(data)

            if self.index.names is not None:
                df.index.name = self.index.names[0]
                df.columns.name = self.index.names[1]

            return df.fillna(0)
        
        raise ValueError("Invalid level")

    def combine_first(self, other):
        nan_pos = np.where(pd.isna(self._data))[0]
        nan_pos_inx = self.index._data[nan_pos]

        other_inx = other.index.get_indexer(nan_pos_inx)
        new_data = self._data.copy()
        new_data[nan_pos] = other._data[other_inx]
        new_index = self.index

        intersect = np.setdiff1d(other.index._data, self.index._data)
        if len(intersect):
            new_index = new_index.append(intersect)
            intersect_data_inx = other.index.get_indexer(intersect)
            new_data = np.concatenate([new_data, other._data[intersect_data_inx]])

        sorted_inx = np.argsort(new_index)
        new_index = new_index[sorted_inx]
        new_data = new_data[sorted_inx]

        return Series(new_data, new_index, self.name)
    
    def _get_start_pos(self, x):
        left, right = 0, len(self.index._data) - 1
        result = None
        
        while left <= right:
            mid = (left + right) // 2
            if self.index._data[mid] <= x:
                result = mid 
                left = mid + 1
            else:
                right = mid - 1

        return result

    def asof(self, label):
        if not self.index.is_monotonic or self.index._data[0] > self.index._data[-1]:
            raise ValueError('asof requires index ot be monotonic and in ascending order')

        inx = self.index.index(label, False)
        if inx != -1:
            return self._data[inx]
        
        if label < self.index._data[0]:
            return None
        
        if label > self.index._data[-1]:
            start = len(self.index) - 1
        else:
            start = self._get_start_pos(label)

        for i in range(start, -1, -1):
            val = self._data[i]
            if not pd.isna(val):
                return val

        return None

    def barplot(self, ax=None, figsize=None, bar_width=0.7, x_offset=0, bar_count=1, 
                bottom_p=None, bottom_n=None, stacked=False, **kwargs):
        from matplotlib.patches import Rectangle

        if ax is None:
            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()
        elif not isinstance(ax, Axes):
            raise TypeError(f'Invalid type for ax: {type(ax)}')

        bar_half = (bar_width * bar_count) / 2

        for pos, val in enumerate(self._data):
            if stacked:
                bottom = bottom_p[pos] if val >= 0 else bottom_n[pos]
                rect = Rectangle((pos - bar_half + x_offset, bottom), bar_width, val, **kwargs)

                if val >= 0:
                    bottom_p[pos] += val
                else:
                    bottom_n[pos] += val
            else:
                rect = Rectangle((pos - bar_half + x_offset, 0), bar_width, val, **kwargs)

            ax.add_patch(rect)

        y_start = np.min(self._data) if np.min(self._data) < 0 else 0

        ax.set_xticks(range(len(self.index._data)))
        ax.set_xticklabels(self.index._data)

        ax.set_xlim(-1, len(self._data))
        ax.set_ylim(y_start, np.max(self._data) * 1.1)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        if bottom_p is not None and bottom_n is not None:
            return bottom_p, bottom_n
        
        return ax

    def plot(self, label=None, ax=None, style=None, alpha=None, bins=None, kind='line', figsize=None, logy=False, 
             title=None, use_index=True, rot=None, xticks=None, 
             yticks=None, xlim=None, ylim=None, grid=False, color=None):
        
        if ax is None:
            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()
        elif not isinstance(ax, Axes):
            raise TypeError(f'Invalid type for ax: {type(ax)}')


        if use_index:
            x = self.index._data
        else:
            x = np.arange(len(self))

        y = self._data

        if kind == 'line':
            if style is not None:
                ax.plot(x, y, style, alpha=alpha, label=label, color=color)
            else:
                ax.plot(x, y, alpha=alpha, label=label, color=color)
                
        elif kind == 'bar':
            ax.bar(x, y, alpha=alpha, label=label, color=color)

        elif kind == 'barh':
            ax.barh(x, y, alpha=alpha, label=label, color=color)

        elif kind == 'hist':
            ax.hist(y, bins=bins, alpha=alpha, label=label, color=color)

        elif kind == 'scatter':
            ax.scatter(x, y, alpha=alpha, label=label, color=color)

        else:
            raise ValueError(f"Invalid kind: {kind}")


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
            ax.set_xticklabels(ax.get_xticks(), rotation=rot)

        if logy:
            ax.set_yscale('log')

        if label is not None:
            ax.legend()

        return ax
    
    def _groupby_multikey(self, key):
        if isinstance(key, Series):
            return key._data, key.name

        if isinstance(key, (list, np.ndarray)):
            return key, ''
        
        if isinstance(key, dict):
            key = [key.get(i, np.nan) for i in self.index]
            return key, ''

        if callable(key):
            key = [key(i) for i in self.index]
            return key, ''
        
        raise TypeError('Invalid type for multi key')

    def _get_multiindex_level(self, key):
        if not isinstance(key, int):
            inx = self.index.names.index(key)
            return self._get_multiindex_level(inx)

        names = self.index.names or ['', '']
        if key == 0:
            return self.index._level_1, names[0]
        if key == 1:
            return self.index._level_2, names[1]
        
        raise ValueError('Invalid index level for MultiIndex')

    def groupby(self, key=None, dropna=True, level=None, group_keys=True):
        if key is not None and level is not None:
            raise TypeError("Cannot specify both 'key' and 'level'")
        
        if key is not None and not callable(key) and \
            not isinstance(key, (np.ndarray, list, Series, dict)):
            raise TypeError(f'"key" type must be "Series"/"list"/"ndarray"/"dict" or func, not: {type(key).__name__}')
        
        key_name = ''
        if level is not None:
            if not isinstance(self.index, MultiIndex):
                raise TypeError('Series index is not MultiIndex!')
            
            if isinstance(level, list):
                if len(level) != 2:
                    raise ValueError('MultiIndex supports 2 levels')
                
                key1, name1 = self._get_multiindex_level(level[0])
                key2, name2 = self._get_multiindex_level(level[1])
                key = [key1, key2]
                key_name = [name1, name2]

            else:
                key, key_name = self._get_multiindex_level(level)
        
        if isinstance(key, Series):
            key_name = key.name
            key = key._data

        if isinstance(key, dict):
            key = [key.get(i, np.nan) for i in self.index]

        if callable(key):
            key = [key(i) for i in self.index]

        is_multi = False
        key_len = len(key) 
        if key_len != len(self._data):
            if isinstance(key, list) and key_len == 2:
                is_multi = True
            else:
                raise ValueError('Invlaid key length')
            
        if is_multi:
            key1, name1 = self._groupby_multikey(key[0])
            key2, name2 = self._groupby_multikey(key[1])
            
            self_len = len(self._data)
            if len(key1) != self_len or len(key2) != self_len:
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

        return SeriesGroupBy(self, groups, key_name, is_multi, group_keys)

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

    def nsmallest(self, n, keep='first'):
        if n > len(self._data):
            n = len(self._data)
        
        positions = np.arange(len(self._data))
        data, pos = self._nsmallest_alg(self._data, n, keep, positions)

        data, pos = np.array(data), np.array(pos) 
        sort_inx = np.argsort(data)

        sort_data = data[sort_inx]
        sort_pos = pos[sort_inx]
        new_inx = self.index._data[sort_pos]
        return Series(sort_data, index=new_inx, name=self.name)

    @staticmethod
    def _nlargest_alg(arr, n, keep, positions):
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

        if len(big) > n:
            return Series._nlargest_alg(big, n, keep, big_pos)

        if len(big) + len(equal) >= n:
            remainder = n - len(big)

            if keep == 'first':
                data = big + equal[:remainder]
                pos = big_pos + equal_pos[:remainder]
            elif keep == 'last':
                data = big + equal[-remainder:]
                pos = big_pos + equal_pos[-remainder:]
            elif keep == 'all':
                data = big + equal
                pos = big_pos + equal_pos
            else:
                raise ValueError("keep must be 'first', 'last', or 'all'")

            return data, pos

        n = n - (len(big) + len(equal))
        data_small, pos_small = Series._nlargest_alg(small, n, keep, small_pos)

        data = big + equal + data_small
        pos = big_pos + equal_pos + pos_small
        return data, pos

    def nlargest(self, n, keep='first'):
        if n > len(self._data):
            n = len(self._data)

        positions = np.arange(len(self._data))
        data, pos = self._nlargest_alg(self._data, n, keep, positions)

        data, pos = np.array(data), np.array(pos)
        sort_idx = np.argsort(data)[::-1] 

        data = data[sort_idx]
        pos = pos[sort_idx]

        inx_data = self.index._data[pos]
        if isinstance(self.index, MultiIndex):
            new_index = MultiIndex.from_tuples(inx_data, self.index.names)
        else:
            new_index = Index(inx_data, self.index.name)

        return Series(data, index=new_index, name=self.name)

    @staticmethod
    def _rank_pos_for_unique(arr, n):
        left = 0
        right = len(arr) - 1
        pos = -1
        while left <= right:
            mid = (right + left) // 2
            mid_num = arr[mid]

            if mid_num > n:
                right = mid - 1
            else:
                left = mid + 1

            if mid_num == n:
                pos = mid
                break

        return pos + 1
    
    @staticmethod
    def _rank_pos_for_notunique(arr, n):
        left = 0
        right = len(arr) - 1
        first = -1
        while left <= right:
            mid = (right + left) // 2
            mid_num = arr[mid]

            if mid_num >= n:
                right = mid - 1
            else:
                left = mid + 1

            if mid_num == n:
                first = mid

        left = first
        right = len(arr) - 1

        last = -1
        while left <= right:
            mid = (right + left) // 2
            mid_num = arr[mid]

            if mid_num <= n:
                left = mid + 1
            else:
                right = mid - 1

            if mid_num == n:
                last = mid

        return (first + 1, last + 1)

    def rank(self, method="average"):
        methods = {
            "average": lambda pos, _: sum(pos) / len(pos),
            "min": lambda pos, _: min(pos),
            "max": lambda pos, _: max(pos),
            "first": lambda pos, item: pos[dict_first_method[item]],
            "dense": lambda _, item: self._rank_pos_for_unique(unique_data, item)
        }

        if method not in methods:
            raise ValueError('Invalid method')

        sorted_data = np.sort(self._data)
        unique_data = np.unique(sorted_data)

        if len(self._data) == len(unique_data):
            gen = (self._rank_pos_for_unique(sorted_data, item) for item in self._data)
            new_data = np.fromiter(gen, dtype=int)

            return Series(new_data, index=self.index, name=self.name, dtype=int)
        
        dict_first_method = {}

        def gen():
            for item in self._data:
                if method == 'dense':
                    yield methods[method](None, item)
                    continue

                first_pos, last_pos = self._rank_pos_for_notunique(sorted_data, item)

                if first_pos == last_pos:
                    yield first_pos
                    continue
                
                if method == 'first':
                    dict_first_method[item] = dict_first_method.get(item, -1) + 1
                    
                all_pos = list(range(first_pos, last_pos + 1))
                yield methods[method](all_pos, item)

        return Series(np.fromiter(gen(), dtype=float), index=self.index, name=self.name, dtype=float)
    
    def sort_index(self, ascending=True, inplace=True):
        pos = self.index._data.argsort()

        if not ascending:
            pos[:] = pos[::-1]
            
        new_index = self.index._data[pos]

        if inplace:
            self._data[:] = self._data[pos]
            self.index = new_index
            return
        
        return Series(self._data[pos], index=new_index, name=self.name, dtype=self._dtype)
    
    def sort_values(self, ascending=True, inplace=False, na_position='last'):
        if self._dtype != np.object_:
            pos = self._data.argsort()
        else:
            pos = np.array(sorted(range(len(self._data)), key=lambda x: (pd.isna(self._data[x]), self._data[x])))

        if ascending:
            if na_position == 'first':
                nan_count = np.sum(pd.isna(self._data))
                pos[:] = np.roll(pos, nan_count)

        else:
            pos[:] = pos[::-1]

            if na_position == 'last':
                nan_count = np.sum(pd.isna(self._data))
                pos[:] = np.roll(pos, -nan_count)

        if inplace:
            self._data[:] = self._data[pos]
            self.index = self.index[pos]
            return

        inx = self.index.__class__(self.index._data[pos], self.index.name)
        return Series(self._data[pos], inx, name=self.name)

    def map(self, arg):
        if isinstance(arg, Series):
            def gen():
                for item in self._data:
                    if (inx := arg.index.index(item, False)) < 0:
                        yield np.nan
                    else:
                        yield arg._data[inx]

            new_data = np.fromiter(gen(), arg._dtype)
            return Series(new_data, index=self.index, name=self.name, dtype=arg._dtype)
        
        if isinstance(arg, dict):
            new_data = np.fromiter((arg.get(item, np.nan) for item in self._data), dtype=np.object_)
            return Series(new_data, index=self.index, name=self.name)
        
        if callable(arg):
            new_data = np.fromiter((arg(item) for item in self._data), dtype=np.object_)
            return Series(new_data, index=self.index, name=self.name)
        
        raise TypeError(f'Invalid type for arg: {type(arg).__name__}')
    
    def median(self):
        ignore_nan_data = self._data[~pd.isna(self._data)]
        sorted_arr = np.sort(ignore_nan_data)
        len_arr = len(sorted_arr)
        
        if len_arr % 2 == 0:
            inx = len_arr // 2
            return (sorted_arr[inx] + sorted_arr[inx - 1]) / 2

        return sorted_arr[math.floor(len_arr / 2)]
    
    def quantile(self, q=0.5):
        if not (0 <= q <= 1):
            return ValueError('param q must be in range 0-1')
        
        if q == 0.5:
            return self.median()
        
        ignore_nan_data = self._data[~pd.isna(self._data)]
        sorted_arr = np.sort(ignore_nan_data)
        len_arr = len(sorted_arr)

        inx = (len_arr - 1) * q

        if (intval := int(inx)) == inx:
            return sorted_arr[intval]
        
        inx1 = math.floor(inx)
        inx2 = math.ceil(inx)
        f = inx - inx1

        # print((sorted_arr[inx1] + sorted_arr[inx2]) / 2)
        return (1 - f) * sorted_arr[inx1] + f * sorted_arr[inx2]
    
    def pct_change(self, periods=1):
        gen = (np.nan if i <= periods -1 else 
               round((num - self._data[i - periods]) / self._data[i - periods], 6) 
               for i, num in enumerate(self._data))
        return Series(np.fromiter(gen, dtype=np.float64), index=self.index, name=self.name, dtype=np.float64)
    
    def var(self, ddof=1):
        if ddof not in {0, 1}:
            raise ValueError(f'ddof must be 1 or 0, not: {ddof}')
        
        ignore_nan_data = self._data[~pd.isna(self._data)]
        mean_val = ignore_nan_data.mean()
        data_minus_mean = ignore_nan_data - mean_val
        data_minus_mean **= 2
        sum_val = np.sum(data_minus_mean)
        return sum_val / (len(ignore_nan_data) - ddof)
    
    def std(self):
        return round(np.std(self._data), 4)
    
    def sum(self):
        return np.nansum(self._data)

    def cumsum(self):
        if not np.issubdtype(self._dtype, np.number):
            raise TypeError('cumsum required numeric series')
        
        return Series(np.cumsum(self._data), self.index, self.name)

    def diff(self, periods=1):
        gen = (np.nan if i <= periods -1 else num - self._data[i - periods]
               for i, num in enumerate(self._data))
        return Series(np.fromiter(gen, dtype=np.float64), index=self.index, name=self.name, dtype=np.float64)
    
    def mean(self):
        return np.round(np.nanmean(self._data), 4)

    def max(self):
        return round(np.nanmax(self._data), 4)
    
    def min(self):
        return round(np.nanmin(self._data), 4)

    def count(self):
        return np.sum(~pd.isna(self._data))

    def searchsorted(self, target_val):
        # return np.searchsorted(self._data, target_val)
        left = 0
        right = len(self._data) - 1
        first = 0
        while left <= right:
            mid = (right + left) // 2
            mid_num = self._data[mid]

            if mid_num >= target_val:
                right = mid - 1
                first = mid
            else:
                left = mid + 1

        return first

    def unique(self):
        union_unsorted = np.unique(self._data, return_index=True)
        return union_unsorted[0][np.argsort(union_unsorted[1])]
    
    def isin(self, other):
        if isinstance(other, Series):
            other = other._data

        return Series(np.isin(self._data, other), self.index, self.name)

    def idxmax(self):
        return self.index._data[np.where(self._data == np.max(self._data))[0][0]]

    def __contains__(self, key):
        return np.any(np.isin(self._index, key)) or np.any(np.isin(self._data, key))
    
    def _op(self, other, func):
        if np.isscalar(other):
            data = func(self._data, other)
            return Series(data, self._index, self.name)
        
        raise TypeError(f'Invalid type: {type(other).__name__}')

    def __gt__(self, other):
        return self._op(other, lambda x, y: x > y)
    
    def __ge__(self, other):
        return self._op(other, lambda x, y: x >= y)

    def __lt__(self, other):
        return self._op(other, lambda x, y: x < y)
    
    def __eq__(self, other):
        return self._op(other, lambda x, y: x == y)
    
    def _mat_ops(self, other, func):
        if isinstance(other, Series):
            index = self.index.union(other.index)
            vals = []

            for key in index:
                if key in other.index and key in self.index:
                    res = func(self[key], other[key])
                else:
                    res = None
    
                vals.append(res)
            # vals = func(self._data, other._data)
            return Series(vals, index, self.name)
        
        return Series(func(self._data, other), self.index, self.name)
    
    def __add__(self, other):
        return self._mat_ops(other, lambda x, y: x + y)
    
    def __sub__(self, other):
        return self._mat_ops(other, lambda x, y: x - y)
        
    def __mul__(self, other):
        return self._mat_ops(other, lambda x, y: x * y)
    
    def __truediv__(self, other):
        return self._mat_ops(other, lambda x, y: x / y)
    
    def __pow__(self, val):
        return Series(self._data**val, self.index, self.name)
    
    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)

    def head(self, n=5):
        return Series(self._data[:n], self.index._data[:n], self.name)

    def tail(self, n=5):
        return Series(self._data[-n:], self.index._data[-n:], self.name)

    def __str__(self):
        if not isinstance(self.index, MultiIndex):
            width = max(len(str(x)) for x in self.index)
            if (inx_name_len := len(self.index.name)) > width:
                width = inx_name_len

            res = f'{self.index.name}\n' if self.index.name else ''

            data = []
            for inx, val in zip(self.index, self._data):
                if self._dtype == 'category':
                    str_row = f'{inx:<{width}}   {self.cat.categories[val]}'
                else:
                    str_row = f'{str(inx):<{width}}   {val}'
                data.append(str_row)

            if isinstance(self.index, (DatetimeIndex, PeriodIndex)):
                footer = f'\nFreq: {self.index.freq}, '
                footer += f'Name: {self.name}, ' if self.name else ''
            else:
                footer = f'\nName: {self.name}, ' if self.name else '\n'
            footer += 'dtype: '

            if hasattr(self._dtype, '__name__'):
                footer += self._dtype.__name__
            else:
                footer += str(self._dtype)

            res = res + '\n'.join(data) + footer

            if self._dtype == 'category':
                res += '\n' + self.cat.type

            return res
        

        inx_width_levels = []
        inx_width_levels.append(max(len(str(x)) for x in self.index._level_1))
        inx_width_levels.append(max(len(str(x)) for x in self.index._level_2))

        inx_names = []
        if self.index.names is not None:
            for i in range(len(inx_width_levels)):
                current_name = self.index.names[i]

                if len(current_name) > inx_width_levels[i]:
                    inx_width_levels[i] = len(current_name)

                inx_names.append(f'{current_name:<{inx_width_levels[i]}}')
            inx_names[-1] += '\n'

        data = []
        for i, (inx, val) in enumerate(zip(self.index, self._data)):
            if inx[0] is None:
                inx = (str(inx[0]), inx[1])

            if inx[1] is None:
                inx = (inx[0], str(inx[1]))

            if i == 0 or inx[0] != self.index._data[i - 1][0]:
                str_row = f'{inx[0]:<{inx_width_levels[0]}}  '
            else:
                str_row = f'{"":<{inx_width_levels[0]}}  '
            
            str_row += f'{inx[1]:<{inx_width_levels[1]}}   {val}'
            data.append(str_row)

        footer = f'\nName: {self.name}, ' if self.name else '\n'
        footer += 'dtype: '

        if hasattr(self._dtype, '__name__'):
            footer += self._dtype.__name__
        else:
            footer += str(self._dtype)

        return '  '.join(inx_names) + '\n'.join(data) + footer

        
        # inx_width_levels = [max(len(str(x)) for x in level) for level in self.index.levels]

        # inx_names = []
        # if self.index.names is not None:
        #     for i in range(len(inx_width_levels)):
        #         current_name = self.index.names[i]

        #         if len(current_name) > inx_width_levels[i]:
        #             inx_width_levels[i] = len(current_name)

        #         inx_names.append(f'{current_name:<{inx_width_levels[i]}}')
        #     inx_names[-1] += '\n'

        # data = []
        # for i, (inx, val) in enumerate(zip(self.index, self._data)):
        #     if i == 0 or self.index.codes[0][i - 1] != inx[0]:
        #         str_row = f'{self.index.levels[0][inx[0]]:<{inx_width_levels[0]}}  '
        #     else:
        #         str_row = f'{'':<{inx_width_levels[0]}}  '
            
        #     str_row += f'{self.index.levels[1][inx[1]]:<{inx_width_levels[1]}}   {val}'
        #     data.append(str_row)

        # footer = f'\nName: {self.name}, ' if self.name else '\n'
        # footer += 'dtype: '

        # if hasattr(self._dtype, '__name__'):
        #     footer += self._dtype.__name__
        # else:
        #     footer += str(self._dtype)

        # return '  '.join(inx_names) + '\n'.join(data) + footer