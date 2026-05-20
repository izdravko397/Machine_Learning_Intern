import numpy as np
import pandas as pd
import math
from find_func import find
from datetime import datetime
import calendar
from abc import ABC, abstractmethod


class Index:
    def __init__(self, data, name='', dtype=np.object_):
        if not hasattr(data, '__iter__') or isinstance(data, str):
            raise TypeError(f'data accepts iterable object, not: {data}')
        
        if isinstance(data, RangeIndex):
            self._data = data.to_numpy()
        elif isinstance(data, Index):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            self._data = data.copy()
        else:
            if isinstance(data[0], tuple):
                self._data = np.fromiter((row for row in data), np.object_)
            else:
                self._data = np.array(data, dtype=dtype)

        nanmask = pd.isna(self._data)
        if np.any(nanmask):
            other_vals = self._data[~nanmask]
            correct_type = np.array([v for v in other_vals]).dtype

            if np.issubdtype(correct_type, np.number):
                self._data = self._data.astype(float)
            else:
                self._data[nanmask] = 'NA'

        self.__sorted_positions = None
        self.__sorted_values = None

        self.name = name
        self._dtype = dtype

    @property
    def _sorted_positions(self):
        if self.__sorted_positions is not None:
            return self.__sorted_positions
        
        self.__sorted_positions = np.argsort(self._data)
        return self.__sorted_positions
    
    @property
    def _sorted_values(self):
        if self.__sorted_values is not None:
            return self.__sorted_values
        
        self.__sorted_values = self._data[self._sorted_positions]
        return self.__sorted_values

    def map(self, func_or_dict):
        if isinstance(func_or_dict, dict):
            new_data = [func_or_dict.get(item, np.nan) for item in self._data]
            return Index(new_data, self.name, self._dtype)
        
        if callable(func_or_dict):
            new_data = [func_or_dict(item) for item in self._data]
            return Index(new_data, self.name, self._dtype)
        
        raise TypeError('Invalid map argument type')

    def copy(self):
        return self.__class__(self._data.copy(), self.name)

    def __getitem__(self, key):
        if isinstance(key, (int, list, np.ndarray, slice)):
            return self._data[key]

        from series import Series
        if isinstance(key, Series):
            if len(key) != len(self):
                raise ValueError('Invalid len for indexing with Series')
            
            return Index(self._data[key._data.astype(bool)], self.name)
        
        raise TypeError(f'Invalid type for key on {self.__class__.__name__}')
    
    def _ser_get_item(self, ser, key):
        from series import Series

        if isinstance(key, slice):
            return Series(ser._data[key], index=self[key], name=ser.name)
        
        if isinstance(key, Series):
            if isinstance(key._data[0], (bool, np.bool)):
                mask = key._data.astype(bool)
                return Series(ser._data[mask], self._data[mask], ser.name)
            
            gen = (ser._data[self.get_loc(i)] for i in key.index)
            return Series(data=np.fromiter(gen, dtype=ser._dtype), index=key.index, name=ser.name, dtype=ser._dtype)

        if isinstance(key, (list, np.ndarray)):
            if isinstance(key[0], bool):
                if isinstance(key, list):
                    key = np.array(key)
                mask = key.astype(bool)
                return Series(ser._data[mask], self._data[mask], ser.name) 
            
            if all(isinstance(el, int) for el in key) and not isinstance(self._data[0], int):
                inx = []
                def gen():
                    for i in key:
                        inx.append(self._data[i])
                        yield ser._data[i]

                return Series(np.fromiter(gen(), ser._dtype), inx, ser.name, ser._dtype)
            
            indexes = [ser.get_loc(label) for label in key]
            gen = (ser._data[i] for i in indexes)
            return Series(data=np.fromiter(gen, dtype=ser._dtype), 
                          index=key, name=ser.name, dtype=ser._dtype)
        
        if isinstance(key, int) and not isinstance(self._data[0], int):
            return ser._data[key]
        
        indexes = self.all_indexes(key)
        indexes_len = len(indexes)
        if not indexes_len:
            raise IndexError(f'invalid label: {key}')
        
        elif indexes_len == 1:
            return ser._data[indexes[0]]
        
        return Series(ser._data[indexes] , index=[key] * indexes_len, name=ser.name, dtype=ser._dtype)

    def get_loc(self, key):
        indexes = self.all_indexes(key)
        indexes_len = len(indexes)

        if not indexes_len:
            raise IndexError(f'Invalid index: {key}')
        
        if indexes_len == 1:
            return indexes[0]
        
        if all(indexes[i - 1] + 1 == indexes[i] for i in range(1, len(indexes))):
            return slice(indexes[0], indexes[-1] + 1)
        
        return indexes
    
    def __contains__(self, label):
        return self.index(label, False) >= 0
    
    def all_indexes(self, label):
        slice_inx = find(self._sorted_values, label)
        return self._sorted_positions[slice_inx[0] : slice_inx[1] + 1]

    def index(self, label, _error=True):
        left = 0
        right = len(self._sorted_values) - 1
        first = -1
        while left <= right:
            mid = (right + left) // 2
            mid_num = self._sorted_values[mid]

            if mid_num >= label:
                right = mid - 1
            else:
                left = mid + 1

            if mid_num == label:
                first = mid

        if first == -1:
            if _error:
                raise IndexError(f'invalid label: {label}')
            else:
                return -1
            
        return self._sorted_positions[first]

    def get_indexer(self, labels):
        if not hasattr(labels, '__iter__') or isinstance(labels, str):
            raise TypeError('parametur labels must be a sequence')
        
        return np.fromiter((self.index(label, False) for label in labels), dtype=int)

    @staticmethod
    def _other_to_Index(other):
        if not hasattr(other, '__iter__') or isinstance(other, str):
            other = np.array([other])
        return Index(other)
        
    def append(self, other):
        if not isinstance(other, Index):
            other = self._other_to_Index(other)

        return Index(np.concatenate((self._data, other._data)), name=self.name)

    def difference(self, other):
        if not hasattr(other, '__iter__') or isinstance(other, str):
            raise TypeError('accepts iterable object')

        if isinstance(other, Index):
            other_data = other._data
        else:
            other_data = np.array(other)
        mask = np.isin(self._data, other_data)

        return Index(self._data[~mask], name=self.name)

    def intersection(self, other):
        if not isinstance(other, Index):
            other = self._other_to_Index(other)

        return Index(np.intersect1d(self._data, other._data), name=self.name)

    def union(self, other):
        if not isinstance(other, Index) or isinstance(other, RangeIndex):
            other = self._other_to_Index(other)

        union_unsorted = np.unique(np.concatenate((self._data, other._data)), return_index=True)
        data = union_unsorted[0][np.argsort(union_unsorted[1])]
        return Index(data, name=self.name)
    
    def isin(self, other):
        if isinstance(other, Index):
            other = other._data

        return np.isin(self._data, other) 
    
    def delete(self, inx):
        if isinstance(inx, Index):
            inx = inx._data.astype(int)

        return Index(np.delete(self._data, inx), name=self.name)
    
    def drop(self, other):
        return Index(self._data[~(self.isin(other))], name=self.name)
    
    def insert(self, inx, item):
        return Index(np.insert(self._data, inx, item), name=self.name)
    
    def unique(self):
        union_unsorted = np.unique(self._data, return_index=True)
        data = union_unsorted[0][np.argsort(union_unsorted[1])]
        return Index(data, name=self.name)
    
    @property
    def is_unique(self):
        return len(self._data) == len(np.unique(self._data))

    @property
    def is_monotonic(self):
        return all(self._data[i] <= self._data[i + 1] for i in range(len(self._data) - 1)) or \
        all(self._data[i] >= self._data[i + 1] for i in range(len(self._data) - 1))

    def __iter__(self):
        return iter(self._data)
    
    def __len__(self):
        return len(self._data)

    def __str__(self):
        return str(self._data)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({str(self)}, dtype={self._dtype.__name__})'




class RangeIndex(Index):
    def __init__(self, stop, start=None, step=1, name=''):
        if start is None:
            start = 0
        else:
            start, stop = stop, start

        if step == 0:
            raise ValueError('step cant be 0')
        
        self._start = start
        self._stop = stop
        self._step = step
        self.name = name
        self.__data = None

    @property
    def _data(self):
        if self.__data is not None:
            return self.__data
        
        self.__data = self.to_numpy()
        return self.__data

    def map(self, func_or_dict):
        index_obj = Index(self.to_numpy(), self.name, int)
        return index_obj.map(func_or_dict)

    def copy(self):
        return RangeIndex(self._start, self._stop, self._step, self.name)

    def __len__(self):
        if self._step > 0:
            return math.ceil((self._stop - self._start) / self._step)

        return math.ceil((self._stop - self._start) / self._step)

    def __iter__(self):
        current = self._start
        if self._step > 0:
            while current < self._stop:
                yield current
                current += self._step
        else:
            while current > self._stop:
                yield current
                current += self._step
    
    def _ser_get_item(self, ser, key):
        from series import Series

        if isinstance(key, slice):
            return Series(ser._data[key], index=self[key], name=ser.name)

        if isinstance(key, Series):
            if isinstance(key._data[0], (bool, np.bool)):
                mask = key._data.astype(bool)
                return Series(ser._data[mask], self._data[mask], ser.name)
            
            gen = (ser._data[ser.get_loc(i)] for i in key.index)
            return Series(data=np.fromiter(gen, dtype=ser._dtype), index=key.index, name=ser.name, dtype=ser._dtype)

        if isinstance(key, (list, np.ndarray)):
            if len(key) != len(self):
                raise ValueError('Invalid len on bool or fancy indexing')
            
            if isinstance(key[0], (bool, np.bool)):
                if isinstance(key, list):
                    key = np.array(key)
                mask = key.astype(bool)
                return Series(ser._data[mask], self._data[mask], ser.name)
            
            indexes = [ser.get_loc(label) for label in key]
            gen = (ser._data[i] for i in indexes)
            return Series(data=np.fromiter(gen, dtype=ser._dtype), 
                          index=key, name=ser.name, dtype=ser._dtype)
        
        indexes = self.all_indexes(key)
        indexes_len = len(indexes)
        if not indexes_len:
            raise IndexError(f'invalid label: {key}')
        
        elif indexes_len == 1:
            return ser._data[indexes[0]]
        
        return Series(ser._data[indexes], index=[key] * indexes_len, name=ser.name, dtype=ser._dtype)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.to_numpy()[key]
        
        res = self._start + key * self._step

        if (self._step > 0 and not (self._start <= res < self._stop)) or \
        (self._step < 0 and not (self._stop < res <= self._start)):
            raise IndexError('index out of range')
        
        return res
    
    def __contains__(self, label):
        return self.index(label, False) >= 0
    
    def all_indexes(self, label):
        return [self.index(label)]
    
    def index(self, label, _error=True):
        if not isinstance(label, int):
            label = int(label)
        
        if self._step > 0:
            in_range = self._start <= label < self._stop
        else:
            in_range = self._stop < label <= self._start

        if not in_range:
            if _error:
                raise ValueError(f"{label} is not in RangeIndex")
            return -1

        if (label - self._start) % self._step != 0:
            if _error:
                raise ValueError(f"{label} is not in RangeIndex")
            return -1

        return (label - self._start) // self._step
    
    def append(self, other):
        if isinstance(other, RangeIndex):
            if self._step == other._step:
                if self._stop == other._start:
                    return RangeIndex(self._start, other._stop, self._step, name=self.name)
        
        index_obj = Index(self.to_numpy(), name=self.name, dtype=np.int64)
        return index_obj.append(other)
    
    def difference(self, other):
        index_obj = Index(self.to_numpy(), name=self.name, dtype=np.int64)
        return index_obj.difference(other)

    def intersection(self, other):
        index_obj = Index(self.to_numpy(), name=self.name, dtype=np.int64)
        return index_obj.intersection(other)

    def union(self, other):
        index_obj = Index(self.to_numpy(), name=self.name, dtype=np.int64)
        return index_obj.union(other)
    
    def isin(self, other):
        index_obj = Index(self.to_numpy(), name=self.name, dtype=np.int64)
        return index_obj.isin(other)
    
    def delete(self, inx):
        index_obj = Index(self.to_numpy(), name=self.name, dtype=np.int64)
        return index_obj.delete(inx)
    
    def drop(self, other):
        index_obj = Index(self.to_numpy(), name=self.name, dtype=np.int64)
        return index_obj.drop(other)
    
    def insert(self, inx, item):
        index_obj = Index(self.to_numpy(), name=self.name, dtype=np.int64)
        return index_obj.insert(inx, item)
    
    def unique(self):
        return RangeIndex(self._start, self._stop, self._step)

    @property
    def is_unique(self):
        return True

    @property
    def is_monotonic(self):
        return True

    def to_numpy(self):
        return np.arange(self._start, self._stop, self._step)
    
    def __str__(self):
        return f'[{", ".join(str(i) for i in range(self._start, self._stop, self._step))}]'

    def __repr__(self):
        return f"RangeIndex(start={self._start}, stop={self._stop}, step={self._step})"
    


class MultiIndex(Index):
    def __init__(self, array, names=None):
        if not hasattr(array, '__iter__') or isinstance(array, str):
            raise ValueError('Array must be iterable')

        if isinstance(array, np.ndarray):
            self._data = array
        elif isinstance(array, (list, tuple)):
            self._data = np.fromiter((row for row in array), dtype=np.object_)
        else:
            self._data = np.fromiter(array, np.object_)

        if names is not None and len(names) != len(self._data[0]):
            raise ValueError('Invalid names count')
        
        self.names = names

        first_tuple_len = len(self._data[0])
        if any(first_tuple_len != len(row) or not isinstance(row, tuple) 
                    for row in self._data):
            raise TypeError('Invalid data format')

        self._level_1 = np.fromiter((d[0] for d in self._data), np.object_)
        self._level_2 = np.fromiter((d[1] for d in self._data), np.object_)

    def append(self, other):
        if not isinstance(other, MultiIndex):
            raise TypeError('Invalid type')
        
        return MultiIndex.from_tuples([*self._data, *other._data], names=self.names)
    
    def unique(self):
        return MultiIndex.from_tuples(np.unique(self._data), names=self.names)

    @property
    def nlevels(self):
        return len(self._data[0])
    
    def __getitem__(self, key):
        return self._data[key]
    
    def index(self, key, _error):
        try:
            return self.get_loc(key)
        except IndexError:
            return -1

    def get_loc(self, key):
        if isinstance(key, tuple):
            key1, key2 = key

            mask_lv_1 = self._level_1 == key1
            mask_lv_2 = self._level_2 == key2

            indexes = np.where(mask_lv_1 & mask_lv_2)[0]
            indexes_len = len(indexes)

            if not indexes_len:
                raise IndexError('Invalid labels combination')
            
            if indexes_len == 1:
                return indexes[0]
            
            return indexes

        indexes = np.where(self._level_1 == key)[0]
        indexes_len = len(indexes)

        if not indexes_len:
            raise KeyError('Invalid level 1 label')
        
        if len(indexes) == 1:
            return indexes[0]
        
        if all(indexes[i - 1] + 1 == indexes[i] for i in range(1, len(indexes))):
            return slice(indexes[0], indexes[-1] + 1)
        
        return indexes
    
    def _ser_get_item(self, ser, key):
        from series import Series
        
        if isinstance(key, list):
            if isinstance(key[0], bool):
                if len(key) != len(self):
                    raise ValueError('Invalid bool indexing')
                
                return Series(ser._data[key], self[key], ser.name)
            
            if all(isinstance(el, int) for el in key):
                return Series(ser._data[key], self[key], ser.name)
            
            indexes = [ser.get_loc(label) for label in key]
            gen = (ser._data[i] for i in indexes)
            return Series(data=np.fromiter(gen, dtype=ser._dtype), 
                          index=key, name=ser.name, dtype=ser._dtype)
        
        indexes = self.get_loc(key)
        label_data = self._level_2[indexes]
        labels = Index(label_data, self.names[1] if self.names else '')
        
        return Series(ser._data[indexes], labels, ser.name)

    @classmethod
    def from_arrays(cls, arrays, names=None):
        if not len(arrays):
            raise ValueError('array is empty')
        
        first_arr_len = len(arrays[0])
        if any(first_arr_len != len(arrays[i]) for i in range(len(arrays))):
            raise ValueError('arrays must be with same length')

        return cls(zip(*arrays), names)
    
    @classmethod
    def from_tuples(cls, array, names=None):
        return cls(array, names)
    
    @property
    def is_monotonic(self):
        return all(self._level_1[i] <= self._level_1[i + 1] for i in range(len(self._level_1) - 1)) or \
        all(self._level_1[i] >= self._level_1[i + 1] for i in range(len(self._level_1) - 1))

    def __iter__(self):
        return iter(self._data)
    
    def __len__(self):
        return len(self._data)

    def __str__(self):
        res = f'{self.__class__.__name__}(['

        data = []
        indent = ' ' * 12
        for r_num, row in enumerate(self._data):
            if r_num > 0:
                str_row = indent + str(row)
            else:
                str_row = str(row)

            data.append(str_row)

        data[-1] += ']'
        
        if self.names is not None:
            names = indent + f'names={self.names})'
        else:
            names = indent + ')'

        data.append(names)

        return res + ',\n'.join(data)
    

class CategoricalIndex(Index):
    def __init__(self, categories, name=''):
        self._data = categories
        self.name = name

    def __iter__(self):
        return iter(self._data)



class Offset(ABC):
    def __init__(self, n: int=1, *args):
        self.n = n
        self.args = args

    @abstractmethod
    def _make_delta(self):
        pass

    @property
    @abstractmethod
    def code(self):
        pass

    def rollforward(self, date):
        if not isinstance(date, pd.Timestamp):
            date = to_datetime(date)

        delta = self._make_delta()
        return date + delta        

    def rollback(self, date):
        if not isinstance(date, pd.Timestamp):
            date = to_datetime(date)

        delta = self._make_delta()
        return date - delta
    
    def __add__(self, other):
        return self.rollforward(other)
    
    def __radd__(self, other):
        return self.rollforward(other)

    def __sub__(self, other):
        return self.rollback(other)
    
    def __rsub__(self, other):
        return self.rollback(other)

    def __mul__(self, other):
        return self.__class__(self.n * other)
    
    def __rmul__(self, other):
        return self.__class__(self.n * other)

    def __repr__(self):
        return f'<{self.n} * {self.__class__.__name__}>'

class Hour(Offset):
    def _make_delta(self):
        return np.timedelta64(self.n, 'h')
    
    @property
    def code(self):
        return 'H'

class Minute(Offset):
    def _make_delta(self):
        return np.timedelta64(self.n, 'm')
    
    @property
    def code(self):
        return 'T'

class Day(Offset):
    def _make_delta(self):
        return np.timedelta64(self.n, 'D')
    
    @property
    def code(self):
        return 'D'
    
class BusinessDay(Offset):
    def _make_delta(self):
        pass
    
    @property
    def code(self):
        return 'B'
    
    def rollforward(self, date):
        if not isinstance(date, pd.Timestamp):
            date = to_datetime(date)

        temp_n = self.n
        while temp_n > 0:
            weekday = date.weekday()

            if weekday < 4 or weekday == 6:
                date += Day()

            elif weekday == 4:
                date += Day(3)

            elif weekday == 5:
                date += Day(2)

            temp_n -= 1

        return date
        
    def rollback(self, date):
        if not isinstance(date, pd.Timestamp):
            date = to_datetime(date)

        temp_n = self.n
        while temp_n > 0:
            weekday = date.weekday()
            if 1 <= weekday <= 4 or weekday == 6:
                date -= Day()
            elif weekday == 0:
                date -= Day(3)
            elif weekday == 5:
                date -= Day(1)

            temp_n -= 1

        return date

class YearEnd(Offset):
    def __init__(self, n=1, month=12, *args):
        super().__init__(n, *args)
        self.month = month

    def _make_delta(self):
        pass

    @property
    def code(self):
        return 'A'

    def rollforward(self, date):
        if not isinstance(date, pd.Timestamp):
            date = to_datetime(date)

        year = date.year + (self.n - 1)
        last_day = calendar.monthrange(year, self.month)[1]
        return pd.Timestamp(year=year, month=self.month, day=last_day)
        
    def rollback(self, date):
        if not isinstance(date, pd.Timestamp):
            date = to_datetime(date)

        year = date.year - (self.n)
        last_day = calendar.monthrange(year, self.month)[1]
        return pd.Timestamp(year=year, month=self.month, day=last_day)
    
    def __mul__(self, other):
        return self.__class__(self.n * other, month=self.month)
    
    def __rmul__(self, other):
        return self.__class__(self.n * other, month=self.month)
    
    
    def __repr__(self):
        return f'<{self.n} * {self.__class__.__name__} month={self.month}>'

class MonthEnd(Offset):
    def _make_delta(self):
        pass

    @property
    def code(self):
        return 'M'

    def rollforward(self, date):
        if not isinstance(date, pd.Timestamp):
            date = to_datetime(date)

        n = self.n - 1 if self.n > 0 else self.n
        val = 11 if date.month + n == 12 else date.month + n
        year = date.year + val // 12
        month = (date.month + n) % 12 or 12
        last_day = calendar.monthrange(year, month)[1]

        month_end = pd.Timestamp(year, month, last_day)
        return month_end    

    def rollback(self, date):
        if not isinstance(date, pd.Timestamp):
            date = to_datetime(date)

        year = date.year + (date.month - self.n - 1) // 12
        month = (date.month - self.n) % 12 or 12
        last_day = calendar.monthrange(year, month)[1]

        month_end = pd.Timestamp(year, month, last_day)
        return month_end
    
class WeekOfMonth(Offset):
    def __init__(self, n = 1, week_num=1, weekday=1, *args):
        self.n = n
        if args:
            week_num = args[0]
            weekday = args[1]

        self.week_num = week_num
        self.weekday = weekday

    def _make_delta(self):
        pass
    
    @property
    def code(self):
        return 'M'

    def rollforward(self, date):
        if not isinstance(date, pd.Timestamp):
            date = to_datetime(date)

        week_num, weekday = self.week_num, self.weekday

        year = date.year + (date.month + self.n - 1) // 12
        month = (date.month + self.n - 1) % 12 or 12

        first_day = pd.Timestamp(year, month, 1)

        first_weekday = first_day.weekday()
        delta_days = (weekday - first_weekday) % 7
        first_occurrence = first_day + np.timedelta64(delta_days, 'D')

        return first_occurrence + np.timedelta64(7 * week_num, 'D')   

    def rollback(self, date):
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)

        week_num, weekday = self.week_num, self.weekday

        year = date.year + (date.month - self.n + 1) // 12
        month = (date.month - self.n) % 12 or 12

        last_day_num = calendar.monthrange(year, month)[1]
        last_day = pd.Timestamp(year, month, last_day_num)

        last_weekday = last_day.weekday()
        delta_days = (last_weekday - weekday) % 7
        last_occurrence = last_day - np.timedelta64(delta_days, 'D')

        return last_occurrence - np.timedelta64(7 * (4 - week_num), 'D')
    
    def __mul__(self, other):
        return self.__class__(self.n * other, week_num=self.week_num, weekday=self.weekday)
    
    def __rmul__(self, other):
        return self.__class__(self.n * other, week_num=self.week_num, weekday=self.weekday)
    
    def __repr__(self):
        return f'<{self.n} * {self.__class__.__name__} week={self.week_num + 1}, weekday={self.weekday + 1}>'

class QuarterEnd(Offset):
    def __init__(self, n=1, month=12, *args):
        super().__init__(n, *args)
        self.month = month

    def _make_delta(self):
        pass

    @property
    def code(self):
        return 'Q'

    def rollforward(self, date):
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(to_datetime(date))

        result = date + MonthEnd(1)

        while (result.month - self.month) % 3 != 0:
            result += MonthEnd(2)

        if result < date:
            result += MonthEnd(4)

        if self.n > 1:
            result += MonthEnd(4 * (self.n - 1))

        return result

    def rollback(self, date):
        if not isinstance(date, pd.Timestamp):
            date = to_datetime(date)
        
        result = date + MonthEnd(0)
        while (result.month - self.month) % 3 != 0:
            result -= MonthEnd(1)

        if result > date:
            result -= MonthEnd(3)

        if self.n > 1:
            result -= MonthEnd(3 * (self.n - 1))

        return result
    
    def __mul__(self, other):
        return self.__class__(self.n * other, month=self.month)
    
    def __rmul__(self, other):
        return self.__class__(self.n * other, month=self.month)
    
    def __repr__(self):
        return f'<{self.n} * {self.__class__.__name__} month={self.month}>'
    
class WeekEnd(Offset):
    def __init__(self, n = 1, weekday=6, *args):
        super().__init__(n, *args)
        self.weekday = weekday

    def _make_delta(self):
        pass

    @property
    def code(self):
        return 'W'

    def rollforward(self, date):
        if not isinstance(date, pd.Timestamp):
            date = to_datetime(date)

        current = date.weekday()
        diff = self.weekday - current
        if diff <= 0:
            diff += 7

        if self.n > 1:
            diff += 7 * (self.n - 1)

        return date + Day(diff)

    def rollback(self, date):
        if not isinstance(date, pd.Timestamp):
            date = to_datetime(date)

        current = date.weekday()
        diff = current - self.weekday
        if diff < 0:
            diff += 7

        if self.n > 1:
            diff += 7 * (self.n - 1)

        return date - Day(diff)
    
    def __mul__(self, other):
        return self.__class__(self.n * other, weekday=self.weekday)
    
    def __rmul__(self, other):
        return self.__class__(self.n * other, weekday=self.weekday)
    
    def __repr__(self):
        return f'<{self.n} * {self.__class__.__name__} weekday={self.weekday}>'

    

frequencies_offset = {
    'M':   lambda n, *args: MonthEnd(n),
    'D':   lambda n, *args: Day(n),
    'B':   lambda n, *args: BusinessDay(n),
    'W':   lambda n, *args: WeekEnd(n, *args),
    'H':   lambda n, *args: Hour(n),
    'T':   lambda n, *args: Minute(n),
    'min': lambda n, *args: Minute(n),
    'WOM': lambda n, *args: WeekOfMonth(n, *args),
    'A':   lambda n, *args: YearEnd(n, *args),
    'Q':   lambda n, *args: QuarterEnd(n, *args),
}

frequencies = ['T', 'H', 'D', 'B', 'W', 'M', 'Q', 'A']

months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", 
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

weekdays = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

def parse_freq(freq):
    import re
    pattern = r"(\d+)?([A-Za-z]+)(?:-(\w+))?"
    matches = re.findall(pattern, freq)[0]

    n = int(matches[0]) if matches[0] else 1
    offset_type = matches[1]
    extra = matches[2]

    if offset_type not in frequencies_offset:
        raise ValueError(f"Invalid freq: {freq}")

    extra1 = ''
    extra2 = ''
    if offset_type in ('A', 'Q'):
        extra1 = months.index(extra) + 1

    elif offset_type == 'W':
        extra1 = weekdays.index(extra)

    elif offset_type == 'WOM':
        extra1 = int(extra[0])
        if not (0 <= extra1 <= 4):
            raise ValueError(f'Invalid freq week num: {freq}')
        
        extra2 = weekdays.index(extra[1:])

    return n, offset_type, extra1, extra2

def get_offset(freq=None, n=None, offset_type=None, extra1=None, extra2=None):
    if n is None:
        n, offset_type, extra1, extra2 = parse_freq(freq)
    return frequencies_offset[offset_type](n, extra1, extra2)



def date_range(start=None, end=None, periods=None, freq='D'):
    if start is None and end is None:
        raise ValueError("Must specify start or end or both")

    if ((start is None and end is not None) or (start is not None and end is None)) and periods is None:
        raise ValueError("Must specify (start or end) and periods")
   
    if start is not None:
        start = to_datetime(start)
    if end is not None:
        end = to_datetime(end)
    
    offset = get_offset(freq)
    if periods is not None:
        if not isinstance(periods, int) or periods <= 0:
            raise ValueError("periods must be a positive integer")
        
        if end is None:
            rng = range(1, periods + 1) if isinstance(offset, (MonthEnd, WeekOfMonth, WeekEnd)) else range(periods)
            dates = [start + offset * i for i in rng] 
        
        elif start is None:
            dates = [end - i * offset for i in range(periods -1, -1, -1)]
        
        else:
            offset = (end - start) / (periods - 1)
            dates = [start + i * offset for i in range(periods)]
            return DatetimeIndex(dates)

        return DatetimeIndex(dates, freq=freq)

    if start is None or end is None:
        raise ValueError("Must specify start and end if no period is submitted")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += offset

    return DatetimeIndex(dates, freq=freq)


def to_datetime(dates):
    if isinstance(dates, (datetime, str, np.datetime64, int)):
        return pd.Timestamp(dates)

    if isinstance(dates, (list, tuple, np.ndarray)):
        return DatetimeIndex(np.array(dates, np.datetime64))
        
    raise TypeError(f'Invalid type for "dates: {type(dates)}"')


class DatetimeIndex(Index):
    def __init__(self, data, freq=None, name='', dtype=None):
        if not isinstance(data, (DatetimeIndex, np.ndarray)):
            data = to_datetime(data)

        if isinstance(data, DatetimeIndex):
            self._data = data._data

        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.datetime64):
            self._data = data

        if dtype is not None:
            self._data = self._data.astype(dtype)
            self._dtype = dtype
        else:
            self._data = self._data.astype('datetime64[ns]')
            self._dtype = self._data.dtype


        self.freq = freq
        self.name = name
        self._sorted_dates = None
        self._is_monotonic = None

    def __getitem__(self, key):
        return self._data[key]
    
    def _ser_get_item(self, ser, key):
        from series import Series

        if isinstance(key, str):
            parse_key = key.replace('/', '-').replace(".", "-")
            parse_key_len = len(parse_key.split('-'))

            freq_map = {1: 'A', 2: 'M', 3: 'D'}
            freq = freq_map.get(parse_key_len)
            if freq is None:
                raise KeyError(f'Invalid str format: {key}')

            if freq == 'D':
                return ser[np.datetime64(key)]
            
            if not self.is_monotonic:
                raise ValueError(f'Invalid indexing in not sort dates: {key}')
            
            offset = get_offset(freq)
            start = to_datetime(key)
            end = start + offset

            return ser[start:end]

        if isinstance(key, (datetime, np.datetime64)):
            inxs = self.all_indexes(key)
            if len(inxs) == 0:
                raise IndexError(f'Invald index: {key}')
            
            if len(inxs) == 1:
                return self._data[inxs[0]]
            
            return Series(ser._data[inxs], DatetimeIndex(self._data[inxs]), ser.name) 

        if isinstance(key, slice):
            if self.is_monotonic:
                start = key.start or self.index[0]
                stop = key.stop or self.index[-1]

                if not isinstance(start, np.datetime64):
                    start = np.datetime64(start)
                if not isinstance(stop, np.datetime64):
                    stop = np.datetime64(stop)

                inxs = []
                for i, date in enumerate(self._data):
                    if start <= date <= stop:
                        inxs.append(i)

                return Series(ser._data[inxs], DatetimeIndex(self._data[inxs]), ser.name)
            
            start = self.index(key.start) if key.start else self[0]
            stop = self.all_indexes(key.stop)[-1] if key.stop else self[-1]

            return Series(ser._data[start:stop], DatetimeIndex(self._data[start:stop]), ser.name)
        
        raise TypeError('invalid type for indexing on DatetimeIndex')
    
    def all_indexes(self, label):
        if not isinstance(label, np.datetime64):
            label = to_datetime(label)

        dates, pos = self._sort_dates()
        slice_inx = find(dates, label)
        return pos[slice_inx[0] : slice_inx[1] + 1]
    
    def index(self, label, _error=True):
        if not isinstance(label, np.datetime64):
            label = to_datetime(label)

        if self.is_monotonic:
            if label < self._data[0] or label > self._data[-1]:
                if _error:
                    raise IndexError(f'Invalid label: {label}')
                else:
                    return -1
            
        dates, pos = self._sort_dates()
        left = 0
        right = len(dates) - 1
        first = -1
        while left <= right:
            mid = (right + left) // 2
            mid_date = dates[mid]

            if mid_date >= label:
                right = mid - 1
            else:
                left = mid + 1

            if mid_date == label:
                first = mid

        if first == -1:
            if _error:
                raise IndexError(f'invalid label: {label}')
            else:
                return -1
            
        return pos[first]
    
    def _sort_dates(self):
        if self._sorted_dates is not None:
            return self._sorted_dates
        
        sorted_pos = np.argsort(self._data)
        sorted_dates = self._data[sorted_pos]
        self._sorted_dates = (sorted_dates, sorted_pos)
        return (sorted_dates, sorted_pos)

    @property
    def is_monotonic(self):
        if self._is_monotonic is not None:
            return self._is_monotonic
        
        self._is_monotonic = all(self._data[i - 1] < self._data[i] 
                for i in range(1, len(self._data)))
        return self._is_monotonic
    
    @staticmethod
    def _date_to_str(date):
        if not pd.isna(date):
            date = date.astype('datetime64[ms]').astype(datetime)
            return date.strftime('%Y-%m-%d %H:%M:%S')
        return str(date)

    def __iter__(self):
        for date in self._data:
            yield self._date_to_str(date)

    def __str__(self):
        datestr = []
        for date in self._data:
            date = self._date_to_str(date)
            datestr.append(date)

        return f'DatetimeIndex({datestr}'\
            f', dtype="{self._dtype}", freq={self.freq})'








def myMultiIndex():
    pass
    # class MultiIndex(Index):
    #     def __init__(self, levels=None, codes=None, names=None):
    #         if names is not None and len(names) != len(levels):
    #             raise ValueError('Invalid names length')
            
    #         for i in range(len(levels)):
    #             if not isinstance(levels[i], Index):
    #                 levels[i] = Index(levels[i])

    #             if names is not None:
    #                 levels[i].name = names[i]

    #         for i in range(len(codes)):
    #             if not isinstance(codes[i], Index):
    #                 codes[i] = Index(codes[i])

    #         self.levels = levels
    #         self.codes = codes
    #         self.names = names

    #     @property
    #     def nlevels(self):
    #         return len(self.levels)

    #     def get_loc(self, key):
    #         if isinstance(key, tuple):
    #             key1, key2 = key

    #             key1_code = self._find_inx_code(self.levels[0], key1)
    #             key2_code = self._find_inx_code(self.levels[1], key2)

    #             if key1_code == -1 or key2_code == -1:
    #                 raise IndexError(f'Invalid labels: {key1}, {key2}')

    #             mask_lv_1 = self.codes[0]._data == key1_code
    #             mask_lv_2 = self.codes[1]._data == key2_code

    #             indexes = np.where(mask_lv_1 & mask_lv_2)[0]
    #             indexes_len = len(indexes)

    #             if not indexes_len:
    #                 raise IndexError('Invalid labels combination')
                
    #             if indexes_len == 1:
    #                 return indexes[0]
                
    #             return indexes

    #         if isinstance(key, list):
    #             all_labels_inx = []
    #             for label in key:
    #                 indexes = self.get_loc(label)
    #                 all_labels_inx.extend(indexes)

    #             return all_labels_inx

    #         if isinstance(key, slice):
    #             start_code = self._find_inx_code(self.levels[0], key.start) if key.start is not None else self.codes[0][0]
    #             stop_code = self._find_inx_code(self.levels[0], key.stop) if key.stop is not None else self.codes[0][-1]

    #             start_pos = self.codes[0].all_indexes(start_code)[0]
    #             stop_pos = self.codes[0].all_indexes(stop_code)[-1] + 1

    #             return slice(start_pos, stop_pos)

    #         key_code = self._find_inx_code(self.levels[0], key)
    #         if key_code == -1:
    #             raise KeyError('Invalid level 1 label')
            
    #         indexes = self.codes[0].all_indexes(key_code)

    #         if len(indexes) == 1:
    #             return indexes[0]
            
    #         if all(indexes[i - 1] + 1 == indexes[i] for i in range(1, len(indexes))):
    #             return slice(indexes[0], indexes[-1] + 1)
            
    #         return indexes

    #     @staticmethod
    #     def _find_inx_code(arr, n):
    #         left = 0
    #         right = len(arr) - 1
    #         pos = -1
    #         while left <= right:
    #             mid = (right + left) // 2
    #             mid_num = arr[mid]

    #             if mid_num > n:
    #                 right = mid - 1
    #             else:
    #                 left = mid + 1

    #             if mid_num == n:
    #                 pos = mid
    #                 break

    #         return pos

    #     @classmethod
    #     def from_arrays(cls, arrays, names=None):
    #         if not len(arrays):
    #             raise ValueError('array is empty')
            
    #         first_arr_len = len(arrays[0])
    #         if any(first_arr_len != len(arrays[i]) for i in range(len(arrays))):
    #             raise ValueError('arrays must be with same length')
            
    #         levels = [np.unique(a) for a in arrays]

    #         codes = [[] for _ in range(len(arrays))]
    #         for row in zip(*arrays):
    #             for i, item in enumerate(row):
    #                 code = cls._find_inx_code(levels[i], item)
    #                 codes[i].append(code)

    #         return cls(levels, codes, names)

    #     def __iter__(self):
    #         return zip(*self.codes)
        
    #     def __len__(self):
    #         return len(self.codes[0])

    #     def __str__(self):
    #         res = f'{self.__class__.__name__}(['

    #         data = []
    #         indent = ' ' * 12
    #         for r_num, code_row in enumerate(zip(*self.codes)):
    #             row_labels = []
    #             for lv, code in enumerate(code_row):
    #                 row_labels.append(str(self.levels[lv][code]))

    #             str_row = '(' + ', '.join(row_labels) + ')'
    #             if r_num > 0:
    #                 str_row = indent + str_row

    #             data.append(str_row)

    #         data[-1] += ']'
            
    #         if self.names is not None:
    #             names = indent + f'names={self.names})'
    #         else:
    #             names = indent + ')'

    #         data.append(names)

    #         return res + ',\n'.join(data)