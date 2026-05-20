from series import Series
from dataframe import DataFrame
import numpy as np
import pandas as pd
from collections import Counter
from index import CategoricalIndex

class Interval:
    def __init__(self, start, stop, right, label=None):
        self.start = round(start, 3)
        self.stop = round(stop, 3)
        self.right = right
        self.label = label

    def __repr__(self):
        if self.right:
            return f'({self.start}, {self.stop}]'
        
        return f'[{self.start}, {self.stop})'

    def __str__(self):
        if self.label:
            return self.label
        
        if self.right:
            return f'({self.start}, {self.stop}]'
        
        return f'[{self.start}, {self.stop})'

class Categorical:
    def __init__(self, values=None, codes=None, categories=None, right=True, labels=None, ordered=False):
        if values is not None:
            categories = np.unique(values)
            codes = np.fromiter(self._code_gen(values, categories), dtype=int)

        self.codes = codes
        if not isinstance(codes, np.ndarray):
            self.codes = np.array(codes)

        self.categories = categories
        if not isinstance(categories, np.ndarray):
            self.categories = np.array(categories)

        self.right = right
        self.labels = labels
        self.ordered = ordered
        
        self.is_intr_cat = True
        if isinstance(self.categories[0], str):
            self.is_intr_cat = False

    @staticmethod
    def _code_gen(data, cats):
        for item in data:
            low = 0
            high = len(cats) - 1

            while low <= high:
                mid = (low + high) // 2
                if cats[mid] == item:
                    yield mid
                    break
                elif item < cats[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
    
    @classmethod
    def from_codes(cls, codes, categories, ordered=False):
        return cls(codes=codes, categories=categories, ordered=ordered)
    
    def set_categories(self, cats):
        if len(cats) != len(set(cats)):
            raise ValueError("Categorical categories must be unique")
        
        def code_gen(data, cats, old_cats):
            for code in data:
                item = old_cats[code]
                low = 0
                high = len(cats) - 1

                while low <= high:
                    mid = (low + high) // 2
                    if cats[mid] == item:
                        yield mid
                        break
                    elif item < cats[mid]:
                        high = mid - 1
                    else:
                        low = mid + 1

        new_codes = np.fromiter(code_gen(self.codes, cats, self.categories), dtype=int)
        return Categorical(codes=new_codes, categories=cats)

    def remove_unused_categories(self):
        unique_codes = np.unique(self.codes)
        new_categories = self.categories[unique_codes]

        return Categorical(self.codes, new_categories)

    def as_ordered(self):
        self.ordered = True

    def as_unordered(self):
        self.ordered = False
    
    def to_series(self, index=None, name=''):
        new_data = [str(self.categories[c]) for c in self.codes]
        return Series(new_data, index, name)
    
    def value_counts(self):
        counts = Counter(self.codes)
        ser_data = np.fromiter((counts.get(i, 0) for i, cat in enumerate(self.categories)), dtype=int)
        index = CategoricalIndex(self.categories)

        return Series(ser_data, index=index, dtype=int)

    @property
    def type(self):
        if self.is_intr_cat:
            side = 'right' if self.right else 'left'
            inter_obj = f'interval[{type(self.categories[0].start).__name__}, {side}]'

        else:
            inter_obj = str(self.categories.dtype)

        cat_len = len(self.categories)
        cat_info = f'Categories ({cat_len}, {inter_obj}): '

        joiner = ' < ' if self.ordered else ', '
        cats = '[' + f'{joiner.join(str(cat) for cat in self.categories)}' + ']'

        return cat_info + cats

    def __str__(self):
        all_cats = '[' + ', '.join(str(self.categories[c]) if c != -1 else 'NaN' for c in self.codes) + ']'
        length = f'\nLength: {len(self.codes)}\n'
        cat_info = self.type

        return all_cats + length + cat_info
    


class CategoricalAccessor:
    def __init__(self, category: Categorical, series: Series):
        self.series = series
        self.category = category
        self.categories = category.categories
        self.type = category.type

    @property
    def codes(self):
        return Series(self.category.codes, self.series.index, self.series.name)

    def value_counts(self):
        return self.category.value_counts()
    
    def remove_unused_categories(self):
        new_data = [self.series.cat.categories[c] for c in self.series._data]
        return Series(new_data, self.series.index, self.series.name, dtype='category')

    def set_categories(self, cats):
        new_cats = self.category.set_categories(cats)
        new_ser_data = np.array([new_cats.categories[code] for code in new_cats.codes])
        new_s = Series(new_ser_data, self.series.index, dtype='category')
    
        new_s._data = CategoricalAccessor(new_cats, new_s)
        return new_s
    
    def __len__(self):
        return len(self.category.codes)
    
    def __iter__(self):
        return iter(self.category.codes)


def _values_to_intervalcodes(x, bins, right): 
    codes = []
    for num in x:
        if (right and not (bins[0] < num <= bins[-1])) or \
        (not right and not (bins[0] <= num < bins[-1])):
            codes.append(-1)
            continue

        low = 0
        high = len(bins)

        while low < high - 1:
            mid = (low + high) // 2
            if right:
                if num <= bins[mid]:
                    high = mid
                else:
                    low = mid
            else:
                if num < bins[mid]:
                    high = mid
                else:
                    low = mid
        
        if 0 <= low < len(bins):
            codes.append(low)
        else:
            codes.append(-1)

    return codes


def cut(x, bins, right=True, labels=None, precision=3):
    if isinstance(bins, int):
        max_val = max(x)
        min_val = min(x)

        part = (max_val - min_val) / bins
        bins = [round(min_val + part * i, precision) for i in range(bins + 1)]
    else:
        bins = sorted(bins)

    cat = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]

    if labels is not None and len(labels) != len(cat):
        raise ValueError('Invalid labels length')
 
    codes = _values_to_intervalcodes(x, bins, right)

    if labels is not None:
        cat = [Interval(c[0], c[1], right, labels[i]) for i, c in enumerate(cat)]
    else:
        cat = [Interval(c[0], c[1], right) for c in cat]

    res = Categorical(codes=codes, categories=cat, right=right, ordered=True, labels=labels)
    if isinstance(x, Series):
        res = res.to_series(x.index, x.name)

    return res


def qcut(x, q, right=True, labels=None):
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    n = len(x)
    sorted_x = np.sort(x)

    if isinstance(q, int):
        quantile_pos = [(i * (n - 1)) / q for i in range(q + 1)]
    else:
        if any(not (0 <= x <= 1) for x in q):
            raise ValueError('Invalid quantiles values')
        
        quantile_pos = [(n - 1) * quan for quan in q]

    quantiles = []
    for pos in quantile_pos:
        if (intval := int(pos)) == pos:
            val =  sorted_x[intval]
        else:
            inx1 = int(np.floor(pos))
            inx2 = int(np.ceil(pos))
            f = pos - inx1
            val = (1 - f) * sorted_x[inx1] + f * sorted_x[inx2]

        quantiles.append(val)

    cat = [[quantiles[i], quantiles[i+1]] for i in range(len(quantiles)-1)]

    if isinstance(q, int):
        if right:
            quantiles[0] -= 0.001
            cat[0][0] -= 0.001
        else:
            quantiles[-1] += 0.001
            cat[-1][1] -= 0.001

    codes = _values_to_intervalcodes(x, quantiles, right)

    if labels is not None: 
        cat = [Interval(c[0], c[1], right, labels[i]) for i, c in enumerate(cat)]
    else:
        cat = [Interval(c[0], c[1], right) for c in cat]

    return Categorical(codes=codes, categories=cat, right=right, labels=labels, ordered=True)


# edges = [0,1,2,3]
# x = [0.5, 1, 2.5]
# print(cut(x, edges, right=False))

# ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# bins = [18, 25, 35, 60, 100]
# age_categories = cut(ages, bins)

# print(age_categories)
# print(age_categories.to_series())
# print(age_categories.codes)
# print(cut(ages, bins, right=False))

# group_names = ["Youth", "YoungAdult", "MiddleAged", "Senior"]
# print(cut(ages, bins, labels=group_names))

# data = np.random.uniform(size=20)
# print(data)
# print(cut(data, 4, precision=2))

# x = [0.5, 1, 2.5]
# a = qcut(x, 3)
# print(a.categories)
# print(a)

# a = qcut(x, 3, right=False)
# print(a.categories)
# print(a)


# data = np.random.standard_normal(100)
# bins = qcut(data, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
# print(bins)
# print(bins.value_counts())

# # data = [1, 2, 3, 4]
# quartiles = qcut(data, 4) #[0, 0.5, 1]
# print(quartiles)
# print(quartiles.value_counts())

# q = qcut(data, [0, 0.1, 0.5, 0.9, 1.])
# q = qcut(data, [0, 0.25, 0.5, 0.75, 1.])
# print(q)
# print(q.value_counts())