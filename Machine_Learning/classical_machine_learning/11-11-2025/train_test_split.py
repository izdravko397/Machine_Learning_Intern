import numpy as np
import pandas as pd
from stratified_shuffle_split import StratifiedShuffleSplit

def to_same_type(data, train_i, test_i):
    if isinstance(data, pd.DataFrame):
        return data.iloc[train_i], data.iloc[test_i]
    
    return data[train_i], data[test_i]


def train_test_split(data, test_size=0.2, random_state=None, stratify=None):
    if not (0 <= test_size <= 1):
        raise ValueError(f'test_size must be in [0, 1] not: {test_size}')
    
    if not isinstance(data, pd.DataFrame) and not (isinstance(data, np.ndarray) and data.ndim == 2):
        raise TypeError(f'Invalid data must be DataFrame or 2d array not: {type(data).__name__}')
    
    if stratify is not None:
        spliter = StratifiedShuffleSplit(test_size=test_size, random_state=random_state)
        train_i, test_i = next(spliter.split(data, stratify))
        return to_same_type(data, train_i, test_i)
    
    if random_state:
        np.random.seed(random_state)

    shuffled_inxs = np.random.permutation(len(data))
    test_part_len = int(len(data) * test_size)
    train_i = shuffled_inxs[test_part_len:]
    test_i = shuffled_inxs[:test_part_len]

    return to_same_type(data, train_i, test_i)




housing = pd.read_csv('datasets/housing/housing.csv')
# print(housing)

housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
print('_____________ Random select _____________')
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, random_state=55)
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(len(strat_test_set))
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
print(len(strat_train_set))

print('_____________ stratify select _____________')
strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

print(housing["income_cat"].value_counts() / len(housing))
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(len(strat_test_set))
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
print(len(strat_train_set))

print('____________ Original Random ____________')
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, random_state=55)
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(len(strat_test_set))
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
print(len(strat_train_set))

print('____________ Original stratify ____________')
from sklearn.model_selection import train_test_split

strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(len(strat_test_set))
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
print(len(strat_train_set))