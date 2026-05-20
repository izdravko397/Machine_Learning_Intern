import numpy as np
import pandas as pd

class StratifiedShuffleSplit:
    def __init__(self, n_splits=10, test_size=0.2, random_state=None):
        if not isinstance(n_splits, int) or n_splits <= 0:
            raise ValueError(f'n_splits must be int > 0')
        
        if not (0 <= test_size <= 1):
            raise ValueError(f'test_size must be in [0, 1] not: {test_size}')
        
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    @staticmethod
    def _check_data(data):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.to_numpy()

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        return data

    def split(self, X, y):
        X = self._check_data(X)
        y = self._check_data(y)

        if len(X) != len(y):
            raise ValueError('X and y must be the same length')
        
        if self.random_state:
            np.random.seed(self.random_state)

        categories = np.unique(y)

        inxs_per_cat = []
        for cat in categories:
            inxs = np.where(y == cat)[0]
            shuffled_inxs = np.random.permutation(inxs)
            inxs_per_cat.append(shuffled_inxs)

        for _ in range(self.n_splits):
            train_inxs = []
            test_inxs = []

            for i, cat in enumerate(categories):
                cat_inxs = inxs_per_cat[i]
                cat_inxs = np.random.permutation(cat_inxs)

                test_len = int(len(cat_inxs) * self.test_size)
                test_inxs.extend(cat_inxs[:test_len])
                train_inxs.extend(cat_inxs[test_len:])

            test_inxs = np.random.permutation(test_inxs)
            train_inxs = np.random.permutation(train_inxs)
            yield train_inxs, test_inxs


# housing = pd.read_csv('datasets/housing/housing.csv')
# # print(housing)

# housing["income_cat"] = pd.cut(housing["median_income"],
#                                 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
#                                 labels=[1, 2, 3, 4, 5])


# splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
# strat_splits = []
# for train_index, test_index in splitter.split(housing, housing["income_cat"]):
#     strat_train_set_n = housing.iloc[train_index]
#     strat_test_set_n = housing.iloc[test_index]
#     strat_splits.append([strat_train_set_n, strat_test_set_n])

# strat_train_set, strat_test_set = strat_splits[0]

# print(housing["income_cat"].value_counts() / len(housing))
# print(len(housing) * 0.2)
# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
# print(len(strat_test_set))
# print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
# print(len(strat_train_set))

# print('____________ Original ____________')
# from sklearn.model_selection import StratifiedShuffleSplit
# splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
# strat_splits = []
# for train_index, test_index in splitter.split(housing, housing["income_cat"]):
#     strat_train_set_n = housing.iloc[train_index]
#     strat_test_set_n = housing.iloc[test_index]
#     strat_splits.append([strat_train_set_n, strat_test_set_n])

# strat_train_set, strat_test_set = strat_splits[0]

# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
# print(len(strat_test_set))
# print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
# print(len(strat_train_set))














# oburkana 
    # def split(self, X, y):
    #     X = self._check_data(X)
    #     y = self._check_data(y)

    #     if len(X) != len(y):
    #         raise ValueError('X and y must be the same length')
        
    #     if self.random_state:
    #         np.random.seed(self.random_state)

    #     categories, counts = np.unique(y, return_counts=True)
    #     count_per_iter = counts // self.n_splits
    #     per_count_start_stop = [[0, 0] for _ in range(len(count_per_iter))]

    #     inxs_per_cat = []
    #     for cat in categories:
    #         inxs = np.where(y == cat)[0]
    #         shuffled_inxs = np.random.permutation(inxs)
    #         inxs_per_cat.append(shuffled_inxs)

    #     for n_iter in range(self.n_splits):
    #         train_inxs = []
    #         test_inxs = []

    #         for i, count in enumerate(count_per_iter):
    #             start, end = per_count_start_stop[i]
    #             end = end + count if n_iter < self.n_splits - 1 else None
    #             subset = inxs_per_cat[i][start:end]

    #             test_len = int(count * self.test_size)
    #             test_inxs.extend(subset[:test_len])
    #             train_inxs.extend(subset[test_len:])

    #             per_count_start_stop[i] = [end, end]
                            
    #         yield train_inxs, test_inxs