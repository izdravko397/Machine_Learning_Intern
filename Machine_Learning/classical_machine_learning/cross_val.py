import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.base import is_classifier, is_regressor
from abc import ABC, abstractmethod

def RMSE(real_data, predicted_data):
    square_diffs = np.square(real_data - predicted_data)
    return np.sqrt(np.mean(square_diffs))

def MAE(real_data, predicted_data):
    diffs = np.abs(real_data - predicted_data)
    return np.mean(diffs)

def RMSPE(real_data, predicted_data):
    square_diffs = np.square((real_data - predicted_data) / real_data)
    return np.sqrt(np.mean(square_diffs))

def MAPE(real_data, predicted_data):
    diffs = np.abs(real_data - predicted_data) / real_data
    return np.mean(diffs)

def accuracy(real_data, predicted_data):
    n_correct = (predicted_data == real_data).sum()
    return n_correct / len(real_data)


scoring_map_regressors = {
    'root_mean_squared_error': RMSE,
    'root_mean_squared_procentage_error': RMSPE,
    'mean_absolute_error': MAE,
    'mean_absolute_procentage_error': MAPE,
}

scoring_map_classifiers = {
    'accuracy': accuracy,
}

def indexing(obj, inxs):
    return obj.iloc[inxs] if isinstance(obj, (pd.DataFrame, pd.Series)) else obj[inxs]


class FoldBase(ABC):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        assert isinstance(n_splits, int) and n_splits > 1, 'n_splits must be int > 1'
        self.n_splits = n_splits

        assert isinstance(shuffle, bool), 'shuffle must be bool'
        self.shuffle = shuffle

        assert random_state is None or isinstance(random_state, int), 'random_state must be int'
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    @abstractmethod
    def _make_folds_inx(self, X, y):
        pass
    
    def split(self, X, y, groups=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        folds_inx = self._make_folds_inx(X, y)
        folds_inx_mask = np.arange(len(folds_inx))

        for i in range(self.n_splits):
            train_inxs = np.concatenate(folds_inx[folds_inx_mask != i]).astype(int)
            test_inxs = folds_inx[i].astype(int)
            yield train_inxs, test_inxs

class KFold(FoldBase):
    def _make_folds_inx(self, X, y):
        inxs = np.arange(len(X))
        if self.shuffle:
            np.random.shuffle(inxs)

        return np.array(np.array_split(inxs, self.n_splits), dtype=np.object_)

class StratifiedKFold(FoldBase):
    def _make_folds_inx(self, X, y):
        classes = np.unique(y)
        class_pos = [np.array_split(np.where(y == cl)[0], self.n_splits) for cl in classes]

        def gen():
            for strat_fold in zip(*class_pos):
                fold = np.concatenate(strat_fold)
                if self.shuffle:
                    np.random.shuffle(fold)

                yield fold

        return np.fromiter(gen(), np.object_)

    


def cross_val_score(estimator, X, y, scoring="root_mean_squared_error", cv=5):
    if is_regressor(estimator):
        score_func = scoring_map_regressors[scoring]
        splitter = KFold(cv, shuffle=True)
    elif is_classifier(estimator):
        score_func = scoring_map_classifiers[scoring]
        splitter = StratifiedKFold(cv, shuffle=True)
    else:
        raise TypeError('Invalid estimator')

    results = np.empty(cv)
    for i, (train_inxs, test_inxs) in enumerate(splitter.split(X, y)):
        fresh_est = clone(estimator)
        X_train = indexing(X, train_inxs)
        y_train = indexing(y, train_inxs)

        uni, coun = np.unique(y_train, return_counts=True)
        # print(f'train set {i}')
        # print(', '.join(f'{cat} - {c / len(y_train)}' for cat, c in zip(uni, coun)))

        X_test = indexing(X, test_inxs)
        y_test = indexing(y, test_inxs)

        uni, coun = np.unique(y_test, return_counts=True)
        # print(f'test set {i}')
        # print(', '.join(f'{cat} - {c / len(y_test)}' for cat, c in zip(uni, coun)))

        fresh_est.fit(X_train, y_train)
        y_predict = fresh_est.predict(X_test)

        results[i] = score_func(y_test, y_predict)

    return results


method_map = {
    'predict': lambda est, X: est.predict(X),
    'predict_proba': lambda est, X: est.predict_proba(X),
    'decision_function': lambda est, X: est.decision_function(X)
}

def get_result_arr(y: np.ndarray, method: str):
    match method:
        case 'predict': 
            dtp = y.dtype
            shape = (len(y),) if y.ndim == 1 else (len(y), len(y[0]))
        case 'decision_function': 
            dtp = float
            shape = (len(y),)
        case 'predict_proba': 
            dtp = np.object_
            shape = (len(y), 2)

    return np.empty(shape=shape, dtype=dtp)

def cross_val_predict(estimator, X, y=None, cv=None, method='predict'):
    if is_regressor(estimator):
        splitter = KFold(cv)
    elif is_classifier(estimator):
        splitter = StratifiedKFold(cv)
    else:
        raise TypeError('Invalid estimator')
    
    method_executer = method_map.get(method)
    if method_executer is None:
        raise ValueError(f'Invalid method: {method}')

    results = get_result_arr(y, method)
    for train_inxs, test_inxs in splitter.split(X, y):
        fresh_est = clone(estimator)
        X_train = indexing(X, train_inxs)
        y_train = indexing(y, train_inxs)
        X_test =  indexing(X, test_inxs)

        fresh_est.fit(X_train, y_train)
        results[test_inxs] = method_executer(fresh_est, X_test)

    return results







# from sklearn.linear_model import LinearRegression
# housing = pd.read_csv('datasets/housing/housing.csv')
# # print(housing.info())
# # print(housing.drop(columns='ocean_proximity').corr())

# X = housing[['median_income']]
# y = housing[['median_house_value']]
# # np.random.seed(42)


# res = cross_val_score(LinearRegression(), X, y, cv=10)
# print(pd.Series(res).describe())

# from sklearn.model_selection import cross_val_score
# res = -cross_val_score(LinearRegression(), X, y,
#                         scoring="neg_root_mean_squared_error", cv=10)

# print(pd.Series(res).describe())




#        ============= accuracy =============
# from sklearn.datasets import fetch_openml
# mnist = fetch_openml('mnist_784', as_frame=False) 

# X, y = mnist.data, mnist.target
# X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# y_train_5 = (y_train == '5')
# y_test_5 = (y_test == '5')

# from sklearn.linear_model import SGDClassifier
# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_5)

# print('------------------ Custom ------------------')
# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


# print('------------------ Original ------------------')
# from sklearn.model_selection import cross_val_score
# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
