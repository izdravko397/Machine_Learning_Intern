import numpy as np
import pandas as pd
from sklearn.base import clone, check_is_fitted
from function_transformer import TransformerMixin
from collections import Counter
from sklearn.base import check_array

class DummyClassifier:
    def __init__(self, *, strategy='prior', random_state=None, constant=None):
        self.strategy = strategy
        self.constant = constant

        if random_state is not None and not isinstance(random_state, int):
            raise TypeError('random_state must be int')
        
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X)
        self.n_features_in_ = len(X[0])
        
        counts = Counter(y)
        self.classes_ = np.array(list(counts.keys()))
        self.n_classes_ = len(counts)
        self.most_frequent_ = counts.most_common(1)[0][0]
        y_len = len(y)
        self.class_prior_ = {cat: count / y_len for cat, count in counts.items()}

        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if self.n_features_in_ != len(X[0]):
            raise ValueError(f'Invalid len of features')

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.strategy in ('most_frequent', 'prior'):
            return np.full(len(X), self.most_frequent_)
        
        elif self.strategy == 'stratified':
            res = self.classes_
            repeat_nums = np.array(list(self.class_prior_.values())) * len(X)
            repeat_nums = repeat_nums.astype(int)
            diff = np.abs(repeat_nums.sum() - len(X))
            repeat_nums[0] += diff

            res = np.repeat(res, repeat_nums)
            np.random.shuffle(res)
            return res
        
        elif self.strategy == 'uniform':
            inxs = np.random.randint(0, len(self.classes_), len(X))
            return self.classes_[inxs]

        elif self.strategy == 'constant':
            return np.full(len(X), self.constant)

        raise ValueError(f'Invalid strategy: {self.strategy}')

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        vals = list(self.class_prior_.values())
        return np.tile(vals, (len(X), 1))


X_train = np.random.standard_normal((10, 3))
y_train = np.repeat([1, 1, 1, 0, 0], 2)

clf = DummyClassifier(strategy='stratified')
clf.fit(X_train, y_train)

print(clf.predict(X_train))

# X_test = [[10], [11], [12]]
# probs = clf.predict_proba(X_test)

# print(probs)
# print(probs.shape)