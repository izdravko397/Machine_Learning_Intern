from sklearn.base import check_array, check_is_fitted, clone
import pandas as pd
import numpy as np


class OneVsRestClassifier:
    def __init__(self, estimator):
        if not hasattr(estimator, 'fit'):
            raise TypeError(f'Invalid estimarot: {type(estimator).__name__}')
        
        self.estimator = estimator
        self._method = self._get_est_method()

    def _get_est_method(self):
        for method in ('predict_proba', 'decision_function'):
            if hasattr(self.estimator, method):
                return method
            
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
            X = check_array(X)
        else:
            X = check_array(X)
            self.feature_names_in_ = np.arange(X.shape[1])

        self.n_features_in_ = X.shape[1]
        classes = np.unique(y)
        estimators = []
        for cls in classes:
            est = clone(self.estimator)
            y_train_cls = y == cls
            est.fit(X, y_train_cls)
            estimators.append(est)
        
        self.estimators_ = estimators
        self.classes_ = classes
        return self
    
    def predict(self, X):
        predictions = self._get_raw_predictions(X)
        inxs = predictions.argmax(axis=1)
        return self.classes_[inxs]

    def _get_raw_predictions(self, X):
        check_is_fitted(self)

        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Invalid features length: {X.shape[1]}')
        
        n_classes = len(self.classes_)
        res = np.empty((len(X), n_classes))
        for i, sample in enumerate(X):
            sample_res = np.empty(n_classes)
            for j, est in enumerate(self.estimators_):
                match self._method:
                    case 'predict_proba': 
                        val = est.predict_proba([sample])[0][1]
                    case 'decision_function':
                        val = est.decision_function([sample])[0]

                sample_res[j] = val
            res[i] = sample_res

        return res
    
    def predict_proba(self, X):
        if self._method != 'predict_proba':
            raise AttributeError(f'{self.estimator.__class__.__name__} has not predict_proba')
        
        return self._get_raw_predictions(X)

    def decision_function(self, X):
        if self._method != 'decision_function':
            raise AttributeError(f'{self.estimator.__class__.__name__} has not decision_function')
        
        return self._get_raw_predictions(X)



class OneVsOneClassifier:
    def __init__(self, estimator):
        if not hasattr(estimator, 'fit'):
            raise TypeError(f'Invalid estimarot: {type(estimator).__name__}')
        
        self.estimator = estimator

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
            X = check_array(X)
        else:
            X = check_array(X)
            self.feature_names_in_ = np.arange(X.shape[1])

        self.n_features_in_ = X.shape[1]
        classes = np.unique(y)
        estimators = []
        classes_comb = []
        classes_inx_comb = set()
        for i, cls1 in enumerate(classes):
            for j, cls2 in enumerate(classes):
                if (j, i) in classes_inx_comb or i == j:
                    continue

                est = clone(self.estimator)
                pos = max(cls1, cls2)

                train_inxs = (y == cls1) | (y == cls2)
                X_train_cls = X[train_inxs]
                y_train_cls = y[train_inxs] == pos

                est.fit(X_train_cls, y_train_cls)
                estimators.append(est)
                classes_comb.append(sorted([cls1, cls2]))
                classes_inx_comb.add((i, j))
        
        self.estimators_ = estimators
        self.classes_comb_ = classes_comb
        self.classes_ = classes
        return self
    
    def _get_raw_predictions(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Invalid features length: {X.shape[1]}')

        n_classes = len(self.classes_)
        results = np.empty((len(X), n_classes))
        for i, sample in enumerate(X):
            sample_predicts = np.zeros(n_classes)
            for j, est in enumerate(self.estimators_):
                pred = est.predict([sample])[0]
                cls = self.classes_comb_[j][int(pred)]
                mask = self.classes_ == cls
                sample_predicts[mask] += 1

            noise = np.random.uniform(-0.3333, 0.3333, size=len(sample_predicts))
            sample_predicts += noise
            results[i] = sample_predicts

        return results
    
    def predict(self, X):
        raw_predictions = self._get_raw_predictions(X)
        inxs = raw_predictions.argmax(axis=1)
        return self.classes_[inxs]
    
    def decision_function(self, X):
        if not hasattr(self.estimator, 'decision_function'):
            raise AttributeError(f'{self.estimator.__class__.__name__} has not decision_function')
        
        return self._get_raw_predictions(X)