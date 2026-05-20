import numpy as np
import pandas as pd
from sklearn.base import check_array, check_is_fitted, clone
from class_score_funcs import cross_val_predict

class ClassifierChain:
    def __init__(self, estimator=None, *, order=None, cv=None, chain_method='predict', random_state=None):
        if not hasattr(estimator, 'fit'):
            raise TypeError('Invalid estimator')
        
        self.estimator = estimator
        
        if not (order is None or isinstance(order, (list, np.ndarray)) or order == 'random'):
            raise TypeError(f'Invalid order: {order}')
         
        self.order = order

        if not (cv is None or (isinstance(cv, int) and cv > 1)):
            raise TypeError('cv must be int > 1')
        
        self.cv = cv
        
        if chain_method not in {'predict', 'predict_proba', 'decision_function'}:
            raise ValueError(f'Invalid chain_method: {chain_method}')
        
        self.chain_method = chain_method 
        
        if not (random_state is None or isinstance(random_state, int)):
            raise TypeError('random_state must be int')
        
        self.random_state = random_state

    def fit(self, X, Y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
            X = check_array(X)
        else:
            X = check_array(X)
            self.feature_names_in_ = np.arange(X.shape[1])

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.order == 'random':
            self.order = np.random.permutation(Y.shape[1])

        if self.order is None:
            self.order = np.arange(Y.shape[1])
        
        if Y.shape[1] != len(self.order):
            raise ValueError(f'Expected {len(self.order)} and receive {Y.shape[1]}')

        estimators = []
        temp_X = X
        for i in self.order:
            fresh_est = clone(self.estimator)
            y_train = Y[:, i]
            fresh_est.fit(temp_X, y_train)
            estimators.append(fresh_est)

            if self.cv is None:
                new_feature = getattr(fresh_est, self.chain_method)(temp_X)
            else:
                new_feature = cross_val_predict(self.estimator, temp_X, y_train, 
                                                self.cv, self.chain_method)
                
            if self.chain_method != 'predict_proba':
                new_feature = new_feature.reshape(-1, 1)

            temp_X = np.hstack([temp_X, new_feature])

        self.order_  = self.order
        self.estimators_ = estimators
        self.n_labels_in_ = Y.shape[1]
        self.n_features_in_ = X.shape[1]
        return self
    
    def _chain_output(self, X, method):
        if not hasattr(self.estimator, method):
            raise AttributeError(f'{self.estimator.__class__.__name__} has not {method}')
        
        check_is_fitted(self)
        X = check_array(X)

        if self.n_features_in_ != X.shape[1]:
            raise ValueError(f'Invalid features length: {X.shape[1]}')
        
        temp_X = X
        results = np.empty((X.shape[0], self.n_labels_in_))
        for i, order in enumerate(self.order_):
            est = self.estimators_[i]
            est_pred = getattr(est, method)(temp_X)
            results[:, order] = est_pred[:, 1] if method == 'predict_proba' else est_pred

            new_feature = est_pred
            if self.chain_method != method:
                new_feature = getattr(est, self.chain_method)(temp_X)

            if self.chain_method != 'predict_proba':
                new_feature = new_feature.reshape(-1, 1)

            temp_X = np.hstack([temp_X, new_feature])

        return results
    
    def predict(self, X):
        return self._chain_output(X, 'predict')
    
    def decision_function(self, X):
        return self._chain_output(X, 'decision_function')

    def predict_proba(self, X):
        return self._chain_output(X, 'predict_proba')
     