import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.base import check_array, check_is_fitted, clone, is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


class AdaBoostClassifier:
    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, random_state=None):
        if estimator is None:
            estimator = DecisionTreeClassifier(max_depth=1)

        if not hasattr(estimator, 'fit'):
            raise TypeError('Invalid estimator')
        
        if not (isinstance(n_estimators, int) and n_estimators > 1):
            raise ValueError('n_estimators must be int > 1')
        
        if not (isinstance(learning_rate, (int, float)) and learning_rate > 0):
            raise ValueError('learning_rate must be int/float > 1')
        
        if not isinstance(random_state, int):
            raise ValueError('random_state must be int')
        
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_samples_, self.n_features_in_ = X.shape

        weights = np.full(self.n_samples_, 1 / self.n_samples_) 
        self.estimators_ = []
        self.estimator_weights_ = np.empty(self.n_estimators)
        for i in range(self.n_estimators):
            clone_est = clone(self.estimator).fit(X, y, sample_weight=weights)
            self.estimators_.append(clone_est)
            y_pred = clone_est.predict(X)
            wrong_preds_mask = y_pred != y

            active_weights_sum = weights[wrong_preds_mask]
            alpha = self.learning_rate * np.log((1 - active_weights_sum) / active_weights_sum) 
            weights[wrong_preds_mask] *= np.exp(alpha)
            self.estimator_weights_[i] = alpha.sum()

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        y_pred = np.empty(X.shape[0], dtype=int)
        for i, sample in enumerate(X):
            ests_pred = np.fromiter((est.predict([sample]).ravel() for est in self.estimators_), 
                                    dtype=int, count=len(self.estimators_))
            
            class_weights = np.empty(self.n_classes_)
            for j, cls in enumerate(self.classes_):
                class_weights[j] = self.estimator_weights_[ests_pred == cls].sum()

            y_pred[i] = self.classes_[class_weights.argmax()]

        return y_pred
    

class GradientBoostingRegressor:
    def __init__(self, loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
                min_impurity_decrease=0.0, random_state=None, max_features=None, max_leaf_nodes=None, 
                validation_fraction=0.1, n_iter_no_change=None, tol=0.0001):
        
        self._estimator = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split,
                                                min_impurity_decrease=min_impurity_decrease, min_samples_leaf=min_samples_leaf,
                                                min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                max_leaf_nodes=max_leaf_nodes)
        
        if loss not in ('squared_error'):
            raise ValueError('Not supported loss')
        
        if learning_rate <= 0:
            raise ValueError('learning_rate must be > 0')

        if not (isinstance(n_estimators, int) and n_estimators > 1):
            raise ValueError('n_estimators must be int > 1')
        
        if not (0 < subsample <= 1):
            raise ValueError('subsample must be 0 < subsample <= 1')

        if not isinstance(random_state, int) and random_state is not None:
            raise TypeError('random_state must be int or None')

        if not (0 < validation_fraction <= 1):
            raise ValueError('validation_fraction must be 0 < validation_fraction <= 1')
        
        if not (isinstance(n_iter_no_change, int) and n_iter_no_change >= 1):
            raise ValueError('n_iter_no_change must be int >= 1')
        
        if tol < 0:
            raise ValueError('tol must be >= 0')
        
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

    def fit(self, X, y):
        X = check_array(X)
        self.n_samples_, self.n_features_in_ = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        val_size = int(self.n_samples_ * self.validation_fraction)
        shuffle_inxs = np.random.permutation(self.n_samples_)
        val_inxs, train_inxs = shuffle_inxs[:val_size], shuffle_inxs[val_size:]

        X_val, y_val = X[val_inxs], y[val_inxs]
        X_train, y_train = X[train_inxs], y[train_inxs]

        train_samples = X_train.shape[0]
        subsample_size = int(train_samples * self.subsample)
        train_inxs = np.arange(train_samples)

        no_change_counter = 0
        last_error_set = y_train.copy()
        last_loss = np.inf
        self.estimators_ = []
        for _ in range(self.n_estimators):
            subset_inxs = np.random.choice(train_inxs, subsample_size, replace=False)
            X_sub, y_sub = X_train[subset_inxs], last_error_set[subset_inxs]

            clone_est = clone(self._estimator).fit(X_sub, self.learning_rate * y_sub)
            self.estimators_.append(clone_est)
            y_val_pred = self.predict(X_val)
            score = MSE(y_val, y_val_pred)

            if last_loss - score < self.tol:
                no_change_counter += 1
                if no_change_counter == self.n_iter_no_change:
                    break
            else:
                no_change_counter = 0

            last_loss = score
            last_error_set -= clone_est.predict(X_train)

        self.n_estimators_ = len(self.estimators_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        preds = self.estimators_[0].predict(X)
        for est in self.estimators_[1:]:
            preds += est.predict(X)

        return preds




class StackingClassifier:
    def __init__(self, estimators, final_estimator=None, cv=None):
        if not isinstance(estimators, list):
            raise TypeError('estimators must be list of tuples')
        
        if all(not (isinstance(name, str) and is_classifier(est)) 
                                        for name, est in estimators):
            raise TypeError('estimators must be list of (name, est)')
        
        if final_estimator is None:
            final_estimator = LogisticRegression()

        if not is_classifier(final_estimator):
            raise TypeError('final_estimator must be classifier')
        
        cv = 5 if cv is None else cv
        if not (isinstance(cv, int) and cv > 1):
            raise ValueError('cv must be int > 1')
            
        self.named_estimators = dict(estimators)
        self.estimators = [est for _, est in estimators]
        self.final_estimator = final_estimator
        self.cv = cv
        self._estimators_method = self._get_estimators_method()

    def _get_estimators_method(self):
        methods = []
        for est in self.estimators:
            m = 'predict'
            if hasattr(est, 'predict_proba'):
                m = 'predict_proba'
            elif hasattr(est, 'decision_function'):
                m = 'decision_function'

            methods.append(m)

        return methods

    def fit(self, X, y):
        X = check_array(X)
        self.n_samples_, self.n_features_in_ = X.shape

        X_final_est = []
        self.estimators_, self.named_estimators_ = [], {}
        for i, (name, est) in enumerate(self.named_estimators.items()):
            est_m = self._estimators_method[i]
            oof_pred = cross_val_predict(clone(est), X, y, cv=self.cv, method=est_m)
            
            if est_m == 'predict_proba' and oof_pred.shape[1] == 2:
                oof_pred = oof_pred[:, 1]

            if oof_pred.ndim == 1:
                oof_pred = oof_pred.reshape(-1, 1)

            X_final_est.append(oof_pred)
            clone_est = clone(est).fit(X, y)
            self.estimators_.append(clone_est)
            self.named_estimators_[name] = clone_est

        clone_fin_est = clone(self.final_estimator).fit(np.hstack(X_final_est), y)
        self.final_estimator_ = clone_fin_est
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        n_samples, n_features = X.shape

        if self.n_features_in_ != n_features:
            raise ValueError('Invalid features length')
        
        X_final_est = []
        for i, est in enumerate(self.estimators_):
            est_m = self._estimators_method[i]
            pred = getattr(est, est_m)(X)

            if est_m == 'predict_proba' and pred.shape[1] == 2:
                pred = pred[:, 1]

            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)

            X_final_est.append(pred)

        return self.final_estimator_.predict(np.hstack(X_final_est))