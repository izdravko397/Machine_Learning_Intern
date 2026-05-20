from sklearn.base import clone, check_array, check_is_fitted, is_classifier
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class VotingClassifier:
    def __init__(self, estimators, voting='hard'):
        if voting not in ('hard', 'soft'):
            raise ValueError(f"voting must be 'hard', 'soft', not: {voting}")
        
        self.voting = voting

        if not isinstance(estimators, list):
            raise ValueError('estimators must be list of tuples') 
        
        self.estimators, self.named_estimators = [], {}
        for name_and_est in estimators:
            if not (isinstance(name_and_est, tuple) and len(name_and_est) == 2):
                raise ValueError('estimators must be list of tuples: (est_name, est)')

            name, est = name_and_est
            if not is_classifier(est):
                raise TypeError('All estimators must be classifiers')
            
            if voting == 'soft' and not hasattr(est, 'predict_proba'):
                raise ValueError('When voting is "soft" all estimators must have predict_proba method')
            
            self.estimators.append(est)
            self.named_estimators[name] = est

    def fit(self, X, y):
        self.estimators_, self.named_estimators_ = [], {}
        for name, est in self.named_estimators.items():
            clone_est = clone(est)
            clone_est.fit(X, y)
            self.estimators_.append(clone_est)
            self.named_estimators_[name] = clone_est

        self.target_dtype_ = y.dtype
        self.calsses_ = np.unique(y)
        return self

    def _soft_voting_predict(self, X):
        results = np.empty(X.shape[0], dtype=self.target_dtype_)

        for i, sample in enumerate(X):
            probabilities = np.empty((len(self.estimators_), len(self.calsses_)))
            for j, est in enumerate(self.estimators_):
                probabilities[j] = est.predict_proba([sample])

            results[i] = (probabilities.sum(axis=0) / len(self.calsses_)).argmax()

        return results

    def _hard_voting_predict(self, X):
        results = np.empty((len(self.estimators_), X.shape[0]), dtype=self.target_dtype_)

        for i, est in enumerate(self.estimators_):
            results[i] = est.predict(X)

        return mode(results, axis=0)[0]
    
    def predict(self, X):
        check_is_fitted(self)

        if self.voting == 'hard':
            return self._hard_voting_predict(X)
        return self._soft_voting_predict(X)

    def score(self, X, y):
        check_is_fitted(self)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    



class BaggingClassifier(VotingClassifier):
    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0,
                 bootstrap=True, bootstrap_features=False, oob_score=False, random_state=None):
        if not hasattr(estimator, 'fit'):
            raise TypeError('Invalid estimator')
        
        self.estimator = estimator

        if not (isinstance(n_estimators, int) and n_estimators > 1):
            raise ValueError('n_estimators must be int > 1')
        self.n_estimators = n_estimators

        if not (isinstance(max_samples, (int, float)) and max_samples > 0):
            raise ValueError('max_samples must be int or float > 0')
        
        self.max_samples = max_samples
        
        if not (isinstance(max_features, (int, float)) and max_features > 0):
            raise ValueError('max_features must be int or float > 0')
        
        self.max_features = max_features

        if not isinstance(bootstrap, bool):
            raise TypeError('bootstrap must be bool')
        
        self.bootstrap = bootstrap
        
        if not isinstance(bootstrap_features, bool):
            raise TypeError('bootstrap_features must be bool')

        self.bootstrap_features = bootstrap_features

        if not isinstance(oob_score, bool):
            raise TypeError('oob_score must be bool')
        
        self.oob_score = oob_score

        if not isinstance(random_state, int) and random_state is not None:
            raise TypeError('random_state must be int or None')
        
        self.random_state = random_state
        self.voting = 'soft' if hasattr(estimator, 'predict_proba') else 'hard'

    @staticmethod
    def _get_subset_size(max_size, train_size):
        if isinstance(max_size, float):
            max_size = int(train_size * max_size)

        return max_size
    
    @staticmethod
    def _get_oob_score(oob_results, y_true):
        preds = np.empty_like(y_true)
        for col_inx in range(oob_results.shape[1]):
            non_nan_mask = ~np.isnan(oob_results[:, col_inx])
            values = oob_results[non_nan_mask, col_inx]

            pred = np.nan
            if len(values) != 0:
                unique, counts = np.unique(values, return_counts=True)
                pred = unique[np.argmax(counts)]
            
            preds[col_inx] = pred

        non_nan_preds = ~np.isnan(preds) 
        preds  = preds[non_nan_preds]
        y_true = y_true[non_nan_preds]

        return accuracy_score(y_true, preds.astype(int))

    def fit(self, X, y):
        self.calsses_ = np.unique(y)
        self.target_dtype_ = y.dtype
        n_samples, n_features = X.shape
        samples_inxs, features_inxs = np.arange(n_samples), np.arange(n_features)

        samples_subset_size  = self._get_subset_size(self.max_samples, n_samples)
        features_subset_size = self._get_subset_size(self.max_features, n_features)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.oob_score:
            oob_preds = np.full((self.n_estimators, y.shape[0]), np.nan)
        
        self.estimators_ = []
        self.est_features_inxs_ = []
        for i in range(self.n_estimators):
            clone_est = clone(self.estimator)
            samples_subset_inxs  = np.random.choice(samples_inxs, samples_subset_size, replace=self.bootstrap)
            features_subset_inxs = np.random.choice(features_inxs, features_subset_size, replace=self.bootstrap_features)
            X_subset = X[np.ix_(samples_subset_inxs, features_subset_inxs)]
            y_subset = y[samples_subset_inxs]

            clone_est.fit(X_subset, y_subset)
            self.estimators_.append(clone_est)
            self.est_features_inxs_.append(features_subset_inxs)

            if self.oob_score:
                oob_subset_mask = ~np.isin(samples_inxs, samples_subset_inxs)
                oob_subset = X[np.ix_(samples_inxs[oob_subset_mask], features_subset_inxs)]
                oob_preds[i, oob_subset_mask] = clone_est.predict(oob_subset)

        if self.oob_score:
            self.oob_score_ = self._get_oob_score(oob_preds, y)

        return self
    
    def _soft_voting_predict(self, X):
        results = np.empty(X.shape[0], dtype=self.target_dtype_)

        for i, sample in enumerate(X):
            probabilities = np.empty((len(self.estimators_), len(self.calsses_)))
            for j, est in enumerate(self.estimators_):
                probabilities[j] = est.predict_proba([sample[self.est_features_inxs_[j]]])

            results[i] = (probabilities.sum(axis=0) / len(self.calsses_)).argmax()

        return results

    def _hard_voting_predict(self, X):
        results = np.empty((len(self.estimators_), X.shape[0]), dtype=self.target_dtype_)

        for i, est in enumerate(self.estimators_):
            results[i] = est.predict(X[:, self.est_features_inxs_[i]])

        return mode(results, axis=0)[0]
    
    def predict(self, X):
        check_is_fitted(self)

        if self.voting == 'hard':
            return self._hard_voting_predict(X)
        return self._soft_voting_predict(X)
    




class RandomForestBase:
    def __init__(self, kind, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, bootstrap=True, 
                oob_score=False, random_state=None, max_samples=None):
        
        self.kind = kind
        cls = DecisionTreeClassifier if kind == 'classifier' else DecisionTreeRegressor
        self._estimator = cls(max_depth=max_depth, min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                 max_features=max_features, max_leaf_nodes=max_leaf_nodes, 
                                                 min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        
        if not (isinstance(n_estimators, int) and n_estimators > 1):
            raise ValueError('n_estimators must be int > 1')
        
        if not isinstance(bootstrap, bool):
            raise TypeError('bootstrap must be bool')
        
        if not isinstance(oob_score, bool):
            raise TypeError('oob_score must be bool')
        
        if not isinstance(random_state, int):
            raise TypeError('random_state must be int')
        
        if not isinstance(max_samples, (int, float)) and max_samples is not None:
            raise TypeError('max_samples must be int/float or None')
        
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap 
        self.oob_score = oob_score
        self.random_state = random_state
        self.max_samples = max_samples

    def _get_oob_score(self, oob_results, y_true):
        preds = np.empty_like(y_true)
        for col_inx in range(oob_results.shape[1]):
            non_nan_mask = ~np.isnan(oob_results[:, col_inx])
            values = oob_results[non_nan_mask, col_inx]

            if len(values) == 0:
                preds[col_inx] = np.nan
                continue
            
            if self.kind == 'classifier':
                unique, counts = np.unique(values, return_counts=True)
                pred = unique[np.argmax(counts)]
            else:
                pred = values.mean()
            
            preds[col_inx] = pred

        non_nan_preds = ~np.isnan(preds) 
        preds  = preds[non_nan_preds]
        y_true = y_true[non_nan_preds]

        if self.kind == 'classifier':
            return accuracy_score(y_true, preds.astype(int))
        return r2_score(y_true, preds)

    def fit(self, X, y):
        X = check_array(X)
        if self.kind == 'regression':
            y = check_array(y).ravel()

        self.n_features_in_ = X.shape[1]
        self.calsses_ = np.unique(y)
        self.target_dtype_ = y.dtype
        n_samples = X.shape[0]

        subset_size = self.max_samples
        if subset_size is None:
            subset_size = n_samples
        elif isinstance(subset_size, float):
            subset_size = int(subset_size * n_samples)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.oob_score:
            oob_preds = np.full((self.n_estimators, y.shape[0]), np.nan)

        self.estimators_ = []
        feature_importances = np.empty((self.n_estimators, self.n_features_in_))
        samples_inxs = np.arange(n_samples)
        for i in range(self.n_estimators):
            subset_inxs = np.random.choice(samples_inxs, subset_size, replace=self.bootstrap)
            X_subset, y_subset = X[subset_inxs], y[subset_inxs] 
            
            clone_est = clone(self._estimator).fit(X_subset, y_subset)
            feature_importances[i] = clone_est.feature_importances_
            self.estimators_.append(clone_est)

            if self.oob_score:
                oob_subset_mask = ~np.isin(samples_inxs, subset_inxs)
                oob_subset = X[oob_subset_mask]
                oob_preds[i, oob_subset_mask] = clone_est.predict(oob_subset)

        if self.oob_score:
            self.oob_score_ = self._get_oob_score(oob_preds, y)

        self.feature_importances_ = feature_importances.mean(axis=0)
        return self
    
    def predict(self, X):
        check_is_fitted(self)

        if self.kind == 'regressor':
            results = np.empty((len(self.estimators_), X.shape[0]), dtype=self.target_dtype_)
            for i, est in enumerate(self.estimators_):
                results[i] = est.predict(X)

            return results.mean(axis=0) 
        
        # classifier soft voting
        results = np.empty(X.shape[0], dtype=self.target_dtype_)
        proba_shape = (len(self.estimators_), len(self.calsses_))
        for i, sample in enumerate(X):
            probabilities = np.empty(proba_shape)
            for j, est in enumerate(self.estimators_):
                probabilities[j] = est.predict_proba([sample])

            results[i] = (probabilities.sum(axis=0) / len(self.calsses_)).argmax()

        return results
        


    
class RandomForestClassifier(RandomForestBase):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, min_weight_fraction_leaf=0, max_features='sqrt', 
                max_leaf_nodes=None, bootstrap=True, oob_score=False, random_state=None, max_samples=None):
        
        super().__init__('classifier', n_estimators, max_depth, min_samples_split, 
                         min_samples_leaf, min_weight_fraction_leaf, max_features, 
                         max_leaf_nodes, bootstrap, oob_score, random_state, max_samples)
        
class RandomForestRegressor(RandomForestBase):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, min_weight_fraction_leaf=0, max_features='sqrt', 
                max_leaf_nodes=None, bootstrap=True, oob_score=False, random_state=None, max_samples=None):
        
        super().__init__('regressor', n_estimators, max_depth, min_samples_split, 
                         min_samples_leaf, min_weight_fraction_leaf, max_features, 
                         max_leaf_nodes, bootstrap, oob_score, random_state, max_samples)



class ExtraTreesClassifier(RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, min_weight_fraction_leaf=0, max_features='sqrt', 
                max_leaf_nodes=None, bootstrap=True, oob_score=False, random_state=None, max_samples=None):
        
        super().__init__(n_estimators, max_depth, min_samples_split, min_samples_leaf, 
                        min_weight_fraction_leaf, max_features, max_leaf_nodes, 
                        bootstrap, oob_score, random_state, max_samples)
        
        self._estimator.splitter = 'random'

class ExtraTreesRegressor(RandomForestRegressor):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, min_weight_fraction_leaf=0, max_features='sqrt', 
                max_leaf_nodes=None, bootstrap=True, oob_score=False, random_state=None, max_samples=None):
        
        super().__init__(n_estimators, max_depth, min_samples_split, min_samples_leaf, 
                        min_weight_fraction_leaf, max_features, max_leaf_nodes, 
                        bootstrap, oob_score, random_state, max_samples)
        
        self._estimator.splitter = 'random'