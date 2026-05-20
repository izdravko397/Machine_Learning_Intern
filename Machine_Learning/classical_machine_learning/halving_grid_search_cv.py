import numpy as np
import pandas as pd
from cross_val_score import *
from sklearn.compose import ColumnTransformer
from itertools import product
from sklearn.base import clone
from sklearn.base import check_is_fitted


class HalvingGridSearchCV:
    def __init__(self, estimator, param_grid, *, scoring='root_mean_squared_error', cv=5, refit=True,
                  factor=3, max_resources='auto', min_resources=100, random_state=None):
        if not hasattr(estimator, 'fit'):
            raise TypeError('Invalid estimator')

        self.estimator = estimator

        if isinstance(param_grid, dict):
            param_grid = [param_grid]

        if not (isinstance(param_grid, list) and all(isinstance(x, dict) for x in param_grid)):
            raise TypeError("param_grid must be dict/list of dicts")
        
        self.param_grid = param_grid

        assert scoring in scoring_map, 'Invalid scoring func' 
        self.scoring = scoring

        assert isinstance(cv, int) and cv > 0, 'cv must be int > 0'
        self.cv = cv

        assert isinstance(refit, bool), 'refit must be bool'
        self.refit = refit

        assert isinstance(factor, int) and factor > 1, 'factor must be int > 1'
        self.factor = factor

        assert max_resources == 'auto' or (isinstance(max_resources, int) and max_resources > 0),\
                                                        'max_resources must be int > 0'
        self.max_resources = max_resources

        assert isinstance(min_resources, int) and min_resources > 0, 'min_resources must be int > 0'
        self.min_resources = min_resources

        assert random_state is None or isinstance(random_state, int), 'random_state must be int'
        self.random_state = random_state

        self._res_row_pattern = self._make_res_row_pattern()
        self._indexing = lambda X, inxs: X.iloc[inxs] if isinstance(X, pd.DataFrame) else X[inxs]

    @staticmethod
    def _gen_params_comb(param_grid):
        values = list(param_grid.values())
        for combination in product(*values):
            yield combination

    def _set_params(self, params_dict):
        est_copy = clone(self.estimator)
        est_copy.set_params(**params_dict)
        return est_copy
    
    def _make_res_row_pattern(self):
        dict_keys = [key for params in self.param_grid for key in params]
        dict_keys += [f'split{i}' for i in range(self.cv)]
        dict_keys += ['mean_test_score', 'std_test_score']
        return {key: None for key in dict_keys}
    
    def _get_best_params(self, best_res, row_inx=0):
        res = {}
        for i, col in enumerate(best_res):
            if col.startswith('split'):
                break
            
            res[col] = best_res.iloc[row_inx, i]

        return res
    
    def _set_res_row(self, split_results, comb):
        res_row = self._res_row_pattern.copy()
        for key, val in comb.items():
            res_row[key] = val

        for i, res in enumerate(split_results):
            res_row[f'split{i}'] = res

        res_row['mean_test_score'] = split_results.mean()
        res_row['std_test_score'] = split_results.std()
        return res_row

    def fit(self, X, y=None, **params):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        max_resources = self.max_resources if isinstance(self.max_resources, int) else len(X)
        min_resources = self.min_resources
        inxs = np.random.permutation(len(X))

        results = []
        for params in self.param_grid:
            param_names = list(params.keys())
            for comb in self._gen_params_comb(params):
                params = dict(zip(param_names, comb))
                est = self._set_params(params)

                resources_inxs = inxs[:min_resources]
                X_resources = self._indexing(X, resources_inxs)
                y_resources = self._indexing(y, resources_inxs)
                split_results = cross_val_score(est, X_resources, y_resources, self.scoring, self.cv)

                res_row = self._set_res_row(split_results, params)
                results.append(res_row)

        df_results_temp = pd.DataFrame(results)

        while True:
            passing_comb_len = len(df_results_temp) / self.factor
            min_resources *= self.factor

            if passing_comb_len < 1 or min_resources > max_resources:
                break
            
            passing_comb_len = int(np.ceil(passing_comb_len))
            df_results_temp = df_results_temp.nsmallest(passing_comb_len, 'mean_test_score')
            np.random.shuffle(inxs)

            current_round_res = []
            for i in range(len(df_results_temp)):
                combination = self._get_best_params(df_results_temp, i)
                est = self._set_params(combination)
                resources_inxs = inxs[:min_resources]
                X_resources = self._indexing(X, resources_inxs)
                y_resources = self._indexing(y, resources_inxs)
                split_results = cross_val_score(est, X_resources, y_resources, self.scoring, self.cv)

                res_row = self._set_res_row(split_results, combination)
                results.append(res_row)
                current_round_res.append(res_row)

            df_results_temp = pd.DataFrame(current_round_res)

        best_res = df_results_temp.nsmallest(1, 'mean_test_score')

        self.best_params_ = self._get_best_params(best_res)
        self.best_score_ = best_res['mean_test_score'].iloc[0]
        self.cv_results_ = results

        if self.refit:
            est = self._set_params(self.best_params_)
            est.fit(X, y)
            self.best_estimator_ = est

        return self
    
    def predict(self, X):
        check_is_fitted(self)
        if not hasattr(self.estimator, 'predict'):
            raise AttributeError(f'{type(self.estimator).__name__} has no predict')
        
        return self.best_estimator_.predict(X)
    
    def transform(self, X):
        check_is_fitted(self)
        if not hasattr(self.estimator, 'transform'):
            raise AttributeError(f'{type(self.estimator).__name__} has no transform')
        
        return self.best_estimator_.transform(X)
    

from sklearn.pipeline import Pipeline
from model_pct_errors import preprocessing, RandomForestRegressor, housing, housing_labels
from sklearn.tree import DecisionTreeRegressor

full_pipeline = Pipeline([
("preprocessing", preprocessing),
("random_forest", RandomForestRegressor(random_state=42)),
])

param_grid = [
    {'preprocessing__geo__n_clusters': [10, 15],
    'random_forest__max_features': [6, 8]},
]

grid_search = HalvingGridSearchCV(full_pipeline, param_grid, cv=3,
        scoring='root_mean_squared_error', min_resources=2000, factor=3, random_state=42)

grid_search.fit(housing, housing_labels)
print(pd.DataFrame(grid_search.cv_results_))
print(grid_search.best_params_)

print()
print('\n\n')

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

grid_search = HalvingGridSearchCV(full_pipeline, param_grid, cv=3,
        scoring='neg_root_mean_squared_error', min_resources=2000, factor=3, random_state=42)

grid_search.fit(housing, housing_labels)
print(pd.DataFrame(grid_search.cv_results_))
print(grid_search.best_params_)