import numpy as np
import pandas as pd
from cross_val_score import *
from sklearn.compose import ColumnTransformer
from itertools import product
from sklearn.base import clone
from sklearn.base import check_is_fitted

class GridSearchCV:
    def __init__(self, estimator, param_grid, *, scoring='root_mean_squared_error', cv=5, refit=True):
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

        self._res_row_pattern = self._make_res_row_pattern()

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
    
    def _get_best_params(self, best_res):
        res = {}
        for i, col in enumerate(best_res):
            if col.startswith('split'):
                break
            
            res[col] = best_res.iloc[0, i]

        return res

    def fit(self, X, y=None, **params):
        results = []
        for params in self.param_grid:
            param_names = list(params.keys())
            for comb in self._gen_params_comb(params):
                params = dict(zip(param_names, comb))
                est = self._set_params(params)
                split_resuts = cross_val_score(est, X, y, self.scoring, self.cv)

                res_row = self._res_row_pattern.copy()
                for key, val in zip(param_names, comb):
                    res_row[key] = val

                for i, res in enumerate(split_resuts):
                    res_row[f'split{i}'] = res

                res_row['mean_test_score'] = split_resuts.mean()
                res_row['std_test_score'] = split_resuts.std()
                results.append(res_row)

        df_results = pd.DataFrame(results)
        best_res = df_results.nsmallest(1, 'mean_test_score')

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
    # {'preprocessing__geo__n_clusters': [5, 8, 10],
    # 'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10],
    'random_forest__max_features': [6, 8]},
]


grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                                scoring='root_mean_squared_error')

grid_search.fit(housing, housing_labels)
print(pd.DataFrame(grid_search.cv_results_))
print('\n\n')

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                                scoring='neg_root_mean_squared_error')

grid_search.fit(housing, housing_labels)
print(pd.DataFrame(grid_search.cv_results_))

# final_model = grid_search.best_estimator_
# print(type(final_model))
# print(type(final_model["random_forest"]))