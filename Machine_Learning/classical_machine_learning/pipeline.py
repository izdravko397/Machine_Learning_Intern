import numpy as np
import pandas as pd


class Pipeline:
    def __init__(self, steps, memory=False):
        if not isinstance(steps, list):
            raise TypeError('Steps must be list')

        self.dict_steps = {}
        for i, (step_name, est) in enumerate(steps):
            if not (isinstance(step_name, str) and hasattr(est, 'fit')):
                raise TypeError('Every step must be (step_name, estimator)')
            
            self.dict_steps[step_name] = i
            
        if hasattr(steps[-1], 'predict'):
            self.predict = self._last_est_method()
        else:
            self.transform = self._last_est_method()
            
        self.steps = steps
        self.memory = memory
        self._storage = {}

    def _last_est_method(self):
        def execute(X, **params):
            X = X.copy()
            for _, est in self.steps[:-1]:
                if self.memory:
                    key = self._make_hash_key(est, X)
                    trans_data = self._storage.get(key)

                    if trans_data is None:
                        trans_data = est.transform(X, **params)
                        self._storage[key] = trans_data
                    X = trans_data
                    continue

                X = est.transform(X, **params)

            last_est = self.steps[-1][1]
            if hasattr(last_est, 'predict'):
                return last_est.predict(X, **params)
            return last_est.transform(X, **params)

        return execute
    
    def __getitem__(self, key):
        return self.steps[self.dict_steps[key]][1]
    
    def _make_hash_key(self, est, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        est_name = est.__class__.__name__
        est_params = tuple(est.get_params().items())
        X_tobytes = X.tobytes()
        return hash((est_name, est_params, X_tobytes))

    def _transform_data(self, X, y=None, **params):
        X = X.copy()
        for _, est in self.steps[:-1]:
            if self.memory:
                key = self._make_hash_key(est, X)
                trans_data = self._storage.get(key)

                if trans_data is None:
                    trans_data = est.fit_transform(X, y, **params)
                    self._storage[key] = trans_data
                X = trans_data
                continue
            
            X = est.fit_transform(X, y, **params)

        return X
    
    def fit(self, X, y=None, **params):
        Xt = self._transform_data(X, y, **params)
        self.steps[-1][1].fit(Xt, y, **params)
        return self
    
    def _attr_check(self, name):
        if not hasattr(self.steps[-1][1], name):
            raise AttributeError(f'Final estimator has not {name}')
        
    def fit_predict(self, X, y=None, **params):
        self._attr_check('fit_predict')
        Xt = self._transform_data(X, y, **params)
        return self.steps[-1][1].fit_predict(Xt, y, **params)
    
    def fit_transform(self, X, y=None, **params):
        self._attr_check('fit_transform')
        Xt = self._transform_data(X, y, **params)
        return self.steps[-1][1].fit_transform(Xt, **params)
    
    def get_feature_names_out(self, input_features=None):
        for _, est in self.steps:
            if not hasattr(est, 'transform'):
                break

            input_features = est.get_feature_names_out(input_features)

        return input_features
    


def make_pipeline(*steps, memory=False):
    name_steps = []
    for est in steps:
        name = est.__class__.__name__.lower()
        name_steps.append((name, est))

    return Pipeline(name_steps, memory=memory)





from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
housing = pd.read_csv('datasets/housing/housing.csv')
housing_num = housing.select_dtypes(include=[np.number])

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), memory=True)
housing_num_prepared = num_pipeline.fit(housing_num).transform(housing_num)
print(housing_num_prepared[:2].round(2))
print(num_pipeline.get_feature_names_out())

housing_num_prepared = num_pipeline.fit_transform(housing_num)
print(housing_num_prepared[:2].round(2))
print(num_pipeline.get_feature_names_out())
# print('\n\n')


# from sklearn.pipeline import make_pipeline
# num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
# housing_num_prepared = num_pipeline.fit_transform(housing_num)
# print(housing_num_prepared[:2].round(2))
# print(num_pipeline.get_feature_names_out())