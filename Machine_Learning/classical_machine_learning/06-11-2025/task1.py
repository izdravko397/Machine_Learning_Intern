import numpy as np
import pandas as pd

class KNeighborsRegressor:
    posible_metric = {'euclidean': lambda X_train, x_new: np.sqrt(np.square(X_train - x_new)), 
                      'manhattan': lambda X_train, x_new: np.abs(X_train - x_new),
    }

    def __init__(self, n_neighbors=5, metric='euclidean'):
        if not (n_neighbors, int) or n_neighbors <= 0:
            raise ValueError(f"The 'n_neighbors' parameter of KNeighborsRegressor must be an int in the range [1, inf). Got {n_neighbors} instead.")
        
        self.n_neighbors = n_neighbors
        
        assert metric in self.posible_metric, 'Invalid "metric"'
        self.metric = metric
        self.X_train = None
        self.y_train = None

    @staticmethod
    def _check_data(data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim != 1:
            raise ValueError('train data must be 1 dim')
        
        return data

    def fit(self, X_train, y_train):
        X_train = self._check_data(X_train)
        y_train = self._check_data(y_train)

        if len(X_train) != len(y_train):
            raise ValueError(f'Found input variables with inconsistent numbers of samples: [{len(X_train)}, {len(y_train)}]')
        
        self.X_train = X_train
        self.y_train = y_train

    def _predict_gen(self, metric_func, new_X):
        for x in new_X:
            metric_distances = metric_func(self.X_train, x)
            n_smallest_inx = np.argpartition(metric_distances, self.n_neighbors)[:self.n_neighbors] 
            yield self.y_train[n_smallest_inx].mean()

    def predict(self, X):
        if self.X_train is None:
            raise ValueError('This KNeighborsRegressor instance is not fitted yet.')
        
        if self.n_neighbors > len(self.X_train):
            raise ValueError(f'Expected n_neighbors <= n_samples_fit, but n_neighbors = {self.n_neighbors}, n_samples_fit = {len(self.X_train)}, n_samples = {len(X)}')
        
        X = self._check_data(X)
        metric_func = self.posible_metric[self.metric]
        return np.fromiter(self._predict_gen(metric_func, X), float, count=len(X))
    
    def __repr__(self):
        return f"KNeighborsRegressor(metric='{self.metric}', n_neighbors={self.n_neighbors})"



data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat["GDP per capita (USD)"].values
y = lifesat["Life satisfaction"].values

# model = KNeighborsRegressor(3, metric='euclidean')
model = KNeighborsRegressor(3, metric='manhattan')
model.fit(X, y)
print(model)


bg_GDP = [1718, 41718, 21718]
print('Custom Model', model.predict(bg_GDP))



from sklearn.neighbors import KNeighborsRegressor
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# model = KNeighborsRegressor(3, metric='euclidean')
model = KNeighborsRegressor(3, metric='manhattan')
model.fit(X, y)

bg_GDP = [[1718], [41718], [21718]]
print('Original Model', model.predict(bg_GDP))
