from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.pipeline import FunctionTransformer, make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin, check_array
# from cross_val_score import cross_val_score
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint
from sklearn.svm import SVR

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


class KNNPrice(BaseEstimator, TransformerMixin):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        model = KNeighborsRegressor(self.k)
        model.fit(X, y)
        self.KNN_ = model
        return self
    
    def transform(self, X):
        return self.KNN_.predict(X).reshape(-1, 1)
        

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

cluster_simil = ClusterSimilarity(n_clusters=15, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

knn_price = KNNPrice(5)

housing = pd.read_csv('datasets/housing/housing.csv')

housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# inxs = np.random.permutation(len(housing))[:5000]

# housing_labels = housing_labels.iloc[inxs]
# housing = housing.iloc[inxs]

# ==================== task1 ====================
# preprocessing = ColumnTransformer([
#         ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
#         ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
#         ("people_per_house", ratio_pipeline(), ["population", "households"]),
#         ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
#                                "households", "median_income"]),
#         ("geo", cluster_simil, ["latitude", "longitude"]),
#         ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
#     ],
#     remainder=default_num_pipeline)

# full_pipeline = Pipeline([
# ("preprocessing", preprocessing),
# ("svr", SVR()),
# ])

# grid = [
# {
#     'svr__kernel': ["linear"],
#     'svr__C': [0.1, 1, 10], 
#     'preprocessing__geo__n_clusters': [10, 15]
# },
# {
#     'svr__kernel': ["rbf"],
#     'svr__C': [0.1, 1, 10],        
#     'svr__gamma': [0.01, 0.1, 1, 10],
#     'preprocessing__geo__n_clusters': [5, 20]
# }
# ]

# grid_search = GridSearchCV(full_pipeline, grid, cv=3,
#                                 scoring='neg_root_mean_squared_error')

# grid_search.fit(housing, housing_labels)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# df = pd.DataFrame(grid_search.cv_results_).sort_values('mean_test_score', ascending=False)
# print(df[['params', 'mean_test_score', 'std_test_score']].head())

# ==================== task2 ====================
# preprocessing = ColumnTransformer([
#         ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
#         ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
#         ("people_per_house", ratio_pipeline(), ["population", "households"]),
#         ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
#                                "households", "median_income"]),
#         ("geo", cluster_simil, ["latitude", "longitude"]),
#         ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
#     ],
#     remainder=default_num_pipeline)

# full_pipeline = Pipeline([
# ("preprocessing", preprocessing),
# ("svr", SVR()),
# ])

# grid = [
# {
#     'svr__kernel': ["linear"],
#     'svr__C': randint(low=1, high=100), 
#     'preprocessing__geo__n_clusters': randint(low=3, high=50)
# },
# {
#     'svr__kernel': ["rbf"],
#     'svr__C': randint(low=1, high=100),        
#     'svr__gamma': randint(low=1, high=5),
#     'preprocessing__geo__n_clusters': randint(low=3, high=50)
# }
# ]

# grid_search = RandomizedSearchCV(full_pipeline, grid, cv=3, n_iter=10,
#                                 scoring='neg_root_mean_squared_error')

# grid_search.fit(housing, housing_labels)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# df = pd.DataFrame(grid_search.cv_results_).sort_values('mean_test_score', ascending=False)
# print(df[['params', 'mean_test_score', 'std_test_score']].head())

# ==================== task3 ====================
# preprocessing = ColumnTransformer([
#         ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
#         ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
#         ("people_per_house", ratio_pipeline(), ["population", "households"]),
#         ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
#                                "households", "median_income"]),
#         ("geo", cluster_simil, ["latitude", "longitude"]),
#         ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
#     ],
#     remainder=default_num_pipeline)

# full_pipeline = Pipeline([
# ("preprocessing", preprocessing),
# ('feature_selection', SelectFromModel(SVR())),
# ("svr", SVR()),
# ])

# grid = {
#     'svr__kernel': ["linear"],
#     'feature_selection__estimator__kernel': ['linear'],
#     'svr__C': randint(low=1, high=100), 
#     'feature_selection__estimator__C': randint(low=1, high=100), 
#     'preprocessing__geo__n_clusters': randint(low=3, high=50)
# }

# grid_search = RandomizedSearchCV(full_pipeline, grid, cv=3, n_iter=10,
#                                 scoring='neg_root_mean_squared_error')

# grid_search.fit(housing, housing_labels)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# df = pd.DataFrame(grid_search.cv_results_).sort_values('mean_test_score', ascending=False)
# print(df[['params', 'mean_test_score', 'std_test_score']].head())

# ==================== task4 ====================
# inxs = np.random.permutation(len(housing))[:100]

# housing_labels = housing_labels.iloc[inxs]
# housing = housing.iloc[inxs]

# preprocessing = ColumnTransformer([
#         ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
#         ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
#         ("people_per_house", ratio_pipeline(), ["population", "households"]),
#         ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
#                                "households", "median_income"]),
#         ("geo", cluster_simil, ["latitude", "longitude"]),
#         ("knn_price", knn_price, ["latitude", "longitude"]),
#         ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
#     ],
#     remainder=default_num_pipeline)

# full_pipeline = Pipeline([
# ("preprocessing", preprocessing),
# ('feature_selection', SelectFromModel(SVR())),
# ("svr", SVR()),
# ])

# grid = {
#     'svr__kernel': ["linear"],
#     'feature_selection__estimator__kernel': ['linear'],
#     'svr__C': randint(low=1, high=100), 
#     'feature_selection__estimator__C': randint(low=1, high=100), 
#     'preprocessing__geo__n_clusters': randint(low=3, high=50),
#     'preprocessing__knn_price__k': randint(low=3, high=10)
# }

# grid_search = RandomizedSearchCV(full_pipeline, grid, cv=2, n_iter=1,
#                                 scoring='neg_root_mean_squared_error')

# grid_search.fit(housing, housing_labels)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# df = pd.DataFrame(grid_search.cv_results_).sort_values('mean_test_score', ascending=False)
# print(df[['params', 'mean_test_score', 'std_test_score']].head())

# ==================== task5 ====================









#---------------------------------------------------------------------
# X = housing[['median_income']]
# y = housing[['median_house_value']]

# rmspe_linear = cross_val_score(LinearRegression(), X, y, 
#                     scoring='root_mean_squared_procentage_error', cv=10).mean()

# mape_linear = cross_val_score(LinearRegression(), X, y, 
#                     scoring='mean_absolute_procentage_error', cv=10).mean()

# rmspe_knn = cross_val_score(KNeighborsRegressor(5), X, y, 
#                     scoring='root_mean_squared_procentage_error', cv=10).mean()

# mape_knn = cross_val_score(KNeighborsRegressor(5), X, y, 
#                     scoring='mean_absolute_procentage_error', cv=10).mean()

# tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
# rmspe_tree_reg = cross_val_score(tree_reg, housing, housing_labels, 
#                     scoring='root_mean_squared_procentage_error', cv=4).mean()

# mape_tree_reg = cross_val_score(tree_reg, housing, housing_labels, 
#                     scoring='mean_absolute_procentage_error', cv=4).mean()

# forest_reg = make_pipeline(preprocessing, RandomForestRegressor(max_features=6, random_state=42))
# rmspe_forest_reg = cross_val_score(forest_reg, housing, housing_labels, 
#                     scoring='root_mean_squared_procentage_error', cv=3).mean()

# mape_forest_reg = cross_val_score(forest_reg, housing, housing_labels, 
#                     scoring='mean_absolute_procentage_error', cv=3).mean()


# df = pd.DataFrame({
#     'RMSPE': [rmspe_linear, rmspe_knn, rmspe_tree_reg, rmspe_forest_reg],
#     'MAPE': [mape_linear, mape_knn, mape_tree_reg, mape_forest_reg]
# }, index=['Linear', 'KNN', 'DecisionTree', 'RandomForest'])

# print(df)