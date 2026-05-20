import numpy as np
import pandas as pd
from pipeline import make_pipeline
from function_transformer import check_is_fitted
from scipy.sparse import csr_matrix, hstack
from collections import defaultdict
from sklearn.compose import make_column_selector
from sparse_matrix import SparseMatrix


class ColumnTransformer:
    def __init__(self, transformers, remainder='drop', sparse_threshold=0.3):
        if not isinstance(transformers, list):
            raise TypeError('transformers must be list')

        for name, trans, cols in transformers:
            if not (isinstance(name, str) and hasattr(trans, 'transform') 
                    and isinstance(cols, (list, np.ndarray, make_column_selector))):
                raise TypeError('Every transformer must be (name, transformer, columns)')
            
        self.transformers = transformers

        if not (0 <= sparse_threshold <= 1):
            raise ValueError('sparse_threshold must be [0, 1]')
        
        self.sparse_threshold = sparse_threshold

        if remainder not in ('passtrgough', 'drop'):
            raise ValueError('Invalid remainder value')
        
        self.remainder = remainder
        self._get_group_data = lambda X, cols: X[cols].values if isinstance(X, pd.DataFrame) else X[:, cols]

    def fit(self, X, y=None, **params):
        for i, (n, trans, cols) in enumerate(self.transformers):
            if isinstance(cols, make_column_selector):
                cols = cols(X)
                self.transformers[i] = (n, trans, cols)

            group_data = self._get_group_data(X, cols)
            trans.fit(group_data, y, **params)

        self.transformers_ = self.transformers
        return self

    def transform(self, X, **params):
        check_is_fitted(self)

        trans_data = []
        selected_cols = []
        res_data_rows = 0
        res_data_cols = 0
        res_data_non_zero_count = 0 
        for _, trans, cols in self.transformers_:
            selected_cols.extend(cols)
            group_data = self._get_group_data(X, cols)
            group_trans_data = trans.transform(group_data, **params)

            res_data_rows += group_trans_data.shape[0]
            res_data_cols += group_trans_data.shape[1]
            res_data_non_zero_count += group_trans_data.nnz if isinstance(group_trans_data, csr_matrix) \
                                                  else np.count_nonzero(group_trans_data)

            trans_data.append(group_trans_data)

        
        if self.remainder == 'passtrgough':
            all_cols = X.columns.to_numpy() if isinstance(X, pd.DataFrame) else np.arange(len(X[0]))
            no_selected_cols = all_cols[~np.isin(all_cols, selected_cols)]
            data = self._get_group_data(X, no_selected_cols)
            res_data_rows += data.shape[0]
            res_data_cols += data.shape[1]
            res_data_non_zero_count += np.count_nonzero(data)
            trans_data.append(data)

        non_zero_el_pct = res_data_non_zero_count / (res_data_rows * res_data_cols)
        if non_zero_el_pct <= self.sparse_threshold:
            trans_data = [csr_matrix(mat) if isinstance(mat, np.ndarray) else mat for mat in trans_data]
            res = hstack(trans_data)
        else:
            trans_data = [mat.toarray() if isinstance(mat, csr_matrix) else mat for mat in trans_data]
            res = np.hstack(trans_data)

        return res 

    def fit_transform(self, X, y=None, **params):
        trans_data = []
        selected_cols = []
        res_data_rows = 0
        res_data_cols = 0
        res_data_non_zero_count = 0 
        for i, (n, trans, cols) in enumerate(self.transformers):
            if isinstance(cols, make_column_selector):
                cols = cols(X)
                self.transformers[i] = (n, trans, cols)

            selected_cols.extend(cols)
            group_data = self._get_group_data(X, cols)
            group_trans_data = trans.fit_transform(group_data, y, **params)

            res_data_rows += group_trans_data.shape[0]
            res_data_cols += group_trans_data.shape[1]
            res_data_non_zero_count += group_trans_data.nnz if isinstance(group_trans_data, csr_matrix) \
                                                  else np.count_nonzero(group_trans_data)

            trans_data.append(group_trans_data)

        if self.remainder == 'passtrgough':
            all_cols = X.columns.to_numpy() if isinstance(X, pd.DataFrame) else np.arange(len(X[0]))
            no_selected_cols = all_cols[~np.isin(all_cols, selected_cols)]
            data = self._get_group_data(X, no_selected_cols)
            res_data_rows += data.shape[0]
            res_data_cols += data.shape[1]
            res_data_non_zero_count += np.count_nonzero(data)
            trans_data.append(data)

        self.transformers_ = self.transformers

        non_zero_el_pct = res_data_non_zero_count / (res_data_rows * res_data_cols)
        if non_zero_el_pct <= self.sparse_threshold:
            trans_data = [csr_matrix(mat) if isinstance(mat, np.ndarray) else mat for mat in trans_data]
            res = hstack(trans_data)
        else:
            trans_data = [mat.toarray() if isinstance(mat, csr_matrix) else mat for mat in trans_data]
            res = np.hstack(trans_data)

        return res 
    
    def get_feature_names_out(self):
        check_is_fitted(self)

        feature_names = []
        for name, trans, cols in self.transformers_:
            feature_names.extend(np.char.add(f'{name}_', trans.get_feature_names_out(cols)))

        return np.array(feature_names)


def make_column_transformer(*transformers, remainder='drop', sparse_threshold=0.3):
    name_transformers = []
    names = defaultdict(int)
    for trans, cols in transformers:
        name = trans.__class__.__name__.lower()
        names[name] += 1
        if (count := names[name]) > 1:
            if count == 2:
                for i, (n, t, c) in enumerate(name_transformers):
                    if n == name:
                        name_transformers[i] = (n + '-1', t, c)
                        break

            name += f'-{count}'
        name_transformers.append((name, trans, cols))

    return ColumnTransformer(name_transformers, remainder, sparse_threshold)









from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
                    "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

housing = pd.read_csv('datasets/housing/housing.csv')

# print(cat_pipeline.fit_transform(housing[["ocean_proximity"]].values)) !!!!!!!!


preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
], sparse_threshold=0.8)

# preprocessing = make_column_transformer(
#     (num_pipeline, make_column_selector(dtype_include=np.number)),
#     (cat_pipeline, make_column_selector(dtype_include=object)),
# )

housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)
print(preprocessing.get_feature_names_out())


print('\n\n')
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
], sparse_threshold=0.8)

# preprocessing = make_column_transformer(
#     (num_pipeline, make_column_selector(dtype_include=np.number)),
#     (cat_pipeline, make_column_selector(dtype_include=object)),
# )

housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)
print(preprocessing.get_feature_names_out())

