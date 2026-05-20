from sklearn.base import check_array, check_is_fitted, TransformerMixin, BaseEstimator
from sklearn.utils.extmath import svd_flip
import numpy as np

class PCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = check_array(X)

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        U, Vt = svd_flip(U, Vt, u_based_decision=False)

        self.components_ = Vt
        self.explained_variance_ = s ** 2 / (X.shape[0] - 1)
        full_variance = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / full_variance

        if self.n_components is None:
            self.n_components_ = self.components_.shape[0]
        elif 0 < self.n_components < 1:
            cumsum = np.cumsum(self.explained_variance_ratio_)
            self.n_components_ = (cumsum < self.n_components).sum() + 1
        elif isinstance(self.n_components, int) and self.n_components <= self.components_.shape[0]:
            self.n_components_ = self.n_components
        else:
            raise ValueError('Invalid n_components')
        
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)

        X_centered = X - self.mean_
        return X_centered @ self.components_[:self.n_components_].T

    def inverse_transform(self, X):
        return (X @ self.components_[:self.n_components_]) + self.mean_




# ----------------- 3D DataSet -----------------
# from scipy.spatial.transform import Rotation

# m = 60
# X = np.zeros((m, 3))  # initialize 3D dataset
# np.random.seed(42)
# angles = (np.random.rand(m) ** 3 + 0.5) * 2 * np.pi  # uneven distribution
# X[:, 0], X[:, 1] = np.cos(angles), np.sin(angles) * 0.5  # oval
# X += 0.28 * np.random.randn(m, 3)  # add more noise
# X = Rotation.from_rotvec([np.pi / 29, -np.pi / 20, np.pi / 4]).apply(X)
# X += [0.2, 0, 0.2]  # shift a bit
# -----------------------------------------------


# custom_pca = PCA(2) 
# custom = custom_pca.fit_transform(X) 
# inverse_custom = custom_pca.inverse_transform(custom)
# print('custom:', custom_pca.n_components_)


# from sklearn.decomposition import PCA
# org_pca = PCA(2)
# org = org_pca.fit_transform(X) 
# inverse_org = org_pca.inverse_transform(org)

# print('Org:', org_pca.n_components_)


# # print(custom == org)
# # print(np.allclose(custom, org, 1))
# print('------------ Transform ------------')
# print(np.hstack([org, custom])) # , np.zeros(len(org)).reshape(-1, 1)
# print('------------ InverseTransform ------------')
# print(np.hstack([inverse_org, inverse_custom])) # , np.zeros(len(org)).reshape(-1, 1)


from sklearn.random_projection import johnson_lindenstrauss_min_dim

class GaussianRandomProjection(TransformerMixin, BaseEstimator):
    def __init__(self, n_components='auto', *, eps=0.1, compute_inverse_components=False, random_state=None):
        if not isinstance(n_components, int) and n_components != 'auto':
            raise ValueError('n_components must be int or auto')
        
        self.n_components = n_components
        self.eps = eps
        self.compute_inverse_components = compute_inverse_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_samples_, self.n_features_in_ = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_components_ = self.n_components
        if self.n_components == 'auto':
            self.n_components_ = johnson_lindenstrauss_min_dim(self.n_samples_, eps=self.eps)

        P = np.random.randn(self.n_components_, self.n_features_in_) / np.sqrt(self.n_components_)
        self.components_ = P
        if self.compute_inverse_components:
            self.inverse_components_ = np.linalg.pinv(P)

        return self
    
    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if self.n_features_in_ != X.shape[1]:
            raise ValueError('Invalid features length')

        return X @ self.components_.T
    
    def inverse_transform(self, X):
        if not self.compute_inverse_components:
            raise ValueError('compute_inverse_components must be True to call inverse_transform')
        
        return X @ self.inverse_components_.T
        


# print(johnson_lindenstrauss_min_dim(60_000, eps=0.4))

m, n = 50, 4000
np.random.seed(42)
X = np.random.randn(m, n)


# print('---------------------- custom ----------------------')
# custom = GaussianRandomProjection(compute_inverse_components=True, random_state=42)
# c_trans = custom.fit_transform(X)
# print('Transform')
# print(c_trans[0])
# print('InverseTransform')
# print(custom.inverse_transform(c_trans)[0])

# print('---------------------- original ----------------------')
# from sklearn.random_projection import GaussianRandomProjection
# org = GaussianRandomProjection(compute_inverse_components=True, random_state=42)
# o_trans = org.fit_transform(X)
# print(o_trans[0])
# print('InverseTransform')
# print(org.inverse_transform(o_trans)[0])


# ---------------------- custom ----------------------
# Transform
# [68.69451881 -0.79981557  0.47445884 ... -1.3609359   1.79177706
#   0.71129981]
# InverseTransform
# [ 0.49671415 -0.1382643   0.64768854 ... -0.3202978   1.64337816
#   0.36064789]
# ---------------------- original ----------------------
# [68.69451881 -0.79981557  0.47445884 ... -1.3609359   1.79177706
#   0.71129981]
# InverseTransform
# [ 0.49671415 -0.1382643   0.64768854 ... -0.3202978   1.64337816
#   0.36064789]





from scipy.sparse import csr_matrix

class SparseRandomProjection(TransformerMixin, BaseEstimator):
    def __init__(self, n_components='auto', *, density='auto', eps=0.1, dense_output=False, 
                 compute_inverse_components=False, random_state=None):
        if not isinstance(n_components, int) and n_components != 'auto':
            raise ValueError('n_components must be int or auto')
        
        self.n_components = n_components
        self.density = density
        self.dense_output = dense_output
        self.eps = eps
        self.compute_inverse_components = compute_inverse_components
        self.random_state = random_state

    def fit(self, X, y=None):
        self.n_samples_, self.n_features_in_ = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.density_ = self.density
        if self.density == 'auto':
            self.density_ = 1 / np.sqrt(self.n_features_in_)

        self.n_components_ = self.n_components
        if self.n_components == 'auto':
            self.n_components_ = johnson_lindenstrauss_min_dim(self.n_samples_, eps=self.eps)

        non_zero_el_col =  int(np.ceil(self.density_ * self.n_features_in_))
        data, rows, cols = [], [], []
        feature_inxs = self.n_features_in_
        for col in range(self.n_components_):
            inxs = np.random.choice(feature_inxs, size=non_zero_el_col, replace=False)
            signs = np.random.choice([1, -1], size=non_zero_el_col)
            vals = signs / np.sqrt(non_zero_el_col)

            data.extend(vals)
            rows.extend(inxs)
            cols.extend([col] * non_zero_el_col)

        self.components_ = csr_matrix((data, (rows, cols)), shape=(self.n_features_in_, self.n_components_)).T
        if self.compute_inverse_components:
            self.inverse_components_ = np.linalg.pinv(self.components_.toarray())

        return self
    
    def transform(self, X):
        check_is_fitted(self)

        if self.n_features_in_ != X.shape[1]:
            raise ValueError('Invalid features length')

        res = X @ self.components_.T
        if self.dense_output and isinstance(res, csr_matrix):
            res = res.toarray()

        return res 
    
    def inverse_transform(self, X):
        if not self.compute_inverse_components:
            raise ValueError('compute_inverse_components must be True to call inverse_transform')
        
        return X @ self.inverse_components_.T
    


print('---------------------- custom ----------------------')
custom = SparseRandomProjection(dense_output=True, compute_inverse_components=True, random_state=42)
c_trans = custom.fit_transform(csr_matrix(X))
print('Transform')
print(c_trans[0])
print(type(c_trans))
print('InverseTransform')
print(custom.inverse_transform(c_trans)[0])

print('---------------------- original ----------------------')
from sklearn.random_projection import SparseRandomProjection
org = SparseRandomProjection(dense_output=True, compute_inverse_components=True, random_state=42)
o_trans = org.fit_transform(csr_matrix(X))
print(o_trans[0])
print(type(o_trans))

print('InverseTransform')
print(org.inverse_transform(o_trans)[0])