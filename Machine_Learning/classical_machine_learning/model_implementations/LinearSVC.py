import numpy as np
from sklearn.base import check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler, add_dummy_feature


class LinearSVC:
    def __init__(self, max_iter=1000, tol=0.0001, eta=0.001,
            n_iter_no_change=5, c=1, random_state=None):
        self.c = c
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        y_b = y.astype(int).copy()
        y_b[y_b == 0] = -1

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)

        self.X_scaler_ = StandardScaler()
        X_scaled = self.X_scaler_.fit_transform(X)
        X_b = add_dummy_feature(X_scaled)
        theta = np.random.randn(X_b.shape[1], 1)
        # theta = np.zeros((X_b.shape[1], 1))
        counter = 0

        for epoch in range(self.max_iter):
            marges = (y_b * (X_b @ theta)).ravel()
            active = marges < 1

            gradiend = theta.copy()
            if np.any(active):
                gradiend -= self.c * np.sum((y_b[active] * X_b[active]), axis=0).reshape(-1, 1)

            theta -= self.eta * gradiend
            if np.linalg.norm(gradiend, ord=2) < self.tol:
                counter += 1
                if counter == self.n_iter_no_change:
                    break
            else:
                counter = 0

        self.n_iter_ = epoch + 1
        self.n_features_in_ = X.shape[1]
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        self.theta_ = theta
        return self

    def predict(self, X):
        decision_function = self.decision_function(X)
        class0, class1 = self.classes_
        return np.where(decision_function >= 0, class1, class0)
    
    def decision_function(self, X):
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Invalid features length')
        
        X_b = self.X_scaler_.transform(X)
        X_b = add_dummy_feature(X_b) 
        return X_b @ self.theta_
    
    def predict_proba(self, X):
        decision_function = self.decision_function(X)
        pos_proba = 1 / (1 + np.exp(-decision_function))
        neg_proba = 1 - pos_proba
        return np.c_[neg_proba, pos_proba]



from sklearn.datasets import load_iris
from sklearn.metrics import f1_score, accuracy_score


iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0).values # Iris virginica

print('--------- Custom ---------')
model = LinearSVC(eta=0.01, random_state=42, n_iter_no_change=5, tol=0.05)
model.fit(X, y)
pred = model.predict(X)
print(model.n_iter_)
print('F1:', f1_score(y, pred))
print('accuracy:', accuracy_score(y, pred))
print(model.coef_.ravel(), model.intercept_[0])
print(model.decision_function(X)[:4])
print(model.predict_proba(X)[:4])
print(y[:4])

print('--------- Original ---------')
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

model = make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=42))
model.fit(X, y)
pred = model.predict(X)
print(model['linearsvc'].n_iter_)
print('F1:', f1_score(y, pred))
print('accuracy:', accuracy_score(y, pred))
print(model['linearsvc'].coef_.ravel(), model['linearsvc'].intercept_[0])
print(model.decision_function(X)[:4])






# def fit(self, X, y): # [-1.37751626 -1.16257315] -1.6600000000000006
#         X = check_array(X)
#         if y.ndim == 1:
#             y = y.reshape(-1, 1)

#         y_b = y.astype(int).copy()
#         y_b[y_b == 0] = -1

#         if self.random_state is not None:
#             np.random.seed(self.random_state)

#         self.classes_ = np.unique(y)

#         self.X_scaler_ = StandardScaler()
#         X_b = self.X_scaler_.fit_transform(X)
#         # theta = np.random.randn(X_b.shape[1], 1)
#         coef = np.zeros((X_b.shape[1], 1))
#         bias = np.array([0.0])
#         counter = 0

#         for epoch in range(self.max_iter):
#             marges = (y_b * (X_b @ coef + bias)).ravel() #/ np.linalg.norm(coef, ord=2)
#             active = marges < 1

#             gradiend_c = coef - self.c * np.sum((y_b[active] * X_b[active]), axis=0).reshape(-1, 1)
#             gradiend_b = -self.c * np.sum(y_b[active], axis=0)

#             coef -= self.eta * gradiend_c
#             bias -= self.eta * gradiend_b

#             if np.linalg.norm(coef, ord=2) < self.tol: #**2 / 2
#                 counter += 1
#                 if counter == self.n_iter_no_change:
#                     break
#             else:
#                 counter = 0

#         self.n_iter_ = epoch + 1
#         self.n_features_in_ = X.shape[1]
#         self.intercept_ = bias
#         self.coef_ = coef
#         self.theta_ = np.vstack([bias, coef])
#         return self