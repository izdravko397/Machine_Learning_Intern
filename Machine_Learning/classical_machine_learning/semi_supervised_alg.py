from sklearn.base import check_array, check_is_fitted, is_classifier, clone
import numpy as np


class SelfTrainingClassifier:
    def __init__(self, estimator, threshold=0.75, criterion='threshold', k_best=10, max_iter=10):
        if not is_classifier(estimator) and not hasattr(estimator, 'predict_proba'):
            raise TypeError('Invalid estimator')

        self.estimator = estimator
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.max_iter = max_iter

    def fit(self, X, y):
        X = check_array(X)
        self.n_samples_, self.n_features_in_ = X.shape

        self.transduction_ = y.copy()
        labeled_mask = (self.transduction_ != -1)
        for i in range(self.max_iter):
            X_labeled, y_labeled = X[labeled_mask], self.transduction_[labeled_mask]
            est = clone(self.estimator)
            unlabeled_mask = np.where(~labeled_mask)[0]
            probs = est.fit(X_labeled, y_labeled).predict_proba(X[unlabeled_mask])

            props_max_vals = probs.max(axis=1)
            match self.criterion:
                case 'threshold':
                    confident_mask = (props_max_vals >= self.threshold)
                    self.transduction_[unlabeled_mask[confident_mask]] = probs[confident_mask].argmax(axis=1)
                    labeled_mask[unlabeled_mask] |= confident_mask 
                case 'k_best':
                    k_best_inxs = props_max_vals.argsort()[-self.k_best:]
                    self.transduction_[unlabeled_mask[k_best_inxs]] = probs[k_best_inxs].argmax(axis=1)
                    labeled_mask[unlabeled_mask[k_best_inxs]] = True
                case _:
                    raise ValueError('criterion must be threshold or k_best')

            if np.all(labeled_mask):
                break

        self.n_iter_ = i + 1
        self.estimator_ = est
        return self
    
    def __getattr__(self, key):
        obj = self.estimator_ if hasattr(self, 'estimator_') else self.estimator
        return getattr(obj, key)


import joblib
from pathlib import Path
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class KMeansLearningClassifier:
    dir_prefix = 'kmeanslearning_models/'

    def __init__(self, n_clusters=8, model_file_name=None, plot_func=None, random_state=None, test=False):
        if Path(self.dir_prefix + model_file_name).exists():
            obj = joblib.load(self.dir_prefix + model_file_name)
            self.__dict__.update(obj.__dict__)
            return

        self.model_file_name = model_file_name
        self.plot_func = plot_func
        self.test = test
        self._kmeans = KMeans(n_clusters, n_init=5, random_state=random_state)
        self._log_reg = LogisticRegression(max_iter=1_000, random_state=random_state)
        
    def fit(self, X):
        X = check_array(X)
        self.X_train_ = X.copy()
        repres_inxs = self._kmeans.fit_transform(X).argmin(axis=0)
        self.representative_samples_ = X[repres_inxs]

        cols = int(np.ceil(np.sqrt(self._kmeans.n_clusters)))
        rows = int(np.ceil(self._kmeans.n_clusters / cols))

        fig, ax = plt.subplots(rows, cols)
        inx = 0
        break_flag = False
        for i in range(rows):
            for j in range(cols):
                self.plot_func(self.representative_samples_[inx], ax[i, j])
                inx += 1
                if inx >= self._kmeans.n_clusters:
                    break_flag = True
                    break

            if break_flag:
                break

        joblib.dump(self, self.dir_prefix + self.model_file_name)
        plt.show()

        if self.test:
            return repres_inxs
        
        return self
    
    def fit_labels(self, labels):
        if len(labels) != len(self.representative_samples_):
            raise ValueError('Invalid number of labels')
        
        y_train_propagated = np.empty(self.X_train_.shape[0], dtype=np.int64)
        for i in range(self._kmeans.n_clusters):
            y_train_propagated[self._kmeans.labels_ == i] = labels[i]

        self._log_reg.fit(self.X_train_, y_train_propagated)
        self.y_train_propagated_ = y_train_propagated
        joblib.dump(self, self.dir_prefix + self.model_file_name)
        return self
    
    def __getattr__(self, key):
        log_reg = self.__dict__.get("_log_reg", None)
        if log_reg is None:
            raise AttributeError(f"{key} not initialized")

        return getattr(log_reg, key)



class ActiveLearningClassifier:
    dir_prefix = 'active_learning/'

    def __init__(self, n_clusters=8, model_file_name=None, plot_func=None, random_state=None, test=False):
        if Path(self.dir_prefix + model_file_name).exists():
            obj = joblib.load(self.dir_prefix + model_file_name)
            self.__dict__.update(obj.__dict__)
            return

        self.model_file_name = model_file_name
        self.plot_func = plot_func
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.test = test
        self._log_reg = LogisticRegression(max_iter=1_000, random_state=random_state, warm_start=True)
        
    def fit(self, X):
        X = check_array(X)
        self._kmeans = KMeans(self.n_clusters, n_init=5, init='random')
        repres_inxs = self._kmeans.fit_transform(X).argmin(axis=0)

        val = X[repres_inxs]
        if hasattr(self, 'representative_samples_'):
            val = np.vstack([self.representative_samples_, val])

        self.representative_samples_ = val

        cols = int(np.ceil(np.sqrt(self.n_clusters)))
        rows = int(np.ceil(self.n_clusters / cols))

        fig, ax = plt.subplots(rows, cols, )
        inx = 0
        break_flag = False
        for i in range(rows):
            for j in range(cols):
                self.plot_func(self.representative_samples_[inx], ax[i, j])
                inx += 1
                if inx >= self.n_clusters:
                    break_flag = True
                    break

            if break_flag:
                break

        joblib.dump(self, self.dir_prefix + self.model_file_name)
        plt.show()
        if self.test:
            return repres_inxs
        return self
    
    def fit_labels(self, labels):
        val = labels
        if hasattr(self, 'representative_labels_'):
            val = np.concatenate([self.representative_labels_, val])
        
        self.representative_labels_ = val

        self._log_reg.fit(self.representative_samples_, self.representative_labels_)
        joblib.dump(self, self.dir_prefix + self.model_file_name)
        return self
    
    def __getattr__(self, key):
        log_reg = self.__dict__.get("_log_reg", None)
        if log_reg is None:
            raise AttributeError(f"{key} not initialized")

        return getattr(log_reg, key)
    



from sklearn.neighbors import KNeighborsClassifier

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def _make_cluster(self, point_inx, neighbors_inxs):
        if point_inx in self._visited_points_inxs:
            return 0

        self.labels_[point_inx] = self._clusters_counter
        self._visited_points_inxs.add(point_inx)

        for i in neighbors_inxs:
            self._make_cluster(i, self.point_neighbors_inxs[i])    
            self.labels_[i] = self._clusters_counter

    def fit(self, X):
        X = check_array(X)
        self.n_samples_, self.n_features_in_ = X.shape 
        self.point_neighbors_inxs = []
        core_sample_indices = []
        components = []

        for i, point in enumerate(X):
            point_dis = np.linalg.norm(X - point, ord=2, axis=1)
            neighbors_mask = (point_dis < self.eps)
            if neighbors_mask.sum() < self.min_samples:
                self.point_neighbors_inxs.append([])
                continue

            core_sample_indices.append(i)
            components.append(point)
            self.point_neighbors_inxs.append(np.where(neighbors_mask)[0])

        self.core_sample_indices_ = np.asarray(core_sample_indices)
        self.components_ = np.asarray(components)

        self.labels_ = np.ones(self.n_samples_) * -1
        self._clusters_counter = 0
        self._visited_points_inxs = set()
        for point_i in self.core_sample_indices_:
            val = self._make_cluster(point_i, self.point_neighbors_inxs[point_i])
            val = 1 if val is None else val
            self._clusters_counter += val

        self.knn_ = KNeighborsClassifier(self.min_samples).fit(X, self.labels_)

        # del self.point_neighbors_inxs
        # del self._visited_points_inxs
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError('Invalid features lenght')
        
        dist, inxs = self.knn_.kneighbors(X, self.min_samples, return_distance=True)
        points_in_eps = (dist <= self.eps).sum(axis=1)

        pred = np.ones(X.shape[0]) * -1
        for i in range(X.shape[0]):
            if points_in_eps[i] < self.min_samples:
                continue

            points_labels = self.labels_[inxs[i]]
            uni, counts = np.unique(points_labels, return_counts=True)
            pred[i] = uni[counts.argmax()]

        return pred