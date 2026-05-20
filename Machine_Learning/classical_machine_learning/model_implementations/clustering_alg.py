from sklearn.base import check_array, check_is_fitted
import numpy as np


class KMeans:
    def __init__(self, n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _centroids_init(self, X):
        if isinstance(self.init, np.ndarray):
            if self.init.shape[0] != self.n_clusters or self.init.shape[1] != self.n_features_in_:
                raise ValueError('init shape must be (n_clusters, n_features_in_)')
            return self.init.copy()
        
        if self.init == 'random':
            ixns = np.random.choice(self.n_samples_, self.n_clusters, False)
            return X[ixns]

        centers = np.empty((self.n_clusters, self.n_features_in_))
        centers[0] = X[np.random.choice(self.n_samples_, 1, False)]
        temp = np.zeros((self.n_samples_, self.n_clusters - 1))
        s_inxs = np.arange(self.n_samples_)
        for i in range(self.n_clusters - 1):
            c = centers[i]
            temp[:, i] = ((X - c) ** 2).sum(axis=1)
            inxs = temp[:, :i + 1].argmin(axis=1)
            min_dist = temp[s_inxs, inxs]
            proba = min_dist / min_dist.sum()
            centers[i + 1] = X[np.random.choice(self.n_samples_, 1, False, p=proba)]

        return centers

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_samples_, self.n_features_in_ = X.shape

        centers, starts, inertials, labels, iters = [], [], [], [], []
        inits = self.n_init
        if isinstance(self.init, np.ndarray):
            inits = 1
        elif inits == 'auto': 
            inits = 10 if self.init == 'random' else 1

        for i_init in range(inits):
            if self.random_state is not None:
                np.random.seed(self.random_state + i_init)

            self.cluster_centers_ = self._centroids_init(X)
            starts.append(self.cluster_centers_.copy())
            for i in range(self.max_iter):
                lab = self.predict(X)
                old_centers = self.cluster_centers_.copy()
                for j in range(self.n_clusters):
                    cluster_mask = lab == j
                    if not np.any(cluster_mask):
                        continue

                    self.cluster_centers_[j] = X[cluster_mask].mean(axis=0)

                if np.linalg.norm(old_centers - self.cluster_centers_, ord=2) <= self.tol:
                    break

            c_per_point = self.cluster_centers_[lab]
            inertials.append(((X - c_per_point) ** 2).sum(axis=1).sum())
            centers.append(self.cluster_centers_.copy())
            labels.append(lab)
            iters.append(i + 1)

        min_iner = np.array(inertials).argmin()
        self.labels_ = labels[min_iner]
        self.inertia_ = inertials[min_iner]
        self.n_iter_ = iters[min_iner]
        self.cluster_centers_ = centers[min_iner]
        self.start_ = starts[min_iner]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError('Invalid features length')

        res = np.empty((X.shape[0], self.n_clusters))
        for i, c in enumerate(self.cluster_centers_):
            res[:, i] = ((X - c) ** 2).sum(axis=1)

        return res
    
    def predict(self, X):
        distances = self.transform(X)
        return distances.argmin(axis=1)
    



class MiniBatchKMeans(KMeans):
    def __init__(self, n_clusters=8, *, init='k-means++', n_init='auto', max_iter=100, 
                tol=0.01, random_state=None, batch_size=1024):
        if init not in ('k-means++', 'random') and not isinstance(init, np.ndarray):
            raise ValueError("init supports 'k-means++', 'random'")
        
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_samples_, self.n_features_in_ = X.shape

        centers, starts, inertials, labels, iters = [], [], [], [], []
        inits = self.n_init
        if isinstance(self.init, np.ndarray):
            inits = 1
        elif inits == 'auto': 
            inits = 10 if self.init == 'random' else 1

        for i_init in range(inits):
            if self.random_state is not None:
                np.random.seed(self.random_state + i_init)

            self.cluster_centers_ = self._centroids_init(X)
            starts.append(self.cluster_centers_.copy())
            num_points_per_cluster = np.zeros(self.n_clusters)
            for i in range(self.max_iter):
                batch_inxs = np.random.choice(self.n_samples_, self.batch_size, False)
                X_batch = X[batch_inxs]

                old_centers = self.cluster_centers_.copy()
                lab = self.predict(X_batch)
                for j in range(self.n_clusters):
                    cluster_mask = lab == j
                    if not np.any(cluster_mask):
                        continue
                    
                    cluster_size = cluster_mask.sum()
                    num_points_per_cluster[j] += cluster_size 
                    eta = cluster_size / num_points_per_cluster[j]
                    old = self.cluster_centers_[j]
                    new = old + eta * (X_batch[cluster_mask].mean(axis=0) - old)
                    self.cluster_centers_[j] = new

                if np.linalg.norm(self.cluster_centers_ - old_centers, ord=2) <= self.tol:
                    break
            
            lab = self.predict(X)
            c_per_point = self.cluster_centers_[lab]
            inertials.append(((X - c_per_point) ** 2).sum(axis=1).sum())
            labels.append(lab)
            centers.append(self.cluster_centers_.copy())
            iters.append(i + 1)

        min_iner = np.array(inertials).argmin()
        self.labels_ = labels[min_iner]
        self.inertia_ = inertials[min_iner]
        self.n_iter_ = iters[min_iner]
        self.cluster_centers_ = centers[min_iner]
        self.start_ = starts[min_iner]
        return self

    def partial_fit(self, X, y=None):
        X = check_array(X)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if not hasattr(self, 'cluster_centers_'):
            self.n_samples_, self.n_features_in_ = X.shape
            self.cluster_points_ = np.zeros(self.n_clusters)
            self.cluster_centers_ = self._centroids_init(X)

        lab = self.predict(X)
        for j in range(self.n_clusters):
            cluster_mask = lab == j
            if not np.any(cluster_mask):
                continue
            
            cluster_size = cluster_mask.sum()
            self.cluster_points_[j] += cluster_size 
            eta = cluster_size / self.cluster_points_[j]
            old = self.cluster_centers_[j]
            new = old + eta * (X[cluster_mask].mean(axis=0) - old)
            self.cluster_centers_[j] = new

        return self







def silhouette_coefficients(X, labels):
    uni_labels = np.unique(labels)
    silhouette_coeff = np.empty(X.shape[0])
    labels_mask = [labels == lb for lb in uni_labels]

    for i, point in enumerate(X):
        p_lab = labels[i]
        dist = np.linalg.norm(X - point, ord=2, axis=1)
        b = np.inf
        for j, lb in enumerate(uni_labels):
            curr_mean_dis = dist[labels_mask[j]].mean()

            if lb == p_lab:
                a = curr_mean_dis
            elif curr_mean_dis < b:
                b = curr_mean_dis

        silhouette_coeff[i] = (b - a) / np.max([a, b])

    return silhouette_coeff

def silhouette_score(X, labels):
    return silhouette_coefficients(X, labels).mean()

def silhouette_diagram(X, labels):
    import matplotlib.pyplot as plt
    uni_lb = np.unique(labels)
    cluster_points = np.array([(labels == lb).sum() for lb in uni_lb])
    n_clusters = len(uni_lb)
    coef = silhouette_coefficients(X, labels)

    height_cluster = (1 / cluster_points) - 0.0001
    y_ticks = []
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots()
    for i, lb in enumerate(uni_lb):
        cluster_coef = np.sort(coef[labels == lb])
        height = height_cluster[i]

        start = i + (cluster_points[i] * height)
        cluster_y = start + np.arange(cluster_points[i]) * height
        y_ticks.append(cluster_y.mean())

        if i == 0:
            min = cluster_y[0]

        elif i == n_clusters - 1:
            max = cluster_y[-1]

        for p_coef, y in zip(cluster_coef, cluster_y):
            ax.barh(y, p_coef, height, color=colors[i])

    ax.vlines(coef.mean(), min, max, 'red', 'dashed')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(np.arange(n_clusters))
    ax.set_ylabel('Cluster')
    ax.set_xlabel('Silhouette Coefficient')
    return ax