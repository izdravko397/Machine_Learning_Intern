from sklearn.base import check_array, check_is_fitted
import numpy as np

class Tree:
    def __init__(self, n_classes=None, n_outputs=1):
        self.n_outputs = n_outputs
        self.n_classes = [n_classes]
        self.children_left = []
        self.children_right = []
        self.feature = []
        self.threshold = []
        self.value = []
        self.impurity = []
        self.n_node_samples = []
        self.weighted_n_node_samples = []

class DecisionTreeBase:
    def __init__(self, kind, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                max_features=None, random_state=None, max_leaf_nodes=None, splitter='best'):
        
        self.kind = kind
        self.criterion = 'gini' if kind == 'classifier' else 'mse'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf 
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.splitter = splitter

        self.cost_func = lambda imp_l, imp_r, m_l, m_r, m: ((imp_l * m_l) / m) + ((imp_r * m_r) / m)

    def _calc_gini(self, samples):
        n = len(samples)
        gini = 1
        for cls in self.classes_:
            gini -= ((samples == cls).sum() / n) **2

        return gini
    
    def _calc_mse(self, samples):
        mean = samples.mean()
        return ((samples - mean) ** 2).mean()

    def _make_tree(self, depth, X, y, previous_score):
        node_id = len(self.tree_.feature)
        node_samples = len(y)

        if self.kind == 'classifier':
            value = np.array([[(y == cls).sum() / node_samples for cls in self.classes_]])
            imp = self._calc_gini(y)
        else:
            value = np.array([[y.mean()]])
            imp = self._calc_mse(y)

        self.tree_.value.append(value)
        self.tree_.impurity.append(imp)
        self.tree_.n_node_samples.append(node_samples)
        self.tree_.weighted_n_node_samples.append(float(node_samples))
        self.tree_.feature.append(-2)
        self.tree_.threshold.append(-2)
        self.tree_.children_left.append(-1)
        self.tree_.children_right.append(-1)

        if (self.max_depth is not None and depth == self.max_depth) or len(np.unique(y)) == 1:
            return node_id
        
        feature_inxs = np.random.choice(self.feature_inxs_, self.max_features_, replace=False)
        feature_inxs_len = len(feature_inxs)
        n_samples = X.shape[0]
        best_t = np.empty(feature_inxs_len)
        best_scores = np.full(feature_inxs_len, np.inf)
        left_masks = np.empty((feature_inxs_len, n_samples), dtype=bool)
        right_masks = np.empty((feature_inxs_len, n_samples), dtype=bool)

        for i, feature_i in enumerate(feature_inxs):
            feature = X[:, feature_i]

            match self.splitter:
                case 'best': thresholds = np.unique(feature)
                case 'random': thresholds = [np.random.uniform(feature.min(), feature.max())] 

            for t in thresholds:
                left_mask = feature <= t
                right_mask = ~left_mask

                if left_mask.sum() < 1 or right_mask.sum() < 1 and self.splitter == 'best':
                    continue
                
                if self.kind == 'classifier':
                    left_imp  = self._calc_gini(y[left_mask])
                    right_imp = self._calc_gini(y[right_mask])
                else:
                    left_imp  = self._calc_mse(y[left_mask])
                    right_imp = self._calc_mse(y[right_mask])
                score = self.cost_func(left_imp, right_imp, left_mask.sum(), right_mask.sum(), n_samples)

                if score < best_scores[i]:
                    best_t[i] = t
                    best_scores[i] = score
                    left_masks[i]  = left_mask
                    right_masks[i] = right_mask

        best_i = np.argmin(best_scores)
        if previous_score <= best_scores[best_i]:
            return node_id
        
        self.tree_.feature[node_id] = feature_inxs[best_i]
        self.tree_.threshold[node_id] = best_t[best_i]
        
        left_ch_id  = self._make_tree(depth + 1, X[left_masks[best_i]], y[left_masks[best_i]], best_scores[best_i])
        right_ch_id = self._make_tree(depth + 1, X[right_masks[best_i]], y[right_masks[best_i]], best_scores[best_i])

        self.tree_.children_left[node_id]  = left_ch_id
        self.tree_.children_right[node_id] = right_ch_id

        return node_id
    
    def _get_max_features(self):
        match self.max_features:
            case None:   return self.n_features_in_
            case 'sqrt': return np.sqrt(self.n_features_in_)
            case 'log2': return np.log2(self.n_features_in_)
            case _:
                if isinstance(self.max_features, float):
                    return self.max_features * self.n_features_in_
                return self.max_features

    def fit(self, X, y):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        self.classes_   = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.max_features_ = int(self._get_max_features())
        self.feature_inxs_ = np.arange(self.n_features_in_)
        self.tree_ = Tree(self.n_classes_)
        self._make_tree(0, X, y, np.inf)

        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        feature = self.tree_.feature
        threshold = self.tree_.threshold
        values = self.tree_.value

        tp = int if self.kind == 'classifier' else float
        predictions = np.empty(X.shape[0], dtype=tp)
        for i, sample in enumerate(X):
            node = 0
            while True:
                f_i = feature[node]
                t = threshold[node]

                if f_i == -2:
                    val = values[node]
                    pred = val.argmax() if self.kind == 'classifier' else val
                    predictions[i] = pred
                    break
                
                f = sample[f_i]
                if f <= t:
                    node = children_left[node]
                else:
                    node = children_right[node]

        return predictions
    

        
class DecisionTreeClassifier(DecisionTreeBase):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                max_features=None, random_state=None, max_leaf_nodes=None, splitter='best'):
        
        super().__init__('classifier', max_depth, min_samples_split, min_samples_leaf,
                        max_features, random_state, max_leaf_nodes, splitter)
        
class DecisionTreeRegressor(DecisionTreeBase):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                max_features=None, random_state=None, max_leaf_nodes=None, splitter='best'):
        
        super().__init__('regressor', max_depth, min_samples_split, min_samples_leaf,
                        max_features, random_state, max_leaf_nodes, splitter)