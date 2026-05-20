import numpy as np
from sklearn.base import clone
from cross_val import scoring_map_regressors, KFold, indexing


def learning_curve(estimator, X, y, *, train_sizes=np.linspace(0.1, 1.0, 5), shuffle=False,
                cv=5, scoring='root_mean_squared_erroe', exploit_incremental_learning=False):
    if not hasattr(estimator, 'fit'):
        raise TypeError('Invalid estimator')
    
    if np.any((train_sizes <= 0) | (train_sizes > 1)):
        raise ValueError('Invalid values in train_sizes')

    if not isinstance(shuffle, bool):
        raise TypeError('shuffle must be bool')
    
    if not isinstance(exploit_incremental_learning, bool):
        raise TypeError('exploit_incremental_learning must be bool')

    if not (isinstance(cv, int) and cv > 1):
        raise ValueError('cv must be int > 1')

    if exploit_incremental_learning and not hasattr(estimator, 'partial_fit'):
        raise AttributeError('estimator has not partial_fit')
    
    score_func = scoring_map_regressors.get(scoring)
    if score_func is None:
        raise ValueError(f'Invalid scoring: {scoring}')
    
    n_ticks = len(train_sizes)
    train_results = np.empty((n_ticks, cv), dtype=float)
    test_results  = np.empty((n_ticks, cv), dtype=float)
    sizes_results = np.empty(n_ticks, dtype=int)

    splitter = KFold(cv, shuffle=shuffle)
    folds_inxs = list(splitter.split(X, y))

    for i, size_pct in enumerate(train_sizes):
        for j, (train_inxs, test_inxs) in enumerate(folds_inxs): 
            size = int(len(train_inxs) * size_pct) or 1
            sized_train_inxs = train_inxs[: size]
            sizes_results[i] = size

            X_train = indexing(X, sized_train_inxs)
            y_train = indexing(y, sized_train_inxs)

            X_test = indexing(X, test_inxs)
            y_test = indexing(y, test_inxs)

            est = clone(estimator)
            if exploit_incremental_learning:
                est.partial_fit(X_train, y_train)
            else:
                est.fit(X_train, y_train)

            y_train_pred = est.predict(X_train)
            y_test_pred =  est.predict(X_test)

            train_results[i, j] = score_func(y_train, y_train_pred)
            test_results[i, j]  = score_func(y_test, y_test_pred)

    return sizes_results, train_results, test_results