import numpy as np

def to_numpy(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    return arr

def binary_conf_mat(y_true, y_pred, unique):
    neg, pos = unique

    matches = y_true == y_pred
    pos_mask_pred = y_pred == pos
    neg_mask_pred = y_pred == neg

    tn = (neg_mask_pred & matches).sum()
    tp = (pos_mask_pred & matches).sum()

    non_matches = ~matches
    pos_mask_true = y_true == pos
    neg_mask_true = y_true == neg

    fn = (pos_mask_true & non_matches).sum()
    fp = (neg_mask_true & non_matches).sum()

    return np.array([[tn, fp], 
                     [fn, tp]])

def confusion_matrix(y_true, y_pred, weight=None):
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    
    classes = np.unique(y_true)
    classes_len = len(classes)
    
    if classes_len == 2:
        return binary_conf_mat(y_true, y_pred, classes)
    
    mat = np.zeros((classes_len, classes_len), dtype=int)
    matches = y_pred == y_true

    for i in range(classes_len):
        for j in range(classes_len):
            if i == j:
                mask = (y_pred == classes[j]) & matches
            else:
                mask = (y_true == classes[i]) & (y_pred == classes[j])

            if weight is not None:
                val = weight[mask].sum()
            else:
                val = mask.sum()

            mat[i, j] = val

    return mat


def confusion_matrix_from_predictions(y_true, y_pred, normalize=None, sample_weight=None, values_format=None):
    mat = confusion_matrix(y_true, y_pred, sample_weight)
    
    if normalize is not None:
        mat = mat.astype(float)
        match normalize:
            case 'true': mat /= mat.sum(axis=1).reshape(-1, 1)
            case 'pred': mat /= mat.sum(axis=0)
            case _:raise ValueError('Invalid normalize')

    if values_format is not None:
        mat = np.vectorize(lambda x: format(x, values_format))(mat)

    return mat


def precision_score(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    tp = conf_mat[1, 1]
    fp = conf_mat[0, 1]

    return tp / (tp + fp)

def recall_score(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    tp, fn = conf_mat[1, 1], conf_mat[1, 0]

    return tp / (tp + fn)



def f1_score(y_true, y_pred, average='binary'):
    if average == 'binary':
        if y_true.ndim != 1:
            raise ValueError("Target is multilabel-indicator but average='binary' is not supported for multilabel data")
        
        conf_mat = confusion_matrix(y_true, y_pred)
        tp, fn, fp = conf_mat[1, 1], conf_mat[1, 0], conf_mat[0, 1]
        return tp / (tp + (fn + fp) / 2)
    
    if average == 'macro':
        n_labels = y_true.shape[1]
        scores = np.empty(n_labels)

        for i in range(n_labels):
            curr_y_true = y_true[:, i]
            curr_y_pred = y_pred[:, i]

            conf_mat = confusion_matrix(curr_y_true, curr_y_pred)
            tp, fn, fp = conf_mat[1, 1], conf_mat[1, 0], conf_mat[0, 1]
            scores[i] = tp / (tp + (fn + fp) / 2)

        return scores.mean()
    
    if average == 'micro':
        n_labels = y_true.shape[1]
        tp, fn, fp = 0, 0, 0

        for i in range(n_labels):
            curr_y_true = y_true[:, i]
            curr_y_pred = y_pred[:, i]

            conf_mat = confusion_matrix(curr_y_true, curr_y_pred)
            tp += conf_mat[1, 1]
            fn += conf_mat[1, 0]
            fp += conf_mat[0, 1]

        return tp / (tp + (fn + fp) / 2)
    
    raise ValueError(f'Invalid average: {average}')


def precision_recall(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    tp, fn, fp = conf_mat[1, 1], conf_mat[1, 0], conf_mat[0, 1]
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    return prec, recall



get_y_pred = lambda y_score, threshold: y_score >= threshold
precision_calc = lambda tp, fp: tp / (tp + fp)
recall_calc = lambda tp, fn: tp / (tp + fn)
fpr_calc = lambda fp, tn: fp / (fp + tn)


def precision_recall_curve(y_true, y_score):
    res_thresholds = np.unique(y_score)
    thresholds_len = len(res_thresholds)

    precisions = np.empty(thresholds_len + 1, dtype=float)
    recalls =    np.empty(thresholds_len + 1, dtype=float)

    neg, pos = np.unique(y_true)
    precisions[0], recalls[0] = (y_true == pos).sum() / len(y_true), 1.

    thresholds = res_thresholds[::-1]
    y_pred = get_y_pred(y_score, thresholds[0])
    conf_mat = confusion_matrix(y_true, y_pred)
    tp, fn, fp = conf_mat[1, 1], conf_mat[1, 0], conf_mat[0, 1]
    precisions[1], recalls[1] = precision_calc(tp, fp), recall_calc(tp, fn)
    last_y_pred = y_pred

    for i, threshold in enumerate(thresholds[1:], start=2):
        y_pred = get_y_pred(y_score, threshold)
        diff_classes = y_true[last_y_pred != y_pred]

        correct_sum = (diff_classes == pos).sum()
        tp += correct_sum
        fn -= correct_sum
        fp += (diff_classes != pos).sum()

        precisions[i], recalls[i] = precision_calc(tp, fp), recall_calc(tp, fn)
        last_y_pred = y_pred
        
    return precisions[::-1], recalls[::-1], res_thresholds



def roc_curve(y_true, y_score):
    thresholds = np.concatenate([[np.inf], np.sort(np.unique(y_score))[::-1]])
    res_thresholds, fpr, recalls = [np.inf], [0], [0]
    neg, pos = np.sort(np.unique(y_true))

    y_pred = get_y_pred(y_score, thresholds[1])
    conf_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_mat.flatten()
    fpr.append(fpr_calc(fp, tn))
    recalls.append(recall_calc(tp, fn))
    res_thresholds.append(thresholds[1])
    last_y_pred = y_pred

    for threshold in thresholds[2:]:
        y_pred = get_y_pred(y_score, threshold)
        diff_classes = y_true[last_y_pred != y_pred]

        correct_sum = (diff_classes == pos).sum()
        tp += correct_sum
        fn -= correct_sum
        fp += (diff_classes != pos).sum()
        tn -= (diff_classes == neg).sum()

        fpr_val, rec_val = fpr_calc(fp, tn), recall_calc(tp, fn)

        if fpr_val != fpr[-1] or rec_val != recalls[-1]:
            fpr.append(fpr_val)
            recalls.append(rec_val)
            res_thresholds.append(threshold)

        last_y_pred = y_pred
        
    return np.array(fpr, float), np.array(recalls, float), np.array(res_thresholds, float)



def roc_auc_score(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    x = np.diff(fpr)
    y = (tpr[1:] + tpr[:-1]) / 2
    return np.sum(x * y)


from cross_val import cross_val_predict

def get_score(estimator, X, y, *, precision=None, recall=None, type='threshold'):
    if precision is not None and recall is not None:
        raise ValueError('get_score expects precision or recall, not both')

    if hasattr(estimator, 'decision_function'):
        method = 'decision_function'
    elif hasattr(estimator, 'predict_proba'):
        method = 'predict_proba'
    
    scores = cross_val_predict(estimator, X, y, cv=3, method=method)
    prec, rec, thresh = precision_recall_curve(y, scores)

    if precision is not None:
        inx = (prec >= precision).argmax()
    elif recall is not None:
        inx = (rec <= recall).argmax()
    else:
        raise ValueError('get_score expects precision or recall')
    
    match type:
        case 'threshold': val = thresh[inx]
        case 'precision': val = prec[inx]
        case 'recall':    val = rec[inx]
        case _: raise ValueError('type must be threshold, precision, recall')

    return val

