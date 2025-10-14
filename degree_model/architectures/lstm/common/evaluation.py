import numpy as np
from sklearn.metrics import confusion_matrix

def adjusted_precision_by_prevalence(y_true, y_pred, prevalences=None, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    supports = cm.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        cond = np.nan_to_num(cm / supports[:, None])

    if prevalences is None:
        total = supports.sum()
        p = supports / total
    elif prevalences == 'uniform':
        p = np.ones_like(supports) / len(supports)
    elif isinstance(prevalences, dict):
        p = np.array([prevalences.get(l, 0.0) for l in labels], dtype=float)
        p = p / p.sum()
    else:
        raise ValueError("prevalences must be None, 'uniform', or dict")

    adjusted_precisions = {}
    for i, lbl in enumerate(labels):
        TP = p[i] * cond[i, i]
        FP = np.sum([p[j] * cond[j, i] for j in range(len(labels)) if j != i])
        denom = TP + FP
        if denom == 0:
            prec = 0.0
        else:
            prec = TP / denom
        adjusted_precisions[int(lbl)] = prec

    return adjusted_precisions, labels, cm, supports
