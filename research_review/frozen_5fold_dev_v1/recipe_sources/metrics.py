"""Binary residue metrics used throughout training and evaluation.

``auc_pr`` is the trapezoidal area under the precision-recall curve, not
average precision. Precision, recall and F1 are macro averages over classes
0 and 1; sensitivity is positive-class recall.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _binary_inputs(labels, preds, *, discrete=False):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    # Accommodate the column vectors accepted by sklearn, but reject matrices.
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds[:, 0]
    if labels.ndim != 1 or preds.ndim != 1 or labels.size == 0:
        raise ValueError("labels and predictions must be nonempty vectors")
    if labels.shape != preds.shape:
        raise ValueError("labels and predictions must have the same shape")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only binary values 0 and 1")
    if not np.isfinite(preds).all():
        raise ValueError("predictions must be finite")
    if discrete and not np.isin(preds, [0, 1]).all():
        raise ValueError("class predictions must contain only 0 and 1")
    return labels, preds


def compute_acc(labels, preds):
    labels, preds = _binary_inputs(labels, preds, discrete=True)
    acc = accuracy_score(labels, preds)
    return acc


def compute_auc_roc(labels, preds):
    labels, preds = _binary_inputs(labels, preds)
    if np.unique(labels).size < 2:
        return float("nan")
    return roc_auc_score(labels, preds)


def compute_auc_pr(labels, preds):
    labels, preds = _binary_inputs(labels, preds)
    # Recall has no denominator when the sample contains no positive labels.
    # sklearn's synthetic endpoint otherwise gives a misleading area of 0.5.
    if not np.any(labels == 1):
        return float("nan")
    p, r, _ = precision_recall_curve(labels, preds)
    auc_pr = auc(r, p)
    return auc_pr


def compute_performance(labels, preds):
    """Return MCC, macro recall/precision/F1, sensitivity and specificity.

    Undefined per-class ratios use zero, including for an absent class.
    """
    labels, preds = _binary_inputs(labels, preds, discrete=True)
    confusion = confusion_matrix(labels, preds, labels=[0, 1])
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    mcc = matthews_corrcoef(labels, preds)
    recall = recall_score(labels, preds, labels=[0, 1], average='macro', zero_division=0)
    precision = precision_score(labels, preds, labels=[0, 1], average='macro', zero_division=0)
    f1 = f1_score(labels, preds, labels=[0, 1], average='macro', zero_division=0)
    sensitivity = TP / float(TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / float(TN + FP) if (TN + FP) > 0 else 0.0
    return mcc, recall, precision, f1, sensitivity, specificity
