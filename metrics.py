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
    roc_curve,
)


def compute_acc(labels, preds):
    acc = accuracy_score(labels, preds)
    return acc


def compute_auc_roc(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    auc_roc = auc(fpr, tpr)
    return auc_roc


def compute_auc_pr(labels, preds):
    p, r, _ = precision_recall_curve(labels, preds)
    auc_pr = auc(r, p)
    return auc_pr


def compute_performance(labels, preds):
    confusion = confusion_matrix(labels, preds, labels=[0, 1])
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    mcc = matthews_corrcoef(labels, preds)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    sensitivity = TP / float(TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / float(TN + FP) if (TN + FP) > 0 else 0.0
    return mcc, recall, precision, f1, sensitivity, specificity
