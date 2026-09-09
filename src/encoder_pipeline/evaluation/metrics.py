import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Macro-averaged precision/recall/F1/PR-AUC over y_true's classes.
    y_pred is the hard class prediction; y_score is (n_samples, n_classes)
    predicted probabilities, needed for PR-AUC."""
    classes = np.arange(y_score.shape[1])  # every class the model can output, not just ones present in this split
    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    return {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "pr_auc": average_precision_score(y_true_bin, y_score, average="macro"),
    }


def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, class_names: list[str],
) -> dict[str, float]:
    """Per-class F1 and PR-AUC, keyed 'f1_<class>' / 'pr_auc_<class>'. Args as
    classification_metrics; class_names indexes y_score's columns. Classes with
    no positives in y_true score 0."""
    classes = np.arange(y_score.shape[1])
    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    f1 = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    pr_auc = average_precision_score(y_true_bin, y_score, average=None)
    return {
        f"{metric}_{name}": float(value)
        for name, f1_c, pr_auc_c in zip(class_names, f1, pr_auc)
        for metric, value in (("f1", f1_c), ("pr_auc", pr_auc_c))
    }
