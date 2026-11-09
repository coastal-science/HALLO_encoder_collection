import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize


def _indicator(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    """y_true as an (n_samples, n_classes) indicator matrix; multi-label y_true already is one."""
    if y_true.ndim == 2:
        return y_true
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    return np.hstack([1 - y_true_bin, y_true_bin]) if y_true_bin.shape[1] == 1 else y_true_bin


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Macro-averaged precision/recall/F1/PR-AUC over y_true's classes.
    y_pred is the hard class prediction; y_score is (n_samples, n_classes)
    predicted probabilities, needed for PR-AUC. Multi-label: y_true / y_pred are (n_samples, n_classes) 0/1."""
    y_true_bin = _indicator(y_true, y_score.shape[1])  # every class the model can output, not just ones in this split
    return {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "pr_auc": average_precision_score(y_true_bin, y_score, average="macro"),
    }


def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, class_names: list[str],
) -> dict[str, float]:
    """Per-class precision, recall, F1 and PR-AUC, keyed '<metric>_<class>'
    (precision/recall/f1/pr_auc). Args as classification_metrics; class_names
    indexes y_score's columns. Classes with no positives in y_true score 0."""
    classes = np.arange(y_score.shape[1])
    y_true_bin = _indicator(y_true, len(classes))
    per_metric = {
        "precision": precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0),
        "pr_auc": average_precision_score(y_true_bin, y_score, average=None),
    }
    return {
        f"{metric}_{name}": float(values[i])
        for i, name in enumerate(class_names)
        for metric, values in per_metric.items()
    }
