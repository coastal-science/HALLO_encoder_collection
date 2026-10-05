import numpy as np

from encoder_pipeline.evaluation.metrics import per_class_metrics


def test_per_class_metrics_keys_and_values_are_per_class():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    y_score = np.eye(3)[y_pred] * 0.7 + 0.1

    out = per_class_metrics(y_true, y_pred, y_score, ["hw", "kw", "noise"])

    assert set(out) == {
        f"{metric}_{name}" for metric in ("precision", "recall", "f1", "pr_auc") for name in ("hw", "kw", "noise")
    }
    assert out["f1_kw"] == 0.8
    assert out["precision_kw"] == 2 / 3  # predicted kw 3x, 2 correct
    assert out["recall_kw"] == 1.0
    assert out["recall_hw"] == 0.5


def test_per_class_metrics_scores_zero_for_class_absent_from_y_true():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 2])
    y_score = np.eye(3)[y_pred] * 0.7 + 0.1

    out = per_class_metrics(y_true, y_pred, y_score, ["a", "b", "c"])

    assert out["pr_auc_c"] == 0.0
    assert out["f1_c"] == 0.0
    assert out["precision_c"] == 0.0
    assert out["recall_c"] == 0.0
