import numpy as np

from encoder_pipeline.evaluation.metrics import per_class_metrics


def test_per_class_metrics_keys_and_values_are_per_class():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    y_score = np.eye(3)[y_pred] * 0.7 + 0.1

    out = per_class_metrics(y_true, y_pred, y_score, ["hw", "kw", "noise"])

    assert set(out) == {"f1_hw", "pr_auc_hw", "f1_kw", "pr_auc_kw", "f1_noise", "pr_auc_noise"}
    assert out["f1_kw"] == 0.8


def test_per_class_metrics_scores_zero_for_class_absent_from_y_true():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 2])
    y_score = np.eye(3)[y_pred] * 0.7 + 0.1

    out = per_class_metrics(y_true, y_pred, y_score, ["a", "b", "c"])

    assert out["pr_auc_c"] == 0.0
    assert out["f1_c"] == 0.0
