from typing import Optional

import numpy as np
import pandas as pd


def predictions_frame(
    split: str, uids: Optional[np.ndarray], y_true: np.ndarray, y_score: np.ndarray, class_names: list[str],
) -> pd.DataFrame:
    """One row per sample of a split: its uid (when known), true and predicted
    label, whether they match, and the score of every class the model outputs
    as prob_<class>. Multi-label (y_true (n_samples, n_classes)): labels are ';'-joined, predicted = prob > 0.5."""
    names = np.asarray(class_names)
    if y_true.ndim == 2:
        y_pred = (y_score > 0.5).astype(y_true.dtype)
        joined = lambda rows: [";".join(names[row > 0.5]) for row in rows]  # noqa: E731
        true_label, pred_label, correct = joined(y_true), joined(y_pred), (y_pred == y_true).all(axis=1)
    else:
        y_pred = y_score.argmax(axis=1)
        true_label, pred_label, correct = names[y_true], names[y_pred], y_pred == y_true
    frame = pd.DataFrame({"split": split, "true_label": true_label, "pred_label": pred_label, "correct": correct})
    if uids is not None:
        frame.insert(0, "uid", uids)
    for i in range(y_score.shape[1]):
        frame[f"prob_{names[i]}"] = y_score[:, i]
    return frame
