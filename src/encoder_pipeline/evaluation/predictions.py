from typing import Optional

import numpy as np
import pandas as pd


def predictions_frame(
    split: str, uids: Optional[np.ndarray], y_true: np.ndarray, y_score: np.ndarray, class_names: list[str],
) -> pd.DataFrame:
    """One row per sample of a split: its uid (when known), true and predicted
    label, whether they match, and the score of every class the model outputs
    as prob_<class>."""
    names = np.asarray(class_names)
    y_pred = y_score.argmax(axis=1)
    frame = pd.DataFrame({
        "split": split, "true_label": names[y_true], "pred_label": names[y_pred], "correct": y_pred == y_true,
    })
    if uids is not None:
        frame.insert(0, "uid", uids)
    for i in range(y_score.shape[1]):
        frame[f"prob_{names[i]}"] = y_score[:, i]
    return frame
