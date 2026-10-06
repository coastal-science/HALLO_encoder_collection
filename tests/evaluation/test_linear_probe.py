import numpy as np

from encoder_pipeline.evaluation.linear_probe import LinearProbe, remap_labels
from encoder_pipeline.evaluation.predictions import predictions_frame


def test_remap_labels_collapses_and_reindexes_to_new_sorted_classes():
    embeddings = {
        "train": (np.zeros((4, 2), np.float32), np.array([0, 1, 2, 3])),  # hw, nrkw, srkw, tkw
    }
    out, classes = remap_labels(embeddings, ["hw", "nrkw", "srkw", "tkw"], {"nrkw": "KW", "srkw": "KW", "tkw": "KW"})

    assert classes == ["KW", "hw"]  # sorted(set) of {hw, KW}
    np.testing.assert_array_equal(out["train"][1], [1, 0, 0, 0])


def test_remap_labels_passes_unmapped_names_through():
    embeddings = {"val": (np.zeros((3, 2), np.float32), np.array([0, 1, 2]))}
    out, classes = remap_labels(embeddings, ["a", "b", "c"], {"b": "a"})

    assert classes == ["a", "c"]
    np.testing.assert_array_equal(out["val"][1], [0, 0, 1])


def test_evaluate_returns_per_sample_scores_for_every_split():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 10)
    x = (rng.normal(size=(20, 4)) + 5 * y[:, None]).astype(np.float32)  # linearly separable
    embeddings = {"train": (x, y), "test": (x[:6], y[:6])}

    metrics, _, scores = LinearProbe(epochs=200, lr=1e-1).evaluate(embeddings, ["a", "b"])

    assert scores["train"].shape == (20, 2) and scores["test"].shape == (6, 2)
    assert float((scores["test"].argmax(axis=1) == y[:6]).mean()) == metrics["test_accuracy"]


def test_predictions_frame_has_one_labelled_row_per_sample():
    y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    frame = predictions_frame("test", np.array([7, 8, 9]), np.array([0, 1, 1]), y_score, ["a", "b"])

    assert list(frame.columns) == ["uid", "split", "true_label", "pred_label", "correct", "prob_a", "prob_b"]
    assert list(frame["uid"]) == [7, 8, 9]
    assert list(frame["pred_label"]) == ["a", "b", "a"]
    assert list(frame["correct"]) == [True, True, False]
    assert "uid" not in predictions_frame("test", None, np.array([0, 1, 1]), y_score, ["a", "b"])
