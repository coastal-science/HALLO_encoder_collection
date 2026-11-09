import numpy as np
import pandas as pd

from encoder_pipeline.embeddings.embed import PerchEmbeddingModel


def _model(loads):
    """A PerchEmbeddingModel without perch_hoplite: loading records each clip, embedding returns its begin time."""
    m = PerchEmbeddingModel.__new__(PerchEmbeddingModel)
    m.embedding_dim, m.batch_size, m.load_workers, m.chunk_size = 3, 4, 2, 2

    def load(path, begin, duration):
        loads.append((path, begin))
        return None if path == "bad.flac" else np.full(8, begin, dtype=np.float32)

    m._safe_load = load
    m._embed = lambda waves: np.stack([np.full(3, w[0]) for w in waves])
    return m


def test_repeated_extract_uses_cache_and_keeps_failures_dropped():
    loads = []
    m = _model(loads)
    clips = pd.DataFrame({"LocalPath": ["a.flac", "a.flac", "b.flac", "bad.flac", "a.flac"],
                          "FileBeginSec": [0.0, 5.0, 0.0, 0.0, 0.0], "Duration": 5.0, "label": [0, 1, 2, 3, 0]})
    e1, y1, keep1 = m.extract(clips)
    assert len(loads) == 4  # 4 unique clips, the repeated a.flac@0 loaded once
    e2, y2, keep2 = m.extract(clips.iloc[::-1].reset_index(drop=True))  # another fold: same clips, other order
    assert len(loads) == 4  # nothing reloaded
    np.testing.assert_array_equal(keep1, [True, True, True, False, True])
    np.testing.assert_array_equal(keep2, keep1[::-1])
    np.testing.assert_array_equal(e2, e1[::-1])
    np.testing.assert_array_equal(y1, [0, 1, 2, 0])
