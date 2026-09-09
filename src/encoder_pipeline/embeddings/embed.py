from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import librosa
import mlflow
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from loguru import logger
from torch.utils.data import DataLoader

from encoder_pipeline.preprocessor.config import SpectrogramConfig



class EmbeddingModel(ABC):
    """Shared interface for embeddings - implementations will perform forward-only pass over a
    DataLoader, returning stacked (embeddings, labels)."""

    @abstractmethod
    def extract(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Runs forward-only over every batch in loader, returning
        (embeddings, labels) stacked across the whole loader."""

class HALLOEmbeddingModel(EmbeddingModel):
    """Embeddings from a checkpoint written by
    encoder_pipeline.model_trainer.train.Trainer.fit"""

    def __init__(self, checkpoint_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = bundle["model"].to(self.device)
        self.spectrogram_config: SpectrogramConfig = bundle["spectrogram_config"]

    def extract(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        embeddings, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for specs, batch_labels in loader:
                specs = specs.to(self.device).unsqueeze(1)
                embeddings.append(self.model.backbone(specs).cpu().numpy())
                labels.append(batch_labels.numpy())
        return np.concatenate(embeddings), np.concatenate(labels)


class PerchEmbeddingModel(EmbeddingModel):
    """Embeddings from a perch_hoplite zoo model (e.g. perch_v2). Ignores the
    spectrogram pipeline entirely: re-reads raw audio at the model's native
    sample rate / window and runs the model's own batched embedder. extract()
    takes a clip-metadata frame, not a spectrogram DataLoader."""

    def __init__(self, preset: str = "perch_v2", batch_size: int = 64, load_workers: int = 8) -> None:
        try:
            from perch_hoplite.zoo import model_configs
        except ImportError as e:
            raise ImportError(
                "perch_hoplite is not installed -- `pip install 'hallo-encoder-collection[perch]'`"
            ) from e
        preset_info = model_configs.get_preset_model_config(preset)
        self.model = preset_info.load_model()
        self.sample_rate = int(preset_info.model_config.sample_rate)
        self.window_s = float(preset_info.model_config.window_size_s)
        self.embedding_dim = int(preset_info.embedding_dim)
        self.batch_size = batch_size
        self.load_workers = load_workers

    def _load_clip(self, path: str, begin_sec: float, duration: float) -> np.ndarray:
        """A window_s-long mono clip at self.sample_rate, centered on the
        annotation and clamped to the file, matching the reference pipeline's
        soundfile+librosa path."""
        info = sf.info(path)
        src_sr, file_dur = info.samplerate, info.frames / info.samplerate
        center = begin_sec + duration / 2
        t0 = max(0.0, min(center - self.window_s / 2, max(0.0, file_dur - self.window_s)))
        audio, _ = sf.read(path, start=int(t0 * src_sr), stop=int((t0 + self.window_s) * src_sr))
        y = (audio[:, 0] if audio.ndim > 1 else audio).astype(np.float32)
        y = librosa.resample(y, orig_sr=src_sr, target_sr=self.sample_rate)
        n = int(self.window_s * self.sample_rate)
        return np.pad(y, (0, max(0, n - len(y))))[:n]

    def extract(self, clips: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """clips: rows with 'LocalPath', 'FileBeginSec', 'Duration', 'label'.
        Returns (embeddings, labels) for the clips that loaded, in row order."""
        rows = list(clips.itertuples(index=False))
        with ThreadPoolExecutor(max_workers=self.load_workers) as pool:
            loaded = list(pool.map(
                lambda r: self._safe_load(r.LocalPath, r.FileBeginSec, r.Duration), rows,
            ))
        waveforms = [w for w in loaded if w is not None]
        labels = [r.label for r, w in zip(rows, loaded) if w is not None]
        n_failed = len(rows) - len(waveforms)
        if n_failed:
            logger.warning("perch: dropped {} of {} clips that failed to load", n_failed, len(rows))
        if not waveforms:
            return np.empty((0, self.embedding_dim), dtype=np.float32), np.empty(0, dtype=np.int64)

        embeddings = []
        for start in range(0, len(waveforms), self.batch_size):
            batch = np.stack(waveforms[start:start + self.batch_size])
            out = self.model.batch_embed(batch)
            embeddings.append(np.asarray(out.pooled_embeddings("mean", "mean")))
        return np.concatenate(embeddings), np.asarray(labels, dtype=np.int64)

    def _safe_load(self, path: str, begin_sec: float, duration: float):
        try:
            return self._load_clip(path, begin_sec, duration)
        except Exception as e:  # soundfile/librosa raise a range of errors on bad files
            logger.error("perch: failed to load {} @ {}s: {}", path, begin_sec, e)
            return None
