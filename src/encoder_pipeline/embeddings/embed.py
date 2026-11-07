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

    def __init__(
        self, preset: str = "perch_v2", batch_size: int = 64, load_workers: int = 8, chunk_size: int = 4096
    ) -> None:
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
        self.chunk_size = chunk_size

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

    def extract(self, clips: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """clips: rows with 'LocalPath', 'FileBeginSec', 'Duration', 'label'.
        Returns (embeddings, labels) for the clips that loaded, in row order,
        plus the boolean mask over clips' rows of which ones those are.
        Repeated clips (e.g. from oversampling) are loaded and embedded once.
        Audio is loaded and embedded chunk_size unique clips at a time."""
        codes, uniques = pd.factorize(pd.MultiIndex.from_frame(clips[["LocalPath", "FileBeginSec", "Duration"]]))
        embeddings = np.empty((len(uniques), self.embedding_dim), dtype=np.float32)
        ok = np.zeros(len(uniques), dtype=bool)
        with ThreadPoolExecutor(max_workers=self.load_workers) as pool:
            for start in range(0, len(uniques), self.chunk_size):
                stop = min(start + self.chunk_size, len(uniques))
                loaded = list(pool.map(lambda key: self._safe_load(*key), uniques[start:stop]))
                chunk_ok = np.array([w is not None for w in loaded])
                ok[start:stop] = chunk_ok
                if chunk_ok.any():
                    embeddings[start:stop][chunk_ok] = self._embed([w for w in loaded if w is not None])
                logger.info("perch: embedded {}/{} unique clips", stop, len(uniques))
        keep = ok[codes]
        n_failed = int((~keep).sum())
        if n_failed:
            logger.warning("perch: dropped {} of {} clips that failed to load", n_failed, len(clips))
        return embeddings[codes[keep]], clips["label"].to_numpy()[keep].astype(np.int64), keep

    def _embed(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Mean-pooled embeddings for equal-length waveforms, batch_size at a time. A short last batch is
        zero-padded to batch_size (padding rows dropped): every new batch shape is another XLA compile +
        autotune on GPU, and those piled up until TF ran out of GPU memory."""
        embeddings = []
        for start in range(0, len(waveforms), self.batch_size):
            batch = np.stack(waveforms[start:start + self.batch_size])
            n = len(batch)
            batch = np.pad(batch, ((0, self.batch_size - n), (0, 0)))
            out = self.model.batch_embed(batch)
            embeddings.append(np.asarray(out.pooled_embeddings("mean", "mean"))[:n])
        return np.concatenate(embeddings)

    def _safe_load(self, path: str, begin_sec: float, duration: float):
        try:
            return self._load_clip(path, begin_sec, duration)
        except Exception as e:  # soundfile/librosa raise a range of errors on bad files
            logger.error("perch: failed to load {} @ {}s: {}", path, begin_sec, e)
            return None


class WhisperEmbeddingModel(PerchEmbeddingModel):
    """Embeddings from a frozen OpenAI Whisper encoder (Hugging Face transformers). Same raw-audio path as
    PerchEmbeddingModel (window_s clips centered on each annotation, resampled, repeated clips embedded once), at
    Whisper's 16 kHz, so content above 8 kHz never reaches the model.

    Whisper's encoder is built for 30 s (3000 frames -> 1500 positions) and transformers refuses any other length;
    a window_s clip gives window_s * 100 frames, so the sinusoidal-initialised (learned) positional table is cut to
    its first window_s * 50 rows -- the usual way of running Whisper on short clips. Features come from Whisper's
    own feature extractor (log10 mel, clamped to max - 8, (x + 4) / 4), padded / cut to exactly window_s."""

    def __init__(
        self, model_name: str = "openai/whisper-base", window_s: float = 5.0, layer: int | None = None,
        peak_normalize: float | None = 0.25, batch_size: int = 64, load_workers: int = 8, chunk_size: int = 4096,
        amp: bool = True,
    ) -> None:
        try:
            from transformers import WhisperFeatureExtractor, WhisperModel
        except ImportError as e:
            raise ImportError("transformers is not installed -- `pip install 'hallo-encoder-collection[whisper]'`") from e
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.sample_rate = int(self.feature_extractor.sampling_rate)
        self.window_s = window_s
        self.n_samples = int(round(window_s * self.sample_rate))
        n_frames = self.n_samples // self.feature_extractor.hop_length
        encoder = WhisperModel.from_pretrained(model_name).get_encoder()
        n_positions = n_frames // (encoder.conv1.stride[0] * encoder.conv2.stride[0])
        if not 0 < n_positions <= encoder.config.max_source_positions or n_positions * 2 != n_frames:
            raise ValueError(f"window_s {window_s} gives {n_frames} frames; need an even count up to 3000 (<= 30 s)")
        positions = encoder.embed_positions.weight.data[:n_positions].clone()
        encoder.embed_positions = torch.nn.Embedding(n_positions, positions.shape[1])
        encoder.embed_positions.weight.data.copy_(positions)
        encoder.config.max_source_positions = n_positions
        self.n_frames = n_frames
        self.encoder = encoder.to(self.device).eval()
        n_states = encoder.config.encoder_layers + 1
        if layer is not None and not -n_states <= layer < n_states:
            raise ValueError(f"layer {layer} out of range for {model_name}'s {n_states} hidden states")
        self.layer = layer
        self.peak_normalize = peak_normalize
        self.amp = amp and self.device.type == "cuda"
        self.embedding_dim = int(encoder.config.d_model)
        self.batch_size = batch_size
        self.load_workers = load_workers
        self.chunk_size = chunk_size
        logger.info("whisper: {} encoder, {} s -> {} frames -> {} positions, layer {}, {}-d",
                    model_name, window_s, n_frames, n_positions, "final" if layer is None else layer,
                    self.embedding_dim)

    def _features(self, batch: np.ndarray) -> torch.Tensor:
        if self.peak_normalize is not None:
            batch = batch - batch.mean(axis=1, keepdims=True)
            peak = np.abs(batch).max(axis=1, keepdims=True)
            batch = np.where(peak > 0, batch * (self.peak_normalize / np.maximum(peak, 1e-12)), batch)
        features = self.feature_extractor(
            list(batch), sampling_rate=self.sample_rate, return_tensors="pt", padding="max_length",
            max_length=self.n_samples, truncation=True, device=self.device.type,
        ).input_features
        return features[..., : self.n_frames].to(self.device)

    @torch.no_grad()
    def _embed(self, waveforms: list[np.ndarray]) -> np.ndarray:
        embeddings = []
        for start in range(0, len(waveforms), self.batch_size):
            features = self._features(np.stack(waveforms[start:start + self.batch_size]))
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                out = self.encoder(features, output_hidden_states=self.layer is not None)
            hidden = out.last_hidden_state if self.layer is None else out.hidden_states[self.layer]
            embeddings.append(hidden.float().mean(dim=1).cpu().numpy())
        return np.concatenate(embeddings)
