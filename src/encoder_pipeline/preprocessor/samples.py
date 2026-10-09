from pathlib import Path

import h5py
import librosa
import numpy as np
from matplotlib.figure import Figure

from encoder_pipeline.preprocessor.config import PreprocessorConfig


def _freq_ticks(config: PreprocessorConfig, n_bins: int, n_ticks: int = 6) -> tuple[np.ndarray, list[str], str]:
    """Row positions, labels and axis title for the frequency axis. In Hz when
    the config pins it down, otherwise the row index."""
    spec_config, sr = config.spectrogram, config.audio_file.resample_sr
    rows = np.linspace(0, n_bins - 1, n_ticks).round().astype(int)
    if spec_config.freq_scale == "mel" and (spec_config.mel.fmax or sr):
        hz = librosa.mel_frequencies(
            n_mels=n_bins, fmin=spec_config.mel.fmin, fmax=spec_config.mel.fmax or sr / 2, htk=spec_config.mel.htk,
        )[rows]
    elif spec_config.freq_scale == "mag" and sr:
        hz = rows * sr / spec_config.stft.n_fft
    else:
        return rows, [str(r) for r in rows], "frequency bin"
    return rows, [f"{f:.0f}" for f in hz], "frequency (Hz)"


def save_sample_images(
    dataset_path: str, config: PreprocessorConfig, out_dir: Path, n_per_class: int, seed: int = 0, n_cols: int = 4,
) -> list[Path]:
    """One <class>.png per class in the dataset's 'Labels', each a grid of
    n_per_class randomly chosen spectrograms titled with their uid. Returns the
    written paths; empty if the dataset has no 'spec' or 'Labels' (e.g. metadata_only)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    window = 2 * config.annotation.time_offset
    paths = []
    with h5py.File(dataset_path, "r") as h5:
        if "spec" not in h5 or "Labels" not in h5:
            return paths
        labels = h5["Labels"].asstr()[:]
        uid_col = config.dataset.uid_col
        uids = h5[uid_col] if uid_col in h5 else None
        rows, tick_labels, freq_title = _freq_ticks(config, h5["spec"].shape[1])
        for label in sorted(set(labels)):
            candidates = np.flatnonzero(labels == label)
            # h5py needs increasing indices
            picked = np.sort(rng.choice(candidates, size=min(n_per_class, len(candidates)), replace=False))
            specs = h5["spec"][picked]
            n_rows = -(-len(picked) // n_cols)
            fig = Figure(figsize=(4 * n_cols, 3 * n_rows), layout="constrained")
            axes = fig.subplots(n_rows, n_cols, squeeze=False)
            for ax in axes.flat[len(picked):]:
                ax.set_axis_off()
            for ax, idx, spec in zip(axes.flat, picked, specs):
                ax.imshow(spec, origin="lower", aspect="auto", cmap="magma", extent=(0, window, -0.5, spec.shape[0] - 0.5))
                ax.set_yticks(rows, tick_labels)
                ax.set_xlabel("time (s)")
                ax.set_ylabel(freq_title)
                uid = idx if uids is None else uids[idx].decode() if isinstance(uids[idx], bytes) else uids[idx]
                ax.set_title(f"{uid_col}={uid}", fontsize=9)
            fig.suptitle(f"{label} ({len(candidates):,} rows)")
            path = out_dir / f"{label}.png"
            fig.savefig(path, dpi=100)
            paths.append(path)
    return paths
