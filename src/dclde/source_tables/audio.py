"""Audio indexing and metadata helpers for DCLDE source-table construction.

Needs file duration and sample rate to place candidate clips,
but it should not read full waveforms. These helpers keep that boundary clear:
index paths once, then inspect individual files with ``soundfile.info``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


@dataclass(frozen=True)
class AudioInfo:
    """Metadata needed to place clips without reading full audio waveforms."""

    relative_path: str
    sample_rate_hz: int
    duration_sec: float
    frames: int


def build_audio_index(dataset_root: Path, audio_extensions: tuple[str, ...]) -> dict[str, list[Path]]:
    """Build a lookup from audio basename to dataset-relative source files.

    Many source annotations name only a file or contain non-portable Windows
    paths. Keeping all matches by basename lets provider-specific adapters apply
    stricter directory rules when identity matters, such as Robert's Bank.
    """

    normalized_exts = {ext.lower().lstrip(".") for ext in audio_extensions}
    index: dict[str, list[Path]] = {}
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower().lstrip(".") not in normalized_exts:
            continue
        relative = path.relative_to(dataset_root)
        index.setdefault(path.name, []).append(relative)
    return index


def inspect_audio(dataset_root: Path, relative_path: str) -> AudioInfo:
    """Read lightweight metadata for one source audio file.
    """

    info = sf.info(dataset_root / relative_path)
    return AudioInfo(
        relative_path=relative_path,
        sample_rate_hz=int(info.samplerate),
        duration_sec=float(info.frames) / float(info.samplerate),
        frames=int(info.frames),
    )
