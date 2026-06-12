"""Normalize DCLDE 2027 annotation sources into source-table rows.

The root ``Annotations.csv`` inventory is the canonical source for normalized
species and ecotype labels. The provider Robert's Bank CSV is used only to
enrich those same events with task-3-specific fields such as pod, call type,
and confidence. The shared annotation object keeps both source row identifiers
so later stages can audit where every source-table row came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CanonicalAnnotation:
    """One source annotation row with normalized labels and traceability.

    Normalized label fields treat known missing-value spellings, such as
    ``NA``, as absent while leaving uncertain labels visible for reports.
    """

    provider: str
    dataset: str
    audio_path: str | None
    source_annotation_path: str
    source_annotation_row: str
    source_filename: str
    source_file_time: str | None
    annotation_start_sec: float | None
    annotation_end_sec: float | None
    freq_min_hz: float | None
    freq_max_hz: float | None
    species: str | None
    ecotype: str | None
    call_type: str | None
    confidence: str | None
    pod: str | None
    annotation_level: str | None
    provider_annotation_path: str | None = None
    provider_annotation_row: str | None = None


def read_root_annotations(dataset_root: Path, audio_index: dict[str, list[Path]]) -> list[CanonicalAnnotation]:
    """Normalize the corpus-wide ``Annotations.csv`` table into canonical rows.

    Audio paths are resolved against the local indexed files by basename because
    the source table stores Windows-style paths that are traceability metadata,
    not portable paths to open.
    """

    path = dataset_root / "Annotations.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows: list[CanonicalAnnotation] = []
    for zero_index, row in frame.iterrows():
        filename = row["Soundfile"]
        provider = row["Provider"]
        dataset = row["Dataset"]
        rows.append(
            CanonicalAnnotation(
                provider=provider,
                dataset=dataset,
                audio_path=_resolve_audio_from_filename(filename, audio_index),
                source_annotation_path="Annotations.csv",
                source_annotation_row=str(row.get("", zero_index + 1)),
                source_filename=filename,
                source_file_time=_blank_to_none(row.get("UTC")),
                annotation_start_sec=_float_or_none(row.get("FileBeginSec")),
                annotation_end_sec=_float_or_none(row.get("FileEndSec")),
                freq_min_hz=_float_or_none(row.get("LowFreqHz")),
                freq_max_hz=_float_or_none(row.get("HighFreqHz")),
                species=_normalize_blank(row.get("ClassSpecies")),
                ecotype=_normalize_blank(row.get("Ecotype")),
                call_type=None,
                confidence=None,
                pod=None,
                annotation_level=_blank_to_none(row.get("AnnotationLevel")),
                provider_annotation_path=None,
                provider_annotation_row=None,
            )
        )
    return rows


def read_roberts_bank_annotations(
    dataset_root: Path,
    audio_index: dict[str, list[Path]],
    *,
    provider: str,
    dataset: str,
    audio_dir: str,
    annotation_path: str,
) -> list[CanonicalAnnotation]:
    """Build canonical Robert's Bank rows by merging two source annotation views.

    ``Annotations.csv`` is the only source consulted for the canonical
    species/ecotype label fields. The provider Robert's Bank CSV is matched
    one-to-one to those same events and contributes only task-3-specific
    enrichment fields such as ``call_type``, ``pod``, and ``confidence``.
    """

    annotation_path_obj = Path(annotation_path)
    root_rows = [
        row
        for row in read_root_annotations(dataset_root, audio_index)
        if row.provider == provider and row.dataset == dataset
    ]
    root_rows_by_key = {_roberts_bank_match_key(row): row for row in root_rows}
    if len(root_rows_by_key) != len(root_rows):
        raise ValueError("Annotations.csv contains duplicate Robert's Bank event keys.")
    frame = pd.read_csv(dataset_root / annotation_path_obj, dtype=str, keep_default_na=False)
    rows: list[CanonicalAnnotation] = []
    matched_keys: set[tuple[str | None, float | None, float | None, float | None, float | None]] = set()
    for zero_index, row in frame.iterrows():
        filename = row["filename"]
        match_key = _roberts_bank_match_key(
            audio_path=_resolve_roberts_bank_audio(filename, audio_index, audio_dir),
            start_sec=_float_or_none(row.get("start")),
            end_sec=_float_or_none(row.get("end")),
            freq_min_hz=_float_or_none(row.get("freq_min")),
            freq_max_hz=_float_or_none(row.get("freq_max")),
        )
        if match_key in matched_keys:
            raise ValueError(f"Robert's Bank provider row {zero_index + 1} duplicates an earlier task-3 event key.")
        root_row = root_rows_by_key.get(match_key)
        if root_row is None:
            raise ValueError(
                f"Robert's Bank provider row {zero_index + 1} did not match any canonical Annotations.csv row."
            )
        matched_keys.add(match_key)
        rows.append(
            CanonicalAnnotation(
                provider=root_row.provider,
                dataset=root_row.dataset,
                audio_path=root_row.audio_path,
                source_annotation_path=root_row.source_annotation_path,
                source_annotation_row=root_row.source_annotation_row,
                source_filename=root_row.source_filename,
                source_file_time=root_row.source_file_time,
                annotation_start_sec=root_row.annotation_start_sec,
                annotation_end_sec=root_row.annotation_end_sec,
                freq_min_hz=root_row.freq_min_hz,
                freq_max_hz=root_row.freq_max_hz,
                species=root_row.species,
                ecotype=root_row.ecotype,
                call_type=_normalize_blank(row.get("call_type")),
                confidence=_normalize_blank(row.get("confidence")),
                pod=_normalize_blank(row.get("pod")),
                annotation_level=root_row.annotation_level,
                provider_annotation_path=annotation_path,
                provider_annotation_row=str(zero_index + 1),
            )
        )
    if matched_keys != set(root_rows_by_key):
        raise ValueError("Annotations.csv contains Robert's Bank rows that did not match the provider task-3 file.")
    return rows


def is_task3_strict(annotation: CanonicalAnnotation) -> tuple[bool, str | None]:
    """Return whether a Robert's Bank row belongs in the strict subset.

    The filter intentionally accepts only confident SRKW KW rows with one named
    catalogue-style call type. Ambiguous ecotypes, generic call labels, compound
    labels, low-confidence rows, and non-KW annotations stay in the full task-3
    source table but are excluded from the strict subset.
    """

    if annotation.species != "KW":
        return False, "task3_strict_non_kw"
    if annotation.ecotype != "SRKW":
        return False, "task3_strict_not_srkw"
    if annotation.confidence not in {"High", "Medium", None}:
        return False, "task3_strict_confidence"
    call_type = annotation.call_type
    if call_type is None:
        return False, "task3_strict_missing_call_type"
    if not _is_discrete_catalogue_call(call_type):
        return False, "task3_strict_non_catalogue_call_type"
    return True, None


def annotation_to_dict(annotation: CanonicalAnnotation) -> dict[str, Any]:
    """Convert one canonical annotation object into source-table row fields."""

    return {
        "provider": annotation.provider,
        "dataset": annotation.dataset,
        "audio_path": annotation.audio_path,
        "source_annotation_path": annotation.source_annotation_path,
        "source_annotation_row": annotation.source_annotation_row,
        "source_filename": annotation.source_filename,
        "source_file_time": annotation.source_file_time,
        "annotation_start_sec": annotation.annotation_start_sec,
        "annotation_end_sec": annotation.annotation_end_sec,
        "freq_min_hz": annotation.freq_min_hz,
        "freq_max_hz": annotation.freq_max_hz,
        "species": annotation.species,
        "ecotype": annotation.ecotype,
        "call_type": annotation.call_type,
        "confidence": annotation.confidence,
        "pod": annotation.pod,
        "annotation_level": annotation.annotation_level,
        "provider_annotation_path": annotation.provider_annotation_path,
        "provider_annotation_row": annotation.provider_annotation_row,
    }


def _resolve_audio_from_filename(filename: str, audio_index: dict[str, list[Path]]) -> str | None:
    """Resolve a source filename to the first matching indexed audio path."""

    matches = audio_index.get(Path(filename).name)
    if not matches:
        matches = audio_index.get(PureWindowsPath(filename).name)
    if not matches:
        return None
    return matches[0].as_posix()


def _resolve_roberts_bank_audio(filename: str, audio_index: dict[str, list[Path]], audio_dir: str) -> str | None:
    """Resolve a filename only if the match lives under the Robert's Bank audio tree."""

    matches = audio_index.get(Path(filename).name, [])
    for match in matches:
        if match.as_posix().startswith(f"{audio_dir}/"):
            return match.as_posix()
    return None


def _blank_to_none(value: object) -> str | None:
    """Normalize blank-like CSV cells to ``None`` for downstream logic."""

    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "NAN"}:
        return None
    return text


def _normalize_blank(value: object) -> str | None:
    """Keep label spelling intact while still treating blank-like cells as missing."""

    text = _blank_to_none(value)
    if text is None:
        return None
    return text.strip()


def _float_or_none(value: object) -> float | None:
    """Parse one optional numeric field from CSV text into a float."""

    text = _blank_to_none(value)
    if text is None:
        return None
    return float(text)


def _roberts_bank_match_key(
    annotation: CanonicalAnnotation | None = None,
    *,
    audio_path: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
) -> tuple[str | None, float | None, float | None, float | None, float | None]:
    """Return the event identity used to match root and provider Robert's Bank rows."""

    if annotation is not None:
        audio_path = annotation.audio_path
        start_sec = annotation.annotation_start_sec
        end_sec = annotation.annotation_end_sec
        freq_min_hz = annotation.freq_min_hz
        freq_max_hz = annotation.freq_max_hz
    return (
        audio_path,
        start_sec,
        end_sec,
        freq_min_hz,
        freq_max_hz,
    )


def _is_discrete_catalogue_call(call_type: str) -> bool:
    """Identify one named catalogue-style call type such as ``S02iii``."""

    blocked_words = {"Unk", "unknown", "whistle", "buzz", "echolocation", "click", "variable"}
    lowered = call_type.lower()
    if any(word.lower() in lowered for word in blocked_words):
        return False
    if any(separator in call_type for separator in ["/", "&", ",", ";", "+"]):
        return False
    return bool(re.fullmatch(r"[ST]\d{2}(?:[a-z]+|[A-Za-z])?", call_type))
