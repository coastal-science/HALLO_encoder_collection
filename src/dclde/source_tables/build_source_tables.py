"""Build source-level DCLDE 2027 audio and annotation tables.

The builder answers inventory questions: what audio exists, what
annotations exist, and where each row came from. Robert's Bank task-3 rows use
canonical labels from the root inventory plus provider pod/call metadata. The
builder reads metadata with ``soundfile.info`` but does not load waveforms,
expand durations, create shifted views, sample background windows, materialize
HDF5, compute spectrograms, or run model workflows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from config import load_yaml_config
from dclde.source_tables.annotations import (
    annotation_to_dict,
    is_task3_strict,
    read_roberts_bank_annotations,
    read_root_annotations,
)
from dclde.source_tables.audio import build_audio_index, inspect_audio

BUILDER_VERSION = "0.1.0"
DEFAULT_AUDIO_EXTENSIONS = ("wav", "flac")

FOLDER_PROVIDER_MAP = {
    "dfo_crp": "DFO_CRP",
    "dfo_wdlp": "DFO_WDLP",
    "onc": "ONC",
    "orcasound": "OrcaSound",
    "scripps": "SIO",
    "simres": "SIMRES",
    "smru": "SMRUConsulting",
    "uaf": "UAF_NGOS",
    "vfpa": "JASCO_VFPA",
}


# ======== Config Models ========

class HoldoutRuleConfig(BaseModel):
    """Explicit source identity for a training holdout."""

    provider: str
    dataset: str
    audio_dir: str
    annotation_path: str


class SourceTableBuildConfig(BaseModel):
    """Validated YAML configuration for DCLDE 2027 source tables."""

    dataset_root: Path
    output_root: Path
    exclude_holdout_rules: list[HoldoutRuleConfig]


# ======== Public Builder API ========

def build_source_tables(config: SourceTableBuildConfig) -> dict[str, Any]:
    """Build the source-facing DCLDE tables used by later dataset stages.

    The output stays close to the original corpus structure: one global audio
    inventory, one non-holdout training-annotation table, and two Robert's Bank
    task-3 tables that separate the full holdout rows from the strict subset.
    """

    task3_holdout = _task3_holdout_rule(config)
    audio_index = build_audio_index(config.dataset_root, DEFAULT_AUDIO_EXTENSIONS)
    root_annotations = read_root_annotations(config.dataset_root, audio_index)
    task3_annotations = read_roberts_bank_annotations(
        config.dataset_root,
        audio_index,
        provider=task3_holdout.provider,
        dataset=task3_holdout.dataset,
        audio_dir=task3_holdout.audio_dir,
        annotation_path=task3_holdout.annotation_path,
    )
    rejects: list[dict[str, Any]] = []

    training_annotations = [row for row in root_annotations if not is_roberts_bank_holdout(row, task3_holdout)]
    audio_records = _build_audio_inventory_rows(source_audio_inventory_records(audio_index, root_annotations), rejects, config)
    readable_audio_paths = {row["audio_path"] for row in audio_records}
    annotation_rows = _build_training_annotations_rows(training_annotations, rejects)
    task3_rows, task3_strict_rows = _build_task3_source_rows(task3_annotations, readable_audio_paths, rejects)

    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    _write_source_table(pd.DataFrame(audio_records), output_root / "audio_inventory")
    _write_source_table(pd.DataFrame(annotation_rows), output_root / "training_annotations")
    _write_source_table(pd.DataFrame(task3_rows), output_root / "task3_roberts_bank")
    _write_source_table(pd.DataFrame(task3_strict_rows), output_root / "task3_roberts_bank_strict")
    _write_rejects(rejects, output_root / "rejects.csv")

    metadata = _metadata(
        config,
        audio_index,
        audio_records,
        annotation_rows,
        task3_rows,
        task3_strict_rows,
        rejects,
    )
    with (output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata


def main() -> None:
    """Run the source-table builder as a small CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config, SourceTableBuildConfig)
    metadata = build_source_tables(config)
    print(json.dumps(metadata["summary_counts"], indent=2, sort_keys=True))


# ======== Source Row Collection ========

def source_audio_inventory_records(audio_index, annotations) -> list[dict[str, Any]]:
    """Assemble the row candidates for the global audio inventory.

    Annotated files reuse provider and dataset metadata already recovered from
    source annotations. Unannotated files fall back to path-derived metadata so
    the inventory still represents the full audio catalog.
    """

    annotation_records = _audio_records_from_annotations(annotations)
    records: dict[str, dict[str, Any]] = {}
    for paths in audio_index.values():
        for path in paths:
            relative_path = path.as_posix()
            records[relative_path] = annotation_records.get(relative_path, _infer_audio_record(path))
    return sorted(records.values(), key=lambda row: row["audio_path"])


def _build_audio_inventory_rows(source_audio_records, rejects, config) -> list[dict[str, Any]]:
    """Finalize the audio inventory by keeping only readable source files.

    Each candidate row is enriched with lightweight audio metadata from
    ``soundfile.info``. Files that cannot be opened stay out of the inventory
    and are written to the reject report instead.
    """

    rows: list[dict[str, Any]] = []
    for record in source_audio_records:
        relative_path = record["audio_path"]
        base = {
            **record,
            "source_filename": Path(relative_path).name,
        }
        try:
            audio_info = inspect_audio(config.dataset_root, relative_path)
        except Exception as exc:
            rejects.append(_reject(base, "unreadable_audio", str(exc)))
            continue
        rows.append({**base, **_audio_fields(audio_info)})
    return sorted(rows, key=lambda row: row["audio_path"])


def is_roberts_bank_holdout(annotation, holdout_rule: HoldoutRuleConfig) -> bool:
    """Mark root annotations that belong to the Robert's Bank task-3 holdout."""

    return annotation.provider == holdout_rule.provider and annotation.dataset == holdout_rule.dataset


def _build_training_annotations_rows(annotations, rejects) -> list[dict[str, Any]]:
    """Filter root annotations down to the usable non-holdout training table."""

    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        row = annotation_to_dict(annotation)
        if annotation.audio_path is None:
            rejects.append(_reject(row, "missing_audio", "Expected source audio file was not found."))
            continue
        if not _valid_times(annotation.annotation_start_sec, annotation.annotation_end_sec):
            rejects.append(_reject(row, "invalid_times", "Annotation start/end times are missing or invalid."))
            continue
        rows.append(row)
    return rows


def _build_task3_source_rows(annotations, readable_audio_paths, rejects) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the full and strict Robert's Bank task-3 source tables.

    The full table keeps every matched Robert's Bank annotation row together
    with validation results. The strict table is the subset that passes the
    task-3 filtering rules defined in ``is_task3_strict``.
    """

    all_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    for annotation in annotations:
        row = annotation_to_dict(annotation)
        row_reasons: list[str] = []
        if annotation.audio_path is None:
            rejects.append(_reject(row, "missing_audio", "Expected Robert's Bank audio file was not found."))
            row_reasons.append("missing_audio")
        elif annotation.audio_path not in readable_audio_paths:
            rejects.append(_reject(row, "unreadable_audio", "Expected Robert's Bank audio file was not readable."))
            row_reasons.append("unreadable_audio")
        if not _valid_times(annotation.annotation_start_sec, annotation.annotation_end_sec):
            rejects.append(_reject(row, "invalid_times", "Task-3 annotation start/end times are missing or invalid."))
            row_reasons.append("invalid_times")

        eligible, reason = is_task3_strict(annotation)
        if row_reasons:
            eligible = False
            reason = ",".join(row_reasons)
        row.update(
            {
                "task3_row_valid": not row_reasons,
                "task3_validation_reason": ",".join(row_reasons) if row_reasons else None,
                "task3_strict_eligible": eligible,
                "task3_strict_exclusion_reason": reason,
            }
        )
        all_rows.append(row)
        if eligible:
            strict_rows.append(row)
    return all_rows, strict_rows


def _audio_records_from_annotations(annotations) -> dict[str, dict[str, Any]]:
    """Reuse annotation-derived provider and dataset metadata at the audio level."""

    records: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        if annotation.audio_path is None:
            continue
        records.setdefault(
            annotation.audio_path,
            {
                "provider": annotation.provider,
                "dataset": annotation.dataset,
                "audio_path": annotation.audio_path,
            },
        )
    return records


# ======== Holdout Rules ========

def _infer_audio_record(path: Path) -> dict[str, str]:
    """Infer minimal inventory metadata for audio files with no annotations."""

    parts = path.parts
    folder = parts[0] if parts else "unknown"
    dataset = parts[2] if len(parts) >= 3 and parts[1] == "audio" else "unknown"
    provider = FOLDER_PROVIDER_MAP.get(folder, folder)
    return {
        "provider": provider,
        "dataset": dataset,
        "audio_path": path.as_posix(),
    }


def _validate_holdout_config(config: SourceTableBuildConfig) -> None:
    """Require one configured holdout rule for the task-3 adapter."""

    if len(config.exclude_holdout_rules) != 1:
        raise ValueError("exclude_holdout_rules must contain exactly one task-3 holdout rule.")


def _task3_holdout_rule(config: SourceTableBuildConfig) -> HoldoutRuleConfig:
    """Return the configured task-3 holdout identity."""

    _validate_holdout_config(config)
    return config.exclude_holdout_rules[0]


# ======== Output Writers ========

def _write_source_table(frame: pd.DataFrame, stem: Path) -> None:
    """Persist one accepted source table in both CSV and Parquet form."""

    frame.to_csv(stem.with_suffix(".csv"), index=False)
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def _write_rejects(rejects: list[dict[str, Any]], path: Path) -> None:
    """Write the reject report, even when the build produced no reject rows."""

    columns = ["reject_reason_code", "reject_reason"]
    if rejects:
        pd.DataFrame(rejects).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=columns).to_csv(path, index=False)


# ======== Metadata ========

def _metadata(
    config,
    audio_index,
    audio_rows,
    annotation_rows,
    task3_rows,
    task3_strict_rows,
    rejects,
):
    """Summarize the build inputs and row counts in a compact metadata sidecar."""

    metadata_config = {
        "task3_holdout_rules": [rule.model_dump(mode="json") for rule in config.exclude_holdout_rules],
    }
    audio_frame = pd.DataFrame(audio_rows)
    annotation_frame = pd.DataFrame(annotation_rows)
    task3_strict_frame = pd.DataFrame(task3_strict_rows)
    return {
        "builder_version": BUILDER_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_root": str(config.dataset_root),
        **metadata_config,
        "config_hash": _hash_json(metadata_config),
        "summary_counts": {
            "indexed_audio_files": sum(len(paths) for paths in audio_index.values()),
            "audio_inventory_rows": len(audio_rows),
            "training_annotations_rows": len(annotation_rows),
            "task3_roberts_bank_rows": len(task3_rows),
            "task3_roberts_bank_strict_rows": len(task3_strict_rows),
            "task3_roberts_bank_valid_source_annotation_rows": int(
                pd.DataFrame(task3_rows).get("task3_row_valid", pd.Series(dtype=bool)).sum()
            ),
            "reject_rows": len(rejects),
            "task3_strict_counts_by_call_type": _value_counts(task3_strict_frame, "call_type"),
            "audio_counts_by_provider": _value_counts(audio_frame, "provider"),
            "annotation_counts_by_provider": _value_counts(annotation_frame, "provider"),
        },
    }


# ======== Shared Utilities ========

def _audio_fields(audio_info):
    """Project ``AudioInfo`` into the columns stored in the audio inventory."""

    return {
        "audio_sample_rate_hz": audio_info.sample_rate_hz,
        "audio_duration_sec": audio_info.duration_sec,
        "audio_frames": audio_info.frames,
    }


def _valid_times(start: float | None, end: float | None) -> bool:
    """Check that an annotation describes a real non-negative time span."""

    return start is not None and end is not None and end > start and start >= 0


def _reject(base: dict[str, Any], reason_code: str, reason: str) -> dict[str, Any]:
    """Attach a machine-readable failure reason to a rejected source row."""

    return {**base, "reject_reason_code": reason_code, "reject_reason": reason}


def _hash_json(data: Any) -> str:
    """Hash JSON-like data deterministically for reproducible metadata values."""

    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    """Return stable JSON-ready counts for one summary column."""

    if frame.empty or column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).sort_index().items()}


if __name__ == "__main__":
    main()
