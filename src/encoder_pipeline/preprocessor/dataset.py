from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

import h5py
import pandas as pd
from loguru import logger
from tqdm import tqdm

from encoder_pipeline.common.file_utils import get_or_create_hashed_file
from encoder_pipeline.preprocessor.annotation import AudioFile, Annotation
from encoder_pipeline.preprocessor.config import AnnotationConfig, AudioFileConfig, DatasetConfig, SpectrogramConfig
from encoder_pipeline.preprocessor.spectrogram import Spectrogram


class Dataset:
    """Generates HDF5 dataset for a given set of audio files, annotations, and spectrogram configuration"""

    def __init__(
        self, spec_config: SpectrogramConfig, audio_file_config: AudioFileConfig, annotation_config: AnnotationConfig,
        dataset_config: DatasetConfig, data_dir: str, run_name: Optional[str] = None,
    ) -> None:
        self.spec_config = spec_config
        self.audio_file_config = audio_file_config
        self.annotation_config = annotation_config
        self.dataset_config = dataset_config
        self.run_name = run_name
        # store resultant hdf5 file with has of config parameters as its file name
        self.out_file, self.content_hash = get_or_create_hashed_file(f"{data_dir}/preprocessor", ".h5", {
                    "spectrogram": self.spec_config.model_dump(),
                    "audio_file": self.audio_file_config.model_dump(),
                    "annotation": self.annotation_config.model_dump(),
                    "annotations_csv": self.dataset_config.annotations_csv,
                    "classes_to_drop": self.dataset_config.classes_to_drop,
                    "run_name": self.run_name,
        })
        # checks if file exists locally - TODO: check mlflow as well
        self.is_materialized = Path(self.out_file).exists()

    def _load_annotations(self) -> pd.DataFrame:
        """Helper to load in annotations.csv file"""
        df = pd.read_csv(self.dataset_config.annotations_csv)
        # check required columns TODO: document specific layout of required csv format for annotations
        for col in ["FileBeginSec", "FileEndSec", "Duration", "CenterTime"]:
            if col not in df.columns:
                raise ValueError(f"required col: {col} not in {self.dataset_config.annotations_csv}")
        if self.dataset_config.uid_col not in df.columns:
            raise ValueError(f"specified uid_col: {self.dataset_config.uid_col} not in {self.dataset_config.annotations_csv}")
        if self.dataset_config.local_file_col not in df.columns:
            raise ValueError(f"specified local_file_col: {self.dataset_config.local_file_col} not in {self.dataset_config.annotations_csv}")
        # convert necessary cols to numeric
        for col in ["FileBeginSec", "FileEndSec", "Duration", "CenterTime"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if self.dataset_config.classes_to_drop:
            if "Labels" not in df.columns:
                raise ValueError(f"classes_to_drop set but 'Labels' col not in {self.dataset_config.annotations_csv}")
            drop = df["Labels"].isin(self.dataset_config.classes_to_drop)
            logger.info("classes_to_drop {}: removing {} of {} rows", self.dataset_config.classes_to_drop, int(drop.sum()), len(df))
            df = df[~drop]

        return df

    @staticmethod
    def _process_file(
        file_path: str, rows: list[dict], spec_config: SpectrogramConfig, audio_file_config: AudioFileConfig,
        annotation_config: AnnotationConfig,
    ) -> list[dict]:
        audio_file = AudioFile(file_path, audio_file_config.resample_sr)
        spec = Spectrogram(spec_config)
        results = []
        for row in rows:
            try:
                annotation = Annotation(audio_file, row["Labels"], row["FileBeginSec"], row["Duration"], annotation_config)
            except ValueError as e:
                logger.error("skipping row {} in {}: {}", row.get("uid", row["_row_index"]), file_path, e)
                continue
            raw = spec.compute_mel(annotation) if spec_config.freq_scale == "mel" else spec.compute_magnitude(annotation)
            # append all cols with the specified row
            results.append({**row, "spec": spec.apply_dynamic(annotation, raw)})
        return results

    def build_hdf5(self, force_rebuild: bool = False) -> Union[str, None]:
        # exit early if already materialized
        if self.is_materialized and not force_rebuild:
            return
        
        tmp_path = f"{self.out_file}.tmp"
        raw_path = f"{self.out_file}.raw.tmp"
        Path(self.out_file).parent.mkdir(parents=True, exist_ok=True)
        df = self._load_annotations().reset_index(drop=True)
        grouped = df.groupby(self.dataset_config.local_file_col)
        metadata_columns = self.dataset_config.metadata_columns or list(df.columns)
        metadata: dict[str, list] = {col: [None] * len(df) for col in metadata_columns}
        with h5py.File(tmp_path, "w") as h5, h5py.File(raw_path, "w") as h5raw, \
                ProcessPoolExecutor(max_workers=self.dataset_config.resolve_max_workers()) as pool:
            n_rows_by_future = {
                pool.submit(
                    Dataset._process_file,
                    file_path,
                    [{"_row_index": idx, **row} for idx, row in zip(rows.index, rows.to_dict("records"))],
                    self.spec_config, self.audio_file_config, self.annotation_config,
                ): len(rows)
                for file_path, rows in grouped
            }
            specs_raw = None
            valid_indices: set[int] = set()
            with tqdm(total=len(df), desc="computing spectrograms", unit="row") as pbar:
                for future in as_completed(list(n_rows_by_future)):
                    # pop + del as we go
                    n_rows = n_rows_by_future.pop(future)
                    results = future.result()
                    for result in results:
                        # create hdf5 dataset to fill in with the rest of the spectrogram data
                        if specs_raw is None:
                            specs_raw = h5raw.create_dataset("spec_raw", shape=(len(df), *result["spec"].shape), dtype=result["spec"].dtype)
                        # insert row in correct index (as_completed(futures) may not be in same order as df)
                        row_index = result["_row_index"]
                        specs_raw[row_index] = result["spec"]
                        valid_indices.add(row_index)
                        for col in metadata_columns:
                            metadata[col][row_index] = result[col]
                    pbar.update(n_rows)
                    del results, future

            if specs_raw is None:
                raise ValueError("every row's Annotation failed -- nothing to write, see preceding skip logs")

            # drop rows whose Annotation failed
            valid_indices = sorted(valid_indices)
            n_skipped = len(df) - len(valid_indices)
            if n_skipped:
                logger.warning("dropping {} of {} rows from the dataset (see preceding skip logs)", n_skipped, len(df))
            specs = h5.create_dataset("spec", shape=(len(valid_indices), *specs_raw.shape[1:]), dtype=specs_raw.dtype)
            for new_pos, old_pos in tqdm(list(enumerate(valid_indices)), desc="writing hdf5", unit="row"):
                specs[new_pos] = specs_raw[old_pos]
            for col, values in metadata.items():
                metadata[col] = [values[i] for i in valid_indices]
            # finnally add in all other metadata cols
            for col, values in metadata.items():
                if pd.api.types.is_numeric_dtype(df[col]):
                    h5.create_dataset(col, data=values)
                else:
                    h5.create_dataset(col, data=[str(v) for v in values], dtype=h5py.string_dtype())
        Path(raw_path).unlink(missing_ok=True)
        Path(tmp_path).rename(self.out_file)
        self.is_materialized = True
