# embedding_models

Trained HALLO encoders kept as standalone checkpoints, each with a sample config that embeds a dataset with
the frozen encoder through the pipeline's embeddings stage, then trains a linear probe on those embeddings.

Folders are named `<encoder>_<pretraining>_<backbone params>_<training spectrograms>`. Training spectrograms
counts the unique clips in the run's train split, before oversampling.

| Folder | Encoder | Pretraining | Params | Train spectrograms | Embedding dim | Input spectrogram | MLflow training run ID |
|---|---|---|---|---|---|---|---|
| `effnet_b1_supervised_6.5M_278k/` | EfficientNet-B1 | supervised classifier, DORI + DCLDE | 6.51M | 278,078 | 1280 | 256 mel × 563 (3 s @ 24 kHz) | `a1e2027254284d009685551bb9c8b5cb` |
| `perch_v2_supervised_10.7M_278k/` | Perch v2 (EfficientNet-B3) | Google's weights, then supervised pretraining, DORI + DCLDE | 10.70M | 277,866 | 1536 | 128 HTK mel × 497 (5 s @ 32 kHz, Perch frontend) | `b0cb422a7cca46a0a37d3ac1c389b9ff` † |
| `ssamba_tiny_ssl_6.9M_915k/` | SSAMBA-tiny | self-supervised on all DORI + DCLDE clips that contain annotations | 6.93M | 914,930 | 192 | 256 mel × 563 (3 s @ 24 kHz) | `54bd84245a234c18b8d640854dd4b4cf` |
| `audio_jepa_vit_base_ssl_85.7M_915k/` | Audio-JEPA ViT-base (EMA target encoder) | self-supervised on all DORI + DCLDE clips that contain annotations  | 85.68M | 914,930 | 768 | 256 mel × 563 (3 s @ 24 kHz) | `580c8a22329142f3b4d19111d69b5290` |

The `.pt` checkpoints are stored with [Git LFS](https://git-lfs.com) (`.gitattributes`). Install `git-lfs` and
run `git lfs install` once before cloning; otherwise the `.pt` files arrive as small text pointers and the
configs fail to load them. In a clone made without it, run `git lfs pull`.

## Running

All config paths are relative to the repo root, so run from there. Use a venv that has `timm`
(e.g. `/home/noah/embed_venv`; the repo's `.venv` lacks it), with the MLflow server at `localhost:5000`:

```bash
PY=/home/noah/embed_venv/bin/python
$PY -m encoder_pipeline.pipeline \
    --config embedding_models/00_base.yaml \
    --override embedding_models/ssamba_tiny_ssl_6.9M_915k/embed.yaml
```

`00_base.yaml` holds the shared settings and each `embed.yaml` is the override for one model, the same
`--config` / `--override` pattern as `configs/`. Each run:

1. builds the preprocessed HDF5 for `annotations_csv`, or reuses a cached one when the preprocessor section matches,
2. skips `model_trainer`, because `embeddings.checkpoint_path` is set,
3. embeds every fold/split with the one frozen checkpoint and writes `data/embeddings/<run_id>/fold<n>.h5`
   (`<split>_embeddings`, `<split>_labels`, `<split>_uids`),
4. trains the linear probe per fold and logs metrics and predictions to MLflow experiment `embedding-models`
   under `model_trainer.run_name`.

## Test dataset

Every `embed.yaml` points at `test_dataset/` by default, so any config can be smoke-tested in a minute or two:

- `dclde_2027_test_annotations.csv`: 240 rows of DCLDE 2027 annotations, with the same 27-column schema as
  `data_raw/DCLDE_2027/*/annotations.csv`. It has 60 each of Background, HW, SRKW and TKW, from 80 audio files
  (3 rows per file, each file used for one class only) spread over all 10 data providers. HW / SRKW / TKW rows are
  real annotations (`uid` = their row number in the original `Annotations.csv`). Background rows are 5 s windows at
  least 10 s from any annotated event (the DCLDE builder's naive method), with `uid` = 10,000,000 + n.
- `dclde_2027_test_splits.csv`: pinned 5-fold splits grouped by audio file. Per class, 2 files are the fixed
  test set of every fold and the rest rotate through val, so every fold's train, val and test sets contain
  every class and no file appears in two splits.
- `build_test_dataset.py`: regenerates both files, byte-identical, from the original DCLDE 2027 release at
  `/data/DCLDE_2027/dclde_2027_killer_whales/`, which every user on the machine can read. It doesn't use
  anything under `data_raw/` or a home directory. It derives the processed columns with
  `scripts/DCLDE_2027/build_dclde_2027_annotations.py`'s own functions. The call-type columns are the exception:
  they need `CreateParentAnnotations.R` and are left empty, since the embedding pipeline doesn't use them.
  Run it from the repo root: `python embedding_models/test_dataset/build_test_dataset.py`.

Only the annotations live in the repo. The audio is read from each row's `LocalPath`
(`/data/DCLDE_2027/dclde_2027_killer_whales/...`). The source files' sample rates range from 16 kHz to
256 kHz and are resampled by the preprocessor. 

## Embedding your own data

Copy a model's `embed.yaml` and change:

- `preprocessor.dataset.annotations_csv`: your annotations (needs `uid`, `LocalPath`, `FileBeginSec`,
  `FileEndSec`, `Duration`, `CenterTime`, `Labels`)
- `model_trainer.run_name`: the MLflow run name
- `model_trainer.dataloader.splits_path`: a pinned splits csv (`uid,fold_0,...` with train/val/test), or
  `null` to compute folds from `n_folds` / `test_size` / `val_size` (grouped by `col_to_group_by`)
- `model_trainer.dataloader.classes` (and `class_label_map` / `classes_to_drop` in the base if your labels differ)
- `embeddings.linear_probe_save: true` if you want each fold's trained probe saved as `fold<n>_linear_probe.pt`

For example, to reproduce the pretraining comparison's frozen probes on DORI VENUS (Strait of Georgia):

```yaml
preprocessor:
  dataset:
    annotations_csv: data_raw/DORI/20261022_190549/annotations_bgcap10000_ft_dori_venus_strait_of_georgia.csv
model_trainer:
  dataloader:
    splits_path: data_raw/DORI/20261022_190549/splits_pretrain_cmp_ft_dori_venus_strait_of_georgia.csv
    drop_unplaceable_windows: true # NaN FileBeginSec / Duration rows
    train_overlap_tol_s: 3.0       # drop train copies of test calls (not needed for the Perch config)
    train_overlap_splits: [test]
```

To re-probe embeddings you already have without re-embedding, set
`embeddings.reuse_embeddings_path: data/embeddings/<run_id>/fold<n>.h5`.
