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

## Downstream results

Taken from the result tables (`summary.csv`) of `configs/pretraining_comparison/compare_all_experiments.ipynb`
(killer whale, DORI) and `configs/adaptation_experiments/compare_adaptation_experiments.ipynb` (blue / fin whale,
on joy-lab), built from their MLflow runs. Values are mean ± std over 5 folds at the val-tuned threshold
(the threshold maximising F1 on val). `--` = not run for this model.

**Killer whale (DORI): SRKW vs rest, 100 % of site labels**

| Model | Method | Site | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|---|
| effnet_b1_supervised_6.5M_278k | fine-tune | Strait of Georgia | 0.758 ± 0.027 | 0.817 ± 0.081 | 0.714 ± 0.052 | 0.828 ± 0.020 |
| | fine-tune | LimeKiln | 0.962 ± 0.011 | 0.966 ± 0.006 | 0.959 ± 0.024 | 0.977 ± 0.009 |
| | fine-tune | Tekteksen | 0.940 ± 0.008 | 0.945 ± 0.019 | 0.935 ± 0.009 | 0.984 ± 0.003 |
| | linear probe | Strait of Georgia | 0.502 ± 0.012 | 0.467 ± 0.037 | 0.547 ± 0.033 | 0.526 ± 0.004 |
| | linear probe | LimeKiln | 0.926 ± 0.021 | 0.943 ± 0.012 | 0.912 ± 0.049 | 0.955 ± 0.001 |
| | linear probe | Tekteksen | 0.760 ± 0.010 | 0.730 ± 0.085 | 0.808 ± 0.077 | 0.868 ± 0.026 |
| perch_v2_supervised_10.7M_278k | fine-tune | Strait of Georgia | 0.882 ± 0.015 | 0.908 ± 0.058 | 0.859 ± 0.025 | 0.931 ± 0.004 |
| | fine-tune | LimeKiln | 0.919 ± 0.005 | 0.919 ± 0.011 | 0.919 ± 0.017 | 0.906 ± 0.018 |
| | fine-tune | Tekteksen | 0.948 ± 0.001 | 0.957 ± 0.020 | 0.940 ± 0.018 | 0.986 ± 0.001 |
| | linear probe | all sites | -- | -- | -- | -- |
| ssamba_tiny_ssl_6.9M_915k | fine-tune | Strait of Georgia | 0.608 ± 0.065 | 0.514 ± 0.106 | 0.765 ± 0.028 | 0.736 ± 0.071 |
| | fine-tune | LimeKiln | 0.911 ± 0.089 | 0.926 ± 0.010 | 0.906 ± 0.150 | 0.946 ± 0.023 |
| | fine-tune | Tekteksen | 0.965 ± 0.004 | 0.978 ± 0.005 | 0.953 ± 0.005 | 0.991 ± 0.003 |
| | linear probe | all sites | -- | -- | -- | -- |
| audio_jepa_vit_base_ssl_85.7M_915k | fine-tune | Strait of Georgia | 0.538 ± 0.150 | 0.587 ± 0.251 | 0.585 ± 0.095 | 0.577 ± 0.211 |
| | fine-tune | LimeKiln | 0.784 ± 0.095 | 0.818 ± 0.110 | 0.769 ± 0.137 | 0.858 ± 0.084 |
| | fine-tune | Tekteksen | 0.811 ± 0.030 | 0.890 ± 0.014 | 0.747 ± 0.052 | 0.915 ± 0.020 |
| | linear probe | all sites | -- | -- | -- | -- |

**Blue / fin whale (Antarctic): each species vs rest, pooled over all 11 deployments**

| Model | Method | Experiment | Species | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|---|---|
| effnet_b1_supervised_6.5M_278k | fine-tune | exp6 single-label | Blue | 0.794 ± 0.004 | 0.773 ± 0.007 | 0.816 ± 0.009 | 0.866 ± 0.009 |
| | fine-tune | exp6 single-label | Fin | 0.792 ± 0.015 | 0.801 ± 0.018 | 0.782 ± 0.015 | 0.850 ± 0.012 |
| | fine-tune | exp7 multi-label | Blue | 0.808 ± 0.008 | 0.815 ± 0.021 | 0.802 ± 0.014 | 0.894 ± 0.005 |
| | fine-tune | exp7 multi-label | Fin | 0.829 ± 0.008 | 0.866 ± 0.016 | 0.796 ± 0.018 | 0.909 ± 0.008 |
| | linear probe* | exp6 single-label | Blue | 0.786 ± 0.005 | 0.765 ± 0.012 | 0.809 ± 0.011 | 0.860 ± 0.010 |
| | linear probe* | exp6 single-label | Fin | 0.795 ± 0.013 | 0.795 ± 0.011 | 0.795 ± 0.014 | 0.850 ± 0.013 |
| | linear probe* | exp7 multi-label | Blue | 0.806 ± 0.007 | 0.807 ± 0.022 | 0.806 ± 0.017 | 0.891 ± 0.005 |
| | linear probe* | exp7 multi-label | Fin | 0.827 ± 0.011 | 0.860 ± 0.011 | 0.797 ± 0.020 | 0.904 ± 0.011 |
| perch_v2_supervised_10.7M_278k | both | all | -- | -- | -- | -- | -- |
| ssamba_tiny_ssl_6.9M_915k | both | all | -- | -- | -- | -- | -- |
| audio_jepa_vit_base_ssl_85.7M_915k | both | all | -- | -- | -- | -- | -- |

\* In the blue / fin experiments, the linear probe is a head trained on the **fine-tuned** model's embeddings; the
killer whale probes are trained on the frozen pretrained encoder.

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
