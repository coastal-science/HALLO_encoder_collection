# File-level comparison from existing clip predictions

Each file's score = max over its annotated clips; decisions: `val_tuned` (file threshold tuned on the fold's val files), `any_clip` (any clip's own decision positive), `vote` (majority of non-Background clip decisions, BirdNET's deployed file rule). Mean ± std over 5 folds; the test files are the same in every fold. `fp_files` = negative test files called positive.

**Caveat:** only annotated clips exist, not whole files. Strait of Georgia KW files hold 1 clip each (file level == clip level for them); its negatives are 18 dedicated background files of 118 clips. LimeKiln has 14 test files, Tekteksen 15 (1 negative). Off-the-shelf detectors only at Strait of Georgia.

## Strait of Georgia: SRKW vs rest (480 test files, 350 positive, 130 negative)

| method | pr_auc | val_tuned f1 | val_tuned precision | val_tuned recall | val_tuned fp_files | any_clip f1 | any_clip precision | any_clip recall | any_clip fp_files | vote f1 | vote precision | vote recall | vote fp_files |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Perch v2 frozen (probe) | 0.974 ± 0.001 | 0.901 ± 0.004 | 0.907 ± 0.019 | 0.894 ± 0.011 | 32.2 | 0.905 ± 0.006 | 0.977 ± 0.004 | 0.842 ± 0.011 | 6.8 | 0.906 ± 0.006 | 0.981 ± 0.004 | 0.842 ± 0.011 | 5.8 |
| Perch v2 fine-tuned (G) | 0.986 ± 0.001 | 0.928 ± 0.005 | 0.936 ± 0.012 | 0.919 ± 0.018 | 22.0 | 0.926 ± 0.004 | 0.953 ± 0.005 | 0.899 ± 0.007 | 15.4 | 0.932 ± 0.002 | 0.966 ± 0.007 | 0.899 ± 0.007 | 11.0 |
| EffNet-B1 ref (FT) | 0.967 ± 0.004 | 0.906 ± 0.004 | 0.865 ± 0.009 | 0.952 ± 0.016 | 52.2 | 0.864 ± 0.023 | 0.931 ± 0.011 | 0.807 ± 0.048 | 21.2 | 0.872 ± 0.021 | 0.950 ± 0.015 | 0.807 ± 0.048 | 15.0 |
| exp_2 EffNet-B1 zero-shot | 0.898 ± 0.000 | 0.847 ± 0.004 | 0.760 ± 0.020 | 0.958 ± 0.022 | 106.4 | 0.756 ± 0.000 | 0.895 ± 0.000 | 0.654 ± 0.000 | 27.0 | 0.758 ± 0.000 | 0.902 ± 0.000 | 0.654 ± 0.000 | 25.0 |
| HALLO encoder v1 | 0.870 ± 0.000 | 0.844 ± 0.002 | 0.735 ± 0.006 | 0.991 ± 0.005 | 125.2 | 0.779 ± 0.000 | 0.816 ± 0.000 | 0.746 ± 0.000 | 59.0 | 0.780 ± 0.000 | 0.820 ± 0.000 | 0.743 ± 0.000 | 57.0 |
| BirdNET (Kait) | 0.878 ± 0.000 | 0.844 ± 0.003 | 0.745 ± 0.012 | 0.974 ± 0.025 | 116.6 | 0.494 ± 0.000 | 0.902 ± 0.000 | 0.340 ± 0.000 | 13.0 | 0.495 ± 0.000 | 0.908 ± 0.000 | 0.340 ± 0.000 | 12.0 |
| Sequence KW detector | 0.764 ± 0.000 | 0.843 ± 0.000 | 0.729 ± 0.000 | 1.000 ± 0.000 | 130.0 | 0.587 ± 0.000 | 0.802 ± 0.000 | 0.463 ± 0.000 | 40.0 | 0.587 ± 0.000 | 0.802 ± 0.000 | 0.463 ± 0.000 | 40.0 |
| Deployed cascade (seq ∧ BirdNET) | 0.882 ± 0.000 | 0.843 ± 0.001 | 0.730 ± 0.001 | 0.998 ± 0.004 | 129.4 | 0.339 ± 0.000 | 0.960 ± 0.000 | 0.206 ± 0.000 | 3.0 | 0.340 ± 0.000 | 0.973 ± 0.000 | 0.206 ± 0.000 | 2.0 |

## Strait of Georgia: KW vs rest (480 test files, 462 positive, 18 negative)

| method | pr_auc | val_tuned f1 | val_tuned precision | val_tuned recall | val_tuned fp_files | any_clip f1 | any_clip precision | any_clip recall | any_clip fp_files | vote f1 | vote precision | vote recall | vote fp_files |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Perch v2 frozen (probe) | 0.991 ± 0.001 | 0.980 ± 0.001 | 0.962 ± 0.000 | 0.998 ± 0.003 | 18.0 | 0.934 ± 0.005 | 0.988 ± 0.002 | 0.885 ± 0.009 | 4.8 | 0.934 ± 0.005 | 0.988 ± 0.002 | 0.885 ± 0.009 | 4.8 |
| Perch v2 fine-tuned (G) | 0.989 ± 0.002 | 0.981 ± 0.000 | 0.963 ± 0.001 | 0.999 ± 0.001 | 17.8 | 0.947 ± 0.007 | 0.974 ± 0.003 | 0.921 ± 0.011 | 11.4 | 0.947 ± 0.007 | 0.974 ± 0.003 | 0.921 ± 0.011 | 11.4 |
| EffNet-B1 ref (FT) | 0.977 ± 0.006 | 0.981 ± 0.001 | 0.963 ± 0.001 | 0.999 ± 0.001 | 17.6 | 0.910 ± 0.012 | 0.967 ± 0.003 | 0.860 ± 0.019 | 13.6 | 0.911 ± 0.012 | 0.967 ± 0.003 | 0.860 ± 0.019 | 13.4 |
| exp_2 EffNet-B1 zero-shot | 0.936 ± 0.000 | 0.979 ± 0.001 | 0.962 ± 0.000 | 0.995 ± 0.002 | 18.0 | 0.803 ± 0.000 | 0.955 ± 0.000 | 0.693 ± 0.000 | 15.0 | 0.803 ± 0.000 | 0.955 ± 0.000 | 0.693 ± 0.000 | 15.0 |
| HALLO encoder v1 | 0.942 ± 0.000 | 0.981 ± 0.000 | 0.963 ± 0.000 | 1.000 ± 0.000 | 18.0 | 0.926 ± 0.000 | 0.958 ± 0.000 | 0.896 ± 0.000 | 18.0 | 0.926 ± 0.000 | 0.958 ± 0.000 | 0.896 ± 0.000 | 18.0 |
| BirdNET (Kait) | 0.953 ± 0.000 | 0.981 ± 0.000 | 0.963 ± 0.000 | 1.000 ± 0.000 | 18.0 | 0.447 ± 0.000 | 0.951 ± 0.000 | 0.292 ± 0.000 | 7.0 | 0.447 ± 0.000 | 0.951 ± 0.000 | 0.292 ± 0.000 | 7.0 |
| Sequence KW detector | 0.976 ± 0.000 | 0.981 ± 0.000 | 0.963 ± 0.000 | 1.000 ± 0.000 | 18.0 | 0.605 ± 0.000 | 0.995 ± 0.000 | 0.435 ± 0.000 | 1.0 | 0.605 ± 0.000 | 0.995 ± 0.000 | 0.435 ± 0.000 | 1.0 |
| Deployed cascade (seq ∧ BirdNET) | 0.977 ± 0.000 | 0.981 ± 0.000 | 0.963 ± 0.000 | 1.000 ± 0.000 | 18.0 | 0.276 ± 0.000 | 0.987 ± 0.000 | 0.160 ± 0.000 | 1.0 | 0.276 ± 0.000 | 1.000 ± 0.000 | 0.160 ± 0.000 | 0.0 |

## LimeKiln: SRKW vs rest (14 test files, 5 positive, 9 negative)

| method | pr_auc | val_tuned f1 | val_tuned precision | val_tuned recall | val_tuned fp_files | any_clip f1 | any_clip precision | any_clip recall | any_clip fp_files | vote f1 | vote precision | vote recall | vote fp_files |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Perch v2 frozen (probe) | 0.766 ± 0.034 | 0.619 ± 0.051 | 0.549 ± 0.046 | 0.720 ± 0.110 | 3.0 | 0.656 ± 0.023 | 0.557 ± 0.032 | 0.800 ± 0.000 | 3.2 | 0.656 ± 0.023 | 0.557 ± 0.032 | 0.800 ± 0.000 | 3.2 |
| Perch v2 fine-tuned (G) | 0.663 ± 0.074 | 0.693 ± 0.051 | 0.614 ± 0.076 | 0.800 ± 0.000 | 2.6 | 0.615 ± 0.000 | 0.500 ± 0.000 | 0.800 ± 0.000 | 4.0 | 0.615 ± 0.000 | 0.500 ± 0.000 | 0.800 ± 0.000 | 4.0 |
| EffNet-B1 ref (FT) | 0.910 ± 0.027 | 0.742 ± 0.033 | 0.693 ± 0.060 | 0.800 ± 0.000 | 1.8 | 0.727 ± 0.000 | 0.667 ± 0.000 | 0.800 ± 0.000 | 2.0 | 0.727 ± 0.000 | 0.667 ± 0.000 | 0.800 ± 0.000 | 2.0 |
| exp_2 EffNet-B1 zero-shot | 0.754 ± 0.000 | 0.730 ± 0.047 | 0.710 ± 0.062 | 0.760 ± 0.089 | 1.6 | 0.727 ± 0.000 | 0.667 ± 0.000 | 0.800 ± 0.000 | 2.0 | 0.727 ± 0.000 | 0.667 ± 0.000 | 0.800 ± 0.000 | 2.0 |

## LimeKiln: KW vs rest (14 test files, 5 positive, 9 negative)

| method | pr_auc | val_tuned f1 | val_tuned precision | val_tuned recall | val_tuned fp_files | any_clip f1 | any_clip precision | any_clip recall | any_clip fp_files | vote f1 | vote precision | vote recall | vote fp_files |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Perch v2 frozen (probe) | 0.756 ± 0.051 | 0.628 ± 0.078 | 0.540 ± 0.037 | 0.760 ± 0.167 | 3.2 | 0.656 ± 0.023 | 0.557 ± 0.032 | 0.800 ± 0.000 | 3.2 | 0.656 ± 0.023 | 0.557 ± 0.032 | 0.800 ± 0.000 | 3.2 |
| Perch v2 fine-tuned (G) | 0.678 ± 0.072 | 0.693 ± 0.051 | 0.614 ± 0.076 | 0.800 ± 0.000 | 2.6 | 0.615 ± 0.000 | 0.500 ± 0.000 | 0.800 ± 0.000 | 4.0 | 0.615 ± 0.000 | 0.500 ± 0.000 | 0.800 ± 0.000 | 4.0 |
| EffNet-B1 ref (FT) | 0.906 ± 0.028 | 0.756 ± 0.040 | 0.720 ± 0.073 | 0.800 ± 0.000 | 1.6 | 0.727 ± 0.000 | 0.667 ± 0.000 | 0.800 ± 0.000 | 2.0 | 0.727 ± 0.000 | 0.667 ± 0.000 | 0.800 ± 0.000 | 2.0 |
| exp_2 EffNet-B1 zero-shot | 0.754 ± 0.000 | 0.744 ± 0.057 | 0.737 ± 0.067 | 0.760 ± 0.089 | 1.4 | 0.667 ± 0.000 | 0.571 ± 0.000 | 0.800 ± 0.000 | 3.0 | 0.727 ± 0.000 | 0.667 ± 0.000 | 0.800 ± 0.000 | 2.0 |

## Tekteksen: SRKW vs rest (15 test files, 14 positive, 1 negative)

| method | pr_auc | val_tuned f1 | val_tuned precision | val_tuned recall | val_tuned fp_files | any_clip f1 | any_clip precision | any_clip recall | any_clip fp_files | vote f1 | vote precision | vote recall | vote fp_files |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Perch v2 frozen (probe) | 1.000 ± 0.000 | 0.993 ± 0.015 | 0.987 ± 0.030 | 1.000 ± 0.000 | 0.2 | 0.920 ± 0.066 | 1.000 ± 0.000 | 0.857 ± 0.113 | 0.0 | 0.920 ± 0.066 | 1.000 ± 0.000 | 0.857 ± 0.113 | 0.0 |
| Perch v2 fine-tuned (G) | 1.000 ± 0.000 | 0.972 ± 0.015 | 0.947 ± 0.030 | 1.000 ± 0.000 | 0.8 | 0.993 ± 0.015 | 0.987 ± 0.030 | 1.000 ± 0.000 | 0.2 | 0.993 ± 0.015 | 0.987 ± 0.030 | 1.000 ± 0.000 | 0.2 |
| EffNet-B1 ref (FT) | 1.000 ± 0.000 | 0.966 ± 0.000 | 0.933 ± 0.000 | 1.000 ± 0.000 | 1.0 | 0.972 ± 0.015 | 0.947 ± 0.030 | 1.000 ± 0.000 | 0.8 | 0.972 ± 0.015 | 0.947 ± 0.030 | 1.000 ± 0.000 | 0.8 |
| exp_2 EffNet-B1 zero-shot | 1.000 ± 0.000 | 0.993 ± 0.015 | 0.987 ± 0.030 | 1.000 ± 0.000 | 0.2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0 |

## Tekteksen: KW vs rest (15 test files, 14 positive, 1 negative)

| method | pr_auc | val_tuned f1 | val_tuned precision | val_tuned recall | val_tuned fp_files | any_clip f1 | any_clip precision | any_clip recall | any_clip fp_files | vote f1 | vote precision | vote recall | vote fp_files |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Perch v2 frozen (probe) | 1.000 ± 0.000 | 0.972 ± 0.015 | 0.947 ± 0.030 | 1.000 ± 0.000 | 0.8 | 0.920 ± 0.066 | 1.000 ± 0.000 | 0.857 ± 0.113 | 0.0 | 0.920 ± 0.066 | 1.000 ± 0.000 | 0.857 ± 0.113 | 0.0 |
| Perch v2 fine-tuned (G) | 1.000 ± 0.000 | 0.979 ± 0.019 | 0.960 ± 0.037 | 1.000 ± 0.000 | 0.6 | 0.993 ± 0.015 | 0.987 ± 0.030 | 1.000 ± 0.000 | 0.2 | 0.993 ± 0.015 | 0.987 ± 0.030 | 1.000 ± 0.000 | 0.2 |
| EffNet-B1 ref (FT) | 1.000 ± 0.000 | 0.966 ± 0.000 | 0.933 ± 0.000 | 1.000 ± 0.000 | 1.0 | 0.972 ± 0.015 | 0.947 ± 0.030 | 1.000 ± 0.000 | 0.8 | 0.972 ± 0.015 | 0.947 ± 0.030 | 1.000 ± 0.000 | 0.8 |
| exp_2 EffNet-B1 zero-shot | 1.000 ± 0.000 | 0.979 ± 0.019 | 0.960 ± 0.037 | 1.000 ± 0.000 | 0.6 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0 |
