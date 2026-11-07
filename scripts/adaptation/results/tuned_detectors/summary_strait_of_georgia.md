# Strait of Georgia: off-the-shelf hallo-tuned-detectors vs fine-tuned arms

Mean ± std over 5 folds, each fold's test clips (the arms' own). (FT) = fine-tuned on the site's train folds.
The test clips are the same in every fold (fixed 10 % test, train / val rotate), so a model that is not trained on the site has std 0 except under val_tuned (its threshold comes from each fold's val clips).

## SRKW vs rest

| method | pr_auc | val_tuned f1 | val_tuned precision | val_tuned recall | argmax f1 | argmax precision | argmax recall | min n_test |
|---|---|---|---|---|---|---|---|---|
| EffNet-B1 ref (FT) | 0.828 ± 0.020 | 0.758 ± 0.027 | 0.817 ± 0.081 | 0.714 ± 0.052 | 0.673 ± 0.069 | 0.588 ± 0.120 | 0.806 ± 0.047 | 2592 |
| SSAMBA A supervised (FT) | 0.774 ± 0.076 | 0.710 ± 0.061 | 0.772 ± 0.165 | 0.685 ± 0.075 | 0.636 ± 0.041 | 0.537 ± 0.069 | 0.788 ± 0.032 | 2592 |
| SSAMBA B SSL source (FT) | 0.777 ± 0.098 | 0.669 ± 0.083 | 0.652 ± 0.108 | 0.702 ± 0.120 | 0.625 ± 0.127 | 0.562 ± 0.140 | 0.714 ± 0.113 | 2592 |
| SSAMBA C5 SSL all (FT) | 0.689 ± 0.053 | 0.606 ± 0.037 | 0.552 ± 0.093 | 0.694 ± 0.073 | 0.599 ± 0.021 | 0.540 ± 0.070 | 0.687 ± 0.062 | 2592 |
| SSAMBA D scratch (FT) | 0.495 ± 0.157 | 0.549 ± 0.118 | 0.484 ± 0.141 | 0.668 ± 0.094 | 0.524 ± 0.163 | 0.470 ± 0.078 | 0.627 ± 0.255 | 2592 |
| Perch v2 probe (site-trained) | 0.930 ± 0.005 | 0.877 ± 0.008 | 0.891 ± 0.032 | 0.863 ± 0.017 | 0.884 ± 0.005 | 0.932 ± 0.014 | 0.840 ± 0.010 | 2592 |
| exp_2 EffNet-B1 zero-shot | 0.305 ± 0.000 | 0.408 ± 0.023 | 0.319 ± 0.042 | 0.586 ± 0.082 | 0.409 ± 0.000 | 0.298 ± 0.000 | 0.650 ± 0.000 | 2592 |
| HALLO encoder v1 | 0.483 ± 0.000 | 0.462 ± 0.005 | 0.476 ± 0.057 | 0.459 ± 0.052 | 0.350 ± 0.000 | 0.229 ± 0.000 | 0.746 ± 0.000 | 2592 |
| BirdNET (Kait) | 0.536 ± 0.000 | 0.512 ± 0.004 | 0.494 ± 0.041 | 0.536 ± 0.035 | 0.457 ± 0.000 | 0.702 ± 0.000 | 0.339 ± 0.000 | 2592 |
| Sequence KW detector | 0.523 ± 0.000 | 0.620 ± 0.003 | 0.584 ± 0.007 | 0.662 ± 0.016 | 0.532 ± 0.000 | 0.624 ± 0.000 | 0.463 ± 0.000 | 2592 |
| Deployed cascade (seq ∧ BirdNET) | 0.733 ± 0.000 | 0.661 ± 0.004 | 0.633 ± 0.010 | 0.692 ± 0.019 | 0.336 ± 0.000 | 0.960 ± 0.000 | 0.203 ± 0.000 | 2592 |

## KW vs rest

| method | pr_auc | val_tuned f1 | val_tuned precision | val_tuned recall | argmax f1 | argmax precision | argmax recall | min n_test |
|---|---|---|---|---|---|---|---|---|
| EffNet-B1 ref (FT) | 0.688 ± 0.038 | 0.706 ± 0.032 | 0.671 ± 0.070 | 0.750 ± 0.044 | 0.611 ± 0.039 | 0.476 ± 0.048 | 0.860 ± 0.019 | 2592 |
| SSAMBA A supervised (FT) | 0.693 ± 0.056 | 0.678 ± 0.058 | 0.657 ± 0.144 | 0.728 ± 0.067 | 0.587 ± 0.043 | 0.449 ± 0.060 | 0.862 ± 0.028 | 2592 |
| SSAMBA B SSL source (FT) | 0.694 ± 0.004 | 0.718 ± 0.034 | 0.671 ± 0.051 | 0.777 ± 0.058 | 0.643 ± 0.073 | 0.523 ± 0.079 | 0.837 ± 0.052 | 2592 |
| SSAMBA C5 SSL all (FT) | 0.641 ± 0.075 | 0.662 ± 0.021 | 0.603 ± 0.049 | 0.740 ± 0.060 | 0.634 ± 0.032 | 0.507 ± 0.044 | 0.851 ± 0.018 | 2592 |
| SSAMBA D scratch (FT) | 0.473 ± 0.128 | 0.577 ± 0.094 | 0.512 ± 0.110 | 0.679 ± 0.132 | 0.570 ± 0.091 | 0.461 ± 0.067 | 0.749 ± 0.137 | 2592 |
| Perch v2 probe (site-trained) | 0.838 ± 0.024 | 0.787 ± 0.043 | 0.759 ± 0.072 | 0.824 ± 0.057 | 0.767 ± 0.032 | 0.678 ± 0.050 | 0.884 ± 0.009 | 2592 |
| exp_2 EffNet-B1 zero-shot | 0.257 ± 0.000 | 0.385 ± 0.018 | 0.274 ± 0.034 | 0.676 ± 0.111 | 0.391 ± 0.000 | 0.273 ± 0.000 | 0.692 ± 0.000 | 2592 |
| HALLO encoder v1 | 0.329 ± 0.000 | 0.420 ± 0.008 | 0.335 ± 0.004 | 0.564 ± 0.040 | 0.331 ± 0.000 | 0.203 ± 0.000 | 0.895 ± 0.000 | 2592 |
| BirdNET (Kait) | 0.516 ± 0.000 | 0.496 ± 0.008 | 0.444 ± 0.050 | 0.574 ± 0.063 | 0.413 ± 0.000 | 0.703 ± 0.000 | 0.293 ± 0.000 | 2592 |
| Sequence KW detector | 0.654 ± 0.000 | 0.669 ± 0.006 | 0.689 ± 0.037 | 0.654 ± 0.043 | 0.558 ± 0.000 | 0.776 ± 0.000 | 0.436 ± 0.000 | 2592 |
| Deployed cascade (seq ∧ BirdNET) | 0.757 ± 0.000 | 0.702 ± 0.011 | 0.738 ± 0.026 | 0.672 ± 0.044 | 0.273 ± 0.000 | 0.987 ± 0.000 | 0.158 ± 0.000 | 2592 |

`argmax` = the model's own decision (BirdNET: backend thresholds; sequence: KW score >= 0.75, counted as SRKW for the SRKW task; cascade: deployed resolver). `val_tuned` = F1-optimal threshold on that fold's val clips.
