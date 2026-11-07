# SRKW vs rest, SSAMBA-tiny C vs EffNet-B1: fine-tuned head and linear probe, mean ± std over folds

## Strait of Georgia

| method | head | pr_auc | argmax precision | argmax recall | argmax f1 | val_tuned precision | val_tuned recall | val_tuned f1 |
|---|---|---|---|---|---|---|---|---|
| effnet_b1_ft | finetune | 0.828 ± 0.020 | 0.588 ± 0.120 | 0.806 ± 0.047 | 0.673 ± 0.069 | 0.817 ± 0.081 | 0.714 ± 0.052 | 0.758 ± 0.027 |
| ssamba_C_ssl_all | finetune | 0.686 ± 0.150 | 0.529 ± 0.117 | 0.637 ± 0.176 | 0.574 ± 0.134 | 0.530 ± 0.113 | 0.729 ± 0.058 | 0.604 ± 0.072 |
| effnet_b1_ft | probe | 0.720 ± 0.057 | 0.599 ± 0.074 | 0.762 ± 0.070 | 0.665 ± 0.033 | 0.628 ± 0.016 | 0.723 ± 0.049 | 0.671 ± 0.019 |
| ssamba_C_ssl_all | probe | 0.667 ± 0.180 | 0.520 ± 0.123 | 0.679 ± 0.124 | 0.586 ± 0.117 | 0.538 ± 0.116 | 0.750 ± 0.045 | 0.617 ± 0.077 |

## LimeKiln

| method | head | pr_auc | argmax precision | argmax recall | argmax f1 | val_tuned precision | val_tuned recall | val_tuned f1 |
|---|---|---|---|---|---|---|---|---|
| effnet_b1_ft | finetune | 0.977 ± 0.009 | 0.968 ± 0.000 | 0.950 ± 0.013 | 0.959 ± 0.007 | 0.966 ± 0.006 | 0.959 ± 0.024 | 0.962 ± 0.011 |
| ssamba_C_ssl_all | finetune | 0.948 ± 0.017 | 0.915 ± 0.015 | 0.906 ± 0.025 | 0.910 ± 0.010 | 0.922 ± 0.009 | 0.884 ± 0.058 | 0.902 ± 0.029 |
| effnet_b1_ft | probe | 0.979 ± 0.009 | 0.965 ± 0.006 | 0.953 ± 0.022 | 0.959 ± 0.011 | 0.965 ± 0.006 | 0.947 ± 0.024 | 0.956 ± 0.010 |
| ssamba_C_ssl_all | probe | 0.951 ± 0.019 | 0.918 ± 0.015 | 0.897 ± 0.050 | 0.906 ± 0.020 | 0.913 ± 0.015 | 0.884 ± 0.054 | 0.898 ± 0.028 |

## Tekteksen

| method | head | pr_auc | argmax precision | argmax recall | argmax f1 | val_tuned precision | val_tuned recall | val_tuned f1 |
|---|---|---|---|---|---|---|---|---|
| effnet_b1_ft | finetune | 0.984 ± 0.003 | 0.950 ± 0.014 | 0.933 ± 0.008 | 0.942 ± 0.006 | 0.945 ± 0.019 | 0.935 ± 0.009 | 0.940 ± 0.008 |
| ssamba_C_ssl_all | finetune | 0.982 ± 0.003 | 0.960 ± 0.016 | 0.932 ± 0.008 | 0.946 ± 0.011 | 0.952 ± 0.029 | 0.935 ± 0.012 | 0.943 ± 0.013 |
| effnet_b1_ft | probe | 0.970 ± 0.007 | 0.890 ± 0.030 | 0.935 ± 0.021 | 0.912 ± 0.013 | 0.930 ± 0.028 | 0.881 ± 0.043 | 0.904 ± 0.014 |
| ssamba_C_ssl_all | probe | 0.982 ± 0.003 | 0.933 ± 0.035 | 0.944 ± 0.014 | 0.938 ± 0.015 | 0.953 ± 0.023 | 0.933 ± 0.014 | 0.943 ± 0.011 |
