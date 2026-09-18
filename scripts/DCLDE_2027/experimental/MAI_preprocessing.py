# script using preprocessing/splitting from Efficient Passive Acoustic Monitoring of Killer  Whales Using a Two-Stage Detection and  Ecotype Classification Cascade, Ruiz et al.

import pandas as pd
from sklearn.model_selection import train_test_split

pp_df = pd.read_csv("/home/noah/HALLO_encoder_collection/data_raw/DCLDE_2027/20260827_090049/annotations.csv")

pp_df = pp_df.drop_duplicates().reset_index(drop=True)

new_pp_df = pp_df[~(pp_df['KW_certain'] == 0.0)]
print(len(new_pp_df))
print(len(pp_df))
print(len(pp_df) - len(new_pp_df))

new_pp_df = new_pp_df[new_pp_df['Duration'] != 0.0]
print(len(new_pp_df))
print(len(pp_df) - len(new_pp_df))

new_pp_df = new_pp_df[new_pp_df['AnnotationLevel'] != 'File']
print(len(new_pp_df))
print(len(pp_df) - len(new_pp_df))

print(pp_df.Labels.value_counts())

# 70/15/15 file-level split, stratified by each file's dominant Labels value so
# NonBio/Bio/KW (stage 1) and SRKW/TKW/SAR/NRKW/OKW (stage 2) proportions hold in each split.
split_df = pp_df[pp_df["Labels"] != "Background"].copy()
file_label = split_df.groupby("Soundfile")["Labels"].agg(lambda s: s.value_counts().idxmax())

train_files, rest_files = train_test_split(file_label.index, test_size=0.3, stratify=file_label, random_state=0)
val_files, test_files = train_test_split(rest_files, test_size=0.5, stratify=file_label.loc[rest_files], random_state=0)

split_df["split"] = split_df["Soundfile"].map(
    {**dict.fromkeys(train_files, "train"), **dict.fromkeys(val_files, "val"), **dict.fromkeys(test_files, "test")}
)
print(split_df.groupby("split")["Labels"].value_counts())

