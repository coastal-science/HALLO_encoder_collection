# Embeddings_MMS_2025

# Datasets

## Training

**DCLDE 2026**

This dataset included clips of marine mammal vocalizations, mostly in the 1 kHz to 12 kHz range

To download the dataset, use the following command:

gsutil -m rsync -r gs://noaa-passive-bioacoustic/dclde/2026/dclde_2026_killer_whales/ ./DESTINATION 


## Test

**Hear My Ship**

This dataset contain audio clips, pictures and video of small vessels that typically do not have AIS information. I believe there is no background baseline.

The dataset has the followiing classes:

- Aux Vessels
- Ferry
- Sail Boat
- Yatch
- Fishing Boat
- Tour Boat
- Motor Boat

Dataset website wiith examples: https://hearmyship.fer.hr/

To download the data, run: 

```bash
python dataset_download_scripts/hear_my_ship_donwload.py --json-file dataset_download_scripts/vessels_hear_my_ship.json --assets sound --out-dir /Your/Output/Directory/
```

This command will download only the audio files. To download the otehr data type, do not set the `--assets` flag.

**QiandaoEar22**

This dataset contains clips of vessel noise for multiple vessel types, as well as background noise

reference: https://arxiv.org/abs/2406.04354

data: https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/duxiaoyang22_mails_ucas_ac_cn/EomiGNu7mO5FmUke62y6Q7IBIP64kpJrJMJOZp_c-qkFAA?e=x8gxuL.



# Tasks

1. Blue Whale detector
    a. Train on DCLDE 2026 (except the JASCO Boundary Pass Data)
    b. test on the Antartic Dataset

2. Call type calssifier
    a. Same model from previous task
    b. Test on the JASCO boundary pass Data.

3. Vessel type Classifier
    a. Same model
    b. Test on the Vessel dataset.


# Representation

1. Mid freq = 0-12kHz
2. low freq = 0-500 Hz
3. Duration 5s
4. Duration 20s
5. Duration 1s
6. Mag Spectrograms
7. Time Correlation
8. Freq Correlation

In summary 18 different models. 19 if we wnat to combine different representations in one model.