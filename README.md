## Project Overview

This project aims to create a collection of audio encoders for downstream underwater acoustic tasks, such as detection, classification, and localization of marine animal sounds.
There have been few efforts to produce encoders useful to underwater acoustics, such as [SurfPerch](https://www.kaggle.com/models/google/surfperch), but they often carry constraints inherited from other environments and tasks, such as speech analysis or bird acoustics.
For example, SurfPerch represents audio as 5s Mel-scaled spectrograms, ranging from 0 to 16 kHz, a representation often used in speech recognition and that has been carried to many terrestrial acoustics tasks by convenience, but that might not be optimal for some underwater (and even terrestrial) acoustic tasks.

The question behind this project is: Can different representations produce better encoders for underwater acoustic tasks?

## Representations

Initially, we will explore 3 ways of representing an audio signal:
1)  A magnitude spectrogram (with linear frequency scale)
2)  A time-similarity matrix, computed as the Pearson correlation coefficients of the time bins in the spectrogram (1)
3)  A frequency-similarity matrix, computed as the Pearson correlation coefficients of the frequency bins in the spectrogram (1)

Additionally, the following segment durations and frequency ranges will be used with each of the above representations:
**Durations**
- 1 second
- 5 seconds
- 20 seconds

**Frequency ranges**
- 0-500 Hz
- 500 - 12000 Hz

## Tasks

The encoders will be evaluated on the following tasks.

**1) Blue whale detection**
Use the encoders to build a detector for the stereotypical low-frequency blue whale calls. 

**2) Vessel classification**
Use  the encoders to build a classifiers that identify the type of vessel producing noise.

**3) KW call-type classification**
Use encoders to classify tonal calls of Southern Resident Killer Whale into call-types. Use a similarity approach to compare inputs to an established call-type catalogue. 


## Datasets

### DCLDE 2027 (training) 

This dataset included clips of marine mammal vocalizations, mostly in the 1 kHz to 12 kHz range.
It contains data from a variety of locations/instruments, and will be used as the main source of training data.
The "Robert's Bank" subset will be excluded from training, because it will be used for testing (task #3) 

### Antarctic Blue (and fin) whales (testing - task #1)

Described in [Miller et al (2021)](https://rdcu.be/fj89R). This dataset contains recordings from multiple instruments/locations. It also contains call-level annotations for blue and fin whales. In other words, the start and end times of calls within the audio file are indicated alongside the call label.

This dataset will be used in task #1-Blue whale detection. The encoders will be used to build detectors that specify the start and end time of every Blue whale call within the selected test files.

### HearMyShip (testing - task #2)

(website with examples: https://hearmyship.fer.hr/)

This dataset contains audio clips, pictures and video of small vessels that typically do not have AIS information. 

The dataset has the following classes:

Aux Vessels  
Ferry  
Sail Boat  
Yatch  
Fishing Boat  
Tour Boat  
Motor Boat  

This dataset will be used in task # 2-Vessel classification. 
The encoders will be used to build classifiers, which will be evaluated on their performance when classifying vessel sounds into different classes. The goal is to assign one label to each 20 s audio clip in the dataset.


### SRKW call-type datasets (testing - task #3)

This dataset is part of the DCLDE dataset, but will be excluded from the training set in order to be used in this task. Specifically, it includes the subset of files recorded by JASCO/ONC/ECHO in Robert's Bank. Audio segments will be extracted around the calls that have been annotated to the call-type level.

In addition, the HALLO SRKW call catalogue will be used as a reference.

In task 3, call-type classifiers will be built using a similarity ranking strategy, where the encoders are used to produce embeddings for each input (calls from Robert's Bank) and then are compared to the embeddings from the reference catalogue. Finally, the similarity indices are ranked to produce the likely call-type.

## Overview of methods

### Pre-processing

Data will be organized into HDF5 databases containing training, validation, tuning (if relevant to the task), and testing sets. 

Within the databases, audio clips will be stored in their time-domain representation (Float 32 one-dimensional arrays) of the specific duration (2, 5, 20s) and sampling rate.


The transformation from the time domain to a time-frequency representation will occur at training/inference time.

### Encoder training

Ecoders will be trained in a self-supervised manner using variational autoencoders. 










