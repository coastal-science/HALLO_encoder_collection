This project aims to create a collection of audio encoders for downstream underwater acoustic tasks, such as detection, classification, and localization of marine animal sounds.
There have been few efforts to produce encoders useful to underwater acoustics, such as [SurfPerch](https://www.kaggle.com/models/google/surfperch), but they often carry constraints inherited from other environments and tasks, such as speech analysis or bird acoustics.
For example, SurfPerch represents audio as 5s Mel-scaled spectrograms, ranging from 0 to 16 kHz, a representation often used in speech recognition and that has been carried to many terrestrial acoustics tasks by convenience, but that might not be optimal for some underwater (and even terrestrial) acoustic tasks.

The question behind this project is: Can different representations produce better encoders for underwater acoustic tasks?

## Representations

Initially, we will explore 3 ways of representing an audio signal:
1) A magnitude spectrogram (with linear frequency scale)
2) A time-similarity matrix, computed as the Pearson correlation coefficients of the time bins in the spectrogram (1)
3) A frequency-similarity matrix, computed as the Pearson correlation coefficients of the frequency bins in the spectrogram (1)

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

**Blue whale detection**
Use the encoders to build a detector for the stereotypical low-frequency blue whale calls. 

**Vessel classification**
Use  the encoders to build a classifiers that identify the type of vessel producing noise.

**KW call-type classification**
Use encoders to classify tonal calls of Southern Resident Killer Whale into call-types. Use a similarity approach to compare inputs to an established call-type catalogue. 



## Datasets


