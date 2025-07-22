# Neural Spike Recordings from Auditory Cortex

This dataset contains multi-unit spike recordings from the auditory cortex of Squirrel monkeys, collected during passive listening tasks. Monkeys were presented with a variety of auditory stimuli, but this documentation focuses on two key stimulus classes used in our study:

- **TIMIT**: English sentences from the TIMIT corpus
- **mVocs**: Monkey vocalizations (e.g., grunts, screams)


---

## 📁 Dataset Structure

The dataset is organized as follows:

```
root/
├── stimuli/ # Metadata about the auditory stimuli
├── sessions/ # One subdirectory per recording session
│ ├── 180413/
│ │ ├── *_MUsp.mat # One file per recording channel
│ │ └── ... # Additional channel files
│ ├── 180420/
│ └── ... # More sessions
├── session_metadata.yml # Annotations about sessions (location, area, hemisphere, etc.)
├── LICENSE.txt # Dataset license
└── README.md # This file
```


### 🧠 Notes

- `stimuli/` contains:
  - `out_sentence_details_timit_all_loudness.mat`
  - `SqMoPhys_MVOCStimcodes.mat`
  - `MonkVocs_15Blocks.wav`
- Each session directory (e.g., `180413/`) contains:
  - Multiple `*_MUsp.mat` files (each corresponding to a recording channel)
- `session_metadata.yml` includes:
  - Session-level annotations (brain area, hemisphere, bad session exclusions)
  - Stimulus repetition counts
  - 2D coordinates for each session

---

## 📄 Description of `*_MUsp.mat` Files

Each `*_MUsp.mat` file contains spike data from one recording channel. The following variables are populated when the file is loaded:

### 🔢 `spike` struct

The `spike` struct contains spike-level data. Key fields:

- `events`: `s × t` matrix of spike waveforms  
  - `s`: number of detected spikes  
  - `t`: number of samples per waveform  
- `spktimes`: vector of spike times (in seconds from recording start)
- `amStimcode`, `fmStimcode`, `dmrStimcode`, `mVocStimcode`, `timitStimcode`: vectors of length `s`, each specifying the stimulus played at the time of the spike
- `trial`: vector of length `s`, mapping each spike to the trial in which it occurred

### 🧪 `trial` struct

The `trial` struct contains trial-level metadata. Key fields:

- `stimon`: vector of stimulus onset times (in seconds)
- `amStimcode`, `fmStimcode`, `dmrStimcode`, `mVocStimcode`, `timitStimcode`: stimulus code vectors, one per trial

---

> **Note:** All stimulus and trial IDs follow **MATLAB-style indexing** (i.e., start from 1).  
> When working in Python, make sure to adjust for zero-based indexing if needed.

---

## 🎧 TIMIT Stimuli

TIMIT stimuli consist of English sentences used during stimulus playback.

- Metadata is stored in: `stimuli/out_sentence_details_timit_all_loudness.mat`
- Main variable: `sentdet` — a list of structs (one per stimulus)

Each element in `sentdet` contains the following key fields:

- `sound`: waveform (numpy array)
- `soundf`: sampling rate (Hz)
- `duration`: total stimulus duration including silence (in seconds)
- `befaft`: tuple `(bef, aft)` specifying silence before and after the sentence
---

## 🐒 mVocs Stimuli

Monkey vocalizations (e.g. grunts, screams) were played as naturalistic stimuli.

- `stimuli/SqMoPhys_MVOCStimcodes.mat`  
  - `mVocsStimCodes`: list of stimulus IDs (MATLAB indexing)
- `stimuli/MonkVocs_15Blocks.wav`  
  - A `.wav` file concatenating all monkey vocalizations with silent gaps

---

## 📜 License

This dataset is shared under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

---

## 📄 Publications Using This Dataset

This dataset has been used in the following studies:

- **Ahmed, B. et al. (2025).**  
  *Deep Neural Networks Explain Spiking Activity in Auditory Cortex.*  
  *PLOS Computational Biology (In press)*

- **Downer, J. D., Bigelow, J., Runfeldt, M., & Malone, B. J. (2021).**  
  *Temporally Precise Population Coding of Dynamic Sounds by Auditory Cortex.*  
  [Journal of Neurophysiology](https://journals.physiology.org/doi/full/10.1152/jn.00709.2020)

---

## 🔗 Citation

If you use this dataset, please cite the dataset itself:

> *Multi-channel auditory cortex electrophysiology in squirrel monkey.*  
> doi:10.5281/zenodo.16175377

This dataset is archived on [Zenodo](https://doi.org/10.5281/zenodo.16175377) and licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).



