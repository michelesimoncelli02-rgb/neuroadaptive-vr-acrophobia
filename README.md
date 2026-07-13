# neuroadaptive-vr-acrophobia
# Toward a Neuroadaptive Virtual Environment for Acrophobia Exposure Therapy

## Overview

This repository contains the software developed as part of my Master's thesis in Biomedical Engineering, *Toward a Neuroadaptive Virtual Environment for Acrophobia Exposure Therapy*.

The project investigates the development of a closed-loop neuroadaptive virtual reality (VR) system capable of estimating a user's fear state from multimodal physiological signals and dynamically adapting the virtual environment in real time. The research combines brain-computer interface technologies, affective computing, signal processing and machine learning to support adaptive exposure therapy.

The repository includes the complete software pipeline developed throughout the project, from offline data analysis and model training to online real-time inference and Unity integration.

---

## Project Objectives

The main objectives of this work were to:

* Design and conduct a human-subject experimental study in immersive virtual reality.
* Acquire synchronized EEG, ECG and EMG signals during VR exposure.
* Develop a robust preprocessing and feature extraction pipeline for multimodal physiological data.
* Train and validate machine learning models capable of estimating fear-related states.
* Integrate the trained classifier into a real-time neuroadaptive framework.
* Dynamically modify the VR environment according to the participant's estimated emotional state.
* Eventually, test the generalising performance of a cross-subject classifier.

---

## Project Structure

The repository is organised to separate raw experimental data, source code, and documentation.

```text
project/
│
├── data/
│   ├── offline/
│   │   ├── P001.xdf
│   │   ├── P001_questionnaires.txt
│   │   ├── P002.xdf
│   │   └── ...
│   │
│   └── online/
│       ├── participant_001/
│       │   ├── pretest.xdf
│       │   ├── fear_classifier.pkl
│       │   └── results_online_exp.txt
│       └── ...
│
├── docs/
│   ├── Consent_Form.pdf
│   ├── Fear_of_Heights_Questionnaire.pdf
│   ├── Igroup_Presence_Questionnaire.pdf
│   ├── Participant_Information_Sheet_Offline.pdf
│   ├── Participant_Information_Sheet_Online.pdf
│   ├── Virtual_Reality_Sickness_Questionnaire.pdf
│   └── README.md
│
├── src/
│   └── xdf_to_mne_raw.py
│
├── Unity/
│   ├── HeightVolume.cs
│   ├── OnlinePlayerControl.cs
│   ├── PresetPlayerMovement.cs
│   └── README.md
│
├── offline_analysis.py
├── offline_analysis_merge.py
├── pretest_analysis.py
└── online_process.py
```
---

## Requirements

The project was developed and tested using **Python 3.14.3**. It is expected to be compatible with recent Python 3 versions (3.11+), although compatibility with earlier releases has not been tested.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

---

## Instrumentation Used

For EEG recording, the LiveAmp wireless amplifier (Brain Products GmbH, Gilching, Germany) was used, with 32 scalp electrodes mounted on a EasyCap EEG cap (easycap GmbH, Herrsching am Ammersee, Germany) according to the international 10–20 electrode placement system, with a sampling frequency of 250 Hz.

ECG was recorded simultaneously using the LiveAmp Extension Box (Brain Products GmbH), an extension module integrated with the LiveAmp amplifier that enables the acquisition of additional physiological signals alongside EEG, and sampled at 250 Hz.

The virtual environment was presented using an **HP Reverb G2 Omnicept Edition** head-mounted display (HMD) (HP Development Company, California, United States).

---

## Research Workflow

The implemented pipeline consists of the following stages:

1. **Experimental Design**

   * Human-subject protocol
   * Virtual reality exposure scenarios
   * Ethical procedures and informed consent

2. **Data Acquisition**

   * 32-channel EEG recording
   * ECG and EMG acquisition
   * Synchronization through Lab Streaming Layer (LSL)

3. **Offline Analysis**

   * Signal preprocessing
   * Artifact handling
   * Feature extraction
   * Statistical analysis
   * Machine learning model development and validation

4. **Online Neuroadaptive Framework**

   * Real-time signal acquisition
   * Online feature computation
   * Continuous fear-state estimation
   * Communication with Unity through LSL
   * Adaptive modification of the virtual environment

---

## Data Availability

The physiological recordings used in this project are **not publicly available** because they contain human participant data collected under ethical approval and informed consent.

The repository therefore contains only the software required to reproduce the processing pipeline.

---

## Author

**Michele Simoncelli**

Master's Thesis in Biengineering for Neuroscience

University of Padova - Technical University of Graz


