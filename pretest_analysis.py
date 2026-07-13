"""
Offline preprocessing, feature extraction and classifier training.

This script processes the physiological recordings collected during the pretest
experiment to generate the participant-specific machine learning model used in
the online neuroadaptive experiment.

The pipeline is the same one applied in the offline_analysis.py.

The participant-specific trained classification pipeline is finally saved and 
later employed during the online adaptive virtual reality experiment.

Author:
    Michele Simoncelli

Project:
    Toward a Neuroadaptive Virtual Environment for Acrophobia Exposure Therapy
"""
# %%
############# IMPORTING MODULES #############
import numpy as np
import mne
from pathlib import Path
from scipy import signal
import matplotlib
matplotlib.use("QtAgg") # only needed for interactive plotting during development
from scipy.signal import find_peaks
from src.xdf_to_mne_raw import XDFLoader

# levels: 0 -> 2 -> 1 -> 3 -> 2 -> 0 -> 1 -> 3 -> 1 -> 3 -> 2 -> 0
levels = [0, 2, 1, 3, 2, 0, 1, 3, 1, 3, 2, 0]

# %%
############# DATA LOADING AND EXPLORATION #############

# Directory containing this script
script_dir = Path(__file__).resolve().parent

# input the participant's code
participant = "participant_XXX"

# File inside another folder
participant_path = script_dir / "data" / "online" / f"{participant}"
file_path = participant_path / "Pretest.xdf"

loader = XDFLoader(file_path)
loader.load_xdf()

# creating a MNE Raw object
raw = loader.create_raw()

# setting as "emg" type the channels used for motion artifact correction  
raw.set_channel_types({
    "TP9": "emg",
    "TP10": "emg"
    })

events, event_id = loader.get_events()

# %%
############# EEG BAND PASS FILTERING #############

# extracting eeg data
eeg_raw = raw.copy().pick(picks="eeg")

# Copy raw data
eeg_filt = eeg_raw.copy()

# Notch filter at 50 Hz
eeg_filt.notch_filter(
    freqs=50,
    verbose=False
)

# Band-pass filter (1–70 Hz)
eeg_filt.filter(
    l_freq=1,
    h_freq=70,
    verbose=False
)

# %%
############# REBLINCA ARTIFACTS CORRECTION #############

# Creating a virtual Fpz channel by averaging Fp1 and Fp2
# Extract the data for the two frontal electrodes
fp1_data = eeg_filt.get_data(picks='Fp1')
fp2_data = eeg_filt.get_data(picks='FP2')
# Compute the average to create the 'virtual Fpz'
fpz_virtual_data = (fp1_data + fp2_data) / 2

# Derive Regr-FPZ (Template for subtraction)
# Filter between 1-7 Hz to isolate blink activity
sfreq = eeg_filt.info['sfreq']
regr_fpz = mne.filter.filter_data(
    fpz_virtual_data,
    sfreq,
    l_freq=1.0,
    h_freq=7.0,
    verbose=False,
)

# Derive Thres-FPZ (Signal for blink detection)
# Step 1: Derivative
deriv = np.diff(regr_fpz, prepend=regr_fpz[:, 0:1])
# Step 2 & 3: Z-score normalization and Squaring
z_deriv = (deriv - np.mean(deriv)) / np.std(deriv)
squared_z = (z_deriv**2).flatten()
# Step 4: Moving average to smooth the two blink peaks
window_size = int(0.1 * sfreq) # 100ms window
moving_avg = np.convolve(squared_z, np.ones(window_size)/window_size, mode='same')
thres_fpz = moving_avg.reshape(1, -1)
mask = (thres_fpz > 1).ravel()

# 4. Compute Regression Coefficients (Bn) and Apply Correction
# We only want to correct the EEG channels
eeg_indices = mne.pick_types(eeg_filt.info, eeg=True)
raw_data = eeg_filt.get_data() # to be consistent with literature nomenclature

for idx in eeg_indices:
    # Estimate Bn using least-squares regression
    # Bn = (Raw_Channel @ Regr_FPZ.T) / (Regr_FPZ @ Regr_FPZ.T)
    ch_data = raw_data[idx, :]
    ref_data = regr_fpz[0, :]
    bn = np.dot(ch_data, ref_data) / np.dot(ref_data, ref_data)
    
    # Conditional Correction: only subtract if Thres-FPZ > 1
    # This preserves signal during blink-free segment
    raw_data[idx, mask] -= bn * regr_fpz[0, mask]

# 5. New object with corrected data
eeg_reb = eeg_filt.copy()
eeg_reb._data = raw_data


# %%
############# MOTION ARTIFACTS CORRECTION #############
'''
# extracting useful data
eeg_data = eeg_reb.get_data(picks="eeg")
emg_data = raw.get_data(picks="emg")  # TP9, TP10

# High-pass EMG reference to isolate muscle activity
emg_hp = mne.filter.filter_data(
    emg_data,
    sfreq=raw.info["sfreq"],
    l_freq=20,
    h_freq=None,
    verbose=False
)

# Average TP9 and TP10
emg_ref = emg_hp.mean(axis=0)

# Copy EEG data
data_corr = eeg_data.copy()

for ch in range(eeg_data.shape[0]):
    y = eeg_data[ch]

    # Least-squares coefficient
    beta = np.dot(y, emg_ref) / np.dot(emg_ref, emg_ref)

    # Remove EMG contribution
    data_corr[ch] = y - beta * emg_ref

eeg_corr = eeg_reb.copy()

eeg_idx = mne.pick_types(eeg_corr.info, eeg=True)
eeg_corr._data[eeg_idx] = data_corr
'''
eeg_corr = eeg_reb.copy()

# %%
############# REFERENCING #############
eeg_corr.set_eeg_reference("average", verbose=False)

# %%
############# BANDPASS FILTERING #############
# band pass filter to remove undesired frequencies [1:40] Hz
eeg_proc = eeg_corr.copy().filter(
    l_freq=1,
    h_freq=40,
    verbose=False
)

# %%
############# ECG PROCESSING #############
# extracting the ECG signal
ecg_raw = raw.copy().pick_channels(["ECG"], verbose=False)

# bandpass filtering [0.5:40] Hz
ecg_filt = ecg_raw.copy().filter(
    l_freq=0.5,
    h_freq=40,
    picks=[ecg_raw.ch_names.index("ECG")],
    verbose=False,
    method="iir"
)
# R peaks detection
ecg = ecg_filt.get_data().ravel()
sfreq = ecg_raw.info["sfreq"]
distance = int(0.4 * sfreq)   # max 150 bpm

peaks, _ = find_peaks(
    ecg,
    distance=distance,
    prominence=np.std(ecg)
)
r_peaks_sec = peaks / sfreq 

# RR intervals computation
rr_intervals = np.diff(r_peaks_sec)  # in seconds

# HR computation
hr = 60 / rr_intervals

# artifacts correction
rr_clean = rr_intervals.copy()
for i in range(len(rr_intervals)):
    start = max(0, i - 5)
    stop = min(len(rr_intervals), i + 5)
    local_med = np.median(rr_intervals[start:stop])
    if abs(rr_intervals[i] - local_med) > 0.2 * local_med:
        rr_clean[i] = np.nan
hr_clean = 60 / rr_clean

# so to avoid nan propagation
valid = ~np.isnan(hr_clean)

# interpolating hr_clean so to have the same samples as the eeg
hr_clean_interp = np.interp(
    eeg_proc.times,   # target times
    r_peaks_sec[1:][valid],         # original HR timestamps
    hr_clean[valid]          # original HR values
)

# adding the hr_clean channel to the raw object of processed data
info_hr = mne.create_info(
    ch_names=["HR"],
    sfreq=raw.info["sfreq"],
    ch_types=["misc"]
)
channel_hr = mne.io.RawArray(
    hr_clean_interp[np.newaxis, :],
    info_hr
)
eeg_proc.add_channels([channel_hr], force_update_info=True)

# %%
############# EPOCHING #############
# creating the epochs, starting from "Start_Trial" and lasting for 10 seconds
epochs = mne.Epochs(
    eeg_proc,
    events,
    event_id={"Start_Trial": event_id["Start_Trial"]},
    tmin=0,
    tmax=10,
    baseline=None,
    preload=True
)

# %%
############# BAD CHANNELS HANDLING #############

# eeg bad channels detection and interpolation on an epoch base
clean_epochs = epochs.copy()

for n_ep, epoch in enumerate(clean_epochs):

    # EEG only (first 30 channels)
    X = epoch[:30, :]   # (channels, time)

    ptp = np.ptp(X, axis=1)
    var = np.var(X, axis=1)

    global_signal = X.mean(axis=0)
    corr = np.array([
        np.corrcoef(X[i], global_signal)[0, 1]
        for i in range(X.shape[0])
    ])

    # thresholds
    bad_ptp = ptp > np.median(ptp) + 3 * np.std(ptp)

    mad_var = np.median(np.abs(var - np.median(var)))
    bad_var = np.abs(var - np.median(var)) > 3 * mad_var

    bad_corr = corr < 0.2

    bad_channels_idx = np.where((bad_ptp + bad_var + bad_corr) >= 2)[0]

    # convert indices -> channel names
    bad_channels = [epoch.ch_names[i] for i in bad_channels_idx]

    # mark bads
    clean_epochs[n_ep].info['bads'] = bad_channels

    # interpolate
    clean_epochs.interpolate_bads(reset_bads=True)

# %%
############# WINDOWING #############

# mantaining the information of the channels
hr_idx = clean_epochs.ch_names.index("HR")
eeg_idx = mne.pick_types(clean_epochs.info, eeg=True)
# before starting using an array structure (window)

window_length = 10.0      # seconds
window_step = 10.0        # seconds

sfreq = clean_epochs.info["sfreq"]

win_samples  = int(window_length * sfreq)
step_samples = int(window_step * sfreq)

all_window_times = []
all_windows = []
all_labels = []
all_groups = []

for ep_idx in range(len(clean_epochs)):

    epoch = clean_epochs[ep_idx]

    data = epoch.get_data()[0]      # (channels, samples)

    n_samples = data.shape[1]

    start = 0

    epoch_start_sec = events[ep_idx, 0] / sfreq

    while start + win_samples <= n_samples:

        stop = start + win_samples

        window_data = data[:, start:stop]

        all_windows.append(window_data) # data

        all_labels.append(
            1 if levels[ep_idx] in [2, 3] else -1 # label
        )

        window_start_sec = epoch_start_sec + start / sfreq

        all_window_times.append(window_start_sec)

        # all windows from same trial share same group
        all_groups.append(ep_idx)
        # so that we can avoid data leakage later

        start += step_samples

# %%
############# FEATURE EXTRACTION #############
n_features = 18
X = np.zeros((len(all_windows), n_features))

for n_win, window in enumerate(all_windows): # working on a window-scale

    # extracting data
    eeg = window[eeg_idx, :]
    n_ch = eeg.shape[0] # number of effective eeg channels
    hr = window[hr_idx, :]

    # PSD computation over 2.5s windows with 50% overlap with Welch method
    nperseg = int(2.5 * sfreq)
    noverlap = int(1.25 * sfreq)
    psd = np.zeros((n_ch, nperseg // 2 + 1))
    for i in range(0, n_ch):
        f_psd, psd[i,:] = signal.welch(
            eeg[i, :],
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap
            )

    # alpha [8-14] Hz
    alpha_8Hz  = (np.abs(f_psd - 8)).argmin()   # pick the frequency closest to 8 Hz
    alpha_14Hz = (np.abs(f_psd - 14)).argmin()  # pick the frequency closest to 14 Hz

    # beta [15-25] Hz
    beta_15Hz = (np.abs(f_psd - 15)).argmin()   # pick the frequency closest to 15 Hz
    beta_25Hz = (np.abs(f_psd - 25)).argmin()   # pick the frequency closest to 25 Hz

    # signal extraction
    alpha_Fp1  = sum(psd[epoch.info["ch_names"].index("Fp1"),  alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
    alpha_Fp2  = sum(psd[epoch.info["ch_names"].index("FP2"),  alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
    alpha_F3   = sum(psd[epoch.info["ch_names"].index("F3"),   alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
    alpha_F4   = sum(psd[epoch.info["ch_names"].index("F4"),   alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
    alpha_F7   = sum(psd[epoch.info["ch_names"].index("F7"),   alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
    alpha_F8   = sum(psd[epoch.info["ch_names"].index("F8"),   alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
    alpha_FC5  = sum(psd[epoch.info["ch_names"].index("FC5"),  alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
    alpha_FC6  = sum(psd[epoch.info["ch_names"].index("FC6"),  alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
    alpha_FT9  = sum(psd[epoch.info["ch_names"].index("FT9"),  alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
    alpha_FT10 = sum(psd[epoch.info["ch_names"].index("FT10"), alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)

    beta_Fp1  = sum(psd[epoch.info["ch_names"].index("Fp1"),  beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
    beta_Fp2  = sum(psd[epoch.info["ch_names"].index("FP2"),  beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
    beta_F3   = sum(psd[epoch.info["ch_names"].index("F3"),   beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
    beta_F4   = sum(psd[epoch.info["ch_names"].index("F4"),   beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
    beta_F7   = sum(psd[epoch.info["ch_names"].index("F7"),   beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
    beta_F8   = sum(psd[epoch.info["ch_names"].index("F8"),   beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
    beta_FC5  = sum(psd[epoch.info["ch_names"].index("FC5"),  beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
    beta_FC6  = sum(psd[epoch.info["ch_names"].index("FC6"),  beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
    beta_FT9  = sum(psd[epoch.info["ch_names"].index("FT9"),  beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
    beta_FT10 = sum(psd[epoch.info["ch_names"].index("FT10"), beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)

    frontal_idx = [
        epoch.ch_names.index(ch)
        for ch in ["Fp1", "FP2", "F3", "F4", "F7", "F8", "Fz"]
    ]
    alpha_power = np.mean(
        psd[frontal_idx][:, alpha_8Hz:alpha_14Hz]
    )
    beta_power = np.mean(
        psd[frontal_idx][:, beta_15Hz:beta_25Hz]
    )
    total_power = np.mean(
        psd[frontal_idx][:, :]
    )

    # Relative Frontal Alpha power
    relative_frontal_alpha =   alpha_power / total_power
    # Relative Frontal Beta power
    relative_frontal_beta =    beta_power / total_power
    # Frontal Alpha/Beta power ration
    frontal_alpha_beta_ratio = alpha_power / beta_power

    # Relative Alpha power for subclusters of electrodes
    frontal_left_idx = [
        epoch.ch_names.index(ch)
        for ch in ["Fp1", "F3", "F7"]
    ]
    frontal_right_idx = [
        epoch.ch_names.index(ch)
        for ch in ["FP2", "F4", "F8"]
    ]
    relative_frontal_left_alpha = np.mean(psd[frontal_left_idx][:, alpha_8Hz:alpha_14Hz]) / total_power
    relative_frontal_right_alpha = np.mean(psd[frontal_right_idx][:, alpha_8Hz:alpha_14Hz]) / total_power

    # Frontal Alpha Asimmetry
    FAA_Fp1_2 =  np.log(alpha_Fp2)  - np.log(alpha_Fp1)
    FAA_F3_4 =   np.log(alpha_F4)   - np.log(alpha_F3)
    FAA_F7_8 =   np.log(alpha_F8)   - np.log(alpha_F7)
    FAA_FC5_6 =  np.log(alpha_FC6)  - np.log(alpha_FC5)
    FAA_FT9_10 = np.log(alpha_FT10) - np.log(alpha_FT9)

    # global FAA
    gFAA = np.mean([FAA_F3_4, FAA_F7_8, FAA_FC5_6, FAA_FT9_10])

    # Frontal Beta Asimmetry
    FBA_Fp1_2 =  np.log(beta_Fp2)  - np.log(beta_Fp1)
    FBA_F3_4 =   np.log(beta_F4)   - np.log(beta_F3)
    FBA_F7_8 =   np.log(beta_F8)   - np.log(beta_F7)
    FBA_FC5_6 =  np.log(beta_FC6)  - np.log(beta_FC5)
    FBA_FT9_10 = np.log(beta_FT10) - np.log(beta_FT9)

    # global FBA
    gFBA = np.mean([FBA_F3_4, FBA_F7_8, FBA_FC5_6, FBA_FT9_10])

    # average HR
    try:
        HR_avg = np.mean(hr)
    except: # if HR data is missing
        HR_avg = 60

    # Window start/stop in absolute time
    window_start_sec = all_window_times[n_win]
    window_stop_sec = window_start_sec + window_length

    # RR intervals whose second R peak falls inside window
    rr_mask = (
        (r_peaks_sec[1:] >= window_start_sec)
        & (r_peaks_sec[1:] < window_stop_sec)
    )

    rr_window = rr_clean[rr_mask]
    rr_window = rr_window[~np.isnan(rr_window)]

    if len(rr_window) >= 3:

        SDNN = np.std(rr_window)

        diff_rr = np.diff(rr_window)

        RMSSD = np.sqrt(
            np.mean(diff_rr ** 2)
        ) if len(diff_rr) > 0 else 0

    else:
        SDNN = 0
        RMSSD = 0

    # saving the features
    X[n_win, :] = [
        HR_avg,
        SDNN,
        RMSSD,
        FAA_Fp1_2,
        FAA_F3_4,
        FAA_F7_8,
        FAA_FC5_6,
        FAA_FT9_10,
        FBA_Fp1_2,
        FBA_F3_4,
        FBA_F7_8,
        FBA_FC5_6,
        FBA_FT9_10,
        #gFAA,
        #gFBA,
        relative_frontal_alpha,
        relative_frontal_left_alpha,
        relative_frontal_right_alpha,
        relative_frontal_beta,
        frontal_alpha_beta_ratio
    ]

# defining the labels
y = np.array(all_labels)
# and the groups
groups = np.array(all_groups)

feats = [
        "HR_avg",
        "SDNN",
        "RMSSD",
        "FAA_Fp1_2",
        "FAA_F3_4",
        "FAA_F7_8",
        "FAA_FC5_6",
        "FAA_FT9_10",
        "FBA_Fp1_2",
        "FBA_F3_4",
        "FBA_F7_8",
        "FBA_FC5_6",
        "FBA_FT9_10",
        #"gFAA",
        #"gFBA",
        "relative_frontal_alpha",
        "relative_frontal_left_alpha",
        "relative_frontal_right_alpha",
        "relative_frontal_beta",
        "frontal_alpha_beta_ratio"
        ]

# %%
############# CLASSIFICATION #############

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

import joblib

model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2
        )

scoring = {
"acc": "accuracy",
"bal_acc": "balanced_accuracy",
"f1": "f1"
}
cv = GroupKFold(n_splits=4)

results_file = Path(f"Pretest_classification_results.txt")

with open(results_file, "w") as f:

    f.write("CLASSIFICATION RESULTS\n")
    f.write("=" * 80 + "\n\n")

    # feature selection
    F, p = f_classif(X, y)
    for feat, fval, pval in zip(feats, F, p):
        print(f"{feat:30s} F={fval:.3f} p={pval:.3f}")
    print()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=5)), # using only the top k features
        ("clf", model)
    ])

    # Evaluating the model with CV to check how well it generalises
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring=scoring,
        n_jobs=-1
    )

    acc = np.mean(scores["test_acc"])
    bal_acc = np.mean(scores["test_bal_acc"])
    f1 = np.mean(scores["test_f1"])

    line = (
        f" ACC={acc:.3f}"
        f" BAL_ACC={bal_acc:.3f}"
        f" F1={f1:.3f}\n"
    )

    print(line.strip())
    f.write(line)

    f.write("\n")

    # Training the model on the whole dataset
    pipeline.fit(X, y)

    # Saving the trained model
    model_path = participant_path / "fear_classifier.pkl"
    joblib.dump(pipeline, model_path)
