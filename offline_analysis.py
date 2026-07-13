"""
Offline fear classification pipeline for subject-specific model development.

This script implements the complete offline processing pipeline used to train
and evaluate participant-specific fear classifiers from synchronized EEG and
ECG recordings acquired during the offline validation experiment. The pipeline
includes signal preprocessing, physiological feature extraction, and machine
learning model comparison.

The main processing stages are:

1. Load EEG, ECG, EMG, and event-marker streams from XDF recordings and
   convert them into MNE Raw objects.
2. Preprocess EEG signals by applying notch and band-pass filtering.
3. Correct ocular artifacts using the REBLINCA algorithm.
4. (Optional) Correct motion artifacts using regression based on TP9 and TP10
   reference channels.
5. Re-reference EEG signals to the common average and apply a final
   1–40 Hz band-pass filter.
6. Preprocess ECG signals, detect R peaks, compute heart rate (HR), RR
   intervals, and heart-rate variability (HRV) metrics, and synchronise the
   HR signal with the EEG recordings.
7. Segment the recordings into 10-second epochs corresponding to individual
   experimental trials.
8. Detect and interpolate bad EEG channels independently for each epoch.
9. Divide each epoch into analysis windows.
10. Extract EEG and ECG features, including:
      - Heart rate (HR)
      - SDNN
      - RMSSD
      - Frontal alpha asymmetry (FAA)
      - Frontal beta asymmetry (FBA)
      - Relative frontal alpha power
      - Relative frontal beta power
      - Frontal alpha/beta power ratio
11. Assign binary labels corresponding to fear and no-fear conditions.
12. Train and evaluate multiple machine-learning classifiers using a pipeline
    composed of feature standardisation, univariate feature selection, and
    classification.
13. Assess classification performance using GroupKFold cross-validation to
    prevent data leakage between windows belonging to the same trial.
14. Save classification results for each participant.

The script processes all participants listed in the `participants` variable
independently, producing participant-specific feature matrices, classifiers,
and performance reports.
"""
# %%
############# IMPORTING MODULES #############
import numpy as np
import mne
from pathlib import Path
from scipy import signal
import matplotlib
matplotlib.use("QtAgg") # only needed for interactive plotting during development
import matplotlib.pyplot as plt
plt.ion()
from scipy.signal import find_peaks
from src.xdf_to_mne_raw import XDFLoader

# the offline acquisitions are named accordingly to the participant's code "PXXX"
participants = ["P001", "P002", "P003", "P004"]

# levels: 0 -> 2 -> 1 -> 3 -> 2 -> 0 -> 1 -> 3 -> 1 -> 3 -> 2 -> 0
levels = [0, 2, 1, 3, 2, 0, 1, 3, 1, 3, 2, 0]

for participant in participants:

    # %%
    ############# DATA LOADING AND EXPLORATION #############
    
    # Directory containing this script
    script_dir = Path(__file__).resolve().parent

    # File inside another folder
    file_path = script_dir / "data" / "offline" / f"{participant}.xdf"

    loader = XDFLoader(file_path)
    loader.load_xdf()

    # creating a MNE Raw object
    raw = loader.create_raw()

    # setting as "emg" type the channels used for motion artifact correction  
    raw.set_channel_types({
        "TP9": "emg",
        "TP10": "emg"
        })
    '''
    # data exploration
    loader.summary()
    print()
    loader.print_streams()
    print()
    '''
    events, event_id = loader.get_events()
    '''
    print()
    raw.plot()
    print()
    print(raw)
    print(raw.info)
    print()
    print(raw.info.keys())
    print()
    print(raw.info["chs"][0].keys())
    print()
    '''

    # %%
    ############# EEG BAND PASS FILTERING #############

    # extracting eeg data
    eeg_raw = raw.copy().pick(picks="eeg")

    # Computing and plotting PSD to spot frequency irregularities
    psd_eeg_raw = eeg_raw.compute_psd(tmax=np.inf, fmax=eeg_raw.info["sfreq"]/2)
    '''
    fig = psd_eeg_raw.plot(average=True, amplitude=False, picks="data", exclude="bads")
    # add some arrows at 50 Hz and its harmonics:
    for ax in fig.axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (50, 100):
            idx = np.searchsorted(freqs, freq)
            ax.arrow(
                x=freqs[idx],
                y=psds[idx] + 18,
                dx=0,
                dy=-12,
                color="red",
                width=0.1,
                head_width=3,
                length_includes_head=True,
            )
    plt.title("PSD of Raw ECG")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (µV²/Hz)")
    '''
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
    '''
    # filtered PSD
    psd_eeg_filt = eeg_filt.compute_psd(tmax=np.inf, fmax=eeg_raw.info["sfreq"]/2)
    
    # comparison and visualisation of the PSDs
    fig, ax = plt.subplots()
    psd_eeg_raw.plot(
        average=True,
        axes=ax,
        spatial_colors=False
    )
    psd_eeg_filt.plot(
        average=True,
        axes=ax,
        spatial_colors=False
    )
    ax.legend(["Raw", "Filtered"])
    ax.set_title("PSD Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (µV²/Hz)")

    # comparison and visualisation of the data
    ch_to_plot = ["Fz", "Cz", "Pz"] # for a few channels
    #duration = 10  # seconds
    fig, axes = plt.subplots(
        len(ch_to_plot),
        1,
        figsize=(14, 6)
    )
    for ax, ch in zip(axes, ch_to_plot):
        ax.plot(eeg_raw.times, eeg_raw.get_data(picks=ch).T)
        ax.plot(eeg_filt.times, eeg_filt.get_data(picks=ch).T)
        ax.legend(["Raw", "Filtered"])
        ax.set_ylabel("Amplitude (mV)")
        ax.set_title(ch)
    ax.set_xlabel("Time (s)")
    '''

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
    '''
    # comparison and visualisation of the data
    ch_to_plot = ["Fz", "Cz", "Pz"] # for a few channels
    #duration = 10  # seconds
    fig, axes = plt.subplots(
        len(ch_to_plot),
        1,
        figsize=(14, 6)
    )
    for ax, ch in zip(axes, ch_to_plot):
        ax.plot(eeg_filt.times, eeg_filt.get_data(picks=ch).T)
        ax.plot(eeg_reb.times, eeg_reb.get_data(picks=ch).T)
        ax.legend(["Filtered", "REBLINCA corrected"])
        ax.set_ylabel("Amplitude (mV)")
        ax.set_title(ch)
    ax.set_xlabel("Time (s)")
    '''

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
    
    # comparison and visualisation of the data
    ch_to_plot = ["Fz", "Cz", "Oz"] # for a few channels
    #duration = 10  # seconds
    fig, axes = plt.subplots(
        len(ch_to_plot),
        1,
        figsize=(14, 6)
    )
    for ax, ch in zip(axes, ch_to_plot):
        ax.plot(eeg_reb.times, eeg_reb.get_data(picks=ch).T)
        ax.plot(eeg_corr.times, eeg_corr.get_data(picks=ch).T)
        ax.legend(["REBLINCA corrected", "REBLINCA + EMG corrected"])
        ax.set_ylabel("Amplitude (mV)")
        ax.set_title(ch)
    ax.set_xlabel("Time (s)")
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
    '''
    # comparison and visualisation of the data
    ch_to_plot = ["Fz", "Cz", "Oz"] # for a few channels
    #duration = 10  # seconds
    fig, axes = plt.subplots(
        len(ch_to_plot),
        1,
        figsize=(14, 6)
    )
    for ax, ch in zip(axes, ch_to_plot):
        ax.plot(eeg_corr.times, eeg_corr.get_data(picks=ch).T)
        ax.plot(eeg_proc.times, eeg_proc.get_data(picks=ch).T)
        ax.legend(["REBLINCA + EMG corrected", "Fully processed"])
        ax.set_ylabel("Amplitude (mV)")
        ax.set_title(ch)
    ax.set_xlabel("Time (s)")
    '''

    # %%
    ############# ECG PROCESSING #############
    # extracting the ECG signal
    ecg_raw = raw.copy().pick_channels(["ECG"])
    '''
    # visualising the ECG data
    plt.figure()
    plt.plot(ecg_raw.times, ecg_raw.get_data()[0])
    plt.title("Raw ECG")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    psd_ecg_raw = ecg_raw.compute_psd(tmax=np.inf, fmax=ecg_raw.info["sfreq"]/2, picks="ecg")
    psd_ecg_raw.plot(average=True, amplitude=False, picks="ecg", exclude="bads")
    '''
    # bandpass filtering [0.5:40] Hz
    ecg_filt = ecg_raw.copy().filter(
        l_freq=0.5,
        h_freq=40,
        picks=[ecg_raw.ch_names.index("ECG")],
        verbose=False,
        method="iir"
    )
    '''
    plt.figure()
    plt.plot(ecg_filt.times, ecg_filt.get_data()[0])
    plt.title("Filtered ECG")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    '''
    '''
    # R peaks detection
    ecg_events, _, _ = mne.preprocessing.find_ecg_events(ecg_filt, ch_name="ECG")

    # R peak times extraction
    sfreq = ecg_filt.info["sfreq"]
    r_peaks_samples = ecg_events[:, 0]
    r_peaks_sec = r_peaks_samples / sfreq

    # RR intervals computation
    rr_intervals = np.diff(r_peaks_sec)  # in seconds
    '''
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
    rr_intervals = np.diff(r_peaks_sec)

    # HR computation
    hr = 60 / rr_intervals
    '''
    # visualising HR over time
    plt.figure()
    plt.plot(r_peaks_sec[1:], hr)
    plt.xlabel("Time (s)")
    plt.ylabel("Heart rate (bpm)")
    plt.title("Instantaneous HR")
    '''
    # artifacts correction
    rr_clean = rr_intervals.copy()
    for i in range(len(rr_intervals)):
        start = max(0, i - 5)
        stop = min(len(rr_intervals), i + 5)
        local_med = np.median(rr_intervals[start:stop])
        if abs(rr_intervals[i] - local_med) > 0.2 * local_med:
            rr_clean[i] = np.nan
    hr_clean = 60 / rr_clean
    '''
    # comparison visualisation
    plt.plot(r_peaks_sec[1:], hr_clean)
    plt.xlabel("Time (s)")
    plt.ylabel("Heart rate (bpm)")
    plt.title("Instantaneous HR") 
    plt.show()
    plt.legend(["Without artifact correction", "With artifact correction"])
    '''
    '''
    # Plot HR
    plt.figure(figsize=(15, 5))
    plt.plot(r_peaks_sec[1:], hr_clean, label="Heart rate")
    # Counters to associate each Start_Trial with its level
    trial_idx = 0
    for ev in events:
        sample = ev[0]
        code = ev[2]
        t = sample / sfreq
        # Start Trial
        if code == event_id["Start_Trial"]:
            plt.axvline(
                t,
                color="green",
                linestyle="--",
                linewidth=1.5
            )
            # Display trial level
            if trial_idx < len(levels):
                plt.text(
                    t,
                    np.nanmax(hr_clean) + 2,
                    f"L{levels[trial_idx]}",
                    rotation=90,
                    color="green",
                    fontsize=9,
                    va="bottom",
                    ha="center"
                )
            trial_idx += 1
        # End Trial
        elif code == event_id["End_Trial"]:
            plt.axvline(
                t,
                color="red",
                linestyle="--",
                linewidth=1.5
            )
    plt.xlabel("Time (s)")
    plt.ylabel("Heart rate (bpm)")
    plt.title("Heart rate with trial boundaries")
    plt.legend(["HR", "Start trial", "End trial"])
    plt.tight_layout()
    plt.show()
    '''
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
    '''
    # data and events are already synchronised
    print(events)
    print(event_id)
    eeg_proc.plot()
    '''
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
    '''
    # printing and visualising useful informations and insights
    print(epochs)
    print()
    print(epochs.info)
    print(epochs.info["ch_names"])
    print()
    print(f"Number of epochs: {len(epochs)}")

    # first epoch
    epochs[0].plot()
    # every epochs
    epochs.plot()
    '''

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
        '''
        print()
        print(f"Epoch {n_ep+1} processed, {len(bad_channels)} bad channels spotted")
        '''

    # %%
    ############# WINDOWING #############

    # indeces for trial with respective levels: 0 or 3
    #idx_trial = np.where(np.isin(levels, [0, 3]))[0]

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

    for ep_idx in range(len(clean_epochs)): # idx_trial:

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
        alpha_14Hz = (np.abs(f_psd - 14)).argmin() # pick the frequency closest to 14 Hz

        # beta [15-25] Hz
        beta_15Hz = (np.abs(f_psd - 15)).argmin()   # pick the frequency closest to 15 Hz
        beta_25Hz = (np.abs(f_psd - 25)).argmin() # pick the frequency closest to 25 Hz

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

    '''
    for i in range(X.shape[1]):
        line = (
            f"{feats[i]:25s}"
            f"{abs(np.mean(X[y==-1, i])-np.mean(X[y==1, i]))/np.std(X[:, i])}"
        )
        print(line.strip())
    '''
    '''
    for i, feat in enumerate(feats):
        plt.figure(figsize=(4, 4))
        plt.boxplot(
            [X[y == -1, i], X[y == 1, i]],
            tick_labels=["No Fear", "Fear"]
        )
        plt.ylabel(feat)
        plt.title(feat)
        plt.tight_layout()
    '''

    # checking features' value and (neuro)physiological lifelikeness
    feature_ranges = {}
    for i, feat in enumerate(feats):

        feature_ranges[feat] = {
            "mean": np.mean(X[:, i]),
            "std": np.std(X[:, i]),
            "min": np.min(X[:, i]),
            "max": np.max(X[:, i])
        }
    # printing statistical values
    for feat, vals in feature_ranges.items():

        print(
            f"{feat:30s}"
            f" mean={vals['mean']:.3f}"
            f" std={vals['std']:.3f}"
            f" min={vals['min']:.3f}"
            f" max={vals['max']:.3f}"
        )

    # %%
    ############# CLASSIFICATION #############

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sklearn.model_selection import LeaveOneOut, StratifiedKFold, GroupKFold
    from sklearn.model_selection import cross_validate

    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_selection import SelectKBest, f_classif

    # classifiers to compare
    models = {
        "SVM_LINEAR_C0.0001":
            SVC(
                kernel="linear",
                C=0.0001,
                class_weight="balanced"
            ),

        "SVM_LINEAR_C0.001":
            SVC(
                kernel="linear",
                C=0.001,
                class_weight="balanced"
            ),

        "SVM_LINEAR_C0.01":
            SVC(
                kernel="linear",
                C=0.01,
                class_weight="balanced"
            ),

        "SVM_LINEAR_C0.1":
            SVC(
                kernel="linear",
                C=0.1,
                class_weight="balanced"
            ),

        "SVM_LINEAR_C1":
            SVC(
                kernel="linear",
                C=1,
                class_weight="balanced"
            ),

        "SVM_LINEAR_C10":
            SVC(
                kernel="linear",
                C=10,
                class_weight="balanced"
            ),

        "SVM_LINEAR_C100":
            SVC(
                kernel="linear",
                C=100,
                class_weight="balanced"
            ),

        "SVM_RBF_C0.1_Gscale":
            SVC(
                kernel="rbf",
                C=0.1,
                gamma="scale",
                class_weight="balanced"
            ),

        "SVM_RBF_C1_Gscale":
            SVC(
                kernel="rbf",
                C=1,
                gamma="scale",
                class_weight="balanced"
            ),

        "SVM_RBF_C10_Gscale":
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced"
            ),

        "SVM_RBF_C0.1_Gauto":
            SVC(
                kernel="rbf",
                C=0.1,
                gamma="auto",
                class_weight="balanced"
            ),

        "SVM_RBF_C1_Gauto":
            SVC(
                kernel="rbf",
                C=1,
                gamma="auto",
                class_weight="balanced"
            ),

        "SVM_RBF_C10_Gauto":
            SVC(
                kernel="rbf",
                C=10,
                gamma="auto",
                class_weight="balanced"
            ),

        "sLDA":
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto"
            ),

        "RF_10":
            RandomForestClassifier(
                n_estimators=10,
                random_state=42
            ),

        "RF_50":
            RandomForestClassifier(
                n_estimators=50,
                random_state=42
            ),

        "RF_100":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),

        "RF_50_md2_msl2_mfsqrt":
            RandomForestClassifier(
                n_estimators=50,
                max_depth=2,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42
            ),

        "RF_100_md2_msl2_mfsqrt":
            RandomForestClassifier(
                n_estimators=100,
                max_depth=2,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42
            ),

        "RF_200_md3_msl2_mfsqrt":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=3,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42
            ),

        "RF_200_md5_msl2_mfsqrt":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42
            ),

        "RF_300_md3_msl1_mfsqrt":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42
            ),

        "RF_300_mdNone_msl4_mfsqrt":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=42
            ),

        "RF_100_md3_msl3_mflog2":
            RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=3,
                max_features="log2",
                random_state=42
            ),

        "RF_500_md5_msl4_mfsqrt":
            RandomForestClassifier(
                n_estimators=500,
                max_depth=5,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=42
            ),

        "GB_100_lr0.05_md2":            
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2
            )
    }

    scoring = {
    "acc": "accuracy",
    "bal_acc": "balanced_accuracy",
    "f1": "f1"
    }
    cv = GroupKFold(n_splits=4)
    '''
    # for manually selecting a subset of features
    X_small = X[:, [
        feats.index("HR_avg"),
        feats.index("gFAA"),
        feats.index("gFBA"),
        feats.index("relative_frontal_alpha"),
        feats.index("frontal_alpha_beta_ratio")
    ]]
    '''

    results_file = Path(f"{participant}_classification_results.txt")

    with open(results_file, "w") as f:

        f.write("CLASSIFICATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"\nSUBJECT: {participant}\n")
        f.write("-" * 80 + "\n")

        print(f"\nProcessing {participant}")

        # feature selection
        F, p = f_classif(X, y)
        for feat, fval, pval in zip(feats, F, p):
            print(f"{feat:30s} F={fval:.3f} p={pval:.3f}")
        print()

        for model_name, model in models.items():

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(f_classif, k=5)), # using only the top k features
                ("clf", model)
            ])

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
                f"{model_name:30s}"
                f" ACC={acc:.3f}"
                f" BAL_ACC={bal_acc:.3f}"
                f" F1={f1:.3f}\n"
            )

            print(line.strip())
            f.write(line)

            f.write("\n")

    plt.close("all")
