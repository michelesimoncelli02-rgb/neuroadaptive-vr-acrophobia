"""
Real-time neuroadaptive control pipeline for the online experiment.

This script implements the complete closed-loop neuroadaptive framework used
during the online virtual reality experiment. It continuously acquires EEG and
ECG signals through Lab Streaming Layer (LSL), preprocesses the physiological
data, extracts neurophysiological and cardiovascular features, and classifies
the participant's current emotional state using a participant-specific machine
learning model trained during the pretest.

Based on the predicted fear level, adaptive commands are streamed back to the
Unity virtual environment to dynamically increase or decrease the exposure
intensity. For each experimental trial, the extracted features, classifier
prediction and corresponding stimulus level are logged for subsequent analysis.

Main processing stages:
    - LSL stream acquisition
    - EEG and ECG preprocessing
    - Blink artifact correction (REBLINCA)
    - Bad channel detection and interpolation
    - Feature extraction
    - Real-time emotional classification
    - Adaptive VR stimulus regulation
    - Experimental results logging

Author:
    Michele Simoncelli

Project:
    Toward a Neuroadaptive Virtual Environment for Acrophobia Exposure Therapy
"""
# %%
import numpy as np
from scipy import signal
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet
import time
from pathlib import Path
import mne
import joblib
from scipy.signal import find_peaks

def main():
  # Loading the pipeline trained in the pretest
  participant = "participant_XXX" 
  participant_path = Path(__file__).resolve().parent / "data" / "online" / f"{participant}"
  model_path = participant_path / "fear_classifier.pkl"
  pipeline = joblib.load(model_path)
  
# %% Streams Setting and Variables Initialisation

  # receive the data Stream
  streams = resolve_stream('type', 'EEG') 
  print("Data stream connected.")

  inlet = StreamInlet(streams[0])

  # Extract valuable information from the LSL stream
  info = inlet.info()
  fs = int(info.nominal_srate())  # [Hz]
  n_channels = info.channel_count()
  
  desc = info.desc()
  channels = desc.child("channels")

  ch_names = []

  ch = channels.child("channel")
  while not ch.empty():
     ch_names.append(ch.child_value("label"))
     ch = ch.next_sibling()

  print(fs)
  print(n_channels)
  print(ch_names)

  if "Aux1" in ch_names:
     ch_names[ch_names.index("Aux1")] = "ECG"

  # handling channels type
  ch_types = ["eeg"] * len(ch_names)
  for idx, ch_name in enumerate(ch_names):
      if ch_name in ["TP9", "TP10"]:
          ch_types[idx] = "emg"

      elif ch_name in ["x_dir", "y_dir", "z_dir"]:
          ch_types[idx] = "misc"

      elif ch_name == "ECG":
          ch_types[idx] = "ecg"
  
  # create a Marker outlet
  info = StreamInfo('CommandStream', 'Markers', 1, 0, 'string', 'myuidw43536')
  outlet = StreamOutlet(info)
  print("Marker stream created.")
  time.sleep(10)
  markername_Increase = ['Go_Up']
  markername_Decrease = ['Go_Down']
  markername_End = ['End_Experiment']
  markername_Start = ["Start_Experiment"]

  # Initialisation: start by increasing the height level to 1.
  outlet.push_sample(markername_Start)
  time.sleep(2)
  outlet.push_sample(markername_Increase)
  
  # Initialise experimental variables
  current_level = 1
  next_level = current_level
  
  # Initialise the results to store
  results = []

  # %% Main Closed Loop Experiment
  n_trials = 10
  for trial in range(n_trials):
      
      print(f"Trial {trial+1}")
    
      buffer = np.empty((0, n_channels))

      while buffer.shape[0] < 20 * fs: # collecting 20 seconds of data

          chunk, _ = inlet.pull_chunk(timeout=1) # (samples, channels)

          if len(chunk):
              buffer = np.vstack((buffer, chunk))

      # now selecting only the last 10 seconds (when the player is not moving)
      buffer = buffer[-10*fs:, :]
      data = buffer.T # (channels, samples) as required by MNE

      # Unit conversion for EEG, ECG, EMG: µV -> V
      for idx, ch_type in enumerate(ch_types):
          if ch_type in ["eeg", "ecg", "emg"]:
              data[idx] *= 1e-6

      # ECG polarity correction
      if "ECG" in ch_names:
          ecg_idx = ch_names.index("ECG")
          # LiveAmp extension box
          data[ecg_idx] *= -1

      if(data.size > 0):
        # transpose to MNE Raw
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=fs,
            ch_types=ch_types
        )
        raw = mne.io.RawArray(data, info)

        # Processing Pipeline
        eeg_raw = raw.copy().pick(picks="eeg")

        # EEG Band-Pass Filtering
        eeg_filt = eeg_raw.copy()
        eeg_filt.notch_filter(
            freqs=50,
            verbose=False
        )
        eeg_filt.filter(
            l_freq=1,
            h_freq=70,
            verbose=False
        )

        # REBLINCA Artifacts Correction
        # Creating a virtual Fpz channel by averaging Fp1 and Fp2
        fpz_virtual_data = (eeg_filt.get_data(picks='Fp1') + eeg_filt.get_data(picks='FP2')) / 2
        # Derive Regr-FPZ (Template for subtraction)
        regr_fpz = mne.filter.filter_data( # 1-7 Hz to isolate blink activity
            fpz_virtual_data,
            fs,
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
        window_size = int(0.1 * fs) # 100ms window
        moving_avg = np.convolve(squared_z, np.ones(window_size)/window_size, mode='same')
        thres_fpz = moving_avg.reshape(1, -1)
        mask = (thres_fpz > 1).ravel()
        # Compute Regression Coefficients (Bn) and apply Correction
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
        # New object with corrected data
        eeg_reb = eeg_filt.copy()
        eeg_reb._data = raw_data
  
        # (NO) Motion Artifact Correction
        eeg_corr = eeg_reb.copy()
    
        # Referencing
        eeg_corr.set_eeg_reference("average", verbose=False)

        # Band-Pass Filtering
        eeg_proc = eeg_corr.copy().filter(
            l_freq=1,
            h_freq=40,
            verbose=False
        )

        # ECG Processing
        ecg_raw = raw.copy().pick_channels(["ECG"])
        # bandpass filtering [0.5:40] Hz
        ecg_filt = ecg_raw.copy().filter(
            l_freq=0.5,
            h_freq=40,
            picks=[ecg_raw.ch_names.index("ECG")],
            verbose=False,
            method="iir"
        )
        # R peaks detection
        '''
        ecg_events, _, _ = mne.preprocessing.find_ecg_events(ecg_filt, ch_name="ECG")
        # R peak times extraction
        r_peaks_samples = ecg_events[:, 0]
        r_peaks_sec = r_peaks_samples / fs
        '''
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
        ''' non reliable online
        for i in range(len(rr_intervals)):
            start = max(0, i - 5)
            stop = min(len(rr_intervals), i + 5)
            local_med = np.median(rr_intervals[start:stop])
            if abs(rr_intervals[i] - local_med) > 0.2 * local_med:
                rr_clean[i] = np.nan
        '''
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
            sfreq=fs,
            ch_types=["misc"]
        )
        channel_hr = mne.io.RawArray(
            hr_clean_interp[np.newaxis, :],
            info_hr
        )
        eeg_proc.add_channels([channel_hr], force_update_info=True)

        # Bad Channels Handling
        clean_data = eeg_proc.copy()
        # EEG only (first 30 channels)
        dat = clean_data.get_data(picks="eeg")   # (channels, time)
        # computing metrics: peak-to-peak and variance and correlation
        ptp = np.ptp(dat, axis=1)
        var = np.var(dat, axis=1)
        global_signal = dat.mean(axis=0)
        corr = np.array([
            np.corrcoef(dat[i], global_signal)[0, 1]
            for i in range(dat.shape[0])
        ])
        # thresholds
        bad_ptp = ptp > np.median(ptp) + 3 * np.std(ptp)
        mad_var = np.median(np.abs(var - np.median(var)))
        bad_var = np.abs(var - np.median(var)) > 3 * mad_var
        bad_corr = corr < 0.2
        # to be marked as bad, a channel must satisfy at least 2 conditions
        bad_channels_idx = np.where((bad_ptp + bad_var + bad_corr) >= 2)[0]
        # convert indices -> channel names
        bad_channels = [clean_data.ch_names[i] for i in bad_channels_idx]
        # mark bads
        clean_data.info['bads'] = bad_channels
        # interpolate
        clean_data.interpolate_bads(reset_bads=True)

        # Feature Extraction
        n_features = 18
        X = np.zeros(n_features)

        # extracting data
        eeg = clean_data.get_data(picks="eeg")
        n_ch = eeg.shape[0] # number of effective eeg channels
        hr = clean_data.get_data(picks="HR")

        # PSD computation over 2s windows with 50% overlap with Welch method
        nperseg = int(2.5 * fs)
        noverlap = int(1.25 * fs)
        psd = np.zeros((n_ch, nperseg // 2 + 1))
        for i in range(0, n_ch):
            f_psd, psd[i,:] = signal.welch(
                eeg[i, :],
                fs=fs,
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
        alpha_Fp1  = sum(psd[clean_data.info["ch_names"].index("Fp1"),  alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
        alpha_Fp2  = sum(psd[clean_data.info["ch_names"].index("FP2"),  alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
        alpha_F3   = sum(psd[clean_data.info["ch_names"].index("F3"),   alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
        alpha_F4   = sum(psd[clean_data.info["ch_names"].index("F4"),   alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
        alpha_F7   = sum(psd[clean_data.info["ch_names"].index("F7"),   alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
        alpha_F8   = sum(psd[clean_data.info["ch_names"].index("F8"),   alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
        alpha_FC5  = sum(psd[clean_data.info["ch_names"].index("FC5"),  alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
        alpha_FC6  = sum(psd[clean_data.info["ch_names"].index("FC6"),  alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
        alpha_FT9  = sum(psd[clean_data.info["ch_names"].index("FT9"),  alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)
        alpha_FT10 = sum(psd[clean_data.info["ch_names"].index("FT10"), alpha_8Hz:alpha_14Hz]) / (alpha_14Hz-alpha_8Hz)

        beta_Fp1  = sum(psd[clean_data.info["ch_names"].index("Fp1"),  beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
        beta_Fp2  = sum(psd[clean_data.info["ch_names"].index("FP2"),  beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
        beta_F3   = sum(psd[clean_data.info["ch_names"].index("F3"),   beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
        beta_F4   = sum(psd[clean_data.info["ch_names"].index("F4"),   beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
        beta_F7   = sum(psd[clean_data.info["ch_names"].index("F7"),   beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
        beta_F8   = sum(psd[clean_data.info["ch_names"].index("F8"),   beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
        beta_FC5  = sum(psd[clean_data.info["ch_names"].index("FC5"),  beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
        beta_FC6  = sum(psd[clean_data.info["ch_names"].index("FC6"),  beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
        beta_FT9  = sum(psd[clean_data.info["ch_names"].index("FT9"),  beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)
        beta_FT10 = sum(psd[clean_data.info["ch_names"].index("FT10"), beta_15Hz:beta_25Hz]) / (beta_25Hz-beta_15Hz)

        frontal_idx = [
            clean_data.ch_names.index(ch)
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
            clean_data.ch_names.index(ch)
            for ch in ["Fp1", "F3", "F7"]
        ]
        frontal_right_idx = [
            clean_data.ch_names.index(ch)
            for ch in ["FP2", "F4", "F8"]
        ]
        relative_frontal_left_alpha = np.mean(psd[frontal_left_idx][:, alpha_8Hz:alpha_14Hz]) / total_power
        relative_frontal_right_alpha = np.mean(psd[frontal_right_idx][:, alpha_8Hz:alpha_14Hz]) / total_power

        # Frontal Alpha Asimmetry
        FAA_Fp1_2  = np.log(alpha_Fp2)-np.log(alpha_Fp1)
        FAA_F3_4   = np.log(alpha_F4)-np.log(alpha_F3)
        FAA_F7_8   = np.log(alpha_F8)-np.log(alpha_F7)
        FAA_FC5_6  = np.log(alpha_FC6)-np.log(alpha_FC5)
        FAA_FT9_10 = np.log(alpha_FT10)-np.log(alpha_FT9)

        # global FAA
        gFAA = np.mean([FAA_F3_4, FAA_F7_8, FAA_FC5_6, FAA_FT9_10])

        # Frontal Beta Asimmetry
        FBA_Fp1_2  = np.log(beta_Fp2)-np.log(beta_Fp1)
        FBA_F3_4   = np.log(beta_F4)-np.log(beta_F3)
        FBA_F7_8   = np.log(beta_F8)-np.log(beta_F7)
        FBA_FC5_6  = np.log(beta_FC6)-np.log(beta_FC5)
        FBA_FT9_10 = np.log(beta_FT10)-np.log(beta_FT9)

        # global FBA
        gFBA = np.mean([FBA_F3_4, FBA_F7_8, FBA_FC5_6, FBA_FT9_10])

        # average HR
        try:
            HR_avg = np.mean(hr)
        except: # if HR data is missing
            HR_avg = 60

        if len(rr_clean) >= 3:

            SDNN = np.std(rr_clean)

            diff_rr = np.diff(rr_clean)

            RMSSD = np.sqrt(
                np.mean(diff_rr ** 2)
            ) if len(diff_rr) > 0 else 0

        else:
            SDNN = 0
            RMSSD = 0

        # saving the features vector
        X[:] = [
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

        # %% Classification 

        prediction = pipeline.predict([X[:]])
        
        # Adaptive exposure logic
        if(prediction == -1): # no fear detected
          
          print('Increase Stimulus')
          
          outlet.push_sample(markername_Increase)

          if next_level < 3:
            next_level = next_level + 1

        elif(prediction == 1): # fear detected
          
          print('Decrease Stimulus')
          
          outlet.push_sample(markername_Decrease)
          
          if next_level > 0:
            next_level = next_level - 1
        
        # Logging results. It stores: EEG and HR features, prediction confidence and stimulus level for each trial.
        results.append(np.concatenate([
            X.copy(),
            [int(prediction[0]), current_level]
        ]))

        # Updating the stimulus level
        current_level = next_level
  
  # After the adaptive loop finishes, the End marker is sent, and the results are saved in a text file.
  time.sleep(2) # wait for 2 seconds before sending the End marker, to ensure that the last trial's data is properly logged and processed.
  
  outlet.push_sample(markername_End)
  
  time.sleep(5) # wait for 5 seconds to ensure that the End marker is sent and received properly before saving the results.
  
  # saving the results
  header = " , ".join(feats + ["prediction", "level"]) # for columns header
  file_path = participant_path / "results_online_exp.txt"
  np.savetxt(
        file_path,
        np.array(results),
        delimiter=" , ",
        header=header,
        comments=""
    )

  time.sleep(5) # wait for 5 seconds to ensure that the results are saved properly before the script ends.

        
if __name__ == '__main__':
    """
    Executes the complete online neuroadaptive experiment.

    The function loads the participant-specific classifier, connects to the
    incoming LSL physiological stream, performs real-time signal processing
    and emotional classification, sends adaptive commands to the Unity
    virtual environment, and stores the experimental results at the end of
    the session.
    """
    main()
