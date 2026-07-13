import numpy as np
import pyxdf
import mne


class XDFLoader:
    """
    Loader for LiveAmp + Unity XDF recordings.

    Assumptions:
    - EEG stream type = "EEG"
    - Unity marker stream name = "UnityMarkers"
    - Aux1 = ECG
    - ECG polarity must be flipped
    - x_dir, y_dir, z_dir = head movement channels
    - EEG/ECG stored in microvolts
    """

    def __init__(self, xdf_path):

        self.xdf_path = xdf_path

        self.streams = None
        self.header = None

        self.eeg_stream = None
        self.marker_stream = None

        self.raw = None

    # --------------------------------------------------
    # Load XDF
    # --------------------------------------------------

    def load_xdf(self):

        self.streams, self.header = pyxdf.load_xdf(self.xdf_path)

        for stream in self.streams:

            stream_name = stream["info"]["name"][0]
            stream_type = stream["info"]["type"][0]

            if stream_type == "EEG":
                self.eeg_stream = stream

            elif stream_name == "UnityMarkers":
                self.marker_stream = stream

        if self.eeg_stream is None:
            raise RuntimeError("No EEG stream found.")

        return self

    # --------------------------------------------------
    # Create MNE Raw
    # --------------------------------------------------

    def create_raw(self, montage="standard_1020"):

        if self.eeg_stream is None:
            raise RuntimeError(
                "Call load_xdf() before create_raw()."
            )

        # ----------------------------------------------
        # Data
        # ----------------------------------------------

        data = np.asarray(
            self.eeg_stream["time_series"],
            dtype=np.float64
        ).T

        sfreq = float(
            self.eeg_stream["info"]["nominal_srate"][0]
        )

        # ----------------------------------------------
        # Channel names
        # ----------------------------------------------

        channels = (
            self.eeg_stream["info"]["desc"][0]
            ["channels"][0]["channel"]
        )

        ch_names = [
            ch["label"][0]
            for ch in channels
        ]

        # Rename Aux1 -> ECG
        if "Aux1" in ch_names:
            ch_names[ch_names.index("Aux1")] = "ECG"

        # ----------------------------------------------
        # Channel types
        # ----------------------------------------------

        ch_types = ["eeg"] * len(ch_names)

        for idx, ch_name in enumerate(ch_names):

            if ch_name == "ECG":
                ch_types[idx] = "ecg"

            elif ch_name in ["x_dir", "y_dir", "z_dir"]:
                ch_types[idx] = "misc"

        # ----------------------------------------------
        # Unit conversion
        # EEG + ECG: µV -> V
        # ----------------------------------------------

        for idx, ch_type in enumerate(ch_types):

            if ch_type in ["eeg", "ecg"]:
                data[idx] *= 1e-6

        # ----------------------------------------------
        # ECG polarity correction
        # ----------------------------------------------

        if "ECG" in ch_names:

            ecg_idx = ch_names.index("ECG")

            # LiveAmp extension box
            data[ecg_idx] *= -1

        # ----------------------------------------------
        # Create MNE object
        # ----------------------------------------------

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=ch_types
        )

        raw = mne.io.RawArray(
            data,
            info,
            verbose=False
        )

        # ----------------------------------------------
        # Montage
        # ----------------------------------------------

        if montage is not None:

            raw.set_montage(
                mne.channels.make_standard_montage(montage),
                on_missing="ignore"
            )

        # ----------------------------------------------
        # Add Unity markers
        # ----------------------------------------------

        self._add_annotations(raw)

        self.raw = raw

        return raw

    # --------------------------------------------------
    # Annotations
    # --------------------------------------------------

    def _add_annotations(self, raw):

        if self.marker_stream is None:
            return

        eeg_start = self.eeg_stream["time_stamps"][0]

        onsets = (
            np.asarray(self.marker_stream["time_stamps"])
            - eeg_start
        )

        descriptions = []

        for marker in self.marker_stream["time_series"]:

            if isinstance(marker, (list, tuple)):
                descriptions.append(str(marker[0]))
            else:
                descriptions.append(str(marker))

        annotations = mne.Annotations(
            onset=onsets,
            duration=np.zeros(len(onsets)),
            description=descriptions
        )

        raw.set_annotations(annotations)

    # --------------------------------------------------
    # Events
    # --------------------------------------------------

    def get_events(self):

        if self.raw is None:
            raise RuntimeError(
                "Create raw first."
            )

        return mne.events_from_annotations(
            self.raw
        )

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def print_streams(self):

        for idx, stream in enumerate(self.streams):

            print(
                f"{idx}: "
                f"{stream['info']['name'][0]} | "
                f"{stream['info']['type'][0]}"
            )

    def summary(self):

        if self.raw is None:
            print("Raw not created yet.")
            return

        print(self.raw)

        print("\nChannel types:")

        for ch_name, ch_type in zip(
            self.raw.ch_names,
            self.raw.get_channel_types()
        ):
            print(f"{ch_name:<8} {ch_type}")