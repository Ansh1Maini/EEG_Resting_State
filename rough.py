import mne
import os
import numpy as np
import pandas as pd
mne.set_log_level("WARNING")

File_Path = "sub-001_task-Rest_eeg (1).set"
OUT_DIR = "eeg_project_figures"
os.makedirs(OUT_DIR, exist_ok=True)

#gaining general info about the dataset
raw = mne.io.read_raw(File_Path, preload=True)
print(raw.info)
print(raw.info["nchan"])
print(raw.info["ch_names"])
fig = raw.plot(duration=10, n_channels=raw.info["nchan"], title="Raw EEG Data", show=False)
fig.savefig(f"{OUT_DIR}/raw_eeg_data.png", dpi=130, bbox_inches="tight")

#Extract EEG data and Filtering

print(raw.get_data())
print(type(raw))
print(raw.get_data().min())
print(raw.get_data().max())
print(raw.get_data().mean())
print(raw.get_data().shape)
print("Channels:", len(raw.ch_names))
print("Samples:", raw.n_times)
print("Sampling frequency:", raw.info["sfreq"], "Hz")


