import mne
import os
import matplotlib.pyplot as plt
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
fig = raw.plot(duration = 30, n_channels=raw.info["nchan"], title = "Raw EEG Data", show = False)
fig.savefig(os.path.join(OUT_DIR, "raw_eeg_data.png"), dpi = 300, bbox_inches = "tight")
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

#inspection of boundary events
print(raw.annotations)
print(raw.annotations.description)
print(raw.annotations.onset)
#since the boundary occurs at the start (0 secs), the need for revising the filtering method was not found..

#Filtering and Re Referencing
raw.filter(l_freq=0.5, h_freq=45.0, fir_design="firwin")
raw.notch_filter(freqs=50.0)

# FIX: title now matches what the code actually does
fig_filtered = raw.plot(duration=30, n_channels=raw.info["nchan"], title="Filtered EEG", show=False)
fig_filtered.savefig(os.path.join(OUT_DIR, "filtered_eeg_data.png"), dpi=300, bbox_inches="tight")

print("Filtering completed successfully!")

#EEG Montage
print(raw.get_montage())
montage = mne.channels.make_standard_montage("standard_1020")   # FIX: extra () removed
raw_montage = raw.copy()
raw_montage.set_montage(montage, match_case = False)
fig = raw_montage.plot_sensors(kind = "topomap", show_names = True, show = False)   # FIX: raw_montage, not raw.montage
fig.savefig(os.path.join(OUT_DIR, "montage_sensors.png"), dpi=300, bbox_inches="tight")

fig = raw.compute_psd().plot(show=False)
fig.savefig(os.path.join(OUT_DIR, "psd_raw.png"), dpi=300, bbox_inches="tight")
fig = raw_montage.compute_psd().plot(show=False)
fig.savefig(os.path.join(OUT_DIR, "psd_raw_montage.png"), dpi=300, bbox_inches="tight")

#ICA and artifact removal

from mne.preprocessing import ICA
ica = ICA(n_components= 20, random_state = 97, max_iter= "auto")
ica.fit(raw_montage)   # FIX: fit on the data that has the montage

# FIX: typo + colon removed; save topographies of all components
figs = ica.plot_components(show=False)
if not isinstance(figs, list):
    figs = [figs]
for i, f in enumerate(figs):
    f.savefig(os.path.join(OUT_DIR, f"ica_components_{i}.png"), dpi=300, bbox_inches="tight")

# FIX: raw_montage, saved to disk
fig = ica.plot_sources(raw_montage, show=False)
fig.savefig(os.path.join(OUT_DIR, "ica_sources.png"), dpi=300, bbox_inches="tight")

# One properties page per component (topography, time course, PSD) for inspection
prop_figs = ica.plot_properties(raw_montage, picks=range(ica.n_components_), show=False)
for i, f in enumerate(prop_figs):
    f.savefig(os.path.join(OUT_DIR, f"ica_properties_IC{i:02d}.png"), dpi=300, bbox_inches="tight")

# After inspecting the saved figures, list the artifact components here, e.g. [0, 3]:
# ica.exclude = []
# raw_clean = raw_montage.copy()
# ica.apply(raw_clean)

