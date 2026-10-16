import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
mne.set_log_level("WARNING")

File_Path = "sub-001_task-Rest_eeg (1).set"
OUT_DIR = "eeg_project_figures"
os.makedirs(OUT_DIR, exist_ok=True)
def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)

#gaining general info about the dataset
raw = mne.io.read_raw(File_Path, preload=True)
print(raw.info)
print("Channels:", raw.info["nchan"], "| Samples:", raw.n_times, "| Sampling frequency:", raw.info["sfreq"], "Hz")
save_fig(raw.plot(duration=30, n_channels=raw.info["nchan"], title="Raw EEG Data", show=False), "raw_eeg_data.png")

#inspection of boundary events
print(raw.annotations)
#since the boundary occurs at the start (0 secs), the need for revising the filtering method was not found..
raw.crop(tmin=5.0)

#Filtering and Re Referencing
raw.filter(l_freq=1.0, h_freq=45.0, fir_design="firwin")
raw.notch_filter(freqs=50.0)
save_fig(raw.compute_psd(fmin=1, fmax=45).plot(show=False), "psd_filtered.png")

#EEG Montage
raw.set_montage(mne.channels.make_standard_montage("standard_1020"), match_case=False)
save_fig(raw.plot_sensors(kind="topomap", show_names=True, show=False), "montage_sensors.png")

#channel interpolation
std = raw.get_data().std(axis=1) * 1e6
print("Lowest std (uV):", sorted(zip(std.round(2), raw.ch_names))[:3])
raw.info["bads"] = [raw.ch_names[i] for i in np.argsort(std)[:2]] + ["AF8"]
print("Interpolating:", raw.info["bads"])
raw.interpolate_bads(reset_bads=True)

#ICA and artifact removal
ica = ICA(n_components=20, random_state=97, max_iter="auto")
ica.fit(raw)

figs = ica.plot_components(show=False)
for i, f in enumerate(figs if isinstance(figs, list) else [figs]):
    save_fig(f, f"ica_components_{i}.png")
save_fig(ica.plot_sources(raw, show=False), "ica_sources.png")
for i, f in enumerate(ica.plot_properties(raw, picks=range(ica.n_components_), show=False)):
    save_fig(f, f"ica_properties_IC{i:02d}.png")

ica.exclude = [0, 12, 14, 16, 18]
#Suspected muscle artifact and eye movement based on visual inspection

raw_before = raw.copy()          #uncleaned copy, taken BEFORE apply
ica.apply(raw)
print("Excluded:", ica.exclude)

for name, r in [("before", raw_before), ("after", raw)]:
    save_fig(r.plot(picks=["Fp1", "Fp2", "Fz", "Cz", "Oz"], start=10, duration=20,
                    scalings=dict(eeg=100e-6), title=f"{name} ICA", show=False), f"{name}_ica.png")
    save_fig(r.compute_psd(fmin=1, fmax=45).plot(show=False), f"psd_{name}_ica.png")

raw.set_eeg_reference("average", projection=False)

#epoching and PSD calculation
epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
ptp = np.ptp(epochs.get_data(), axis=2).max(axis=1)
epochs.drop_bad(reject=dict(eeg=np.percentile(ptp, 90)))
print("Epochs kept:", len(epochs))

psd = epochs.compute_psd(method="welch", fmin=1, fmax=45, n_fft=1000)
save_fig(psd.plot(show=False), "psd_epochs.png")
bands = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 45)}
save_fig(psd.plot_topomap(bands=bands, normalize=True, show=False), "band_topomaps.png")

#bar graphical representation of the PSD Analysis
spec = psd.average()
f = spec.freqs
d = spec.get_data() * 1e12
power = np.array([d[:, (f >= lo) & (f < hi)].sum(axis=1) * (f[1] - f[0])
                  for lo, hi in bands.values()]).T
rel = power / power.sum(axis=1, keepdims=True) * 100

# bar graphs
for data, ylabel, name in [(power, "Power (uV^2)", "bar_abs_power.png"),
                           (rel, "Relative power (%)", "bar_rel_power.png")]:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(list(bands), data.mean(0), yerr=data.std(0), capsize=5, edgecolor="black")
    ax.set_ylabel(ylabel)
    ax.set_title("Mean band power across channels")
    save_fig(fig, name)

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(rel.mean(0), labels=list(bands), autopct="%1.1f%%", startangle=90)
save_fig(fig, "pie_rel_power.png")