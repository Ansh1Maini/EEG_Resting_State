import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

mne.set_log_level("WARNING")

FILE_PATH = "sub-001_task-Rest_eeg (1).set"
OUT_DIR = "eeg_project_figures"
os.makedirs(OUT_DIR, exist_ok=True)

MANUAL_EXCLUDE = [0, 2, 7, 14]
REJECT_PERCENTILE = 90

def save_figure(fig, filename):
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

raw = mne.io.read_raw(FILE_PATH, preload=True)
raw.crop(tmin=5.0)
print(f"Channels: {raw.info['nchan']} | Duration: {round(raw.times[-1], 1)}s")

raw.filter(l_freq=1.0, h_freq=45.0, fir_design="firwin")
raw.notch_filter(freqs=50.0)

montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, match_case=False)

save_figure(raw.plot_sensors(kind="topomap", show_names=True, show=False), "montage_sensors.png")
save_figure(raw.compute_psd(fmin=1, fmax=45).plot(show=False), "psd_filtered.png")

raw.info["bads"] = ["CP1", "P1", "AF8"]
raw.interpolate_bads(reset_bads=True)

ica = ICA(n_components=20, random_state=97, max_iter="auto")
ica.fit(raw)

save_figure(ica.plot_components(show=False), "ica_components_overview.png")
save_figure(ica.plot_sources(raw, show=False), "ica_sources.png")

ica.exclude = MANUAL_EXCLUDE
print(f"ICA components marked for manual exclusion: {ica.exclude}")

raw_before = raw.copy()
ica.apply(raw)

save_figure(ica.plot_overlay(raw_before, exclude=ica.exclude, show=False), "ica_overlay.png")

raw.set_eeg_reference("average", projection=False)
raw.save(os.path.join(OUT_DIR, "resting_clean_raw.fif"), overwrite=True)

epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
peak_to_peak = np.ptp(epochs.get_data(), axis=2).max(axis=1)
epochs.drop_bad(reject=dict(eeg=np.percentile(peak_to_peak, REJECT_PERCENTILE)))
print(f"Epochs retained after rejection: {len(epochs)}")

psd = epochs.compute_psd(method="welch", fmin=1, fmax=45)
save_figure(psd.plot(show=False), "psd_epochs.png")

bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45)
}

save_figure(psd.plot_topomap(bands=bands, normalize=True, show=False), "band_topomaps.png")

spec = psd.average()
f = spec.freqs
d = spec.get_data() * 1e12
power = np.array([d[:, (f >= lo) & (f < hi)].sum(axis=1) * (f[1] - f[0])
                  for lo, hi in bands.values()]).T
rel = power / power.sum(axis=1, keepdims=True) * 100

pd.DataFrame({"band": list(bands), "relative_%": rel.mean(0).round(1), "sd_%": rel.std(0).round(1)}
             ).to_csv(os.path.join(OUT_DIR, "band_power_table.csv"), index=False)

for data, ylabel, name in [(power, "Power (uV^2)", "bar_abs_power.png"),
                           (rel, "Relative power (%)", "bar_rel_power.png")]:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(list(bands), data.mean(0), yerr=data.std(0), capsize=5, edgecolor="black")
    ax.set_ylabel(ylabel)
    ax.set_title("Mean band power across channels")
    save_figure(fig, name)

print("Pipeline execution completed successfully!")