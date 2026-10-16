import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
# Suppress extensive MNE output
mne.set_log_level("WARNING")

#setup
FILE_PATH = "sub-001_task-Rest_eeg (1).set"
OUT_DIR = "eeg_project_figures"
os.makedirs(OUT_DIR, exist_ok=True)

MANUAL_EXCLUDE = None  # Set to a list of integers if you want manual override (e.g., [0, 2])
MAX_BAD_CHANNELS = 6
REJECT_PERCENTILE = 90

def save_figure(fig, filename):
    """Helper function to save figures cleanly and close them."""
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

#loading data
print("Loading raw EEG dataset...")
raw = mne.io.read_raw(FILE_PATH, preload=True)

# Crop first 5 seconds to remove startup/amplifier transients
raw.crop(tmin=5.0)
print(f"Channels: {raw.info['nchan']} | Duration: {round(raw.times[-1], 1)}s")

# Apply bandpass filter (1–45 Hz) and notch filter (50 Hz powerline)
raw.filter(l_freq=1.0, h_freq=45.0, fir_design="firwin")
raw.notch_filter(freqs=50.0)

# Set standard 10-20 electrode layout
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, match_case=False)

# Save initial sensor and filtered PSD plots
save_figure(raw.plot_sensors(kind="topomap", show_names=True, show=False), "montage_sensors.png")
save_figure(raw.compute_psd(fmin=1, fmax=45).plot(show=False), "psd_filtered.png")

#bad channel detection
ch_std = raw.get_data().std(axis=1) * 1e6

# Robust z-score calculation using Median Absolute Deviation (MAD)
median_std = np.median(ch_std)
mad = np.median(np.abs(ch_std - median_std))
robust_z = (ch_std - median_std) / (1.4826 * mad)

# Identify flat channels and noisy outliers
flat_channels = [raw.ch_names[i] for i in range(len(ch_std)) if ch_std[i] < (0.05 * median_std)]
noisy_indices = np.argsort(-robust_z)
noisy_channels = [raw.ch_names[i] for i in noisy_indices if robust_z[i] > 5.0 and raw.ch_names[i] not in flat_channels]

# Combine bad channels up to the max limit
bad_list = flat_channels + noisy_channels[:max(0, MAX_BAD_CHANNELS - len(flat_channels))]
print(f"Identified bad channels for interpolation: {bad_list}")

# Apply bad channels and interpolate
raw.info["bads"] = bad_list
raw.interpolate_bads(reset_bads=True)

# ==========================================
# 4. INDEPENDENT COMPONENT ANALYSIS (ICA)
# ==========================================
print("Fitting ICA model...")
ica = ICA(n_components=20, random_state=97, max_iter="auto")
ica.fit(raw)

# Save ICA component topomaps and properties for inspection
save_figure(ica.plot_components(show=False), "ica_components_overview.png")
save_figure(ica.plot_sources(raw, show=False), "ica_sources.png")

# Automatically detect eye-blink and muscle artifact components
eye_idx, _ = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2", "AF7", "AF8"])
muscle_idx, muscle_scores = ica.find_bads_muscle(raw)

# Combine automated artifact selections
auto_exclude = sorted(list(set(eye_idx) | set(muscle_idx)))
ica.exclude = MANUAL_EXCLUDE if MANUAL_EXCLUDE is not None else auto_exclude
print(f"ICA components marked for exclusion: {ica.exclude}")

# Create copy for comparison, then apply cleaning
raw_before = raw.copy()
ica.apply(raw)

# Save cleaning verification overlay plot
save_figure(ica.plot_overlay(raw_before, exclude=ica.exclude, show=False), "ica_overlay.png")

# Re-reference to average reference
raw.set_eeg_reference("average", projection=False)
raw.save(os.path.join(OUT_DIR, "resting_clean_raw.fif"), overwrite=True)

# Create 2-second epochs and drop high-amplitude outlier epochs
epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
peak_to_peak = np.ptp(epochs.get_data(), axis=2).max(axis=1)
epochs.drop_bad(reject=dict(eeg=np.percentile(peak_to_peak, REJECT_PERCENTILE)))
print(f"Epochs retained after rejection: {len(epochs)}")

# Compute Power Spectral Density (PSD) via Welch's method
psd = epochs.compute_psd(method="welch", fmin=1, fmax=45)
save_figure(psd.plot(show=False), "psd_epochs.png")

# Define frequency bands for power extraction
bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45)
}
save_figure(psd.plot_topomap(bands=bands, normalize=True, show=False), "band_topomaps.png")
print("Pipeline execution complete successfully!")