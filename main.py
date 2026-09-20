import mne
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from pyprep.find_noisy_channels import NoisyChannels
import os

mne.set_log_level("WARNING")

FILE_PATH = "sub-001_task-Rest_eeg(1).set"
OUT_DIR = "eeg_project_figures"
os.makedirs(OUT_DIR, exist_ok=True)

raw = mne.io.read_raw_eeglab(FILE_PATH, preload=True)
print(f"Loaded: {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz, "
      f"{raw.n_times / raw.info['sfreq']:.1f} s")


raw.drop_channels(["FT9", "PO3", "POz"])
print(f"Channels after removal: {len(raw.ch_names)}")

raw_original = raw.copy()
fig = raw_original.plot(title="Before Filtering", show=False)
fig.savefig(f"{OUT_DIR}/01_before_filtering.png", dpi=130, bbox_inches="tight")
plt.close(fig)

psd_check = raw_original.compute_psd(fmax=80, method="welch")
psds_c, freqs_c = psd_check.get_data(return_freqs=True)
mean_c = psds_c.mean(axis=0)
print("\nPower near 50Hz vs 60Hz (dB) on unfiltered data:")
for f0 in [50, 60]:
    idx = np.argmin(np.abs(freqs_c - f0))
    print(f"  {f0} Hz: {10 * np.log10(mean_c[idx]):.1f} dB")
# Neither showed a sharp spike here, so no notch filter is applied below.


raw_filtered = raw_original.copy()
raw_filtered.filter(l_freq=1.0, h_freq=45.0, fir_design="firwin")

fig2 = raw_filtered.plot(title="After Filtering", show=False)
fig2.savefig(f"{OUT_DIR}/02_after_filtering.png", dpi=130, bbox_inches="tight")
plt.close(fig2)


noisy_detector = NoisyChannels(raw_filtered, random_state=97)
noisy_detector.find_all_bads()
bad_channels = noisy_detector.get_bads()

print(f"\nBad by deviation: {noisy_detector.bad_by_deviation}")
print(f"Bad by high-frequency noise: {noisy_detector.bad_by_hf_noise}")
print(f"Bad by low correlation with neighbors: {noisy_detector.bad_by_correlation}")
print(f"Combined bad channels: {bad_channels}")

raw_filtered.info["bads"] = bad_channels
if bad_channels:
    raw_filtered.interpolate_bads(reset_bads=True)
    print(f"Interpolated {len(bad_channels)} bad channel(s) from neighbors.")
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter="auto")
ica.fit(raw_filtered)
sources = ica.get_sources(raw_filtered).get_data()
sfreq = raw_filtered.info["sfreq"]
frontal_proxy = raw_filtered.copy().pick(["Fp1", "Fp2"]).get_data().mean(axis=0)
blink_corr = [abs(np.corrcoef(sources[i], frontal_proxy)[0, 1]) for i in range(sources.shape[0])]
print("\n=== Blink correlation (r vs Fp1/Fp2 proxy) ===")
for i, r in enumerate(blink_corr):
    print(f"IC{i}: r = {r:.3f}")
BLINK_THRESHOLD = 0.4
blink_components = [i for i, r in enumerate(blink_corr) if r > BLINK_THRESHOLD]
print(f"Blink components (r > {BLINK_THRESHOLD}): {blink_components}")
muscle_ratios = []
for i in range(sources.shape[0]):
    f, p = welch(sources[i], fs=sfreq, nperseg=int(sfreq * 2))
    total = p[(f >= 1) & (f <= 45)].sum()
    hf = p[(f >= 20) & (f <= 45)].sum()
    muscle_ratios.append(hf / total)

print("\n=== Muscle-artifact screening (high-freq power ratio, DIAGNOSTIC ONLY) ===")
for i, r in enumerate(muscle_ratios):
    flag = "  <-- candidate" if r > 0.7 and i not in blink_components else ""
    print(f"IC{i}: ratio = {r:.3f}{flag}")

muscle_candidates = [i for i, r in enumerate(muscle_ratios) if r > 0.7 and i not in blink_components]
print(f"\nTop muscle-artifact candidates (ratio > 0.7): {muscle_candidates}")
print("These are NOT auto-excluded. Inspect their saved property plots "
      "(broad, non-dipolar topography + rising-with-frequency spectrum = "
      "real muscle artifact) before adding any to ica.exclude by hand.")

if muscle_candidates:
    props_fig = ica.plot_properties(raw_filtered, picks=muscle_candidates, show=False)
    for idx, f in zip(muscle_candidates, props_fig):
        f.savefig(f"{OUT_DIR}/ica_muscle_candidate_IC{idx}.png", dpi=110, bbox_inches="tight")
        plt.close(f)

ica.exclude = blink_components

ica_figs = ica.plot_components(show=False)
ica_figs.savefig(f"{OUT_DIR}/03_ica_components.png", dpi=130, bbox_inches="tight")
plt.close(ica_figs)

ica.apply(raw_filtered)
raw_cleaned = raw_filtered.copy()

fig3 = raw_cleaned.plot(title="After ICA", show=False)
fig3.savefig(f"{OUT_DIR}/04_after_ica.png", dpi=130, bbox_inches="tight")
plt.close(fig3)
raw_cleaned.set_eeg_reference("average")

psd = raw_cleaned.compute_psd(fmin=1.0, fmax=45.0, method="welch")
fig_psd = psd.plot(show=False)
fig_psd.savefig(f"{OUT_DIR}/05_psd.png", dpi=130, bbox_inches="tight")
plt.close(fig_psd)
fig_topomap = psd.plot_topomap(
    bands={
        "Delta (1-4 Hz)": (1, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-12 Hz)": (8, 12),
        "Beta (12-30 Hz)": (12, 30),
        "Gamma (30-45 Hz)": (30, 45),
    },
    show=False,
)
fig_topomap.savefig(f"{OUT_DIR}/06_psd_band_topomaps.png", dpi=130, bbox_inches="tight")
plt.close(fig_topomap)

bands = {
    "Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 12),
    "Beta": (12, 30), "Gamma": (30, 45),
}
band_powers = {band: psd.get_data(fmin=lo, fmax=hi).mean() for band, (lo, hi) in bands.items()}
total_power = sum(band_powers.values())
percentages = {band: (p / total_power) * 100 for band, p in band_powers.items()}

print("\n=== Relative Band Power ===")
for band, pct in percentages.items():
    print(f"{band}: {pct:.2f}%")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(percentages.keys(), percentages.values(), color="#2F5496")
ax.set_xlabel("Frequency Band")
ax.set_ylabel("Relative Power (%)")
ax.set_title("Relative EEG Band Power (final corrected pipeline)")
for band, pct in percentages.items():
    ax.text(band, pct, f"{pct:.1f}%", ha="center", va="bottom")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/07_relative_band_power.png", dpi=130, bbox_inches="tight")
plt.close(fig)

print(f"\nDone. Figures saved to: {OUT_DIR}/")
