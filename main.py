import mne
import numpy as np
import matplotlib.pyplot as plt
import os
mne.set_log_level("Warning")

FILE_PATH  = "sub-001_task-Rest_eeg.set"
save_dir = os.path.expanduser("~/Documents/EEG Project")
os.makedirs(save_dir, exist_ok=True)
raw = mne.io.read_raw(FILE_PATH, preload= True)
print(raw.info)
print(raw)
print(raw.ch_names)

#removing channels specified for exclusion in the doc
raw.drop_channels(["FT9", "PO3", "POz"])

print(f"Channels after removal {len(raw.ch_names)}")
print(raw.ch_names)
raw_original = raw.copy()
fig = raw_original.plot(title = "Before Filtering")
fig.savefig(f"{save_dir}/Before_Filtering.png", dpi = 130, bbox_inches='tight')
#filtering
raw_filtered = raw_original.copy()
raw_filtered.resample(500)
raw_filtered.filter(l_freq = 1, h_freq = 45, fir_design = "firwin")
raw_filtered.notch_filter(freqs = [50], fir_design = "firwin") #powersupplyhum
fig2 = raw_filtered.plot(title = "After Filtering")
fig2.savefig(f"{save_dir}/After_Filtering.png", dpi = 130, bbox_inches='tight')

#ICA
ica = mne.preprocessing.ICA(
    n_components = 20,
    random_state = 97,
    max_iter = "auto"
)

#Fit ICA on filtered EEG
ica.fit(raw_filtered)
print(ica)

#icaplotting
ica_figs = ica.plot_components(show=False)
ica_figs.savefig(
        f"{save_dir}/ICA.png",
        dpi = 130,
        bbox_inches = 'tight'
    )
ica.exclude = [0,3,6,14,16]
ica.apply(raw_filtered)
raw_cleaned = raw_filtered.copy()
print(raw_cleaned)
fig3 = raw_cleaned.plot(title = "After ICA")
fig3.savefig(f"{save_dir}/After_ICA.png")

#PSD
psd = raw_cleaned.compute_psd(fmin = 1.0, fmax = 45.0, method = "welch")
fig_psd = psd.plot()
fig_psd.savefig(f"{save_dir}/PSD.png", dpi = 130, bbox_inches='tight')

# Save Topomap of PSD across key standard frequency bands
fig_topomap = psd.plot_topomap(
    bands={
        "Delta (1-4 Hz)": (1, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-12 Hz)": (8, 12),
        "Beta (12-30 Hz)": (12, 30),
        "Gamma (30-45 Hz)": (30, 45)
    },
    show=False
)
fig_topomap.savefig(f"{save_dir}/PSD_Band_Topomaps.png", dpi=130, bbox_inches="tight")

print("PSD analysis executed and plots saved successfully.")


# Relative band power (%)

bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 45)
}

band_powers = {}

for band, (low, high) in bands.items():
    band_powers[band] = psd.get_data(
        fmin=low,
        fmax=high
    ).mean()

total_power = sum(band_powers.values())

percentages = {
    band: (power / total_power) * 100
    for band, power in band_powers.items()
}

# Print percentages
print("\nRelative Band Power:")
for band, percentage in percentages.items():
    print(f"{band}: {percentage:.2f}%")


# Create bar graph
fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(percentages.keys(), percentages.values())

ax.set_xlabel("Frequency Band")
ax.set_ylabel("Relative Power (%)")
ax.set_title("Relative EEG Band Power")

# Add percentage values above bars
for band, percentage in percentages.items():
    ax.text(
        band,
        percentage,
        f"{percentage:.1f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

# Save directly to EEG Project folder
fig.savefig(
    f"{save_dir}/Relative_Band_Power_Percentages.png",
    dpi=130,
    bbox_inches="tight"
)

plt.close(fig)