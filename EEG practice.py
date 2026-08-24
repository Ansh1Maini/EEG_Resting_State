import mne
import numpy as np
import os
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import matplotlib.pyplot as plt


mne.set_log_level("WARNING")

FILE_PATH = "1_N170.set"
OUT_DIR = "n170_figures"
os.makedirs(OUT_DIR, exist_ok=True)

