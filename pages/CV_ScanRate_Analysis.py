import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# === 1. Load all CSV files in the current folder ===
cv_files = sorted(glob.glob("*.csv"))
n_files = len(cv_files)

# === 2. Colormap setup ===
cmap = plt.colormaps.get_cmap('plasma').resampled(n_files)

# === 3. Start plotting ===
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))

for i, file in enumerate(cv_files):
    try:
        # === Read the CSV and skip non-numeric headers if needed ===
        df = pd.read_csv(file)
        
        # === Extract Voltage and Current ===
        voltage = df["Working Electrode (V)"].astype(float).values
        current = df["Current (A)"].astype(float).values * 1e6  # Convert A → µA

        # === Create label from filename ===
        legend_label = os.path.splitext(os.path.basename(file))[0].replace("_", " ")

        # === Plot ===
        plt.plot(voltage, current, label=legend_label, color=cmap(i), linewidth=2)

    except Exception as e:
        print(f"❌ Error in file {file}: {e}")

# === 4. Final plot formatting ===
plt.xlabel("Voltage (V)", color='white')
plt.ylabel("Current (µA)", color='white')
plt.title("CV Overlay Plot (Current in µA)", color='white')
plt.tick_params(colors='white')
plt.legend(fontsize=8)
plt.grid(False)
plt.tight_layout()
plt.savefig("CV_overlay_uA.png", dpi=600, facecolor='black')
plt.show()
