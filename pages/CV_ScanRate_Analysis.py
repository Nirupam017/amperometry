import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")
st.title("🔬 CV Analysis Tool")

mode = st.sidebar.radio("Select Mode", ["Overlay CV Curves", "Scan Rate Study"])

bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

# ========== OVERLAY MODE ==========
if mode == "Overlay CV Curves":
    st.header("📈 CV Curve Overlay")
    uploaded_files = st.file_uploader("Upload CV CSV Files", type="csv", accept_multiple_files=True)

    if uploaded_files:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)

        for i, file in enumerate(uploaded_files):
            with st.expander(f"Customize {file.name}"):
                label = st.text_input(f"Legend Label for {file.name}", value=file.name, key=f"label_{i}")
                color = st.color_picker(f"Color for {file.name}", key=f"color_{i}")

            df = pd.read_csv(file)
            df.dropna(subset=["Working Electrode (V)", "Current (A)"], inplace=True)
            voltage = df["Working Electrode (V)"].values
            current = df["Current (A)"].values * 1e6

            ax.plot(voltage, current, label=label, color=color, linewidth=2)

        ax.set_xlabel("Voltage (V)", fontsize=14, color=text_color)
        ax.set_ylabel("Current (µA)", fontsize=14, color=text_color)
        ax.set_title("CV Curve Overlay", fontsize=16, fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
        ax.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)
        st.pyplot(fig)

# ========== SCAN RATE MODE ==========
elif mode == "Scan Rate Study":
    st.header("📉 Scan Rate Study with Peak Detection")
    uploaded_files = st.file_uploader("Upload CV Files with Different Scan Rates", type="csv", accept_multiple_files=True)

    if uploaded_files:
        records_anodic = []
        records_cathodic = []

        for i, file in enumerate(uploaded_files):
            with st.expander(f"Details for {file.name}"):
                scan_rate = st.number_input(f"Scan Rate (V/s) for {file.name}", key=f"rate_{i}", min_value=0.0, step=0.01)

            df = pd.read_csv(file)
            df.dropna(subset=["Working Electrode (V)", "Current (A)"], inplace=True)
            voltage = df["Working Electrode (V)"].values
            current = df["Current (A)"].values * 1e6

            # Detect anodic peak (positive current)
            anodic_peaks, _ = find_peaks(current, prominence=0.5)
            cathodic_peaks, _ = find_peaks(-current, prominence=0.5)

            anodic_peak_current = np.nan
            cathodic_peak_current = np.nan

            if len(anodic_peaks) > 0:
                idx = anodic_peaks[np.argmax(current[anodic_peaks])]
                anodic_peak_current = current[idx]
            if len(cathodic_peaks) > 0:
                idx = cathodic_peaks[np.argmax(-current[cathodic_peaks])]
                cathodic_peak_current = current[idx]

            if not np.isnan(anodic_peak_current):
                records_anodic.append((scan_rate, anodic_peak_current))
            if not np.isnan(cathodic_peak_current):
                records_cathodic.append((scan_rate, cathodic_peak_current))

        def plot_peak_vs_scan_rate(records, peak_type):
            if not records:
                st.warning(f"No {peak_type} peaks detected.")
                return

            records = sorted(records)
            scan_rates = np.array([r[0] for r in records]).reshape(-1, 1)
            peak_currents = np.array([r[1] for r in records])

            model = LinearRegression().fit(scan_rates, peak_currents)
            y_pred = model.predict(scan_rates)
            r2 = r2_score(peak_currents, y_pred)

            fig, ax = plt.subplots(figsize=(8, 6), facecolor=bg_color)
            ax.set_facecolor(bg_color)
            ax.scatter(scan_rates, peak_currents, color="dodgerblue", edgecolor="black", s=80, label="Data Points")
            ax.plot(scan_rates, y_pred, color="red", linewidth=2,
                    label=f"Fit: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
            ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(facecolor=bg_color, edgecolor=text_color, boxstyle='round'),
                    color=text_color)
            ax.set_xlabel("Scan Rate (V/s)", fontsize=14, color=text_color)
            ax.set_ylabel(f"{peak_type} Peak Current (µA)", fontsize=14, color=text_color)
            ax.set_title(f"{peak_type} Peak Current vs. Scan Rate", fontsize=16, fontweight="bold", color=text_color)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)
            ax.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)
            st.pyplot(fig)

        plot_peak_vs_scan_rate(records_anodic, "Anodic")
        plot_peak_vs_scan_rate(records_cathodic, "Cathodic")
