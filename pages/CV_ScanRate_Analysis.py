import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(layout="wide")
st.title("🔬 CV Analysis Tool")

# Sidebar mode selector
mode = st.sidebar.radio("Select Mode", ["Overlay CV Curves", "Scan Rate Study"])

# Theme controls
bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

# Helper function for peak detection
def detect_peaks(voltage, current):
    smoothed = savgol_filter(current, window_length=31, polyorder=3)
    peaks_pos, _ = find_peaks(smoothed, prominence=1)
    peaks_neg, _ = find_peaks(-smoothed, prominence=1)
    return smoothed, peaks_pos, peaks_neg

# Overlay CV Curves
if mode == "Overlay CV Curves":
    st.header("📈 CV Curve Overlay")
    uploaded_files = st.file_uploader("Upload CV CSV Files", type="csv", accept_multiple_files=True)

    if uploaded_files:
        cv_data = []
        for i, file in enumerate(uploaded_files):
            with st.expander(f"Customize {file.name}"):
                label = st.text_input(f"Legend Label for {file.name}", value=file.name, key=f"label_{i}")
                color = st.color_picker(f"Color for {file.name}", key=f"color_{i}")
            df = pd.read_csv(file)
            df.dropna(subset=["Working Electrode (V)", "Current (A)"], inplace=True)
            voltage = df["Working Electrode (V)"].values
            current = df["Current (A)"].values * 1e6  # A to µA
            smoothed, peaks_pos, peaks_neg = detect_peaks(voltage, current)
            cv_data.append((voltage, current, smoothed, peaks_pos, peaks_neg, label, color))

        # Plot CV overlays
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        for voltage, current, smoothed, peaks_pos, peaks_neg, label, color in cv_data:
            ax.plot(voltage, smoothed, label=label, color=color, linewidth=2)
            ax.plot(voltage[peaks_pos], smoothed[peaks_pos], 'o', color='green', label="Anodic Peaks")
            ax.plot(voltage[peaks_neg], smoothed[peaks_neg], 'x', color='magenta', label="Cathodic Peaks")

        ax.set_xlabel("Voltage (V)", fontsize=14, color=text_color)
        ax.set_ylabel("Current (µA)", fontsize=14, color=text_color)
        ax.set_title("CV Curve Overlay with Peak Detection", fontsize=16, fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
        ax.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color, loc='best')
        st.pyplot(fig)

# Scan Rate Study
elif mode == "Scan Rate Study":
    st.header("📉 Scan Rate Study")
    uploaded_files = st.file_uploader("Upload CV Files with Different Scan Rates", type="csv", accept_multiple_files=True)

    if uploaded_files:
        records = []
        for i, file in enumerate(uploaded_files):
            with st.expander(f"Details for {file.name}"):
                scan_rate = st.number_input(f"Scan Rate (V/s) for {file.name}", key=f"rate_{i}", min_value=0.0, step=0.01)
            df = pd.read_csv(file)
            df.dropna(subset=["Working Electrode (V)", "Current (A)"], inplace=True)
            voltage = df["Working Electrode (V)"].values
            current = df["Current (A)"].values * 1e6
            smoothed, peaks_pos, peaks_neg = detect_peaks(voltage, current)

            anodic_peak = smoothed[peaks_pos[np.argmax(smoothed[peaks_pos])]] if len(peaks_pos) > 0 else np.nan
            cathodic_peak = smoothed[peaks_neg[np.argmin(smoothed[peaks_neg])]] if len(peaks_neg) > 0 else np.nan

            st.write(f"Anodic Peak: {anodic_peak:.2f} µA, Cathodic Peak: {cathodic_peak:.2f} µA")
            records.append((scan_rate, anodic_peak, cathodic_peak))

        # Filter NaNs
        records = [r for r in records if not np.isnan(r[1]) and not np.isnan(r[2])]
        if records:
            scan_rates = np.array([r[0] for r in records]).reshape(-1, 1)
            anodic_peaks = np.array([r[1] for r in records])
            cathodic_peaks = np.array([r[2] for r in records])

            model_a = LinearRegression().fit(scan_rates, anodic_peaks)
            model_c = LinearRegression().fit(scan_rates, cathodic_peaks)

            pred_a = model_a.predict(scan_rates)
            pred_c = model_c.predict(scan_rates)

            fig, ax = plt.subplots(figsize=(8, 6), facecolor=bg_color)
            ax.set_facecolor(bg_color)
            ax.scatter(scan_rates, anodic_peaks, color='red', edgecolor='black', label='Anodic Peak')
            ax.plot(scan_rates, pred_a, color='darkred', linestyle='--')

            ax.scatter(scan_rates, cathodic_peaks, color='blue', edgecolor='black', label='Cathodic Peak')
            ax.plot(scan_rates, pred_c, color='navy', linestyle='--')

            ax.set_xlabel("Scan Rate (V/s)", fontsize=14, color=text_color)
            ax.set_ylabel("Peak Current (µA)", fontsize=14, color=text_color)
            ax.set_title("Peak Current vs Scan Rate", fontsize=16, fontweight="bold", color=text_color)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)
            ax.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)
            st.pyplot(fig)
        else:
            st.warning("No valid peak data found to plot.")
