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
            cv_data.append((voltage, current, label, color))

        # Plot CV overlays
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        for voltage, current, label, color in cv_data:
            ax.plot(voltage, current, label=label, color=color, linewidth=2)

        ax.set_xlabel("Voltage (V)", fontsize=14, color=text_color)
        ax.set_ylabel("Current (µA)", fontsize=14, color=text_color)
        ax.set_title("CV Curve Overlay", fontsize=16, fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
        ax.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)
        st.pyplot(fig)

# Scan Rate Study
elif mode == "Scan Rate Study":
    st.header("📉 Scan Rate Study")
    uploaded_files = st.file_uploader("Upload CV Files with Different Scan Rates", type="csv", accept_multiple_files=True)

    if uploaded_files:
        anodic_peaks = []
        cathodic_peaks = []

        for i, file in enumerate(uploaded_files):
            with st.expander(f"Details for {file.name}"):
                scan_rate = st.number_input(f"Scan Rate (V/s) for {file.name}", key=f"rate_{i}", min_value=0.0, step=0.01)

            df = pd.read_csv(file)
            df.dropna(subset=["Working Electrode (V)", "Current (A)"], inplace=True)
            voltage = df["Working Electrode (V)"].values
            current_raw = df["Current (A)"].values * 1e6  # A to µA
            current = savgol_filter(current_raw, 31, 3)  # Smoothing

            # Peak detection
            anodic_idx, _ = find_peaks(current, prominence=1)
            cathodic_idx, _ = find_peaks(-current, prominence=1)

            anodic_current = current[anodic_idx].max() if len(anodic_idx) > 0 else None
            cathodic_current = current[cathodic_idx].min() if len(cathodic_idx) > 0 else None

            if anodic_current is not None and cathodic_current is not None:
                anodic_peaks.append((scan_rate, anodic_current))
                cathodic_peaks.append((scan_rate, cathodic_current))
            else:
                st.warning(f"No peaks found in {file.name}")

            # Plot with peaks
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(voltage, current, label="Smoothed CV", color="navy")
            if len(anodic_idx) > 0:
                ax.plot(voltage[anodic_idx], current[anodic_idx], "ro", label="Anodic Peaks")
            if len(cathodic_idx) > 0:
                ax.plot(voltage[cathodic_idx], current[cathodic_idx], "go", label="Cathodic Peaks")

            ax.set_title(f"Peak Detection in {file.name}")
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Current (µA)")
            ax.legend()
            st.pyplot(fig)

        # Final plot: Scan rate vs peak current
        if anodic_peaks and cathodic_peaks:
            anodic_peaks.sort()
            cathodic_peaks.sort()
            scan_rates_a = np.array([x[0] for x in anodic_peaks]).reshape(-1, 1)
            currents_a = np.array([x[1] for x in anodic_peaks])
            scan_rates_c = np.array([x[0] for x in cathodic_peaks]).reshape(-1, 1)
            currents_c = np.array([x[1] for x in cathodic_peaks])

            model_a = LinearRegression().fit(scan_rates_a, currents_a)
            model_c = LinearRegression().fit(scan_rates_c, currents_c)
            pred_a = model_a.predict(scan_rates_a)
            pred_c = model_c.predict(scan_rates_c)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.scatter(scan_rates_a, currents_a, color='red', label='Anodic Peak')
            ax2.plot(scan_rates_a, pred_a, 'r--', label=f'Anodic Fit (R²={r2_score(currents_a, pred_a):.2f})')
            ax2.scatter(scan_rates_c, currents_c, color='blue', label='Cathodic Peak')
            ax2.plot(scan_rates_c, pred_c, 'b--', label=f'Cathodic Fit (R²={r2_score(currents_c, pred_c):.2f})')

            ax2.set_xlabel("Scan Rate (V/s)", fontsize=14)
            ax2.set_ylabel("Peak Current (µA)", fontsize=14)
            ax2.set_title("Peak Current vs Scan Rate", fontsize=16, fontweight="bold")
            ax2.legend()
            st.pyplot(fig2)
