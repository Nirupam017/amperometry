import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
        records = []
        for i, file in enumerate(uploaded_files):
            with st.expander(f"Details for {file.name}"):
                scan_rate = st.number_input(f"Scan Rate (V/s) for {file.name}", key=f"rate_{i}", min_value=0.0, step=0.01)
            df = pd.read_csv(file)
            df.dropna(subset=["Working Electrode (V)", "Current (A)"], inplace=True)
            voltage = df["Working Electrode (V)"].values
            current = df["Current (A)"].values * 1e6
            peak_current = current[np.argmax(np.abs(current))]  # Max abs peak
            records.append((scan_rate, peak_current))

        # Sort and convert
        records = sorted(records)
        scan_rates = np.array([r[0] for r in records]).reshape(-1, 1)
        peak_currents = np.array([r[1] for r in records])

        model = LinearRegression().fit(scan_rates, peak_currents)
        y_pred = model.predict(scan_rates)
        r2 = r2_score(peak_currents, y_pred)

        # Plot scan rate study
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        ax.scatter(scan_rates, peak_currents, color="dodgerblue", edgecolor="black", s=80, label="Data Points")
        ax.plot(scan_rates, y_pred, color="red", linewidth=2, label=f"Fit: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

        ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(facecolor=bg_color, edgecolor=text_color, boxstyle='round'),
                color=text_color)

        ax.set_xlabel("Scan Rate (V/s)", fontsize=14, color=text_color)
        ax.set_ylabel("Peak Current (µA)", fontsize=14, color=text_color)
        ax.set_title("Scan Rate vs. Peak Current", fontsize=16, fontweight="bold", color=text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
        ax.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)
        st.pyplot(fig)
