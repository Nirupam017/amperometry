import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

st.set_page_config(layout="wide")
st.title("🔬 Advanced CV Analysis Tool")

# === Theme ===
bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

# === Font ===
st.sidebar.markdown("### 🎨 Font")
font_size = st.sidebar.slider("Font Size", 8, 20, 14)
font_weight = st.sidebar.selectbox("Font Weight", ["normal", "bold"])

# === Axis Control ===
st.sidebar.markdown("### 📏 Axis Control")
use_custom_x = st.sidebar.checkbox("Manual X-axis")
use_custom_y = st.sidebar.checkbox("Manual Y-axis")

if use_custom_x:
    x_min = st.sidebar.number_input("X min", value=-1.0)
    x_max = st.sidebar.number_input("X max", value=0.2)

if use_custom_y:
    y_min = st.sidebar.number_input("Y min", value=-80.0)
    y_max = st.sidebar.number_input("Y max", value=20.0)

# === Calibration ===
st.sidebar.markdown("### 🧪 Calibration")
do_calibration = st.sidebar.checkbox("Enable Calibration")

use_peak = False
target_voltage = None

if do_calibration:
    mode = st.sidebar.radio("Mode", ["Manual Voltage", "Auto Peak"])

    if mode == "Manual Voltage":
        target_voltage = st.sidebar.number_input("Voltage (V)", value=-0.2)
    else:
        use_peak = True
        peak_type = st.sidebar.selectbox("Peak Type", ["Anodic", "Cathodic"])

    use_avg = st.sidebar.checkbox("Use averaging window", value=True)
    window = st.sidebar.slider("Window size", 1, 20, 5)

# === Plot Settings ===
color_palette = plt.get_cmap("tab10")

# === Upload ===
uploaded_files = st.file_uploader("Upload CV CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    fig, ax = plt.subplots(figsize=(8,6), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    calibration_currents = []
    concentrations = []

    for i, file in enumerate(uploaded_files):

        with st.expander(f"{file.name} settings"):
            label = st.text_input(f"Label {i}", file.name, key=f"label_{i}")
            conc = st.number_input(f"Concentration (µM) {i}", value=20.0, key=f"conc_{i}")

        df = pd.read_csv(file)
        df.dropna(subset=["Working Electrode (V)", "Current (A)"], inplace=True)

        voltage = df["Working Electrode (V)"].values
        current = df["Current (A)"].values * 1e6

        color = color_palette(i % color_palette.N)

        ax.plot(voltage, current, label=label, linewidth=2, color=color)

        # === Calibration Extraction ===
        if do_calibration:

            if use_peak:
                if peak_type == "Anodic":
                    idx = np.argmax(current)
                else:
                    idx = np.argmin(current)
            else:
                idx = (np.abs(voltage - target_voltage)).argmin()

            if use_avg:
                start = max(0, idx - window)
                end = idx + window
                current_val = np.mean(current[start:end])
            else:
                current_val = current[idx]

            calibration_currents.append(current_val)
            concentrations.append(conc)

            # Mark point on graph
            ax.scatter(voltage[idx], current[idx], color=color, s=50)

    # === Axis apply ===
    if use_custom_x and x_min < x_max:
        ax.set_xlim(x_min, x_max)

    if use_custom_y and y_min < y_max:
        ax.set_ylim(y_min, y_max)

    # === Labels ===
    ax.set_xlabel("Voltage (V)", fontsize=font_size, color=text_color)
    ax.set_ylabel("Current (µA)", fontsize=font_size, color=text_color)
    ax.set_title("CV Overlay", fontsize=font_size+2, color=text_color)

    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)

    ax.legend(facecolor=bg_color, labelcolor=text_color)

    # === Inset Zoom ===
    axins = inset_axes(ax, width="40%", height="40%", loc="lower right")
    for i, file in enumerate(uploaded_files):
        df = pd.read_csv(file)
        df.dropna(subset=["Working Electrode (V)", "Current (A)"], inplace=True)

        voltage = df["Working Electrode (V)"].values
        current = df["Current (A)"].values * 1e6

        color = color_palette(i % color_palette.N)
        axins.plot(voltage, current, color=color)

    axins.set_xlim(-0.4, 0.0)
    axins.set_ylim(-30, 10)
    axins.set_title("Zoom")

    st.pyplot(fig)

    # === Calibration Plot ===
    if do_calibration and len(concentrations) > 1:

        st.subheader("📊 Calibration Curve")

        conc_array = np.array(concentrations)
        curr_array = np.array(calibration_currents)

        slope, intercept = np.polyfit(conc_array, curr_array, 1)
        fit = slope * conc_array + intercept

        r2 = 1 - (np.sum((curr_array - fit)**2) / np.sum((curr_array - np.mean(curr_array))**2))

        fig2, ax2 = plt.subplots()

        ax2.scatter(conc_array, curr_array)
        ax2.plot(conc_array, fit, linestyle="--")

        ax2.set_xlabel("Concentration (µM)")
        ax2.set_ylabel("Current (µA)")
        ax2.set_title(f"Sensitivity = {slope:.3f} µA/µM | R² = {r2:.4f}")

        st.pyplot(fig2)

        st.write(f"**Sensitivity:** {slope:.3f} µA/µM")
        st.write(f"**R²:** {r2:.4f}")
