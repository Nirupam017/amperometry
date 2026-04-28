import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

st.set_page_config(layout="wide")
st.title("🔬 Advanced CV Analysis Tool (Peak-Based Calibration)")

# === Theme ===
bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

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

# === Calibration Settings ===
st.sidebar.markdown("### 🧪 Calibration")
do_calibration = st.sidebar.checkbox("Enable Calibration")

if do_calibration:
    st.sidebar.markdown("### 🎯 Peak Selection")

    use_peak_mode = st.sidebar.checkbox("Use Peak Voltages")

    if use_peak_mode:
        peak_upper_v = st.sidebar.number_input("Upper Peak Voltage (Oxidation)", value=0.1)
        peak_lower_v = st.sidebar.number_input("Lower Peak Voltage (Reduction)", value=-0.25)

        calibration_mode = st.sidebar.selectbox(
            "Calibration Type",
            ["Upper Peak", "Lower Peak", "Both"]
        )
    else:
        target_voltage = st.sidebar.number_input("Voltage (V)", value=-0.25)

    direction = st.sidebar.selectbox(
        "Scan branch",
        ["Full", "Forward", "Reverse"]
    )

# === Upload ===
uploaded_files = st.file_uploader(
    "Upload CV CSV files",
    type="csv",
    accept_multiple_files=True
)

color_palette = plt.get_cmap("tab10")

# === Data Store ===
data_store = []

if uploaded_files:
    for file in uploaded_files:
        try:
            file.seek(0)
            df = pd.read_csv(file)

            if df.empty:
                st.warning(f"{file.name} is empty. Skipping.")
                continue

        except Exception:
            st.warning(f"{file.name} could not be read. Skipping.")
            continue

        # === Column Detection ===
        possible_voltage = ["Working Electrode (V)", "Ewe/V", "Voltage"]
        possible_current = ["Current (A)", "I/A", "Current"]

        v_col = next((c for c in possible_voltage if c in df.columns), None)
        i_col = next((c for c in possible_current if c in df.columns), None)

        if v_col is None or i_col is None:
            st.warning(f"{file.name} missing required columns.")
            continue

        df.dropna(subset=[v_col, i_col], inplace=True)

        voltage = df[v_col].values
        current = df[i_col].values * 1e6  # µA

        data_store.append((file.name, voltage, current))

# === Plot ===
if data_store:

    fig, ax = plt.subplots(figsize=(8,6), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    cal_upper = []
    cal_lower = []
    concentrations = []

    for i, (name, voltage, current) in enumerate(data_store):

        with st.expander(f"{name} settings"):
            label = st.text_input(f"Label {i}", name, key=f"label_{i}")
            conc = st.number_input(f"Concentration (µM) {i}", value=20.0, key=f"conc_{i}")

        # === Branch Selection ===
        if do_calibration and direction != "Full":
            mid = len(voltage) // 2
            if direction == "Forward":
                voltage_use = voltage[:mid]
                current_use = current[:mid]
            else:
                voltage_use = voltage[mid:]
                current_use = current[mid:]
        else:
            voltage_use = voltage
            current_use = current

        color = color_palette(i % color_palette.N)
        ax.plot(voltage, current, label=label, linewidth=2, color=color)

        # === Calibration Extraction ===
        if do_calibration:

            concentrations.append(conc)

            if use_peak_mode:

                upper_current = np.interp(peak_upper_v, voltage_use, current_use)
                lower_current = np.interp(peak_lower_v, voltage_use, current_use)

                cal_upper.append(upper_current)
                cal_lower.append(lower_current)

                ax.scatter(peak_upper_v, upper_current, color=color, s=60, edgecolors='black')
                ax.scatter(peak_lower_v, lower_current, color=color, s=60, edgecolors='black')

            else:
                current_val = np.interp(target_voltage, voltage_use, current_use)
                cal_upper.append(current_val)

                ax.scatter(target_voltage, current_val, color=color, s=60, edgecolors='black')

    # === Axis ===
    if use_custom_x and x_min < x_max:
        ax.set_xlim(x_min, x_max)

    if use_custom_y and y_min < y_max:
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Voltage (V)", color=text_color)
    ax.set_ylabel("Current (µA)", color=text_color)
    ax.set_title("CV Overlay", color=text_color)

    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)

    ax.legend(facecolor=bg_color, labelcolor=text_color)

    # === Inset ===
    axins = inset_axes(ax, width="40%", height="40%", loc="lower right")

    for i, (name, voltage, current) in enumerate(data_store):
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

        if use_peak_mode:

            if calibration_mode in ["Upper Peak", "Both"]:

                curr_array = np.array(cal_upper)

                slope, intercept = np.polyfit(conc_array, curr_array, 1)
                fit = slope * conc_array + intercept

                r2 = 1 - (np.sum((curr_array - fit)**2) /
                          np.sum((curr_array - np.mean(curr_array))**2))

                fig_u, ax_u = plt.subplots()
                ax_u.scatter(conc_array, curr_array)
                ax_u.plot(conc_array, fit, linestyle="--")

                ax_u.set_title(f"Upper Peak | Sensitivity={slope:.3f}, R²={r2:.4f}")
                ax_u.set_xlabel("Concentration (µM)")
                ax_u.set_ylabel("Current (µA)")

                st.pyplot(fig_u)

            if calibration_mode in ["Lower Peak", "Both"]:

                curr_array = np.array(cal_lower)

                slope, intercept = np.polyfit(conc_array, curr_array, 1)
                fit = slope * conc_array + intercept

                r2 = 1 - (np.sum((curr_array - fit)**2) /
                          np.sum((curr_array - np.mean(curr_array))**2))

                fig_l, ax_l = plt.subplots()
                ax_l.scatter(conc_array, curr_array)
                ax_l.plot(conc_array, fit, linestyle="--")

                ax_l.set_title(f"Lower Peak | Sensitivity={slope:.3f}, R²={r2:.4f}")
                ax_l.set_xlabel("Concentration (µM)")
                ax_l.set_ylabel("Current (µA)")

                st.pyplot(fig_l)

        else:

            curr_array = np.array(cal_upper)

            slope, intercept = np.polyfit(conc_array, curr_array, 1)
            fit = slope * conc_array + intercept

            r2 = 1 - (np.sum((curr_array - fit)**2) /
                      np.sum((curr_array - np.mean(curr_array))**2))

            fig2, ax2 = plt.subplots()
            ax2.scatter(conc_array, curr_array)
            ax2.plot(conc_array, fit, linestyle="--")

            ax2.set_xlabel("Concentration (µM)")
            ax2.set_ylabel(f"Current at {target_voltage} V (µA)")
            ax2.set_title(f"Sensitivity = {slope:.3f} µA/µM | R² = {r2:.4f}")

            st.pyplot(fig2)
