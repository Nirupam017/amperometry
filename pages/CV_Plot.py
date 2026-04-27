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

# === Scan Direction Selection ===
st.sidebar.markdown("### 🔄 Scan Selection")
scan_mode = st.sidebar.selectbox(
    "Select Scan Branch",
    ["Full", "Forward", "Reverse"]
)

# === Calibration ===
st.sidebar.markdown("### 🧪 Calibration")
do_calibration = st.sidebar.checkbox("Enable Calibration")

if do_calibration:
    target_voltage = st.sidebar.number_input("Voltage (V)", value=-0.25)

# === Upload ===
uploaded_files = st.file_uploader(
    "Upload CV CSV files",
    type="csv",
    accept_multiple_files=True
)

color_palette = plt.get_cmap("tab10")

data_store = []

# === Load Data Safely ===
if uploaded_files:
    for file in uploaded_files:
        try:
            file.seek(0)
            df = pd.read_csv(file)
        except:
            st.warning(f"{file.name} unreadable. Skipping.")
            continue

        if df.empty:
            st.warning(f"{file.name} empty. Skipping.")
            continue

        # Auto detect columns
        v_col = next((c for c in df.columns if "V" in c or "Potential" in c), None)
        i_col = next((c for c in df.columns if "A" in c or "Current" in c), None)

        if v_col is None or i_col is None:
            st.warning(f"{file.name} missing columns.")
            continue

        df.dropna(subset=[v_col, i_col], inplace=True)

        voltage = df[v_col].values
        current = df[i_col].values * 1e6

        # === Detect scan direction using slope ===
        dV = np.gradient(voltage)

        if scan_mode == "Forward":
            mask = dV > 0
            voltage = voltage[mask]
            current = current[mask]

        elif scan_mode == "Reverse":
            mask = dV < 0
            voltage = voltage[mask]
            current = current[mask]

        data_store.append((file.name, voltage, current))

# === Plot ===
if data_store:

    fig, ax = plt.subplots(figsize=(8,6), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    calibration_currents = []
    concentrations = []

    for i, (name, voltage, current) in enumerate(data_store):

        with st.expander(f"{name} settings"):
            label = st.text_input(f"Label {i}", name, key=f"label_{i}")
            conc = st.number_input(f"Concentration (µM) {i}", value=20.0, key=f"conc_{i}")

        color = color_palette(i % color_palette.N)
        ax.plot(voltage, current, label=label, color=color, linewidth=2)

        # === Calibration ===
        if do_calibration:
            # Interpolated value (accurate)
            current_val = np.interp(target_voltage, voltage, current)

            calibration_currents.append(current_val)
            concentrations.append(conc)

            # Plot SAME value (fix mismatch)
            ax.scatter(target_voltage, current_val, color=color, s=60, edgecolors='black')

    # === Axis scaling ===
    if use_custom_x and x_min < x_max:
        ax.set_xlim(x_min, x_max)

    if use_custom_y and y_min < y_max:
        ax.set_ylim(y_min, y_max)

    # === Styling ===
    ax.set_xlabel("Voltage (V)", fontsize=font_size, color=text_color)
    ax.set_ylabel("Current (µA)", fontsize=font_size, color=text_color)
    ax.set_title("CV Overlay", fontsize=font_size+2, color=text_color)

    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)

    ax.legend()

    # === Inset Zoom ===
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
        curr_array = np.array(calibration_currents)

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

        st.write(f"**Sensitivity:** {slope:.3f} µA/µM")
        st.write(f"**R²:** {r2:.4f}")
