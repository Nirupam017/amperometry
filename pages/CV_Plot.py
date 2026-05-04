import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔬 CV Analysis Tool (User-Controlled)")

# === Theme ===
bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

# === Calibration Settings ===
st.sidebar.markdown("### 🧪 Measurement Settings")

target_voltage = st.sidebar.number_input("Voltage (V)", value=-0.25)

scan_choice = st.sidebar.selectbox(
    "Select Scan",
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
            df = pd.read_csv(file)
        except:
            st.warning(f"{file.name} failed to load")
            continue

        possible_voltage = ["Working Electrode (V)", "Ewe/V", "Voltage"]
        possible_current = ["Current (A)", "I/A", "Current"]

        v_col = next((c for c in possible_voltage if c in df.columns), None)
        i_col = next((c for c in possible_current if c in df.columns), None)

        if v_col is None or i_col is None:
            st.warning(f"{file.name} missing required columns")
            continue

        df.dropna(subset=[v_col, i_col], inplace=True)

        voltage = df[v_col].values
        current = df[i_col].values * 1e6  # µA

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
            conc = st.number_input(f"Concentration (µM) {i}", value=10.0, key=f"conc_{i}")

        color = color_palette(i % color_palette.N)
        ax.plot(voltage, current, label=label, linewidth=2, color=color)

        # === Split CV safely ===
        voltage = np.array(voltage)
        current = np.array(current)

        turning_idx = np.argmax(voltage)

        if turning_idx == 0 or turning_idx == len(voltage) - 1:
            v_use = voltage
            i_use = current
        else:
            if scan_choice == "Forward":
                v_use = voltage[:turning_idx]
                i_use = current[:turning_idx]
            elif scan_choice == "Reverse":
                v_use = voltage[turning_idx:]
                i_use = current[turning_idx:]
            else:
                v_use = voltage
                i_use = current

        # === Interpolation ===
        try:
            current_val = np.interp(target_voltage, v_use, i_use)
        except:
            current_val = np.nan

        calibration_currents.append(current_val)
        concentrations.append(conc)

        # Plot marker
        if not np.isnan(current_val):
            ax.scatter(
                target_voltage,
                current_val,
                color=color,
                edgecolors='black',
                s=80,
                zorder=5
            )

    ax.set_xlabel("Voltage (V)", color=text_color)
    ax.set_ylabel("Current (µA)", color=text_color)
    ax.set_title("CV Overlay", color=text_color)

    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)

    ax.legend(facecolor=bg_color, labelcolor=text_color)

    st.pyplot(fig)

    # === Calibration ===
    if len(concentrations) > 1:

        st.subheader("📊 Calibration Curve")

        conc = np.array(concentrations)
        curr = np.array(calibration_currents)

        # remove NaNs
        mask = ~np.isnan(curr)
        conc = conc[mask]
        curr = curr[mask]

        if len(conc) > 1:

            slope, intercept = np.polyfit(conc, curr, 1)
            fit = slope * conc + intercept

            r2 = 1 - (np.sum((curr - fit)**2) /
                      np.sum((curr - np.mean(curr))**2))

            fig2, ax2 = plt.subplots()
            ax2.scatter(conc, curr)
            ax2.plot(conc, fit, linestyle="--")

            ax2.set_xlabel("Concentration (µM)")
            ax2.set_ylabel(f"Current at {target_voltage} V (µA)")
            ax2.set_title(f"Sensitivity = {slope:.3f} µA/µM | R² = {r2:.4f}")

            st.pyplot(fig2)

            st.write(f"**Sensitivity:** {slope:.3f} µA/µM")
            st.write(f"**R²:** {r2:.4f}")
