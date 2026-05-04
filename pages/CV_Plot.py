import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔬 CV Peak Analysis Tool (Oxidation + Reduction)")

# === Theme ===
bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

# === Upload ===
uploaded_files = st.file_uploader(
    "Upload CV CSV files",
    type="csv",
    accept_multiple_files=True
)

color_palette = plt.get_cmap("tab10")

# === Peak Detection Functions ===
def find_oxidation_peak(voltage, current):
    idx = np.argmax(current)
    return voltage[idx], current[idx]

def find_reduction_peak(voltage, current):
    idx = np.argmin(current)
    return voltage[idx], current[idx]

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

    ox_currents = []
    red_currents = []
    concentrations = []

    for i, (name, voltage, current) in enumerate(data_store):

        with st.expander(f"{name} settings"):
            label = st.text_input(f"Label {i}", name, key=f"label_{i}")
            conc = st.number_input(f"Concentration (µM) {i}", value=10.0, key=f"conc_{i}")

        color = color_palette(i % color_palette.N)
        ax.plot(voltage, current, label=label, linewidth=2, color=color)

        # === Split forward & reverse ===
        turning_idx = np.argmax(voltage)

        forward_v = voltage[:turning_idx]
        forward_i = current[:turning_idx]

        reverse_v = voltage[turning_idx:]
        reverse_i = current[turning_idx:]

        # === Oxidation peak (max current from both branches) ===
        Ep_f, Ip_f = find_oxidation_peak(forward_v, forward_i)
        Ep_r, Ip_r = find_oxidation_peak(reverse_v, reverse_i)

        if Ip_f > Ip_r:
            Ep_ox, Ip_ox = Ep_f, Ip_f
        else:
            Ep_ox, Ip_ox = Ep_r, Ip_r

        # === Reduction peak (min current from both branches) ===
        Ep_f_r, Ip_f_r = find_reduction_peak(forward_v, forward_i)
        Ep_r_r, Ip_r_r = find_reduction_peak(reverse_v, reverse_i)

        if Ip_f_r < Ip_r_r:
            Ep_red, Ip_red = Ep_f_r, Ip_f_r
        else:
            Ep_red, Ip_red = Ep_r_r, Ip_r_r

        # Store
        ox_currents.append(Ip_ox)
        red_currents.append(Ip_red)
        concentrations.append(conc)

        # Plot peaks
        ax.scatter(Ep_ox, Ip_ox, color=color, edgecolors='black', s=80, zorder=5)
        ax.scatter(Ep_red, Ip_red, color=color, edgecolors='black', s=80, marker='s', zorder=5)

    ax.set_xlabel("Voltage (V)", color=text_color)
    ax.set_ylabel("Current (µA)", color=text_color)
    ax.set_title("CV Overlay with Peak Detection", color=text_color)

    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)

    ax.legend(facecolor=bg_color, labelcolor=text_color)

    st.pyplot(fig)

    # === Oxidation Calibration ===
    if len(concentrations) > 1:

        st.subheader("📊 Oxidation Peak Calibration")

        conc = np.array(concentrations)
        ox = np.array(ox_currents)

        slope, intercept = np.polyfit(conc, ox, 1)
        fit = slope * conc + intercept

        r2 = 1 - (np.sum((ox - fit)**2) /
                  np.sum((ox - np.mean(ox))**2))

        fig2, ax2 = plt.subplots()
        ax2.scatter(conc, ox)
        ax2.plot(conc, fit, linestyle="--")

        ax2.set_xlabel("Concentration (µM)")
        ax2.set_ylabel("Oxidation Peak Current (µA)")
        ax2.set_title(f"Sensitivity = {slope:.3f} µA/µM | R² = {r2:.4f}")

        st.pyplot(fig2)

    # === Reduction Calibration ===
    if len(concentrations) > 1:

        st.subheader("📊 Reduction Peak Calibration")

        red = np.array(red_currents)

        slope, intercept = np.polyfit(conc, red, 1)
        fit = slope * conc + intercept

        r2 = 1 - (np.sum((red - fit)**2) /
                  np.sum((red - np.mean(red))**2))

        fig3, ax3 = plt.subplots()
        ax3.scatter(conc, red)
        ax3.plot(conc, fit, linestyle="--")

        ax3.set_xlabel("Concentration (µM)")
        ax3.set_ylabel("Reduction Peak Current (µA)")
        ax3.set_title(f"Sensitivity = {slope:.3f} µA/µM | R² = {r2:.4f}")

        st.pyplot(fig3)
