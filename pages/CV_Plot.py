import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔬 CV Analysis Tool (User-Controlled)")

# =========================
# Theme
# =========================
bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])

bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

# =========================
# Calibration Option
# =========================
st.sidebar.markdown("### 📊 Calibration Options")

enable_calibration = st.sidebar.checkbox(
    "Enable Calibration Curve",
    value=False
)

# =========================
# Upload Files
# =========================
uploaded_files = st.file_uploader(
    "Upload CV CSV files",
    type="csv",
    accept_multiple_files=True
)

color_palette = plt.get_cmap("tab10")

# =========================
# Data Store
# =========================
data_store = []

if uploaded_files:

    for file in uploaded_files:

        try:
            df = pd.read_csv(file)

        except:
            st.warning(f"{file.name} failed to load")
            continue

        # Possible column names
        possible_voltage = [
            "Working Electrode (V)",
            "Ewe/V",
            "Voltage"
        ]

        possible_current = [
            "Current (A)",
            "I/A",
            "Current"
        ]

        v_col = next(
            (c for c in possible_voltage if c in df.columns),
            None
        )

        i_col = next(
            (c for c in possible_current if c in df.columns),
            None
        )

        if v_col is None or i_col is None:
            st.warning(f"{file.name} missing required columns")
            continue

        df.dropna(subset=[v_col, i_col], inplace=True)

        voltage = df[v_col].values
        current = df[i_col].values * 1e6  # Convert to µA

        data_store.append(
            (file.name, voltage, current)
        )

# =========================
# Main Plot
# =========================
if data_store:

    fig, ax = plt.subplots(
        figsize=(8, 6),
        facecolor=bg_color
    )

    ax.set_facecolor(bg_color)

    calibration_currents = []
    concentrations = []

    for i, (name, voltage, current) in enumerate(data_store):

        color = color_palette(i % color_palette.N)

        # =====================================
        # USER SETTINGS FOR EACH CURVE
        # =====================================
        with st.expander(f"{name} Settings"):

            label = st.text_input(
                f"Label {i}",
                value=name,
                key=f"label_{i}"
            )

            # ---------- Calibration Inputs ----------
            if enable_calibration:

                st.markdown("### Calibration Parameters")

                conc = st.number_input(
                    f"Concentration (µM) - Curve {i}",
                    value=float(i + 1),
                    key=f"conc_{i}"
                )

                scan_choice = st.selectbox(
                    f"Peak Region - Curve {i}",
                    ["Full", "Forward", "Reverse"],
                    key=f"scan_{i}"
                )

                peak_voltage = st.number_input(
                    f"Voltage for Peak Current (V) - Curve {i}",
                    value=0.0,
                    format="%.4f",
                    key=f"peakv_{i}"
                )

        # =====================================
        # Plot Full CV
        # =====================================
        ax.plot(
            voltage,
            current,
            linewidth=2,
            color=color,
            label=label
        )

        # =====================================
        # Calibration Extraction
        # =====================================
        if enable_calibration:

            voltage = np.array(voltage)
            current = np.array(current)

            # Find turning point
            turning_idx = np.argmax(voltage)

            # Split scans
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

            # Sort for interpolation
            sort_idx = np.argsort(v_use)

            v_use = v_use[sort_idx]
            i_use = i_use[sort_idx]

            # Interpolate current at chosen voltage
            try:

                current_val = np.interp(
                    peak_voltage,
                    v_use,
                    i_use
                )

            except:

                current_val = np.nan

            calibration_currents.append(current_val)
            concentrations.append(conc)

            # Mark chosen point
            if not np.isnan(current_val):

                ax.scatter(
                    peak_voltage,
                    current_val,
                    color=color,
                    edgecolors="black",
                    s=90,
                    zorder=5
                )

    # =====================================
    # Plot Styling
    # =====================================
    ax.set_xlabel(
        "Voltage (V)",
        color=text_color
    )

    ax.set_ylabel(
        "Current (µA)",
        color=text_color
    )

    ax.set_title(
        "CV Overlay",
        color=text_color
    )

    ax.tick_params(colors=text_color)

    for spine in ax.spines.values():
        spine.set_color(text_color)

    ax.legend(
        facecolor=bg_color,
        labelcolor=text_color
    )

    st.pyplot(fig)

    # =====================================
    # Calibration Curve
    # =====================================
    if enable_calibration:

        st.subheader("📊 Calibration Curve")

        conc = np.array(concentrations)
        curr = np.array(calibration_currents)

        # Remove NaNs
        mask = ~np.isnan(curr)

        conc = conc[mask]
        curr = curr[mask]

        if len(conc) > 1:

            # Linear Fit
            slope, intercept = np.polyfit(
                conc,
                curr,
                1
            )

            fit = slope * conc + intercept

            # R²
            r2 = 1 - (
                np.sum((curr - fit) ** 2)
                /
                np.sum((curr - np.mean(curr)) ** 2)
            )

            # Plot Calibration
            fig2, ax2 = plt.subplots(figsize=(6, 5))

            ax2.scatter(
                conc,
                curr,
                s=80
            )

            ax2.plot(
                conc,
                fit,
                linestyle="--"
            )

            ax2.set_xlabel("Concentration (µM)")
            ax2.set_ylabel("Peak Current (µA)")

            ax2.set_title(
                f"Sensitivity = {slope:.4f} µA/µM | R² = {r2:.4f}"
            )

            st.pyplot(fig2)

            st.write(f"### Sensitivity: {slope:.4f} µA/µM")
            st.write(f"### R²: {r2:.4f}")

        else:

            st.warning(
                "Need at least 2 valid calibration points."
            )
