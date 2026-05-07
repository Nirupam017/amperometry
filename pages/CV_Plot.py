import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔬 CV Overlay Tool")

# =========================
# Theme
# =========================
bg_choice = st.sidebar.selectbox(
    "Background Color",
    ["White", "Black"]
)

bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

# =========================
# Peak Detection Toggle
# =========================
enable_peak = st.sidebar.checkbox(
    "Enable Peak Current Detection",
    value=False
)

# =========================
# Upload CSV Files
# =========================
uploaded_files = st.file_uploader(
    "Upload CV CSV files",
    type="csv",
    accept_multiple_files=True
)

color_palette = plt.get_cmap("tab10")

# =========================
# Store Data
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

            st.warning(
                f"{file.name} missing required columns"
            )

            continue

        df.dropna(
            subset=[v_col, i_col],
            inplace=True
        )

        voltage = df[v_col].values
        current = df[i_col].values * 1e6  # Convert to µA

        data_store.append(
            (file.name, voltage, current)
        )

# =========================
# Plotting
# =========================
if data_store:

    fig, ax = plt.subplots(
        figsize=(9, 7),
        facecolor=bg_color
    )

    ax.set_facecolor(bg_color)

    # =========================
    # Loop Through Files
    # =========================
    for i, (name, voltage, current) in enumerate(data_store):

        color = color_palette(i % color_palette.N)

        # =========================
        # Settings Per Curve
        # =========================
        with st.expander(f"{name} Settings"):

            label = st.text_input(
                f"Label {i}",
                value=name,
                key=f"label_{i}"
            )

            line_width = st.slider(
                f"Line Width {i}",
                min_value=1,
                max_value=6,
                value=2,
                key=f"lw_{i}"
            )

            # =========================
            # Peak Detection Inputs
            # =========================
            if enable_peak:

                peak_voltage = st.number_input(
                    f"Voltage for Curve {i} (V)",
                    value=0.0,
                    format="%.4f",
                    key=f"peak_voltage_{i}"
                )

                curve_part = st.selectbox(
                    f"Select Branch {i}",
                    [
                        "Top (Oxidation)",
                        "Bottom (Reduction)"
                    ],
                    key=f"curve_part_{i}"
                )

        # =========================
        # Plot Curve
        # =========================
        ax.plot(
            voltage,
            current,
            label=label,
            linewidth=line_width,
            color=color
        )

        # =========================
        # Peak Detection
        # =========================
        if enable_peak:

            voltage = np.array(voltage)
            current = np.array(current)

            # Turning point
            turning_idx = np.argmax(voltage)

            # =========================
            # Safety fallback
            # =========================
            if (
                turning_idx <= 1
                or
                turning_idx >= len(voltage) - 1
            ):

                v_use = voltage
                i_use = current

            else:

                if curve_part == "Top (Oxidation)":

                    v_use = voltage[:turning_idx]
                    i_use = current[:turning_idx]

                else:

                    v_use = voltage[turning_idx:]
                    i_use = current[turning_idx:]

            # =========================
            # Additional Safety
            # =========================
            if len(v_use) > 0:

                idx = np.argmin(
                    np.abs(v_use - peak_voltage)
                )

                selected_voltage = v_use[idx]
                selected_current = i_use[idx]

                # Plot Marker
                ax.scatter(
                    selected_voltage,
                    selected_current,
                    color=color,
                    edgecolors="black",
                    s=140,
                    zorder=10
                )

                # Display Values
                st.write(f"### {label}")

                st.write(
                    f"Selected Voltage: "
                    f"{selected_voltage:.4f} V"
                )

                st.write(
                    f"Peak Current: "
                    f"{selected_current:.4f} µA"
                )

            else:

                st.warning(
                    f"Could not detect branch for {label}"
                )

    # =========================
    # Styling
    # =========================
    ax.set_xlabel(
        "Voltage (V)",
        color=text_color,
        fontsize=15
    )

    ax.set_ylabel(
        "Current (µA)",
        color=text_color,
        fontsize=15
    )

    ax.set_title(
        "CV Overlay",
        color=text_color,
        fontsize=20
    )

    ax.tick_params(
        colors=text_color,
        labelsize=12
    )

    for spine in ax.spines.values():
        spine.set_color(text_color)

    legend = ax.legend(
        facecolor=bg_color,
        edgecolor=text_color,
        fontsize=11
    )

    for text in legend.get_texts():
        text.set_color(text_color)

    st.pyplot(fig)
