import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
# Inset Zoom Toggle
# =========================

enable_inset = st.sidebar.checkbox(
    "Enable Inset Zoom",
    value=False
)

if enable_inset:

    inset_xmin = st.sidebar.number_input(
        "Inset Min Voltage (V)",
        value=-0.20,
        format="%.3f"
    )

    inset_xmax = st.sidebar.number_input(
        "Inset Max Voltage (V)",
        value=0.20,
        format="%.3f"
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

        except Exception:
            st.warning(f"{file.name} failed to load")
            continue

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
        current = df[i_col].values * 1e6

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
    # Create Inset Axis
    # =========================

    if enable_inset:

        axins = inset_axes(
            ax,
            width="35%",
            height="35%",
            loc="upper right"
        )

        axins.set_facecolor(bg_color)

    # =========================
    # Loop Through Files
    # =========================

    for i, (name, voltage, current) in enumerate(data_store):

        color = color_palette(i % color_palette.N)

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
        # Main Plot
        # =========================

        ax.plot(
            voltage,
            current,
            label=label,
            linewidth=line_width,
            color=color
        )

        # =========================
        # Inset Plot
        # =========================

        if enable_inset:

            mask = (
                (voltage >= inset_xmin)
                &
                (voltage <= inset_xmax)
            )

            if np.any(mask):

                axins.plot(
                    voltage[mask],
                    current[mask],
                    linewidth=line_width,
                    color=color
                )

        # =========================
        # Peak Detection
        # =========================

        if enable_peak:

            voltage = np.array(voltage)
            current = np.array(current)

            turning_idx = np.argmax(voltage)

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

            if len(v_use) > 0:

                idx = np.argmin(
                    np.abs(v_use - peak_voltage)
                )

                selected_voltage = v_use[idx]
                selected_current = i_use[idx]

                ax.scatter(
                    selected_voltage,
                    selected_current,
                    color=color,
                    edgecolors="black",
                    s=140,
                    zorder=10
                )

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
    # Main Axis Styling
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

    # =========================
    # Inset Formatting
    # =========================

    if enable_inset:

        axins.set_xlim(
            inset_xmin,
            inset_xmax
        )

        axins.tick_params(
            colors=text_color,
            labelsize=8
        )

        for spine in axins.spines.values():
            spine.set_color(text_color)

        all_y = []

        for _, voltage, current in data_store:

            mask = (
                (voltage >= inset_xmin)
                &
                (voltage <= inset_xmax)
            )

            all_y.extend(current[mask])

        if len(all_y) > 0:

            ymin = min(all_y)
            ymax = max(all_y)

            margin = 0.10 * (ymax - ymin)

            if margin == 0:
                margin = 1

            axins.set_ylim(
                ymin - margin,
                ymax + margin
            )

        ax.indicate_inset_zoom(
            axins,
            edgecolor=text_color
        )

    st.pyplot(fig)
