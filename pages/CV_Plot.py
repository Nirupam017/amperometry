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
        current = df[i_col].values * 1e6  # µA

        data_store.append(
            (file.name, voltage, current)
        )

# =========================
# Plot CV Overlay
# =========================
if data_store:

    fig, ax = plt.subplots(
        figsize=(8, 6),
        facecolor=bg_color
    )

    ax.set_facecolor(bg_color)

    for i, (name, voltage, current) in enumerate(data_store):

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

        color = color_palette(i % color_palette.N)

        ax.plot(
            voltage,
            current,
            label=label,
            linewidth=line_width,
            color=color
        )

    # =========================
    # Styling
    # =========================
    ax.set_xlabel(
        "Voltage (V)",
        color=text_color,
        fontsize=14
    )

    ax.set_ylabel(
        "Current (µA)",
        color=text_color,
        fontsize=14
    )

    ax.set_title(
        "CV Overlay",
        color=text_color,
        fontsize=16
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
        fontsize=10
    )

    for text in legend.get_texts():
        text.set_color(text_color)

    st.pyplot(fig)
