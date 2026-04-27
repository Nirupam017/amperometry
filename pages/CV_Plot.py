import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔬 CV Analysis Tool")

# === Background and Theme ===
bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

# === Font Customization ===
st.sidebar.markdown("### 🎨 Font Customization")
font_size = st.sidebar.slider("Font Size", min_value=8, max_value=20, value=14)
font_weight = st.sidebar.selectbox("Font Weight", ["normal", "bold"])

# === Axis Scaling Controls ===
st.sidebar.markdown("### 📏 Axis Scale Settings")

use_custom_x = st.sidebar.checkbox("Set X-axis range manually")
use_custom_y = st.sidebar.checkbox("Set Y-axis range manually")

x_min, x_max = None, None
y_min, y_max = None, None

if use_custom_x:
    x_min = st.sidebar.number_input("X min", value=-1.0, step=0.1)
    x_max = st.sidebar.number_input("X max", value=1.0, step=0.1)

if use_custom_y:
    y_min = st.sidebar.number_input("Y min (μA)", value=-10.0, step=1.0)
    y_max = st.sidebar.number_input("Y max (μA)", value=10.0, step=1.0)

# === Professional Color Palette ===
color_palette = plt.get_cmap("tab10")

# === File Upload ===
st.header("📈 CV Curve Overlay")
uploaded_files = st.file_uploader(
    "Upload CV CSV Files",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    for i, file in enumerate(uploaded_files):
        with st.expander(f"Customize {file.name}"):
            label = st.text_input(
                f"Legend Label for {file.name}",
                value=file.name,
                key=f"label_{i}"
            )

        df = pd.read_csv(file)
        df.dropna(
            subset=["Working Electrode (V)", "Current (A)"],
            inplace=True
        )

        voltage = df["Working Electrode (V)"].values
        current = df["Current (A)"].values * 1e6  # Convert A → μA

        color = color_palette(i % color_palette.N)

        ax.plot(
            voltage,
            current,
            label=label,
            linewidth=2,
            color=color
        )

    # === Apply Manual Axis Scaling ===
    if use_custom_x:
        if x_min < x_max:
            ax.set_xlim(x_min, x_max)
        else:
            st.sidebar.warning("⚠️ X min must be less than X max")

    if use_custom_y:
        if y_min < y_max:
            ax.set_ylim(y_min, y_max)
        else:
            st.sidebar.warning("⚠️ Y min must be less than Y max")

    # === Styling ===
    ax.set_xlabel(
        "Voltage (V)",
        fontsize=font_size,
        fontweight=font_weight,
        color=text_color
    )
    ax.set_ylabel(
        "Current (μA)",
        fontsize=font_size,
        fontweight=font_weight,
        color=text_color
    )
    ax.set_title(
        "CV Curve Overlay",
        fontsize=font_size + 2,
        fontweight=font_weight,
        color=text_color
    )

    ax.tick_params(colors=text_color, labelsize=font_size)

    for spine in ax.spines.values():
        spine.set_color(text_color)

    ax.legend(
        facecolor=bg_color,
        edgecolor=text_color,
        labelcolor=text_color
    )

    st.pyplot(fig)
