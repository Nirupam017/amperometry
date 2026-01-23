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

# === Professional Color Cycle ===
# Change palette name if needed: "tab10", "Set2", "Dark2", "Paired"
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
        current = df["Current (A)"].values * 1e9  # A → nA

        # === Automatically assigned professional color ===
        color = color_palette(i % color_palette.N)

        ax.plot(
            voltage,
            current,
            label=label,
            linewidth=2,
            color=color
        )

    # === Styling ===
    ax.set_xlabel(
        "Voltage (V)",
        fontsize=font_size,
        fontweight=font_weight,
        color=text_color
    )
    ax.set_ylabel(
        "Current (nA)",
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
