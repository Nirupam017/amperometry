import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("🔬 EIS Analysis Tool (Nyquist & Bode)")

# =========================
# BACKGROUND & THEME
# =========================
bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

# =========================
# FONT SETTINGS
# =========================
st.sidebar.markdown("### 🎨 Font Customization")
font_size = st.sidebar.slider("Font Size", 8, 20, 14)
font_weight = st.sidebar.selectbox("Font Weight", ["normal", "bold"])

# =========================
# COLOR PALETTE
# =========================
color_palette = plt.get_cmap("tab10")

# =========================
# FILE UPLOADER
# =========================
st.header("📂 Upload EIS CSV Files")
uploaded_files = st.file_uploader(
    "Upload CSV files",
    type="csv",
    accept_multiple_files=True
)

# =========================
# MAIN PLOTTING
# =========================
if uploaded_files:

    col1, col2 = st.columns(2)

    # =========================
    # NYQUIST PLOT
    # =========================
    with col1:
        st.subheader("🔵 Nyquist Plot")

        fig_nyq, ax_nyq = plt.subplots(figsize=(6, 6), facecolor=bg_color)
        ax_nyq.set_facecolor(bg_color)

        for i, file in enumerate(uploaded_files):

            with st.expander(f"Customize {file.name}"):
                label = st.text_input(
                    f"Legend label",
                    value=file.name,
                    key=f"nyq_label_{i}"
                )

            df = pd.read_csv(file)

            Z_real = df.iloc[:, 8].values   # Column I → Z'
            Z_imag = df.iloc[:, 9].values   # Column J → Z'' (already negative)

            color = color_palette(i % color_palette.N)

            ax_nyq.plot(
                Z_real,
                Z_imag,          # plot −Z″ (Nyquist convention)
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=4,
                color=color,
                label=label
            )

        ax_nyq.set_xlabel("Z′ (Ω)", fontsize=font_size,
                          fontweight=font_weight, color=text_color)
        ax_nyq.set_ylabel("−Z″ (Ω)", fontsize=font_size,
                          fontweight=font_weight, color=text_color)

        ax_nyq.tick_params(colors=text_color, labelsize=font_size)
        ax_nyq.grid(True, which="both", alpha=0.3)

        for spine in ax_nyq.spines.values():
            spine.set_color(text_color)

        ax_nyq.legend(facecolor=bg_color, edgecolor=text_color,
                      labelcolor=text_color)

        st.pyplot(fig_nyq)

    # =========================
    # BODE PLOT
    # =========================
    with col2:
        st.subheader("🟢 Bode Plot")

        fig_bode, ax_mag = plt.subplots(figsize=(7, 6), facecolor=bg_color)
        ax_mag.set_facecolor(bg_color)
        ax_phase = ax_mag.twinx()

        for i, file in enumerate(uploaded_files):

            df = pd.read_csv(file)

            freq = df.iloc[:, 6].values     # Column G → Frequency
            Z_mag = df.iloc[:, 7].values    # Column H → |Z|
            phase = df.iloc[:, 10].values  # Column K → Phase (already negative)

            color = color_palette(i % color_palette.N)

            # |Z|
            ax_mag.loglog(
                freq,
                Z_mag,
                'o-',
                linewidth=2,
                markersize=4,
                color=color
            )

            # Phase
            ax_phase.semilogx(
                freq,
                phase,
                's--',
                linewidth=2,
                markersize=4,
                color=color
            )

        ax_mag.set_xlabel("Frequency (Hz)", fontsize=font_size,
                          fontweight=font_weight, color=text_color)
        ax_mag.set_ylabel("|Z| (Ω)", fontsize=font_size,
                          fontweight=font_weight, color="red")
        ax_phase.set_ylabel("Phase (degrees)", fontsize=font_size,
                            fontweight=font_weight, color="blue")

        ax_mag.tick_params(axis='x', colors=text_color, labelsize=font_size)
        ax_mag.tick_params(axis='y', colors="red", labelsize=font_size)
        ax_phase.tick_params(axis='y', colors="blue", labelsize=font_size)

        ax_mag.grid(True, which="both", alpha=0.3)

        for spine in ax_mag.spines.values():
            spine.set_color(text_color)

        st.pyplot(fig_bode)
