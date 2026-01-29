import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔬 EIS Analysis Tool")

# ===============================
# Sidebar styling
# ===============================
bg_choice = st.sidebar.selectbox("Background", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

font_size = st.sidebar.slider("Font Size", 10, 20, 14)
font_weight = st.sidebar.selectbox("Font Weight", ["normal", "bold"])

color_palette = plt.get_cmap("tab10")

# ===============================
# File uploader
# ===============================
uploaded_files = st.file_uploader(
    "Upload EIS CSV file(s)",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:

    col1, col2 = st.columns(2)

    # =====================================================
    # NYQUIST PLOT
    # =====================================================
    with col1:
        st.subheader("Nyquist Plot")

        fig_nyq, ax_nyq = plt.subplots(figsize=(6, 6), facecolor=bg_color)
        ax_nyq.set_facecolor(bg_color)

        for i, file in enumerate(uploaded_files):
            file.seek(0)   # 🔥 CRITICAL FIX
            df = pd.read_csv(file)

            Z_real = pd.to_numeric(df.iloc[:, 9], errors="coerce")   # Column J
            Z_imag = pd.to_numeric(df.iloc[:,10], errors="coerce")  # Column K (ALREADY POSITIVE)

            mask = ~(Z_real.isna() | Z_imag.isna())
            Z_real = Z_real[mask]
            Z_imag = Z_imag[mask]

            ax_nyq.plot(
                Z_real,
                Z_imag,
                'o-',
                linewidth=2,
                markersize=4,
                color=color_palette(i),
                label=file.name
            )

        ax_nyq.set_xlabel("Z′ (Ω)", fontsize=font_size,
                          fontweight=font_weight, color=text_color)
        ax_nyq.set_ylabel("Z″ (Ω)", fontsize=font_size,
                          fontweight=font_weight, color=text_color)
        ax_nyq.set_title("Nyquist Plot", fontsize=font_size + 2,
                         fontweight=font_weight, color=text_color)

        ax_nyq.grid(True, which="both", alpha=0.3)
        ax_nyq.legend()
        st.pyplot(fig_nyq)

    # =====================================================
    # BODE PLOT
    # =====================================================
    with col2:
        st.subheader("Bode Plot")

        fig_bode, ax_mag = plt.subplots(figsize=(7, 5), facecolor=bg_color)
        ax_mag.set_facecolor(bg_color)
        ax_phase = ax_mag.twinx()

        for i, file in enumerate(uploaded_files):
            file.seek(0)   # 🔥 CRITICAL FIX
            df = pd.read_csv(file)

            freq  = pd.to_numeric(df.iloc[:, 6], errors="coerce")  # G
            Zmag  = pd.to_numeric(df.iloc[:, 7], errors="coerce")  # H
            phase = pd.to_numeric(df.iloc[:, 8], errors="coerce")  # I

            mask = ~(freq.isna() | Zmag.isna() | phase.isna())
            freq  = freq[mask]
            Zmag  = Zmag[mask]
            phase = phase[mask]

            order = np.argsort(freq)
            freq  = freq.iloc[order]
            Zmag  = Zmag.iloc[order]
            phase = phase.iloc[order]

            ax_mag.loglog(freq, Zmag, 'o',
                          color=color_palette(i), markersize=4)
            ax_phase.semilogx(freq, phase, 'o',
                              color=color_palette(i), markersize=4)

        ax_mag.set_xlabel("Frequency (Hz)", fontsize=font_size,
                          fontweight=font_weight, color=text_color)
        ax_mag.set_ylabel("|Z| (Ω)", fontsize=font_size,
                          fontweight=font_weight, color="red")
        ax_phase.set_ylabel("Phase (degrees)", fontsize=font_size,
                            fontweight=font_weight, color="blue")

        ax_mag.grid(True, which="both", alpha=0.3)
        st.pyplot(fig_bode)
