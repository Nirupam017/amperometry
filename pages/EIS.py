import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔬 EIS Analysis Tool")

bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

font_size = st.sidebar.slider("Font Size", 8, 20, 14)
font_weight = st.sidebar.selectbox("Font Weight", ["normal", "bold"])

color_palette = plt.get_cmap("tab10")

uploaded_files = st.file_uploader(
    "Upload EIS CSV Files",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:

    col1, col2 = st.columns(2)

    # =====================
    # NYQUIST
    # =====================
    with col1:
        fig_nyq, ax_nyq = plt.subplots(figsize=(6, 6), facecolor=bg_color)
        ax_nyq.set_facecolor(bg_color)

        for i, file in enumerate(uploaded_files):
            file.seek(0)  # 🔥 FIX
            df = pd.read_csv(file)

            Z_real = df.iloc[:, 8].values   # Column I
            Z_imag = df.iloc[:, 9].values   # Column J (already negative)

            ax_nyq.plot(
                Z_real,
                Z_imag,
                'o-',
                color=color_palette(i),
                linewidth=2,
                markersize=4,
                label=file.name
            )

        ax_nyq.set_xlabel("Z′ (Ω)", fontsize=font_size,
                          fontweight=font_weight, color=text_color)
        ax_nyq.set_ylabel("−Z″ (Ω)", fontsize=font_size,
                          fontweight=font_weight, color=text_color)

        ax_nyq.grid(True, which="both", alpha=0.3)
        ax_nyq.legend()
        st.pyplot(fig_nyq)

    # =====================
    # BODE
    # =====================
    with col2:
        fig_bode, ax_mag = plt.subplots(figsize=(7, 6), facecolor=bg_color)
        ax_mag.set_facecolor(bg_color)
        ax_phase = ax_mag.twinx()

        for i, file in enumerate(uploaded_files):
            file.seek(0)  # 🔥 FIX AGAIN
            df = pd.read_csv(file)

            freq = df.iloc[:, 6].values     # Column G
            Z_mag = df.iloc[:, 7].values    # Column H
            phase = df.iloc[:, 10].values  # Column K (already negative)

            ax_mag.loglog(freq, Z_mag, 'o-', color=color_palette(i), linewidth=2)
            ax_phase.semilogx(freq, phase, 's--', color=color_palette(i), linewidth=2)

        ax_mag.set_xlabel("Frequency (Hz)", fontsize=font_size,
                          fontweight=font_weight, color=text_color)
        ax_mag.set_ylabel("|Z| (Ω)", fontsize=font_size,
                          fontweight=font_weight, color="red")
        ax_phase.set_ylabel("Phase (°)", fontsize=font_size,
                            fontweight=font_weight, color="blue")

        ax_mag.grid(True, which="both", alpha=0.3)
        st.pyplot(fig_bode)
