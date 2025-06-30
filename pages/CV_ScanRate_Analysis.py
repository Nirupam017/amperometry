import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🔬 CV Plotter (Auto Column Detection + µA)")

uploaded_files = st.file_uploader("Upload CV CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.colormaps.get_cmap('viridis').resampled(len(uploaded_files))

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Try reading with automatic header guess
            raw = pd.read_csv(uploaded_file, skip_blank_lines=True, header=None)
            
            # Find row where "Current" or "Voltage" likely starts
            header_row = None
            for idx, row in raw.iterrows():
                if row.astype(str).str.contains("Current", case=False).any():
                    header_row = idx
                    break

            if header_row is None:
                st.warning(f"⚠️ No valid header found in {uploaded_file.name}")
                continue

            # Re-read with correct header
            df = pd.read_csv(uploaded_file, header=header_row)
            df = df.apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)

            # Auto column detection
            voltage_col = [col for col in df.columns if "Voltage" in col or "Working" in col]
            current_col = [col for col in df.columns if "Current" in col]

            if not voltage_col or not current_col:
                st.warning(f"⚠️ Couldn't detect voltage/current columns in {uploaded_file.name}")
                continue

            voltage = df[voltage_col[0]].values
            current_uA = df[current_col[0]].values * 1e6

            ax.plot(voltage, current_uA, label=uploaded_file.name.replace("_", " ").replace(".csv", ""), color=cmap(i), linewidth=2)

        except Exception as e:
            st.error(f"❌ Error in file {uploaded_file.name}: {e}")

    # Final formatting
    ax.set_title("CV Overlay Plot (Current in µA)", fontsize=16, color='white')
    ax.set_xlabel("Voltage (V)", fontsize=14, color='white')
    ax.set_ylabel("Current (µA)", fontsize=14, color='white')
    ax.tick_params(colors='white')
    ax.legend(fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    st.pyplot(fig)

    # Download high-res
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=600, facecolor='black')
    st.download_button("📥 Download CV Overlay", data=buf.getvalue(), file_name="cv_overlay_uA.png", mime="image/png")
