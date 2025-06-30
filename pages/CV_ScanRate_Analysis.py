import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🌌 CV Plotter with Colormap & µA Scaling")

uploaded_files = st.file_uploader("Upload CV CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.colormaps.get_cmap('viridis').resampled(len(uploaded_files))

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            df = pd.read_csv(uploaded_file, header=1)
            df = df.apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)

            if df.shape[1] <= 7:
                st.warning(f"⚠️ {uploaded_file.name} has fewer than 8 columns.")
                continue

            voltage = df.iloc[:, 5].values
            current_uA = df.iloc[:, 7].values * 1e6  # Convert from A to µA

            label = uploaded_file.name.replace("_", " ").replace(".csv", "")

            ax.plot(voltage, current_uA, label=label, color=cmap(i), linewidth=2)

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")

    # Final formatting
    ax.set_title("CV Overlay Plot (Current in µA)", fontsize=16, color='white')
    ax.set_xlabel("Voltage (V)", fontsize=14, color='white')
    ax.set_ylabel("Current (µA)", fontsize=14, color='white')
    ax.tick_params(colors='white')
    ax.legend(fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    st.pyplot(fig)

    # Save high-res PNG
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=600, facecolor='black')
    st.download_button(
        label="📥 Download High-Res Dark PNG",
        data=buf.getvalue(),
        file_name="CV_overlay_dark.png",
        mime="image/png"
    )
