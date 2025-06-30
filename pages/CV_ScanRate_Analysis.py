import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Streamlit config
st.set_page_config(page_title="CV Plotter", layout="centered", page_icon="🔬")
st.title("🔬 Cyclic Voltammetry Overlay Plot (Current in µA)")

# File uploader
uploaded_files = st.file_uploader("Upload one or more CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.colormaps.get_cmap('plasma').resampled(len(uploaded_files))

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            df = pd.read_csv(uploaded_file)

            # Ensure expected columns exist
            if "Working Electrode (V)" not in df.columns or "Current (A)" not in df.columns:
                st.warning(f"⚠️ Skipping {uploaded_file.name}: required columns not found.")
                continue

            voltage = df["Working Electrode (V)"].astype(float).values
            current_uA = df["Current (A)"].astype(float).values * 1e6  # Convert A → µA

            label = uploaded_file.name.replace("_", " ")
            ax.plot(voltage, current_uA, label=label, color=cmap(i), linewidth=2)

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")

    ax.set_xlabel("Voltage (V)", color='white')
    ax.set_ylabel("Current (µA)", color='white')
    ax.set_title("CV Overlay Plot (Current in µA)", color='white')
    ax.tick_params(colors='white')
    ax.legend(fontsize=8)
    ax.grid(False)
    fig.tight_layout()

    st.pyplot(fig)
