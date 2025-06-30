import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from scipy.stats import linregress
import matplotlib.cm as cm

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("📈 CV Overlay + Scan Rate Study Tool")

# Theme toggle
theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
plt.style.use('dark_background' if theme == "Dark" else 'default')
bg_color = 'black' if theme == "Dark" else 'white'
font_color = 'white' if theme == "Dark" else 'black'

# Function to read file robustly
def read_file_safely(file):
    try:
        file.seek(0)
        df = pd.read_csv(file, header=None)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        if df.shape[1] >= 2:
            return df
    except Exception as e:
        st.error(f"❌ Could not read file {file.name}: {e}")
    return None

# --- CV Overlay Section ---
st.header("1️⃣ CV Overlay")
num_cv_files = st.number_input("Number of CV files:", 1, 20, 1)
cv_files = st.file_uploader("Upload CV files (CSV only):", type=['csv'], accept_multiple_files=True, key='cv')

# User-defined voltage range
col1, col2 = st.columns(2)
with col1:
    v_start = st.number_input("Start Voltage (V)", value=-0.8)
with col2:
    v_end = st.number_input("End Voltage (V)", value=0.8)

if len(cv_files) == num_cv_files:
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    cmap1 = cm.get_cmap('viridis', num_cv_files)

    for i, file in enumerate(cv_files):
        df = read_file_safely(file)
        if df is not None:
            voltage = df.iloc[:, 0].values
            current = df.iloc[:, 1].values

            # Filter by user voltage range
            mask = (voltage >= v_start) & (voltage <= v_end)
            voltage = voltage[mask]
            current = current[mask]

            label = file.name.split('.')[0].replace('_', ' ')
            ax1.plot(voltage, current, label=label, color=cmap1(i), linewidth=2)

    ax1.set_title("CV Overlay", color=font_color, fontsize=15, fontweight='bold')
    ax1.set_xlabel("Voltage (V)", color=font_color, fontsize=13, fontweight='bold')
    ax1.set_ylabel("Current (A)", color=font_color, fontsize=13, fontweight='bold')
    ax1.tick_params(colors=font_color)
    ax1.legend(fontsize=9)
    ax1.set_facecolor(bg_color)
    st.pyplot(fig1)

    buf1 = BytesIO()
    fig1.savefig(buf1, dpi=600, facecolor=bg_color)
    st.download_button("📥 Download CV Overlay", buf1.getvalue(), file_name="cv_overlay.png")
