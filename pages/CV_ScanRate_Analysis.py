import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from scipy.signal import savgol_filter
from scipy.stats import linregress
import matplotlib.cm as cm

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("📈 CV Overlay + Scan Rate Study Tool")

theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
plt.style.use('dark_background' if theme == "Dark" else 'default')
bg_color = 'black' if theme == "Dark" else 'white'
font_color = 'white' if theme == "Dark" else 'black'

def read_file_safely(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    for sep in [',', '\t', ';']:
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=sep)
            if df.shape[1] >= 2:
                return df
        except:
            continue
    st.error(f"❌ Could not read file: {file.name}")
    return None

# --- CV Overlay Plot ---
st.header("1️⃣ CV Overlay")
num_cv_files = st.number_input("Number of CV files:", 1, 20, 3)
cv_files = st.file_uploader("Upload CV files (CSV/XLSX):", type=['csv', 'xlsx'], accept_multiple_files=True, key='cv')

# Ask user for Voltage (X-axis) range
st.subheader("Set Voltage Range for Plotting (X-axis)")
v_min = st.number_input("Start Voltage (V):", value=-1.0)
v_max = st.number_input("End Voltage (V):", value=1.0)

if len(cv_files) == num_cv_files:
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    cmap1 = cm.get_cmap('viridis', num_cv_files)

    for i, file in enumerate(cv_files):
        df = read_file_safely(file)
        if df is not None:
            voltage = df.iloc[:, 0].values
            current = df.iloc[:, 1].values

            # Filter data within user-defined voltage range
            mask = (voltage >= v_min) & (voltage <= v_max)
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

# --- Scan Rate Study ---
st.header("2️⃣ Scan Rate Study")
num_sr_files = st.number_input("Number of scan rate files:", 1, 10, 3)
scanrate_files, scanrates, peak_currents = [], [], []

for i in range(num_sr_files):
    col1, col2 = st.columns([3, 1])
    with col1:
        file = st.file_uploader(f"Upload file #{i+1}", type=['csv', 'xlsx'], key=f'srfile{i}')
    with col2:
        rate = st.number_input(f"Scan rate #{i+1} (mV/s):", key=f'srrate{i}', value=50.0)

    if file:
        df = read_file_safely(file)
        if df is not None:
            scanrate_files.append(df)
            scanrates.append(rate)

if len(scanrate_files) == num_sr_files:
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    cmap2 = cm.get_cmap('plasma', num_sr_files)

    for i, (df, rate) in enumerate(zip(scanrate_files, scanrates)):
        voltage = df.iloc[:, 0].values
        current = df.iloc[:, 1].values
        label = f"{rate:.0f} mV/s"
        ax2.plot(voltage, current, label=label, color=cmap2(i), linewidth=2)
        peak_current = np.max(np.abs(current))
        peak_currents.append(peak_current * 1e6)  # µA

    ax2.set_title("Scan Rate Curves", color=font_color, fontsize=15, fontweight='bold')
    ax2.set_xlabel("Voltage (V)", color=font_color, fontsize=13, fontweight='bold')
    ax2.set_ylabel("Current (A)", color=font_color, fontsize=13, fontweight='bold')
    ax2.tick_params(colors=font_color)
    ax2.legend(fontsize=9)
    ax2.set_facecolor(bg_color)
    st.pyplot(fig2)

    buf2 = BytesIO()
    fig2.savefig(buf2, dpi=600, facecolor=bg_color)
    st.download_button("📥 Download Scan Rate Plot", buf2.getvalue(), file_name="scan_rate_plot.png")

    # === Randles-Sevcik plot ===
    st.subheader("📊 3️⃣ Randles-Sevcik Plot: Ip vs √Scan Rate")
    scanrates_array = np.sqrt(np.array(scanrates))
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    slope, intercept, r_val, *_ = linregress(scanrates_array, peak_currents)
    fit_line = slope * scanrates_array + intercept

    ax3.scatter(scanrates_array, peak_currents, color='blue', label='Data')
    ax3.plot(scanrates_array, fit_line, color='red', label=f"Fit: y = {slope:.2f}√v + {intercept:.2f}")
    ax3.text(0.05, 0.9, f"$R^2$ = {r_val**2:.4f}", transform=ax3.transAxes,
             fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
    ax3.set_xlabel("√Scan Rate (√mV/s)", fontsize=13, fontweight='bold')
    ax3.set_ylabel("Peak Current (µA)", fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    buf3 = BytesIO()
    fig3.savefig(buf3, dpi=600)
    st.download_button("📥 Download Randles Plot", buf3.getvalue(), file_name="randles_sevcik_plot.png")
