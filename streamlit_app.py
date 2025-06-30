import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === Page Setup ===
st.set_page_config(layout="wide", page_title="Electrochemistry Dashboard", page_icon="⚛️")

# === THEME ===
with st.sidebar:
    st.markdown("## 🔌 Appearance")
    theme_mode = st.selectbox("Choose Theme", ["Smart Light", "Deep Dark"])
    if theme_mode == "Smart Light":
        bg_color = "white"
        text_color = "black"
    else:
        bg_color = "#111"
        text_color = "white"

    trace_color = st.color_picker("Raw Trace Color", "#1E90FF")
    line_color = st.color_picker("Smoothed Line Color", "#FF595E")
    label_fontsize = st.slider("Font Size", 10, 18, 13)

    st.markdown("## 📥 Upload Data")
    uploaded_file = st.file_uploader("Drop your CSV file here", type="csv")

    if uploaded_file:
        file_label = st.text_input("Legend Label for Curve", value=uploaded_file.name.split(".")[0])
        scan_rate = st.number_input("Scan Rate (mV/s) for this file", value=50)

# === Load and Process Data ===
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_columns = ["Elapsed Time (s)", "Current (A)", "Working Electrode (V)"]

    if not all(col in df.columns for col in required_columns):
        st.error("CSV missing required columns. Expecting: Elapsed Time (s), Current (A), Working Electrode (V)")
    else:
        df.dropna(inplace=True)
        df['Current_nA'] = df["Current (A)"] * 1e9

        # === PLOT CV OVERLAY ===
        with st.expander("🔍 CV Overlay Plot"):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(df["Working Electrode (V)"], df['Current_nA'], label=file_label, color=trace_color, linewidth=1.5)
            ax.set_xlabel("Voltage (V)", fontsize=label_fontsize)
            ax.set_ylabel("Current (nA)", fontsize=label_fontsize)
            ax.set_title("Cyclic Voltammetry", fontsize=label_fontsize + 2, color=text_color)
            ax.grid(True)
            ax.legend()
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(text_color)
            st.pyplot(fig)

        # === SCAN RATE STUDY ===
        with st.expander("📊 Scan Rate Study (Peak Analysis)"):
            pos_peak = df.loc[df['Current_nA'].idxmax()]
            neg_peak = df.loc[df['Current_nA'].idxmin()]
            st.markdown(f"**Positive Peak Current:** {pos_peak['Current_nA']:.2f} nA at {pos_peak['Working Electrode (V)']:.2f} V")
            st.markdown(f"**Negative Peak Current:** {neg_peak['Current_nA']:.2f} nA at {neg_peak['Working Electrode (V)']:.2f} V")

            if 'scan_data' not in st.session_state:
                st.session_state.scan_data = []

            if st.button("Add to Scan Rate Plot"):
                st.session_state.scan_data.append({
                    'Scan Rate (mV/s)': scan_rate,
                    'Ipa (nA)': pos_peak['Current_nA'],
                    'Ipc (nA)': neg_peak['Current_nA']
                })

            if st.session_state.scan_data:
                scan_df = pd.DataFrame(st.session_state.scan_data)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.scatter(scan_df['Scan Rate (mV/s)'], scan_df['Ipa (nA)'], color='crimson', label='Ipa')
                ax2.scatter(scan_df['Scan Rate (mV/s)'], scan_df['Ipc (nA)'], color='navy', label='Ipc')
                ax2.set_xlabel("Scan Rate (mV/s)", fontsize=label_fontsize)
                ax2.set_ylabel("Peak Current (nA)", fontsize=label_fontsize)
                ax2.set_title("Scan Rate vs Peak Currents", fontsize=label_fontsize + 1, color=text_color)
                ax2.legend()
                fig2.patch.set_facecolor(bg_color)
                ax2.set_facecolor(bg_color)
                ax2.tick_params(colors=text_color)
                for spine in ax2.spines.values():
                    spine.set_edgecolor(text_color)
                st.pyplot(fig2)

                st.download_button("📉 Download Scan Rate Data", scan_df.to_csv(index=False), file_name="scanrate_study.csv")

# === Footer ===
st.markdown("---")
st.markdown("Made with ❤️ for electrochemical research.")
