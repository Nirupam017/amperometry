import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout="wide", page_title="Electrochem Analyzer", page_icon="🔬")

# --- Sidebar Settings ---
with st.sidebar:
    st.markdown("## 🎨 Appearance")
    theme_mode = st.radio("Theme", ["Light", "Dark"])
    trace_color = st.color_picker("Raw Trace Color", "#1E90FF")
    line_color = st.color_picker("Smoothed Line Color", "#FF595E")
    st.markdown("---")
    st.markdown("## 📥 Upload & Settings")
    uploaded_file = st.file_uploader("Drop CSV here", type="csv")
    if uploaded_file:
        file_label = st.text_input("Legend Label", value=uploaded_file.name.split(".")[0])
        scan_rate = st.number_input("Scan Rate (mV/s)", value=50.0)

# --- Theme Setup ---
if theme_mode == "Dark":
    bg_color, text_color = "#111", "white"
    plt.style.use("dark_background")
else:
    bg_color, text_color = "white", "black"
    plt.style.use("default")

# --- Header ---
st.markdown(
    f"<div style='text-align:center; padding:10px; background-color:{bg_color};'>"
    f"<h1 style='color:#FF595E;'>🔬 ElectroChem Analyzer</h1>"
    f"<h4 style='color:gray;'>Smart CV & Scan‑Rate Dashboard</h4>"
    "</div>",
    unsafe_allow_html=True
)
st.markdown("---")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "Working Electrode (V)" not in df or "Current (A)" not in df:
        st.error("CSV missing required columns.")
    else:
        df = df.dropna(subset=["Working Electrode (V)", "Current (A)"])
        df["Current_µA"] = df["Current (A)"] * 1e6

        # --- CV Overlay Plot ---
        st.subheader("📈 CV Overlay")
        fig, ax = plt.subplots(figsize=(7,5), facecolor=bg_color)
        ax.plot(df["Working Electrode (V)"], df["Current_µA"],
                color=trace_color, label=file_label, linewidth=1.8)
        ax.set_facecolor(bg_color)
        ax.set_xlabel("Voltage (V)", color=text_color)
        ax.set_ylabel("Current (µA)", color=text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
        ax.legend(facecolor=bg_color, edgecolor=text_color)
        st.pyplot(fig)

        # --- Scan Rate Analysis ---
        st.subheader("📉 Scan Rate Study")
        pos_peak = df.iloc[df["Current_µA"].idxmax()]
        neg_peak = df.iloc[df["Current_µA"].idxmin()]
        ipa = float(pos_peak["Current_µA"])
        ipc = float(neg_peak["Current_µA"])
        st.markdown(f"- **Anodic Peak (Ipa):** {ipa:.2f} µA at {pos_peak['Working Electrode (V)']:.3f} V")
        st.markdown(f"- **Cathodic Peak (Ipc):** {ipc:.2f} µA at {neg_peak['Working Electrode (V)']:.3f} V")

        # Track scan rate data across runs
        if "scan_records" not in st.session_state:
            st.session_state.scan_records = []
        if st.button("🟢 Add to Scan Rate Study"):
            st.session_state.scan_records.append({"scan": scan_rate, "ipa": ipa, "ipc": ipc})

        records = st.session_state.scan_records
        if records:
            df2 = pd.DataFrame(records)
            model = LinearRegression().fit(df2[["scan"]], df2["ipa"])
            ipa_pred = model.predict(df2[["scan"]])
            r2 = r2_score(df2["ipa"], ipa_pred)

            fig2, ax2 = plt.subplots(figsize=(7,5), facecolor=bg_color)
            ax2.scatter(df2["scan"], df2["ipa"], label="Ipa", color="crimson", s=60)
            ax2.plot(df2["scan"], ipa_pred, color="crimson", linestyle="--")
            ax2.set_facecolor(bg_color)
            ax2.set_xlabel("Scan Rate (mV/s)", color=text_color)
            ax2.set_ylabel("Ipa (µA)", color=text_color)
            ax2.tick_params(colors=text_color)
            for spine in ax2.spines.values():
                spine.set_edgecolor(text_color)
            ax2.text(0.05,0.95,f"R²={r2:.3f}", transform=ax2.transAxes,
                     bbox=dict(facecolor=bg_color, edgecolor=text_color),
                     color=text_color)
            st.pyplot(fig2)
            st.download_button("📄 Download Scan Rate Data", df2.to_csv(index=False), file_name="scan_rate_data.csv")

st.markdown("---")
st.markdown("<div style='text-align:center;'>Made for electrochemist minds.</div>", unsafe_allow_html=True)

