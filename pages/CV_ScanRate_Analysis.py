import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🔬 Cyclic Voltammetry (CV) Plotter")

# Upload single or multiple CSV files
uploaded_files = st.file_uploader("Upload CV CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    fig, ax = plt.subplots(figsize=(10, 6))

    for uploaded_file in uploaded_files:
        try:
            # Read file without header, auto-convert to numeric
            df = pd.read_csv(uploaded_file, header=None)
            df = df.apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)

            if df.shape[1] < 8:
                st.warning(f"⚠️ File {uploaded_file.name} must have at least 8 columns.")
                continue

            voltage = df.iloc[:, 5]  # Column 6
            current = df.iloc[:, 7]  # Column 8

            ax.plot(voltage, current, label=uploaded_file.name, linewidth=2)
        except Exception as e:
            st.error(f"❌ Error with {uploaded_file.name}: {e}")

    ax.set_title("CV Overlay Plot", fontsize=16, fontweight='bold')
    ax.set_xlabel("Voltage (V)", fontsize=14)
    ax.set_ylabel("Current (A)", fontsize=14)
    ax.grid(True)
    ax.legend(loc="best", fontsize=9)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    st.pyplot(fig)

    # Export button
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=600, bbox_inches='tight', facecolor="white")
    st.download_button(
        label="📥 Download High-Res PNG",
        data=buf.getvalue(),
        file_name="cv_overlay_plot.png",
        mime="image/png"
    )
