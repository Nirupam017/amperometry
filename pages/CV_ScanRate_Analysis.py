import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🔬 Cyclic Voltammetry (CV) Plotter")

uploaded_files = st.file_uploader("Upload CV CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    fig, ax = plt.subplots(figsize=(10, 6))

    for uploaded_file in uploaded_files:
        try:
            # Read with headers
            df = pd.read_csv(uploaded_file)

            # Ensure numeric conversion
            df = df.apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)

            if 'Working Electrode (V)' not in df.columns or 'Current (A)' not in df.columns:
                st.warning(f"⚠️ {uploaded_file.name} is missing required columns.")
                continue

            voltage = df['Working Electrode (V)']
            current = df['Current (A)']

            ax.plot(voltage, current, label=uploaded_file.name, linewidth=2)

            # Optional debug
            st.write(f"Preview of {uploaded_file.name}")
            st.dataframe(df[['Working Electrode (V)', 'Current (A)']].head(5))

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")

    ax.set_title("CV Overlay Plot", fontsize=16, fontweight='bold')
    ax.set_xlabel("Voltage (V)", fontsize=14)
    ax.set_ylabel("Current (A)", fontsize=14)
    ax.grid(True)
    ax.legend(loc="best", fontsize=9)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=600, bbox_inches='tight', facecolor="white")
    st.download_button(
        label="📥 Download High-Res PNG",
        data=buf.getvalue(),
        file_name="cv_overlay_plot.png",
        mime="image/png"
    )
