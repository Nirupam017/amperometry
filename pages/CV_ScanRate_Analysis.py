import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🔬 Cyclic Voltammetry (CV) Plotter (Current in µA)")

uploaded_files = st.file_uploader("Upload CV CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    fig, ax = plt.subplots(figsize=(10, 6))

    for uploaded_file in uploaded_files:
        try:
            # Read CSV assuming first row is header (adjust if needed)
            df = pd.read_csv(uploaded_file, header=1)

            # Convert all entries to numeric (handle scientific notation like 3.84E-06)
            df = df.apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)

            # Use column 6 (index 5) for voltage and column 8 (index 7) for current
            if df.shape[1] <= 7:
                st.warning(f"⚠️ {uploaded_file.name} has fewer than 8 columns.")
                continue

            voltage = df.iloc[:, 5]
            current_uA = df.iloc[:, 7] * 1e6  # Convert A to µA

            #if voltage.empty or current_uA.empty:
              #  st.warning(f"⚠️ No valid voltage/current data in {uploaded_file.name}")
              #  continue

            ax.plot(voltage, current_uA, label=uploaded_file.name, linewidth=2)

            # Preview data
            st.subheader(f"🔍 Preview: {uploaded_file.name}")
            st.dataframe(pd.DataFrame({
                "Voltage (V)": voltage.head(5),
                "Current (µA)": current_uA.head(5)
            }))

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")

    # Final plot formatting
    ax.set_title("CV Overlay Plot", fontsize=16, fontweight='bold')
    ax.set_xlabel("Voltage (V)", fontsize=14)
    ax.set_ylabel("Current (µA)", fontsize=14)
    ax.grid(True)
    ax.legend(loc="best", fontsize=9)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    st.pyplot(fig)

    # Download button
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=600, bbox_inches='tight', facecolor="white")
    st.download_button(
        label="📥 Download High-Res PNG",
        data=buf.getvalue(),
        file_name="cv_overlay_plot.png",
        mime="image/png"
    )
