import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🔬 CV Analysis Tool")

bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

st.header("📈 CV Curve Overlay with Manual Peak Marking")
uploaded_files = st.file_uploader("Upload CV CSV Files", type="csv", accept_multiple_files=True)

# Session state for peak storage
if "peaks" not in st.session_state:
    st.session_state.peaks = []

if uploaded_files:
    fig = go.Figure()

    for i, file in enumerate(uploaded_files):
        with st.expander(f"Customize {file.name}"):
            label = st.text_input(f"Legend Label for {file.name}", value=file.name, key=f"label_{i}")
            color = st.color_picker(f"Color for {file.name}", key=f"color_{i}")

        df = pd.read_csv(file)
        df.dropna(subset=["Working Electrode (V)", "Current (A)"], inplace=True)
        voltage = df["Working Electrode (V)"].values
        current = df["Current (A)"].values * 1e6

        fig.add_trace(go.Scatter(x=voltage, y=current, mode="lines", name=label, line=dict(color=color)))

    fig.update_layout(
        title="CV Curve Overlay (Click to mark peaks)",
        xaxis_title="Voltage (V)",
        yaxis_title="Current (µA)",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        dragmode="pan"
    )

    # Show plot
    selected_point = st.plotly_chart(fig, use_container_width=True)

    # Manual Peak Marking
    st.markdown("### 🧪 Mark Peaks")
    peak_type = st.radio("Peak Type", ["Anodic", "Cathodic"])
    peak_voltage = st.number_input("Voltage of Peak (V)", format="%.4f")
    peak_current = st.number_input("Current of Peak (µA)", format="%.2f")
    if st.button("➕ Add Peak"):
        st.session_state.peaks.append({
            "Type": peak_type,
            "Voltage (V)": peak_voltage,
            "Current (µA)": peak_current
        })

    # Show marked peaks
    if st.session_state.peaks:
        st.markdown("### 📍 Marked Peaks")
        st.dataframe(pd.DataFrame(st.session_state.peaks))

        # Optionally allow download
        csv = pd.DataFrame(st.session_state.peaks).to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download Peaks CSV", csv, "marked_peaks.csv", "text/csv")

    # Option to reset
    if st.button("❌ Clear All Peaks"):
        st.session_state.peaks = []
