import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide")

st.title("🔬 Interactive CV Analysis Tool")

# =========================
# Upload Files
# =========================
uploaded_files = st.file_uploader(
    "Upload CV CSV files",
    type="csv",
    accept_multiple_files=True
)

enable_calibration = st.sidebar.checkbox(
    "Enable Calibration Curve",
    value=False
)

data_store = []

# =========================
# Read Files
# =========================
if uploaded_files:

    for file in uploaded_files:

        try:
            df = pd.read_csv(file)

        except:
            st.warning(f"Failed loading {file.name}")
            continue

        possible_voltage = [
            "Working Electrode (V)",
            "Ewe/V",
            "Voltage"
        ]

        possible_current = [
            "Current (A)",
            "I/A",
            "Current"
        ]

        v_col = next((c for c in possible_voltage if c in df.columns), None)
        i_col = next((c for c in possible_current if c in df.columns), None)

        if v_col is None or i_col is None:
            st.warning(f"{file.name} missing columns")
            continue

        voltage = df[v_col].values
        current = df[i_col].values * 1e6

        data_store.append(
            (file.name, voltage, current)
        )

# =========================
# Plot Interactive CV
# =========================
if data_store:

    fig = go.Figure()

    concentrations = []
    picked_currents = []

    for i, (name, voltage, current) in enumerate(data_store):

        with st.expander(f"{name} Settings"):

            label = st.text_input(
                f"Label {i}",
                value=name,
                key=f"label_{i}"
            )

            if enable_calibration:

                conc = st.number_input(
                    f"Concentration (µM) - Curve {i}",
                    value=float(i + 1),
                    key=f"conc_{i}"
                )

        fig.add_trace(
            go.Scatter(
                x=voltage,
                y=current,
                mode="lines",
                name=label,
                line=dict(width=3),
            )
        )

    fig.update_layout(
        title="Interactive CV Overlay",
        xaxis_title="Voltage (V)",
        yaxis_title="Current (µA)",
        height=700
    )

    st.markdown("## Click on peak points")

    selected_points = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Selected Points
    # =========================
    if selected_points:

        st.subheader("📍 Selected Points")

        calibration_data = []

        for idx, point in enumerate(selected_points):

            x = point["x"]
            y = point["y"]
            curve_number = point["curveNumber"]

            curve_name = data_store[curve_number][0]

            st.write(
                f"Curve: {curve_name} | "
                f"Voltage: {x:.4f} V | "
                f"Current: {y:.4f} µA"
            )

            if enable_calibration:

                conc = st.number_input(
                    f"Concentration for selected point {idx}",
                    value=float(idx + 1),
                    key=f"calib_{idx}"
                )

                calibration_data.append(
                    [conc, y]
                )

        # =========================
        # Calibration Curve
        # =========================
        if enable_calibration and len(calibration_data) > 1:

            calib_df = pd.DataFrame(
                calibration_data,
                columns=["Concentration", "Current"]
            )

            conc = calib_df["Concentration"].values
            curr = calib_df["Current"].values

            slope, intercept = np.polyfit(conc, curr, 1)

            fit = slope * conc + intercept

            r2 = 1 - (
                np.sum((curr - fit) ** 2)
                /
                np.sum((curr - np.mean(curr)) ** 2)
            )

            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    x=conc,
                    y=curr,
                    mode="markers",
                    name="Data"
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=conc,
                    y=fit,
                    mode="lines",
                    name="Fit"
                )
            )

            fig2.update_layout(
                title=f"Sensitivity = {slope:.4f} µA/µM | R² = {r2:.4f}",
                xaxis_title="Concentration (µM)",
                yaxis_title="Peak Current (µA)"
            )

            st.plotly_chart(fig2, use_container_width=True)

            st.write(f"### Sensitivity: {slope:.4f} µA/µM")
            st.write(f"### R²: {r2:.4f}")
