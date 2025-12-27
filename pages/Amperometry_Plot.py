import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ======================
# Sidebar Inputs
# ======================
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
label = st.sidebar.text_input("Label", value="sensor1")

start_time = st.sidebar.number_input("Plot Start Time (s)", value=1500)
end_time = st.sidebar.number_input("Plot End Time (s)", value=2000)

overlay_raw = st.sidebar.checkbox("Overlay raw trace", value=True)
show_spike_arrows = st.sidebar.checkbox("Show Spike Concentration Labels", value=True)

spike_start = st.sidebar.number_input("Spike Start Time (s)", value=1500)
spike_interval = st.sidebar.number_input("Spike Interval (s)", value=50)
spike_count = st.sidebar.number_input("Number of Spikes", value=10)
conc_per_spike = st.sidebar.number_input("Concentration per Spike (µM)", value=20.0)

ROLLING_WINDOW = 20
SPIKE_WINDOW = 5

# ======================
# Load Data
# ======================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    TIME_COL = df.columns[3]
    CURRENT_COL = df.columns[6]

    plot_df = df[(df[TIME_COL] >= start_time) & (df[TIME_COL] <= end_time)].copy()

    time = plot_df[TIME_COL].values
    current_nA = plot_df[CURRENT_COL].values

    smoothed = (
        pd.Series(current_nA)
        .rolling(window=ROLLING_WINDOW, center=True)
        .mean()
        .values
    )

    # ======================
    # Spike Logic
    # ======================
    spike_times = np.arange(
        spike_start,
        spike_start + spike_interval * spike_count,
        spike_interval
    )

    concentrations = np.arange(
        conc_per_spike,
        conc_per_spike * (spike_count + 1),
        conc_per_spike
    )

    spike_currents = []

    for t in spike_times:
        mask = (plot_df[TIME_COL] >= t - SPIKE_WINDOW) & \
               (plot_df[TIME_COL] <= t + SPIKE_WINDOW)
        spike_currents.append(plot_df.loc[mask, CURRENT_COL].mean())

    y = np.array(spike_currents)
    X = concentrations.reshape(-1, 1)

    # ======================
    # Linear Regression
    # ======================
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)

    # ======================
    # ✅ CORRECT LOD (baseline noise)
    # ======================
    baseline_mask = plot_df[TIME_COL] < spike_start
    baseline_current = plot_df.loc[baseline_mask, CURRENT_COL]

    baseline_std = baseline_current.std()
    LOD = (3 * baseline_std) / slope

    # ======================
    # Plotting
    # ======================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Plot A: Chronoamperometry ----
    if overlay_raw:
        ax1.plot(time, current_nA, linewidth=0.5, alpha=0.6)

    ax1.plot(time, smoothed, linewidth=1.8, color="red")

    if show_spike_arrows:
        for t, conc in zip(spike_times, concentrations):
            yval = np.interp(t, time, smoothed)
            ax1.annotate(
                f"{int(conc)} µM",
                xy=(t, yval),
                xytext=(t, yval + 5),
                arrowprops=dict(arrowstyle="->", lw=1)
            )

    ax1.set_xlim(start_time, end_time)
    ax1.set_xticks(spike_times)
    ax1.set_xticklabels(spike_times.astype(int))

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Current (nA)")
    ax1.set_title("A", loc="left", fontweight="bold")

    # ---- Plot B: Calibration ----
    ax2.scatter(concentrations, y, s=70, edgecolors="black")
    ax2.plot(concentrations, y_pred, linewidth=2)

    box_text = (
        f"R² = {r2:.4f}\n"
        f"LOD = {LOD:.2f} µM\n"
        f"Sensitivity = {slope:.2f} nA/µM\n"
        f"y = {slope:.2f}x + {intercept:.2f}"
    )

    ax2.text(
        0.05, 0.95, box_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", pad=0.4)
    )

    ax2.set_xlabel("Concentration (µM)")
    ax2.set_ylabel("Current (nA)")
    ax2.set_xticks(concentrations)
    ax2.set_title("B", loc="left", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)

    # ======================
    # Downloads
    # ======================
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)

    st.download_button(
        "📷 Download Figure",
        buf.getvalue(),
        file_name=f"{label}_figure.png",
        mime="image/png"
    )

    result_df = pd.DataFrame({
        "Spike Time (s)": spike_times,
        "Concentration (µM)": concentrations,
        "Avg Current (nA)": y,
        "Predicted Current (nA)": y_pred
    })

    st.download_button(
        "📄 Download CSV Result Table",
        result_df.to_csv(index=False),
        file_name=f"{label}_results.csv"
    )

    st.success("✅ Analysis complete")

else:
    st.warning("Please upload a CSV file.")
