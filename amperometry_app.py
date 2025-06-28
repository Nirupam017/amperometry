import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.signal import find_peaks

st.set_page_config(layout="wide")
st.title("🔬 Amperometry Analysis Tool")

st.sidebar.header("Upload and Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

label = st.sidebar.text_input("Label", value="sensor1")
start_time = st.sidebar.number_input("Start Time (s)", value=280)
end_time = st.sidebar.number_input("End Time (s)", value=499)
auto_detect = st.sidebar.checkbox("Auto-detect spikes", value=False)
overlay_raw = st.sidebar.checkbox("Overlay raw trace", value=True)

# Default for manual spike setup
spike_start = st.sidebar.number_input("Spike Start (s)", value=300)
spike_interval = st.sidebar.number_input("Spike Interval (s)", value=20)
spike_count = st.sidebar.number_input("Spike Count", value=10)
conc_per_spike = st.sidebar.number_input("Conc/Spike (µM)", value=20.0)

yaxis_min = st.sidebar.number_input("Y-axis Min (nA)", value=-20)
yaxis_max = st.sidebar.number_input("Y-axis Max (nA)", value=100)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    TIME_COL = "Elapsed Time (s)"
    CURRENT_COL = "Current (A)"
    SPIKE_WINDOW = 2
    ROLLING_WINDOW = 15

    plot_df = df[(df[TIME_COL] >= start_time) & (df[TIME_COL] <= end_time)].copy()
    time = plot_df[TIME_COL].values
    current_nA = plot_df[CURRENT_COL].values * 1e9
    smoothed = pd.Series(current_nA).rolling(window=ROLLING_WINDOW, center=True).mean().values

    # Auto spike detection
    if auto_detect:
        peaks, _ = find_peaks(smoothed, distance=spike_interval * 5, height=np.nanmean(smoothed) + 5)
        spike_times = time[peaks][:spike_count]
        concentrations = np.arange(conc_per_spike, conc_per_spike * (len(spike_times) + 1), conc_per_spike)
    else:
        spike_times = np.arange(spike_start, spike_start + spike_interval * spike_count, spike_interval)
        concentrations = np.arange(conc_per_spike, conc_per_spike * spike_count + 1, conc_per_spike)

    spike_currents = []
    valid_concs = []
    valid_spike_times = []

    for i, t in enumerate(spike_times):
        mask = (plot_df[TIME_COL] >= t - SPIKE_WINDOW) & (plot_df[TIME_COL] <= t + SPIKE_WINDOW)
        window = plot_df.loc[mask, CURRENT_COL]
        if not window.empty:
            avg = window.mean()
            spike_currents.append(avg * 1e9)
            valid_concs.append(concentrations[i])
            valid_spike_times.append(t)

    if len(spike_currents) >= 2:
        X = np.array(valid_concs).reshape(-1, 1)
        y = np.array(spike_currents)
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, y_pred)

        baseline = df[(df[TIME_COL] >= start_time) & (df[TIME_COL] < spike_start - 5)]
        baseline_std = np.std(baseline[CURRENT_COL].values) * 1e9
        LOD = (3 * baseline_std) / slope

        # --- PLOTS ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        if overlay_raw:
            ax1.plot(time, current_nA, color='lightgray', linewidth=0.8, label="Raw")

        ax1.plot(time, smoothed, color='red', linewidth=1.5, label="Smoothed")
        for t, conc in zip(spike_times, concentrations):
            yval = np.interp(t, time, smoothed)
            ax1.annotate(f"{conc:.0f} µM", xy=(t, yval), xytext=(t, yval + 5),
                         arrowprops=dict(arrowstyle='->'), ha='center', fontsize=9)

        ax1.set_xlabel("t /s", fontsize=14)
        ax1.set_ylabel("Current /nA", fontsize=14)
        ax1.set_title("A", loc='left', fontsize=16, fontweight='bold')
        ax1.set_xticks(np.arange(start_time, end_time + 1, spike_interval))
        ax1.set_ylim(yaxis_min, yaxis_max)
        ax1.legend()
        for spine in ax1.spines.values():
            spine.set_linewidth(1.2)

        ax2.errorbar(valid_concs, y, fmt='o', color='red', label="Data", yerr=2)
        ax2.plot(valid_concs, y_pred, color='black', label="Fit")
        ax2.text(0.55, 0.15,
                 f'y = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r2:.3f}\nLOD = {LOD:.2f} µM\nSensitivity = {slope:.2f} nA/µM',
                 transform=ax2.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='black'))
        ax2.set_xlabel("Concentration /µM", fontsize=14)
        ax2.set_ylabel("Current /nA", fontsize=14)
        ax2.set_title("B", loc='left', fontsize=16, fontweight='bold')
        ax2.set_xticks(valid_concs)
        ax2.legend(loc='lower right', fontsize=10)
        for spine in ax2.spines.values():
            spine.set_linewidth(1.2)

        plt.tight_layout()
        st.pyplot(fig)

        # --- Download buttons ---
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button("📷 Download Figure", buf.getvalue(), file_name=f"{label}_figure.png", mime="image/png")

        result_df = pd.DataFrame({
            "Spike Time (s)": valid_spike_times,
            "Concentration (µM)": valid_concs,
            "Avg Current (nA)": y,
            "Predicted Current (nA)": y_pred
        })
        st.download_button("📄 Download CSV Result Table", result_df.to_csv(index=False), file_name=f"{label}_results.csv")

        st.success(f"✅ Done! Label: **{label}**")
        st.markdown(f"- **Sensitivity**: `{slope:.2f} nA/µM`")
        st.markdown(f"- **LOD**: `{LOD:.2f} µM`")
        st.markdown(f"- **R²**: `{r2:.4f}`")
    else:
        st.warning("⚠️ Not enough valid spikes detected.")
