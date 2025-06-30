import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

st.set_page_config(layout="wide")
st.title("🔬 Amperometry Analysis Tool")

# Theme toggle
theme = st.sidebar.radio("Theme Mode", ["Light", "Dark"], index=0)
bg_color = "black" if theme == "Dark" else "white"
text_color = "white" if theme == "Dark" else "black"
raw_color = "steelblue"  # Better than gray
smooth_color = "#FF595E"

# Sidebar inputs
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
label = st.sidebar.text_input("Label", value="sensor1")
start_time = st.sidebar.number_input("Start Time (s)", value=280)
end_time = st.sidebar.number_input("End Time (s)", value=499)
overlay_raw = st.sidebar.checkbox("Overlay raw trace", value=True)
custom_yaxis = st.sidebar.checkbox("Set Y-axis manually", value=False)
custom_xaxis = st.sidebar.checkbox("Set X-axis manually", value=False)
enable_inset = st.sidebar.checkbox("Enable Inset Zoom Plot", value=False)

spike_start = st.sidebar.number_input("Spike Start (s)", value=300)
spike_interval = st.sidebar.number_input("Spike Interval (s)", value=20)
spike_count = st.sidebar.number_input("Spike Count", value=10)
conc_per_spike = st.sidebar.number_input("Conc/Spike (µM)", value=20.0)

if custom_yaxis:
    yaxis_min = st.sidebar.number_input("Y-axis Min (nA)", value=0.0)
    yaxis_max = st.sidebar.number_input("Y-axis Max (nA)", value=100.0)
if custom_xaxis:
    xaxis_min = st.sidebar.number_input("X-axis Min (s)", value=start_time)
    xaxis_max = st.sidebar.number_input("X-axis Max (s)", value=end_time)
if enable_inset:
    inset_start = st.sidebar.number_input("Inset Start Time (s)", value=int(start_time + 40))
    inset_end = st.sidebar.number_input("Inset End Time (s)", value=int(end_time - 40))

# Main processing
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
            valid_concs.append(int(concentrations[i]))
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

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=bg_color)

        # --- Panel A ---
        if overlay_raw:
            ax1.plot(time, current_nA, color=raw_color, linewidth=0.8)
        ax1.plot(time, smoothed, color=smooth_color, linewidth=1.5)

        for t, conc in zip(spike_times, concentrations):
            yval = np.interp(t, time, smoothed)
            ax1.annotate(f"{int(conc)} µM", xy=(t, yval), xytext=(t, yval + 3),
                         arrowprops=dict(arrowstyle='->', color=text_color),
                         ha='center', fontsize=9, fontweight='bold', color=text_color)

        ax1.set_xlabel("Time (s)", fontsize=14, fontweight='bold', color=text_color)
        ax1.set_ylabel("Current (nA)", fontsize=14, fontweight='bold', color=text_color)
        ax1.set_title("A", loc='left', fontsize=16, fontweight='bold', color=text_color)

        xticks = np.arange(start_time, end_time + 1, spike_interval)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([str(int(x)) for x in xticks], fontweight='bold', color=text_color)
        ax1.tick_params(axis='both', labelsize=12, width=1.5, colors=text_color)
        ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', color=text_color)

        if custom_yaxis:
            ax1.set_ylim(yaxis_min, yaxis_max)
        else:
            y_min = np.nanmin(smoothed)
            y_max = np.nanmax(smoothed)
            margin = (y_max - y_min) * 0.1
            ax1.set_ylim(y_min - margin, y_max + margin)

        if custom_xaxis:
            ax1.set_xlim(xaxis_min, xaxis_max)
        else:
            ax1.set_xlim(start_time, end_time)

        for spine in ax1.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(text_color)

        # Optional inset zoom
        if enable_inset:
            axins = inset_axes(ax1, width="35%", height="35%", loc='upper left',
                               bbox_to_anchor=(0.1, 0.9, 1, 1), bbox_transform=ax1.transAxes)
            inset_mask = (time >= inset_start) & (time <= inset_end)
            axins.plot(time[inset_mask], smoothed[inset_mask], color=smooth_color, linewidth=1.5)
            axins.set_xlim(inset_start, inset_end)
            axins.set_ylim(np.nanmin(smoothed[inset_mask]) - 2, np.nanmax(smoothed[inset_mask]) + 2)
            axins.tick_params(axis='both', labelsize=10)
            for spine in axins.spines.values():
                spine.set_linewidth(1.0)
            mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="black", lw=1.2)

        # --- Panel B: Calibration ---
        ax2.scatter(valid_concs, y, color=smooth_color, edgecolors='black', s=60)
        ax2.plot(valid_concs, y_pred, color='white' if theme == "Dark" else 'black', linewidth=2)

        box_text = '\n'.join([
            rf"$R^2$ = {r2:.4f}",
            rf"LOD = {LOD:.2f} µM",
            rf"Sensitivity = {slope:.2f} nA/µM",
            rf"y = {slope:.2f}x + {intercept:.2f}"
        ])
        props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
        ax2.text(0.05, 0.95, box_text, transform=ax2.transAxes,
                 fontsize=11, verticalalignment='top', bbox=props, fontweight='bold', color='black')

        ax2.set_xlabel("Concentration (µM)", fontsize=14, fontweight='bold', color=text_color)
        ax2.set_ylabel("Current (nA)", fontsize=14, fontweight='bold', color=text_color)
        ax2.set_title("B", loc='left', fontsize=16, fontweight='bold', color=text_color)
        ax2.set_xticks(valid_concs)
        ax2.set_xticklabels([str(int(x)) for x in valid_concs], fontweight='bold', color=text_color)
        ax2.set_yticklabels(ax2.get_yticks(), fontweight='bold', color=text_color)
        ax2.tick_params(axis='both', labelsize=12, width=1.5, colors=text_color)

        for spine in ax2.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(text_color)

        plt.tight_layout()
        st.pyplot(fig)

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





