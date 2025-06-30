import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")
st.title("🔬 Amperometry Analysis Tool")

# === Sidebar Customization ===
bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

trace_color = st.sidebar.color_picker("Pick Raw Trace Color", "#1E90FF")
line_color = st.sidebar.color_picker("Pick Smoothed Line Color", "#FF595E")

# === User Inputs ===
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
label = st.sidebar.text_input("Label", value="sensor1")

start_time = st.sidebar.number_input("Start Time (s)", value=280)
end_time = st.sidebar.number_input("End Time (s)", value=499)
overlay_raw = st.sidebar.checkbox("Overlay raw trace", value=True)

spike_start = st.sidebar.number_input("Spike Start (s)", value=300)
spike_interval = st.sidebar.number_input("Spike Interval (s)", value=20)
spike_count = st.sidebar.number_input("Spike Count", value=10)
conc_per_spike = st.sidebar.number_input("Conc/Spike (µM)", value=20.0)

# Optional inset
st.sidebar.markdown("### 🔍 Inset Plot")
inset_enabled = st.sidebar.checkbox("Show Inset", value=False)
inset_start = st.sidebar.number_input("Inset Start", value=380)
inset_end = st.sidebar.number_input("Inset End", value=420)

# === Processing and Plotting ===
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    TIME_COL = "Elapsed Time (s)"
    CURRENT_COL = "Current (A)"
    SPIKE_WINDOW = 2
    ROLLING_WINDOW = 15

    plot_df = df[(df[TIME_COL] >= start_time) & (df[TIME_COL] <= end_time)].copy()
    time = plot_df[TIME_COL].values
    current_nA = plot_df[CURRENT_COL].values * 1e9

    # remove abnormal jumps
    diffs = np.abs(np.diff(current_nA, prepend=current_nA[0]))
    current_nA[diffs > 5] = np.nan

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
        plt.rcParams['font.family'] = 'Arial'

        # Plot A
        if overlay_raw:
            ax1.plot(time, current_nA, color=trace_color, linewidth=0.5, alpha=0.7)
        ax1.plot(time, smoothed, color=line_color, linewidth=1.5)

        for t, conc in zip(spike_times, concentrations):
            yval = np.interp(t, time, smoothed)
            ax1.annotate(f"{int(conc)} µM", xy=(t, yval), xytext=(t, yval + 3),
                         arrowprops=dict(arrowstyle='->', color=text_color),
                         ha='center', fontsize=9, fontweight='bold', color=text_color)

        ax1.set_xlabel("Time (s)", fontsize=14, fontweight='bold', color=text_color)
        ax1.set_ylabel("Current (nA)", fontsize=14, fontweight='bold', color=text_color)
        ax1.set_title("A", loc='left', fontsize=16, fontweight='bold', color=text_color)
        ax1.set_xticks(np.arange(start_time, end_time + 1, spike_interval))
        ax1.tick_params(axis='both', labelsize=12, width=1.5, colors=text_color, labelcolor=text_color)
        for spine in ax1.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(text_color)

        # Plot B
        ax2.scatter(valid_concs, y, color=line_color, edgecolors='black', s=60)
        ax2.plot(valid_concs, y_pred, color=text_color, linewidth=2)

        box_text = '\n'.join([
            f"R² = {r2:.4f}",
            f"LOD = {LOD:.2f} µM",
            f"Sensitivity = {slope:.2f} nA/µM",
            f"y = {slope:.2f}x + {intercept:.2f}"
        ])
        props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
        ax2.text(0.05, 0.95, box_text, transform=ax2.transAxes,
                 fontsize=11, verticalalignment='top', bbox=props, fontweight='bold', color='black')

        ax2.set_xlabel("Concentration (µM)", fontsize=14, fontweight='bold', color=text_color)
        ax2.set_ylabel("Current (nA)", fontsize=14, fontweight='bold', color=text_color)
        ax2.set_title("B", loc='left', fontsize=16, fontweight='bold', color=text_color)
        ax2.set_xticks(valid_concs)
        ax2.tick_params(axis='both', labelsize=12, width=1.5, colors=text_color, labelcolor=text_color)
        for spine in ax2.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(text_color)

        plt.tight_layout()
        st.pyplot(fig)

        # Inset Plot
        if inset_enabled:
            inset_df = df[(df[TIME_COL] >= inset_start) & (df[TIME_COL] <= inset_end)].copy()
            inset_time = inset_df[TIME_COL].values
            inset_current = inset_df[CURRENT_COL].values * 1e9
            inset_smooth = pd.Series(inset_current).rolling(window=ROLLING_WINDOW, center=True).mean().values

            fig2, ax_inset = plt.subplots(figsize=(6, 3))
            if overlay_raw:
                ax_inset.plot(inset_time, inset_current, color=trace_color, linewidth=0.5, alpha=0.7)
            ax_inset.plot(inset_time, inset_smooth, color=line_color, linewidth=1.5)
            ax_inset.set_xlabel("Time (s)", fontsize=13, fontweight='bold')
            ax_inset.set_ylabel("Current (nA)", fontsize=13, fontweight='bold')
            ax_inset.set_title(f"Inset: {inset_start}-{inset_end} s", fontsize=14, fontweight='bold')
            ax_inset.tick_params(axis='both', labelsize=11, width=1.5)
            st.pyplot(fig2)

        # Downloads
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







