import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")

# =========================
# Sidebar inputs
# =========================
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

plot_start = st.sidebar.number_input("Plot start time (s)", value=1450)
spike_start = st.sidebar.number_input("Spike start time (s)", value=1500)
spike_end   = st.sidebar.number_input("Spike end time (s)", value=1950)
spike_interval = st.sidebar.number_input("Spike interval (s)", value=50)

conc_per_spike = st.sidebar.number_input("Concentration per spike (µM)", value=20.0)

SPIKE_WINDOW = 5      # seconds for averaging
SMOOTH_WINDOW = 20

# =========================
# Load data
# =========================
if uploaded_file is None:
    st.warning("Upload a CSV file")
    st.stop()

df = pd.read_csv(uploaded_file)

TIME_COL = df.columns[3]     # time column
CURR_COL = df.columns[6]     # current column

# =========================
# Generate spike times
# =========================
spike_times = np.arange(spike_start, spike_end + 1, spike_interval)
concentrations = np.arange(
    conc_per_spike,
    conc_per_spike * (len(spike_times) + 1),
    conc_per_spike
)

# =========================
# Plot window (visual only)
# =========================
plot_df = df[df[TIME_COL] >= plot_start].copy()

time = plot_df[TIME_COL].values
current = plot_df[CURR_COL].values

smooth = (
    pd.Series(current)
    .rolling(SMOOTH_WINDOW, center=True)
    .mean()
    .values
)

# =========================
# Extract spike currents
# =========================
spike_currents = []

for t in spike_times:
    mask = (df[TIME_COL] >= t - SPIKE_WINDOW) & (df[TIME_COL] <= t + SPIKE_WINDOW)
    spike_currents.append(df.loc[mask, CURR_COL].mean())

y = np.array(spike_currents)
X = concentrations.reshape(-1, 1)

# =========================
# Linear regression
# =========================
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

slope = model.coef_[0]
intercept = model.intercept_
r2 = r2_score(y, y_pred)

# =========================
# ✅ LOD from TRUE baseline (1450–1500)
# =========================
baseline_mask = (df[TIME_COL] >= plot_start) & (df[TIME_COL] < spike_start)
baseline_current = df.loc[baseline_mask, CURR_COL]

baseline_std = baseline_current.std()
LOD = (3 * baseline_std) / slope

# =========================
# Plotting
# =========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ---- Plot A ----
ax1.plot(time, smooth, color="red", linewidth=1.8, label="Smoothed")
ax1.plot(time, current, color="gray", alpha=0.4, linewidth=0.5, label="Raw")

for t, c in zip(spike_times, concentrations):
    yval = np.interp(t, time, smooth)
    ax1.annotate(
        f"{int(c)} µM",
        xy=(t, yval),
        xytext=(t, yval + 4),
        arrowprops=dict(arrowstyle="->", lw=1)
    )

ax1.set_xlim(plot_start, spike_end + spike_interval)
ax1.set_xticks(spike_times)
ax1.set_xticklabels(spike_times.astype(int))
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Current (nA)")
ax1.set_title("A", loc="left", fontweight="bold")

# ---- Plot B ----
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
    va="top",
    bbox=dict(boxstyle="round", pad=0.4)
)

ax2.set_xticks(concentrations)
ax2.set_xlabel("Concentration (µM)")
ax2.set_ylabel("Current (nA)")
ax2.set_title("B", loc="left", fontweight="bold")

plt.tight_layout()
st.pyplot(fig)

# =========================
# Downloads
# =========================
buf = BytesIO()
fig.savefig(buf, dpi=300, format="png")

st.download_button(
    "Download Figure",
    buf.getvalue(),
    file_name="chrono_calibration.png",
    mime="image/png"
)

result_df = pd.DataFrame({
    "Spike time (s)": spike_times,
    "Concentration (µM)": concentrations,
    "Avg current (nA)": y
})

st.download_button(
    "Download Results CSV",
    result_df.to_csv(index=False),
    file_name="results.csv"
)

st.success("✅ Done — logic now exactly matches your experiment")
