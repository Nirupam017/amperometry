import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("🔬 CV Overlay + Scan Rate Study Tool")

st.sidebar.header("⚙️ Settings")
bg_choice = st.sidebar.selectbox("Background Color", ["White", "Black"])
bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

uploaded_files = st.sidebar.file_uploader("Upload CV CSV Files", type="csv", accept_multiple_files=True)

cv_curves = []  # [(voltage, current, label, color), ...]
scanrate_data = []  # [(scan_rate, anodic_peak, cathodic_peak, label)]

if uploaded_files:
    for i, file in enumerate(uploaded_files):
        st.sidebar.markdown(f"### File {i+1}: {file.name}")
        label = st.sidebar.text_input(f"Legend Label for {file.name}", value=f"CV_{i+1}")
        scan_rate = st.sidebar.number_input(f"Scan Rate (mV/s) for {file.name}", min_value=1.0, value=50.0, step=1.0)
        curve_color = st.sidebar.color_picker(f"Curve Color for {file.name}", value=f"#{np.random.randint(0x1000000):06x}")

        try:
            df = pd.read_csv(file)
            if "Working Electrode (V)" not in df.columns or "Current (A)" not in df.columns:
                st.error(f"File '{file.name}' does not contain required columns.")
                continue

            voltage = df["Working Electrode (V)"].values
            current = df["Current (A)"].values * 1e6  # convert to uA for clarity

            # Find anodic and cathodic peak currents
            anodic_peak = np.max(current)
            cathodic_peak = np.min(current)

            cv_curves.append((voltage, current, label, curve_color))
            scanrate_data.append((scan_rate, anodic_peak, cathodic_peak, label))

        except Exception as e:
            st.error(f"Error reading '{file.name}': {e}")

    # === Plot Overlaid CV Curves ===
    fig1, ax1 = plt.subplots(figsize=(8, 6), facecolor=bg_color)
    ax1.set_facecolor(bg_color)
    for voltage, current, label, color in cv_curves:
        ax1.plot(voltage, current, label=label, color=color, linewidth=2)
    ax1.set_xlabel("Potential (V vs ref)", fontsize=14, color=text_color, fontweight='bold')
    ax1.set_ylabel("Current (µA)", fontsize=14, color=text_color, fontweight='bold')
    ax1.set_title("Overlaid CV Curves", fontsize=16, color=text_color, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=12, colors=text_color)
    ax1.legend(fontsize=10)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(text_color)
    st.pyplot(fig1)

    # === Plot Scan Rate Study ===
    scan_rates = np.array([s[0] for s in scanrate_data])
    anodic_peaks = np.array([s[1] for s in scanrate_data])
    cathodic_peaks = np.array([s[2] for s in scanrate_data])

    fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor=bg_color)
    ax2.set_facecolor(bg_color)

    sqrt_scan = np.sqrt(scan_rates)
    ax2.scatter(sqrt_scan, anodic_peaks, label="Anodic Peak", color='red', s=60)
    ax2.scatter(sqrt_scan, cathodic_peaks, label="Cathodic Peak", color='blue', s=60)

    # Linear fit and display equation and R²
    for peak_data, name, color in [(anodic_peaks, "Anodic", 'red'), (cathodic_peaks, "Cathodic", 'blue')]:
        model = LinearRegression()
        model.fit(sqrt_scan.reshape(-1, 1), peak_data)
        pred = model.predict(sqrt_scan.reshape(-1, 1))
        r2 = model.score(sqrt_scan.reshape(-1, 1), peak_data)
        slope = model.coef_[0]
        intercept = model.intercept_
        ax2.plot(sqrt_scan, pred, color=color, linestyle='--', label=f"{name} Fit: y={slope:.2f}x+{intercept:.2f}, R²={r2:.3f}")

    ax2.set_xlabel("√Scan Rate (mV/s)^0.5", fontsize=14, color=text_color, fontweight='bold')
    ax2.set_ylabel("Peak Current (µA)", fontsize=14, color=text_color, fontweight='bold')
    ax2.set_title("Scan Rate Study", fontsize=16, color=text_color, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.tick_params(axis='both', labelsize=12, colors=text_color)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(text_color)

    st.pyplot(fig2)

    # Optionally export CSV summary
    summary_df = pd.DataFrame(scanrate_data, columns=["Scan Rate (mV/s)", "Anodic Peak (uA)", "Cathodic Peak (uA)", "Label"])
    st.download_button("📄 Download Scan Rate Summary", summary_df.to_csv(index=False), file_name="scanrate_summary.csv")
