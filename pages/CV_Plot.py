import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔬 CV Overlay Tool")

# =========================
# Theme
# =========================
bg_choice = st.sidebar.selectbox(
    "Background Color",
    ["White", "Black"]
)

bg_color = "white" if bg_choice == "White" else "black"
text_color = "black" if bg_color == "white" else "white"

# =========================
# Peak Analysis Settings
# =========================
st.sidebar.markdown("## 📍 Peak Current Analysis")

enable_peak = st.sidebar.checkbox(
    "Enable Peak Current Detection",
    value=False
)

target_voltage = st.sidebar.number_input(
    "Voltage (V)",
    value=0.0,
    format="%.4f"
)

peak_choice = st.sidebar.selectbox(
    "Which Curve?",
    ["Top Curve", "Bottom Curve"]
)

# =========================
# Upload Files
# =========================
uploaded_files = st.file_uploader(
    "Upload CV CSV files",
    type="csv",
    accept_multiple_files=True
)

color_palette = plt.get_cmap("tab10")

# =========================
# Data Store
# =========================
data_store = []

if uploaded_files:

    for file in uploaded_files:

        try:
            df = pd.read_csv(file)

        except:
            st.warning(f"{file.name} failed to load")
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

        v_col = next(
            (c for c in possible_voltage if c in df.columns),
            None
        )

        i_col = next(
            (c for c in possible_current if c in df.columns),
            None
        )

        if v_col is None or i_col is None:

            st.warning(
                f"{file.name} missing required columns"
            )

            continue

        df.dropna(
            subset=[v_col, i_col],
            inplace=True
        )

        voltage = df[v_col].values
        current = df[i_col].values * 1e6  # µA

        data_store.append(
            (file.name, voltage, current)
        )

# =========================
# Plot CV Overlay
# =========================
if data_store:

    fig, ax = plt.subplots(
        figsize=(8, 6),
        facecolor=bg_color
    )

    ax.set_facecolor(bg_color)

    peak_results = []

    for i, (name, voltage, current) in enumerate(data_store):

        with st.expander(f"{name} Settings"):

            label = st.text_input(
                f"Label {i}",
                value=name,
                key=f"label_{i}"
            )

            line_width = st.slider(
                f"Line Width {i}",
                min_value=1,
                max_value=6,
                value=2,
                key=f"lw_{i}"
            )

        color = color_palette(i % color_palette.N)

        ax.plot(
            voltage,
            current,
            label=label,
            linewidth=line_width,
            color=color
        )

        # =========================
        # Peak Current Extraction
        # =========================
        if enable_peak:

            try:

                voltage = np.array(voltage)
                current = np.array(current)

                # Find closest voltage index
                idx = np.argmin(
                    np.abs(voltage - target_voltage)
                )

                current_at_voltage = current[idx]

                peak_results.append({
                    "label": label,
                    "voltage": voltage[idx],
                    "current": current_at_voltage,
                    "color": color
                })

            except:
                pass

    # =========================
    # Find Top or Bottom Curve
    # =========================
    if enable_peak and len(peak_results) > 0:

        if peak_choice == "Top Curve":

            selected_peak = max(
                peak_results,
                key=lambda x: x["current"]
            )

        else:

            selected_peak = min(
                peak_results,
                key=lambda x: x["current"]
            )

        # Mark selected point
        ax.scatter(
            selected_peak["voltage"],
            selected_peak["current"],
            color=selected_peak["color"],
            edgecolors="black",
            s=120,
            zorder=10
        )

        # Display Result
        st.subheader("📌 Selected Peak")

        st.write(
            f"**Curve:** {selected_peak['label']}"
        )

        st.write(
            f"**Voltage:** {selected_peak['voltage']:.4f} V"
        )

        st.write(
            f"**Current:** {selected_peak['current']:.4f} µA"
        )

    # =========================
    # Styling
    # =========================
    ax.set_xlabel(
        "Voltage (V)",
        color=text_color,
        fontsize=14
    )

    ax.set_ylabel(
        "Current (µA)",
        color=text_color,
        fontsize=14
    )

    ax.set_title(
        "CV Overlay",
        color=text_color,
        fontsize=16
    )

    ax.tick_params(
        colors=text_color,
        labelsize=12
    )

    for spine in ax.spines.values():
        spine.set_color(text_color)

    legend = ax.legend(
        facecolor=bg_color,
        edgecolor=text_color,
        fontsize=10
    )

    for text in legend.get_texts():
        text.set_color(text_color)

    st.pyplot(fig)
