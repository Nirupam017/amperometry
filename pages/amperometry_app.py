import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Sidebar settings
    st.sidebar.title("Plot Settings")
    bg_color = st.sidebar.radio("Background color", ("White", "Black"))
    
    # Auto switch text color
    if bg_color == "White":
        face_color = "white"
        text_color = "black"
    else:
        face_color = "black"
        text_color = "white"
    
    xlabel = st.sidebar.text_input("X-axis label", "Time (s)")
    ylabel = st.sidebar.text_input("Y-axis label", "Amplitude")
    legend_labels = st.sidebar.text_input("Legend labels (comma-separated)", "Signal 1,Signal 2")
    legends = [label.strip() for label in legend_labels.split(',')]

    # Main plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=face_color)
    ax1.set_facecolor(face_color)
    ax2.set_facecolor(face_color)

    ax1.plot(df.iloc[:, 0], df.iloc[:, 1], label=legends[0], color='tab:blue')
    ax2.plot(df.iloc[:, 0], df.iloc[:, 2], label=legends[1], color='tab:orange')

    # Axes labels and titles
    for ax, title in zip([ax1, ax2], legends):
        ax.set_xlabel(xlabel, color=text_color)
        ax.set_ylabel(ylabel, color=text_color)
        ax.set_title(title, color=text_color)
        ax.legend(facecolor=face_color, edgecolor=text_color, labelcolor=text_color)
        ax.tick_params(colors=text_color)

    st.pyplot(fig)

    # Inset plot for zoom
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Inset Plot Zoom Range")
    zoom_start = st.sidebar.number_input("Start index", min_value=0, max_value=len(df)-1, value=0)
    zoom_end = st.sidebar.number_input("End index", min_value=0, max_value=len(df)-1, value=len(df)-1)

    if zoom_start < zoom_end:
        fig2, ax_inset = plt.subplots(figsize=(6, 3), facecolor=face_color)
        ax_inset.set_facecolor(face_color)

        ax_inset.plot(df.iloc[zoom_start:zoom_end, 0], df.iloc[zoom_start:zoom_end, 1], label=legends[0], color='tab:blue')
        ax_inset.plot(df.iloc[zoom_start:zoom_end, 0], df.iloc[zoom_start:zoom_end, 2], label=legends[1], color='tab:orange')

        ax_inset.set_xlabel(xlabel, color=text_color)
        ax_inset.set_ylabel(ylabel, color=text_color)
        ax_inset.set_title("Zoomed Inset", color=text_color)
        ax_inset.legend(facecolor=face_color, edgecolor=text_color, labelcolor=text_color)
        ax_inset.tick_params(colors=text_color)

        st.pyplot(fig2)
    else:
        st.warning("Start index must be less than End index.")
