#  Amperometry Analysis Tool

An interactive and publication-ready tool for analysing amperometric biosensor data. Built using **Python**, **Streamlit**, and **scikit-learn**, it extracts spike currents, plots real-time traces, generates calibration curves, and calculates sensitivity and LOD.

##  Try It Online

👉 https://amperometry-plotter.streamlit.app/  
*(No installation required. Just upload your CSV and go!)*

---

## 🔍 Features

- 📈 Real-time amperometry trace with optional smoothed overlay
- 🎯 Spike annotation with custom timing and concentrations
- 📉 Calibration curve with:
- Linear regression fit
- $R^2$, Sensitivity, and LOD calculation
- 📤 Downloadable figures and data in CSV

---

## 📂Input Format

CSV file with at least two columns:

| Elapsed Time (s) | Current (A) |
|------------------|-------------|
| 0.00             | 2.34E-10    |
| ...              | ...         |

---

## How It Works

1. Upload your amperometry `.csv` file
2. Enter:
   - Start and End times for plotting
   - Spike start time, interval, and total count
   - Concentration per spike
3. The app will:
   - Extract average current near each spike (±2 seconds)
   - Plot the trace with spike annotations
   - Fit and display a calibration curve
   - Show sensitivity, LOD, and goodness-of-fit ($R^2$)
   - Provide export buttons for the figure and CSV

---

