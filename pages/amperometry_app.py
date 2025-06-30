import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Page settings
st.set_page_config(page_title="Glutamate Sensor Plot", layout="wide")

# Toggle background color
dark_mode = st.checkbox("Dark Mode", value=True)

# Set background and text colors
if dark_mode:
    bg_color = 'black'
    text_color = 'white'
else:
    bg_color = 'white'
    text_color = 'black'

# Sample data (replace with your actual data)
time = np.linspace(280, 490, 1000)
true_current = 0.14 * (time - 280) + 56
noise = np.random.normal(0, 1.5, size=true_current.shape)
current = true_current + noise

concentrations = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
currents_avg = 0.14 * concentrations + 51.97  # linear trend
noise2 = np.random.normal(0, 1.2, size=concentrations.shape)
currents = currents_avg + noise2

# Fit linear regression
X = concentrations.reshape(-1, 1)
y = currents
model = LinearRegression().fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y)
sensitivity = slope
LOD = 3 * np.std(noise2) / sensitivity

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=bg_color)
fig.subplots_adjust(wspace=0.4)

# --- Plot A: Time vs Current ---
ax1.plot(time, current, color='deepskyblue', linewidth=0.8, label='Raw current')
ax1.plot(time, true_current, color='salmon', linewidth=2, label='Smoothed')
ax1.set_xlabel('Time (s)', fontsize=12, weight='bold', color=text_color)
ax1.set_ylabel('Current (nA)', fontsize=12, weight='bold', color=text_color)
ax1.set_title('A', loc='left', fontsize=16, weight='bold', color=text_color)
ax1.set_facecolor(bg_color)
ax1.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)

# Set tick colors and spine colors
ax1.tick_params(colors=text_color)
for spine in ax1.spines.values():
    spine.set_color(text_color)
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_color(text_color)

# --- Plot B: Calibration Curve ---
ax2.scatter(concentrations, currents, color='lightcoral', s=60, edgecolor='black')
x_fit = np.linspace(0, 210, 100)
y_fit = model.predict(x_fit.reshape(-1, 1))
ax2.plot(x_fit, y_fit, color='white' if dark_mode else 'black', linestyle='--', linewidth=2)

ax2.set_xlabel('Concentration (μM)', fontsize=12, weight='bold', color=text_color)
ax2.set_ylabel('Current (nA)', fontsize=12, weight='bold', color=text_color)
ax2.set_title('B', loc='left', fontsize=16, weight='bold', color=text_color)
ax2.set_facecolor(bg_color)

# Add annotation box
equation = f"$R^2$ = {r_squared:.4f}\nLOD = {LOD:.2f} μM\nSensitivity = {sensitivity:.2f} nA/μM\ny = {slope:.2f}x + {intercept:.2f}"
ax2.text(0.05, 0.95, equation, transform=ax2.transAxes,
         fontsize=12, color=text_color,
         verticalalignment='top',
         bbox=dict(facecolor=bg_color, edgecolor=text_color))

# Set tick and spine colors
ax2.tick_params(colors=text_color)
for spine in ax2.spines.values():
    spine.set_color(text_color)
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_color(text_color)

# Show in Streamlit
st.pyplot(fig)



