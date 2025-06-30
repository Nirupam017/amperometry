import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Set theme (you can change this value)
background = st.selectbox("Choose background color:", ["white", "black"])

# Determine color scheme
bg_color = 'white' if background == 'white' else 'black'
text_color = 'black' if background == 'white' else 'white'

# Generate sample data
np.random.seed(0)
time = np.linspace(280, 490, 500)
true_current = 55 + 0.05 * (time - 280) + np.floor((time - 280) / 20)
noise = np.random.normal(0, 1.5, size=time.shape)
current = true_current + noise

# Calibration data
concentrations = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
currents = 0.14 * concentrations + 51.97 + np.random.normal(0, 1.2, size=concentrations.shape)

# Fit linear model
model = LinearRegression()
model.fit(concentrations.reshape(-1, 1), currents)
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(concentrations.reshape(-1, 1), currents)

# Calculate LOD = 3 * std(blank) / slope
blank_std = np.std(currents[:3])  # using first 3 as example
LOD = 3 * blank_std / slope
sensitivity = slope

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=bg_color)
fig.subplots_adjust(wspace=0.4)

# Plot A - Time vs Current
ax1.plot(time, current, color='deepskyblue', linewidth=0.8, label='Raw current')
ax1.plot(time, true_current, color='salmon', linewidth=2, label='Smoothed')
ax1.set_xlabel('Time (s)', fontsize=12, weight='bold', color=text_color)
ax1.set_ylabel('Current (nA)', fontsize=12, weight='bold', color=text_color)
ax1.set_title('A', loc='left', fontsize=16, weight='bold', color=text_color)
ax1.set_facecolor(bg_color)
ax1.tick_params(colors=text_color)
ax1.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)

# Update tick and spine colors for ax1
for spine in ax1.spines.values():
    spine.set_color(text_color)
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_color(text_color)

# Plot B - Calibration Curve
ax2.scatter(concentrations, currents, color='lightcoral', s=60, edgecolor='black')
x_fit = np.linspace(0, 210, 100)
y_fit = model.predict(x_fit.reshape(-1, 1))
ax2.plot(x_fit, y_fit, color=text_color, linestyle='--', linewidth=2)

ax2.set_xlabel('Concentration (μM)', fontsize=12, weight='bold', color=text_color)
ax2.set_ylabel('Current (nA)', fontsize=12, weight='bold', color=text_color)
ax2.set_title('B', loc='left', fontsize=16, weight='bold', color=text_color)
ax2.set_facecolor(bg_color)
ax2.tick_params(colors=text_color)

# Update tick and spine colors for ax2
for spine in ax2.spines.values():
    spine.set_color(text_color)
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_color(text_color)

# Annotation box with dynamic color
equation = f"$R^2$ = {r_squared:.4f}\nLOD = {LOD:.2f} μM\nSensitivity = {sensitivity:.2f} nA/μM\ny = {slope:.2f}x + {intercept:.2f}"
ax2.text(0.05, 0.95, equation, transform=ax2.transAxes,
         fontsize=12, color=text_color,
         verticalalignment='top',
         bbox=dict(facecolor=bg_color, edgecolor=text_color))

# Show plot in Streamlit
st.pyplot(fig)




