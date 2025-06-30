# Inside your plotting section, just replacing the relevant parts

# Main plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=bg_color)
plt.rcParams['font.family'] = 'Arial'
ax1.set_facecolor(bg_color)
ax2.set_facecolor(bg_color)

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

    fig2, ax_inset = plt.subplots(figsize=(6, 3), facecolor=bg_color)
    ax_inset.set_facecolor(bg_color)
    
    if overlay_raw:
        ax_inset.plot(inset_time, inset_current, color=trace_color, linewidth=0.5, alpha=0.7)
    ax_inset.plot(inset_time, inset_smooth, color=line_color, linewidth=1.5)

    ax_inset.set_xlabel("Time (s)", fontsize=13, fontweight='bold', color=text_color)
    ax_inset.set_ylabel("Current (nA)", fontsize=13, fontweight='bold', color=text_color)
    ax_inset.set_title(f"Inset: {inset_start}-{inset_end} s", fontsize=14, fontweight='bold', color=text_color)
    ax_inset.tick_params(axis='both', labelsize=11, width=1.5, colors=text_color, labelcolor=text_color)
    for spine in ax_inset.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(text_color)

    st.pyplot(fig2)

