import pandas as pd
from pathlib import Path
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import spotpy
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.font_manager as fm
import socket
host = socket.gethostname()

# Set the font
path_to_palatinottf = '/home/phillip/Downloads/Palatino.ttf'
fm.fontManager.addfont(path_to_palatinottf)
plt.rcParams["font.family"] = "Palatino"


if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'

file_name = 'test_file'

## Files and paths
data_path = '/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/notebook_example_for_spot/'
obs = pd.read_csv(data_path + 'obs_runoff_example.csv', parse_dates=['Date'], index_col='Date')
obs = obs['2000-01-01':'2020-12-31']['Qobs']
# m^3/s to mm
obs = obs * 86400 / (295.67484249904464 * 1000000) * 1000
dream_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/parameters/uncertainty/'
dream_file = dream_path + file_name
output = dream_file + '/'
os.makedirs(output, exist_ok=True)

results = spotpy.analyser.load_csv_results(dream_file)
fields = [word for word in results.dtype.names if word.startswith("sim")]

## Fig 1
start_date = datetime(2000, 1, 1)
end_date = datetime(2020, 12, 31)
num_days = (end_date - start_date).days
dates = [start_date + timedelta(days=i) for i in range(num_days + 1)]

colors = [('#e4665c'), ('#5e37a2'), ('#04647c')]
cmap = LinearSegmentedColormap.from_list('custom', colors, N=20)

fig = plt.figure(figsize=(16, 5))
ax = plt.subplot(1, 1, 1)

percentiles = {}
for field in fields:
    percentiles[field] = np.percentile(results[field][-100:-1], range(5, 96, 5))

for idx, perc in enumerate(range(5, 96, 5)):
    perc_values = [percentiles[field][idx] for field in fields]
    color = cmap(idx / 20)
    ax.plot(dates, perc_values, color=color, linestyle='solid', alpha=0.7, zorder=2)

ax.plot(dates, np.array(obs), color="black", label='Observations')
ax.set_ylim(0, max([percentiles[field][-1] for field in fields]) * 1.13)
ax.set_xlim(start_date, end_date)
ax.set_ylabel('Runoff [mm w.e.]')
ax.set_facecolor('lightgrey')
ax.grid(color='white', linestyle='-', linewidth=0.5, zorder=1)

cax = ax.inset_axes([0.8, 0.83, 0.027, 0.03])
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal')
cbar.ax.text(1.45, 0.1, 'Parameter Uncertainty', fontsize=14, transform=cbar.ax.transAxes)
cbar.ax.set_xticks([0, 1])
cbar.ax.set_xticklabels(['P5', 'P95'], fontsize=10)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=14, bbox_to_anchor=(0.929, 0.98), frameon=False)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

fig.savefig(output + file_name + '_param_unc_long.png', dpi=300)
plt.clf()  # Clear the figure to avoid overlap

## Create dataframe with annual mean obs and percentile timeseries
percentile_data = {f'q{perc}': [] for perc in range(5, 96, 5)}
percentiles = {}

for field in fields:
    percentiles[field] = np.percentile(results[field][-100:-1], range(5, 96, 5))

for idx, perc in enumerate(range(5, 96, 5)):
    perc_values = [percentiles[field][idx] for field in fields]
    percentile_data[f'q{perc}'] = perc_values

perc_data = {
    'Observations': obs,
    **percentile_data
}
perc_df = pd.DataFrame(perc_data, index=dates)

perc_df_ann = perc_df.copy()
perc_df_ann["month"] = perc_df_ann.index.month
perc_df_ann["day"] = perc_df_ann.index.day
perc_df_ann = perc_df_ann.groupby(["month", "day"]).mean()
perc_df_ann["date"] = pd.date_range(perc_df.index[0], freq='D', periods=len(perc_df_ann)).strftime('%Y-%m-%d')
perc_df_ann = perc_df_ann.set_index(perc_df_ann["date"])
perc_df_ann.index = pd.to_datetime(perc_df_ann.index)

## Plot long-term annual means
fig = plt.figure(figsize=(7, 5), dpi=400)  # Create a new figure for the second plot
ax = plt.gca()  # Get the current axis

colors = [('#e4665c'), ('#5e37a2'), ('#04647c')]
cmap = LinearSegmentedColormap.from_list('custom', colors, N=20)

ax.plot(perc_df_ann.index, perc_df_ann['Observations'], color="black", label='Observations', zorder=3)

for idx, perc in enumerate(range(5, 96, 5)):
    color = cmap(idx / 20)
    ax.plot(perc_df_ann.index, perc_df_ann[f'q{perc}'], color=color, linestyle='solid', alpha=0.7, zorder=2)

ax.set_ylim(0, 32)  # Set y-axis limit for the second figure
ax.set_ylabel('Runoff [mm w.e.]', fontsize=14)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
plt.xticks(fontsize=14)  # Set the font size of the x-axis tick labels
plt.yticks(fontsize=14)  # Set the font size of the x-axis tick labels
ax.set_facecolor('lightgrey')
ax.grid(color='white', linestyle='-', linewidth=0.5, zorder=1)

# Autoscale x-axis without extra padding
ax.autoscale(enable=True, axis='x', tight=True)

plt.tight_layout()

# # Create a white rectangle box that covers part of the plot area
# white_box = Rectangle((0.68, 0.753), 0.28, 0.2, transform=ax.transAxes, color='white', zorder=4)
# # Add the box to the plot
# ax.add_patch(white_box)
#
# cbar_ax = ax.inset_axes([0.6924, 0.82, 0.042, 0.03])
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, orientation='horizontal')
# cbar.ax.text(1.38, 0.08, 'Parameter Uncertainty', fontsize=9, transform=cbar.ax.transAxes)
# cbar.ax.set_xticks([0, 1])
# cbar.ax.set_xticklabels(['P5', 'P95'], fontsize=7)
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, fontsize=9, bbox_to_anchor=(0.885, 0.935), frameon=False, bbox_transform=ax.transAxes)


fig.savefig(output + file_name + '_param_unc_annual.png', dpi=300)
plt.clf()
# plt.show()
plt.close()  # Close the figure to free memory

