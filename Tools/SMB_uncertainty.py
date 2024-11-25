## Imports
import numpy as np
from scipy.stats import rankdata
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import spotpy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import os
import contextlib
import sys
import socket
import pandas as pd
import plotly.io as pio
from matilda.core import matilda_simulation
from statsmodels.graphics.tsaplots import plot_acf
import HydroErr as he
from spotpy.objectivefunctions import mae, rmse
from pathlib import Path

host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
sys.path.append(home + '/Ana-Lena_Phillip/data/tests_and_tools/Tools')
sys.path.append(home + '/EBA-CA/Repositories/matilda_edu/tools')
from matilda_resampler import MatildaBulkSampler, doy_col, extract_resamples
from helpers import dict_to_pickle, pickle_to_dict, write_yaml, read_yaml
import matplotlib.font_manager as fm
from helpers import dict_to_pickle, pickle_to_dict

import matplotlib.colors as mcolors
path_to_palatinottf = '/home/phillip/Downloads/Palatino.ttf'
fm.fontManager.addfont(path_to_palatinottf)


pio.renderers.default = "browser"

## Load ensemble means

matilda_scenarios = pickle_to_dict(home + '/EBA-CA/Repositories/matilda_edu/output/cmip6/adjusted/matilda_scenario_input.pickle')

def calculate_ensemble_mean(data):
    # Initialize the dictionary to store the ensemble mean dataframes for SSP2 and SSP5
    ensemble_mean_dict = {}

    # Iterate through each scenario (SSP2, SSP5)
    for scenario in data.keys():
        # Extract all models for the current scenario
        models = data[scenario]
        # Create a list of DataFrames from all models
        dfs = [df.set_index('TIMESTAMP') for df in models.values()]
        # Concatenate all DataFrames along the TIMESTAMP axis
        concatenated_df = pd.concat(dfs, axis=1, keys=models.keys())
        # Calculate the mean across all models for T2 and RRR
        ensemble_mean = concatenated_df.groupby(level=1, axis=1).mean()
        # Reset the index to get TIMESTAMP back as a column
        ensemble_mean.reset_index(inplace=True)
        # Store the result in the dictionary under the scenario's name
        ensemble_mean_dict[scenario] = ensemble_mean[['TIMESTAMP', 'T2', 'RRR']]

    return ensemble_mean_dict

# Apply the function to the data
ensemble_mean_result = calculate_ensemble_mean(matilda_scenarios)

## MATILDA Setup
data_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/notebook_example_for_spot/'

glacier_profile = pd.read_csv(data_path + 'glacier_profile.csv')

full_period = {'set_up_end': '1999-12-31', 'set_up_start': '1998-01-01',
               'sim_end': '2100-12-31', 'sim_start': '2000-01-01'}

settings = {
    # 'obs': obs,
    'area_cat': 295.67484249904464,
    'area_glac': 31.829413146585885,
    'ele_cat': 3293.491688025922,
    'ele_dat': 3335.67,
    'ele_glac': 4001.8798828125,
    'elev_rescaling': True,
    'freq': 'Y',
    'lat': 42.18280043250193,
    'glacier_profile': glacier_profile,
    'plots': False
}

fix_val = {
    # Fix:
    'RFS': 0.15,
    'SFCF': 1,
    'CET': 0,
    # Step 1:
    'PCORR': 0.58,
    'lr_temp': -0.006,
    'lr_prec': 0.0015,

    # Step 2:
    "TT_diff": 0.76198,
    'TT_snow': -1.44646,
    'CFMAX_snow': 3.3677
}

parameters = {
    'BETA': 1.0,
    'FC': 99.15976,
    'K0': 0.01,
    'K1': 0.01,
    'K2': 0.15,
    'LP': 0.998,
    'MAXBAS': 2.0,
    'PERC': 0.09232826,
    'UZL': 126.411575,
    'CWH': 0.000117,
    'AG': 0.54930484
}

## Define melt rate parameter range: CFMAX_ice = CFMAX_snow * CFMAX_rel
# low: 0.626 --> smb_00-18 = -0.156 (shean_mean)
# hi: 1.814 --> smb_14-20 = -0.704 (WGMS_karab)

cfmax_rel = {'min': 0.626, 'max': 1.814, 'barandun': 1.2556936}

# 3.3677 * 0.626 = 2.1081802 mm/K/d
# 3.3677 * 1.814 = 6.1090078000000005 mm/K/d

## Run MATILDA

# smb_unc = dict()
#
# for s in ensemble_mean_result.keys():
#     smb_unc[s] = pd.DataFrame()
#     for c in cfmax_rel.keys():
#         # Run the simulation
#         results = matilda_simulation(input_df=ensemble_mean_result[s], CFMAX_rel=cfmax_rel[c], **full_period,
#                                      **settings, **parameters, **fix_val)
#
#         # Resample runoff data annually and calculate glacier contribution in %
#         runoff = results[0][['ice_melt_on_glaciers', 'total_runoff']].resample('Y').sum()
#         runoff[f'glacier_contribution_{c}'] = (runoff['ice_melt_on_glaciers'] / runoff['total_runoff']) * 100
#
#         # Prepare glacier data
#         glaciers = pd.DataFrame()
#         glaciers[[f'glacier_area_{c}', f'glacier_mass_mmwe_{c}']] = results[5][['glacier_area', 'glacier_mass_mmwe']][
#                                                                     1:]
#
#         # When glacier area is 0, glacier mass is as well
#         glaciers.loc[glaciers[f'glacier_area_{c}'] == 0, f'glacier_mass_mmwe_{c}'] = 0
#
#         # Convert glacier mass from mm to m
#         glaciers[f'glacier_mass_mmwe_{c}'] = glaciers[f'glacier_mass_mmwe_{c}'] / 1000
#
#         # Shift the glaciers DataFrame index to match runoff index (last day of the year)
#         glaciers.index = glaciers.index + pd.offsets.YearEnd(0)
#
#         # Merge the two DataFrames on the TIMESTAMP index
#         merged_df = pd.merge(runoff[f'glacier_contribution_{c}'], glaciers, left_index=True, right_index=True,
#                              how='inner')
#
#         # Apply 3-year rolling mean to all columns in merged_df
#         merged_df_rolling = merged_df.rolling(window=3).mean()
#
#         # Concatenate the rolling mean data to the final DataFrame
#         smb_unc[s] = pd.concat([smb_unc[s], merged_df_rolling], axis=1)

## Create figures

# path_to_palatinottf = '/home/phillip/Downloads/palatino-linotype-font/pala.ttf'
# fm.fontManager.addfont(path_to_palatinottf)
# plt.rcParams["font.family"] = "Palatino Linotype"

import scienceplots
plt.style.use(['science', 'grid'])

# Y-axis labels for each subplot
y_labels = ['Glacier Area [km²]', 'Glacier Mass [m.w.e.]', 'Runoff Contribution from Glaciers [\%]']

# Variable names for subplots
variables = ['glacier_area', 'glacier_mass_mmwe', 'glacier_contribution']

# Color settings for SSP2 and SSP5
colors = {'SSP2': 'orange', 'SSP5': 'darkblue'}

# Custom legend for uncertainty (shaded area)
uncertainty_patch_ssp2 = mpatches.Patch(color=colors['SSP2'], alpha=0.3, label='SSP2 others')
uncertainty_patch_ssp5 = mpatches.Patch(color=colors['SSP5'], alpha=0.3, label='SSP5 others')

# Add both uncertainty patches and the barandun lines to the legend in desired order
handles = [
    Line2D([], [], color=colors['SSP2'], linestyle='--', linewidth=0.7, label='SSP2 Barandun et.al.'),
    uncertainty_patch_ssp2,
    Line2D([], [], color=colors['SSP5'], linestyle='--', linewidth=0.7, label='SSP5 Barandun et.al.'),
    uncertainty_patch_ssp5
]


# Create a list of step values between min and max for cfmax_rel in steps of 10%
cfmax_rel_values = np.linspace(cfmax_rel['min'], cfmax_rel['max'], 21)

## Run MATILDA

# smb_unc = dict()
#
# for s in ensemble_mean_result.keys():
#     smb_unc[s] = pd.DataFrame()
#
#     for c in cfmax_rel.keys():
#         # Original runs for min, max, and barandun
#         results = matilda_simulation(input_df=ensemble_mean_result[s], CFMAX_rel=cfmax_rel[c], **full_period,
#                                      **settings, **parameters, **fix_val)
#
#         # Same processing steps as before for these runs
#         runoff = results[0][['ice_melt_on_glaciers', 'total_runoff']].resample('Y').sum()
#         runoff[f'glacier_contribution_{c}'] = (runoff['ice_melt_on_glaciers'] / runoff['total_runoff']) * 100
#
#         glaciers = pd.DataFrame()
#         glaciers[[f'glacier_area_{c}', f'glacier_mass_mmwe_{c}']] = results[5][['glacier_area', 'glacier_mass_mmwe']][
#                                                                     1:]
#         glaciers.loc[glaciers[f'glacier_area_{c}'] == 0, f'glacier_mass_mmwe_{c}'] = 0
#         glaciers[f'glacier_mass_mmwe_{c}'] = glaciers[f'glacier_mass_mmwe_{c}'] / 1000
#         glaciers.index = glaciers.index + pd.offsets.YearEnd(0)
#
#         merged_df = pd.merge(runoff[f'glacier_contribution_{c}'], glaciers, left_index=True, right_index=True,
#                              how='inner')
#         merged_df_rolling = merged_df.rolling(window=3).mean()
#         smb_unc[s] = pd.concat([smb_unc[s], merged_df_rolling], axis=1)
#
#     # Now run for step values between min and max
#     for val in cfmax_rel_values:
#         rounded_val = round(val, 3)  # Round to 3 decimal places
#
#         # Run the simulation
#         results = matilda_simulation(input_df=ensemble_mean_result[s], CFMAX_rel=val, **full_period,
#                                      **settings, **parameters, **fix_val)
#
#         # Resample runoff data and calculate glacier contribution
#         runoff = results[0][['ice_melt_on_glaciers', 'total_runoff']].resample('Y').sum()
#         glacier_contribution = (runoff['ice_melt_on_glaciers'] / runoff['total_runoff']) * 100
#
#         # Convert glacier_contribution from Series to DataFrame and assign a column name
#         glacier_contribution_df = pd.DataFrame(glacier_contribution,
#                                                columns=[f'glacier_contribution'])
#
#         # Prepare glacier data
#         glaciers = pd.DataFrame()
#         glaciers['glacier_area'] = results[5]['glacier_area'][1:]
#         glaciers['glacier_mass_mmwe'] = results[5]['glacier_mass_mmwe'][1:] / 1000
#
#         # When glacier area is 0, glacier mass is as well
#         glaciers.loc[glaciers['glacier_area'] == 0, 'glacier_mass_mmwe'] = 0
#
#         # Adjust the index of the glaciers DataFrame to match the runoff DataFrame
#         glaciers.index = glaciers.index + pd.offsets.YearEnd(0)
#
#         # Merge the glacier_contribution DataFrame with glaciers DataFrame
#         merged_df = pd.merge(glacier_contribution_df, glaciers, left_index=True, right_index=True, how='inner')
#
#         # Apply 3-year rolling mean
#         merged_df_rolling = merged_df.rolling(window=3).mean()
#
#         # Concatenate the rolling mean data to the final DataFrame
#         smb_unc[s] = pd.concat([smb_unc[s], merged_df_rolling.add_suffix(f'_step_{rounded_val}')], axis=1)

## Save/load pickels

# dict_to_pickle(smb_unc, "/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/figures/uncertainty/smb_uncertainty.pickle")

smb_unc = pickle_to_dict("/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/figures/uncertainty/smb_uncertainty.pickle")
# smb_unc = pickle_to_dict"/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/figures/uncertainty/smb_uncertainty_5perc.pickle")


##

# Calculate the min and max values across all step runs for shading
for scenario in smb_unc.keys():
    for var in ['glacier_area', 'glacier_mass_mmwe', 'glacier_contribution']:
        step_cols = [col for col in smb_unc[scenario].columns if var in col and 'step' in col]
        smb_unc[scenario][f'{var}_min_all_steps'] = smb_unc[scenario][step_cols].min(axis=1)
        smb_unc[scenario][f'{var}_max_all_steps'] = smb_unc[scenario][step_cols].max(axis=1)

# Define color maps for the two scenarios
blue_cmap = mcolors.LinearSegmentedColormap.from_list("blue_gradient", ["lightblue", "darkblue"])
orange_cmap = mcolors.LinearSegmentedColormap.from_list("orange_gradient", ["yellow", "darkorange"])

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(4, 10), sharex=True)

variables = ['glacier_area', 'glacier_mass_mmwe', 'glacier_contribution']
y_labels = ['Glacier Area [km²]', 'Glacier Mass [m.w.e.]', 'Runoff Contribution from Glaciers [\%]']
colors = {'SSP2': 'orange', 'SSP5': 'darkblue'}

for i, var in enumerate(variables):
    ax = axs[i]

    for scenario in smb_unc.keys():
        color = colors[scenario]

        # Choose the correct color map based on the scenario
        cmap = orange_cmap if scenario == 'SSP2' else blue_cmap

        # Plot min and max lines corresponding to the min and max parameter values (with respective line widths)
        ax.plot(smb_unc[scenario].index, smb_unc[scenario][f'{var}_min'], color=color, alpha=0.4, linewidth=1,
                zorder=2)
        ax.plot(smb_unc[scenario].index, smb_unc[scenario][f'{var}_max'], color=color, alpha=0.8, linewidth=1.5, zorder=2)

        # Shading between the min and max values of all runs
        ax.fill_between(smb_unc[scenario].index, smb_unc[scenario][f'{var}_min_all_steps'],
                        smb_unc[scenario][f'{var}_max_all_steps'], color=color, alpha=0.3, zorder=2)

        # Plot the barandun line
        ax.plot(smb_unc[scenario].index, smb_unc[scenario][f'{var}_barandun'], label=f'{scenario} Barandun',
                color=color, linestyle='--', linewidth=1, zorder=2)

        # # Plot the step lines with a color gradient
        # for idx, val in enumerate(cfmax_rel_values):
        #     rounded_val = round(val, 3)
        #
        #     # Normalize the index for the colormap (0 to 1 range)
        #     normalized_idx = idx / (len(cfmax_rel_values) - 1)
        #
        #     # Get the color from the colormap
        #     step_color = cmap(normalized_idx)
        #
        #     # Plot the step lines with the color gradient
        #     ax.plot(smb_unc[scenario].index, smb_unc[scenario][f'{var}_step_{rounded_val}'], color=step_color, alpha=0.3,
        #             linewidth=0.3, zorder=1)

        ax.set_ylabel(y_labels[i], fontsize=12, fontweight='bold')
        ax.grid(True, color='lightgrey', linewidth=0.5, alpha=0.7, zorder=1)
        ax.autoscale(enable=True, axis='x', tight=True)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax)

        # Adjust the font size of the x-axis labels
        ax.tick_params(axis='x', labelsize=9)  # Set the desired font size for x-axis labels

# Custom legend and final adjustments
fig.legend(handles=handles, loc='lower center', ncol=2, frameon=True, fontsize=10, prop={'weight': 'bold'})
plt.tight_layout()
fig.subplots_adjust(bottom=0.09)

# Save the figure
fig.savefig(home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/figures/uncertainty/smb_uncertainty.png', dpi=400)
# fig.show()
