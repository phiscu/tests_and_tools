## Imports
import numpy as np
from scipy.stats import rankdata
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import spotpy
import matplotlib.pyplot as plt
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
path_to_palatinottf = '/home/phillip/Downloads/Palatino.ttf'
fm.fontManager.addfont(path_to_palatinottf)


pio.renderers.default = "browser"

data_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/parameters/'

## Paths

# Converged
# data = pd.read_csv(data_path + 'converged/' + 'new_fast/' + 'DEMCz_Paper_1_NOFIX_conlim-08_thin-1_burnIn500_100k_chains8_loglikeKGE_savesim_lr_temp55-65_SMB400_CFMAX35-8.csv')
# data = pd.read_csv(data_path + 'converged/' + 'new_fast/' + 'DEMCz_Paper_1_INTERNAL_conlim-08_thin-1_burnIn500_100k_chains10_loglikeKGE_savesim_lr_temp55-65_SMB400_CFMAX35-8.csv')
# data = pd.read_csv(data_path + 'converged/' + 'new_fast/' + 'DEMCz_Paper_1_NEW-FAST005_conlim-08_thin-1_burnIn500_100k_chains8_loglikeKGE_savesim_lr_temp55-65_SMB430_CFMAX35-8.csv')
# data = pd.read_csv(data_path + 'converged/' + 'new_fast/' + 'DEMCz_Paper_1_NEW-FAST005_conlim-08_thin-1_burnIn500_100k_chains10_loglikeKGE_savesim_lr_temp55-65_SMB430_CFMAX35-8.csv')
# Normal:
# data = pd.read_csv(data_path + 'normal/' + 'DEMCz_Paper_1_timing-param-fix_PCORR64_amount-param-normal_conlim-08_thin-1_burnIn500_400k_chains8_loglikeKGE_SMB430_2000-2017.csv')
# data = pd.read_csv(data_path + 'normal/' + 'DEMCz_Paper_1_FAST1-5M-006_PCORR64_amount-param-normal_conlim-08_thin-1_burnIn500_400k_chains8_loglikeKGE_SMB430_2000-2017.csv')
# SWE:
# data = pd.read_csv('/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/parameters/SWE/DEMCz_Paper_1_SWE-Final_Step3_conlim-08_thin-1_burnIn500_400k_chains10_loglikeKGE_2000-2017.csv')
# data = pd.read_csv(data_path + 'SWE/' + 'DEMCz_Paper_1_SWE-Final_Step3_pareto-bounds_conlim-08_thin-1_burnIn500_400k_chains10_loglikeKGE_2000-2017.csv')
data = pd.read_csv(
    '/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/parameters/SWE/DEMCz_Paper_1_SnowCal-update_Step3_conlim-08_thin-1_burnIn500_400k_chains10_loglikeKGE_2000-2017.csv')
# data = pd.read_csv(
# '/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/parameters/SWE/DEMCz_Paper_1_SnowCal-update_Step3_conlim-08_thin-1_burnIn500_400k_chains8_loglikeKGE_2000-2017.csv')

## Filters

data = data[data['like1'] > 800]            # 700 =~0.85    max. 895 =~0.87
data = data[data['like2'] < 50]
# data = data.tail(100)

print(min(data['parCFMAX_rel']), max(data['parCFMAX_rel']))

## Posterior distribution

burn_in_period = 0  # Example burn-in period
thin_factor = 1  # Example thinning factor
thinned_samples = data.iloc[burn_in_period::thin_factor]

# Create a 4x5 matrix of subplots
fig, axs = plt.subplots(5, 5, figsize=(20, 16))

# Plot posterior distributions for each parameter in the matrix of subplots
for i, parameter in enumerate(thinned_samples.columns[:-1]):  # Exclude the 'chain' column
    row = i // 5
    col = i % 5
    sns.kdeplot(thinned_samples[parameter], shade=True, ax=axs[row, col])
    axs[row, col].set_xlabel(parameter)
    axs[row, col].set_ylabel('Density')
    axs[row, col].set_title(f'Posterior Distribution of {parameter}')

plt.tight_layout()
plt.show()

## Identify best run

# Create a custom text that includes the index number for each data point
custom_text = [f'Index: {index}<br>like1: {like1}<br>like2: {like2}' for (index, like1, like2) in
               zip(data.index, data['like1'], data['like2'])]

# Create a 2D scatter plot with custom text
fig = go.Figure(data=go.Scatter(
    x=data['like1'],
    y=data['like2'],
    mode='markers',
    text=custom_text,  # Assign custom text to each data point
    hoverinfo='text',  # Show custom text when hovering
))

# Update layout
fig.update_layout(
    xaxis_title='Loglike Kling-Gupta-Efficiency score',
    yaxis_title='MAE of mean annual SMB',
    #title='2D Scatter Plot of like1 and like2 with parPCORR Color Ramp',
    margin=dict(l=0, r=0, b=0, t=40)  # Adjust margins for better visualization
)

# Show the plot
fig.show()

## Get parameters

best = data[data.index == 8826]         # 8826 38319
# Filter columns with the prefix 'par'
par_columns = [col for col in best.columns if col.startswith('par')]

# Create a dictionary with keys as column names without the 'par' prefix
parameters = {col.replace('par', ''): best[col].values[0] for col in par_columns}

# Print the dictionary
print(parameters)

## List of dicts to resample

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

par_columns = [col for col in data.columns if col.startswith('par')]
param_list = []
for index, row in data.iterrows():
    param = {col.replace('par', ''): row[col] for col in par_columns}
    param_list.append(param)

for p in param_list:
    p.update(fix_val)


##
# fix_more = {'LP': 0.7, 'MAXBAS': 3.0}
#
# parameters.update(fix_more)


## MATILDA:

data_path = '/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/notebook_example_for_spot/'
obs = pd.read_csv(data_path + 'obs_runoff_example.csv')
df = pd.read_csv(data_path + 'era5.csv')
glacier_profile = pd.read_csv(data_path + 'glacier_profile.csv')

full_period = {'set_up_end': '1999-12-31', 'set_up_start': '1998-01-01',
               'sim_end': '2020-12-31', 'sim_start': '2000-01-01'}
calibration_period = {'set_up_end': '1999-12-31', 'set_up_start': '1998-01-01',
                      'sim_end': '2017-12-31', 'sim_start': '2000-01-01'}
validation_period = {'set_up_end': '2017-12-31', 'set_up_start': '2016-01-01',
                     'sim_end': '2020-12-31', 'sim_start': '2018-01-01'}

settings = {
    'input_df': df,
    'obs': obs,
    'area_cat': 295.67484249904464,
    'area_glac': 31.829413146585885,
    'ele_cat': 3293.491688025922,
    'ele_dat': 3335.67,
    'ele_glac': 4001.8798828125,
    'elev_rescaling': True,
    'freq': 'D',
    'lat': 42.18280043250193,
    'glacier_profile': glacier_profile,
    'plot_type': 'all'
}

# fix_val = {'PCORR': 0.64, 'SFCF': 1, 'CET': 0}
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

# FAST 1M: good fit, shift in spring and autumn, MB perfect
# parameters = {'lr_temp': -0.0057516084, 'lr_prec': 0.0015256472, 'BETA': 5.6014814, 'FC': 323.61023, 'K0': 0.124523245, 'K1': 0.01791149, 'K2': 0.006872296, 'LP': 0.5467752, 'MAXBAS': 5.325173, 'PERC': 2.9256027, 'UZL': 354.0794, 'TT_snow': 0.5702063, 'TT_diff': 1.9629607, 'CFMAX_ice': 5.2739882, 'CFMAX_rel': 1.2821848, 'CWH': 0.05004947, 'AG': 0.5625456, 'RFS': 0.2245709}
# FAST005 (old FAST): perfect fit, MB 0.08 m lower (-0.5)
# parameters = {'lr_temp': -0.0065, 'BETA': 1.9256144, 'FC': 159.13263, 'K1': 0.015555069, 'K2': 0.00107, 'PERC': 1.814131, 'UZL': 411.4885, 'TT_snow': -1.5, 'CFMAX_ice': 5.47608, 'CWH': 0.11245408}
# FAST005 (old FAST) 2 : good fit, shift in spring and autumn, MB very good
# parameters = {'lr_temp': -0.0065, 'BETA': 2.2036834, 'FC': 165.63278, 'K1': 0.0103, 'K2': 0.00107, 'PERC': 1.8354536, 'UZL': 370.14172, 'TT_snow': 0.30567998, 'CFMAX_ice': 5.801351, 'CWH': 0.18327774}

# INTERNAL 10chains not converged: best fit yet, MB perfect - validation period bad (0.56)
# parameters = {'lr_temp': -0.0065, 'lr_prec': 0.002, 'BETA': 5.923893, 'FC': 423.99445, 'K0': 0.34201974, 'K1': 0.0102, 'K2': 0.00106, 'LP': 0.4893323, 'MAXBAS': 2.3684337, 'PERC': 1.63444, 'UZL': 398.43973, 'TT_snow': 0.2792817, 'TT_diff': 2.5, 'CFMAX_ice': 6.499742, 'CFMAX_rel': 1.3407481, 'CWH': 0.2, 'AG': 0.6276681, 'RFS': 0.0995919}

# Random results from normal_DEMCz (DEMCz_Paper_1_timing-param-fix_PCORR64_amount-param-normal_conlim-08_thin-1_burnIn500_400k_chains8_loglikeKGE_SMB430_2000-2017.csv):
# superb in both likes (0.88, diffMB=-0.14; OK fit for validation period (0.7) --> lr_temp exceed bounds...:
# parameters = {'lr_temp': -0.00522, 'lr_prec': 0.00262, 'BETA': 1.7596115, 'TT_snow': -1.7349747, 'TT_diff': 2.8170776, 'CFMAX_ice': 3.6867228, 'CFMAX_rel': 0.7901475, 'FC': 127.04292, 'K0': 0.2840262, 'K1': 0.012222029, 'K2': 0.00119, 'LP': 0.92943203, 'MAXBAS': 2.0706086, 'PERC': 1.159848, 'UZL': 480.79425, 'CWH': 0.06322092, 'AG': 0.87717545, 'RFS': 0.21784203}
# superb in both likes (0.87, diffMB=0.002; OK fit for validation period (0.77) --> lr_temp and CFMAX_ice are out of your own bounds...:
# parameters = {'lr_temp': -0.00522, 'lr_prec': 0.00262, 'BETA': 1.3601397, 'TT_snow': -1.7347374, 'TT_diff': 2.9, 'CFMAX_ice': 3.3, 'CFMAX_rel': 0.725, 'FC': 50.2, 'K0': 0.034036454, 'K1': 0.0101, 'K2': 0.00119, 'LP': 1.0, 'MAXBAS': 2.0, 'PERC': 0.87882555, 'UZL': 499.0, 'CWH': 0.112680875, 'AG': 0.999, 'RFS': 0.12816271}
# almost the same as above:
# parameters = {'lr_temp': -0.00522, 'lr_prec': 0.00262, 'BETA': 2.29, 'TT_snow': -1.7105813, 'TT_diff': 2.9, 'CFMAX_ice': 3.3, 'CFMAX_rel': 0.725, 'FC': 215.32507, 'K0': 0.29593968, 'K1': 0.0101, 'K2': 0.00119, 'LP': 1.0, 'MAXBAS': 2.0, 'PERC': 1.0314698, 'UZL': 499.0, 'CWH': 0.09771314, 'AG': 0.999, 'RFS': 0.12379003}

# Best results LHS workflow:
# fix_val = {'PCORR': 0.64, 'SFCF': 1, 'CET': 0, 'lr_temp': -0.00605, 'lr_prec': 0.00117, 'TT_diff': 1.36708, 'CFMAX_rel': 1.81114}
# Great in both likes (0.85, diffMB=-0.005; great fit for validation period (0.81); systematical shift in spring and autumn
# parameters = {'BETA': 1.007054, 'FC': 302.78784, 'K1': 0.0130889015, 'K2': 0.0049547367, 'PERC': 0.8058457, 'UZL': 482.38788, 'TT_snow': -0.418914, 'CFMAX_ice': 5.592482, 'CWH': 0.10325227}
# Great KGE (0.86), good in diffMB=-0.056; great fit for validation period (0.79); fewer systematical shift in spring and autumn
# parameters = {'BETA': 4.469486, 'FC': 291.8014, 'K1': 0.013892675, 'K2': 0.0045123543, 'PERC': 1.4685482, 'UZL': 430.45685, 'TT_snow': -1.3098346, 'CFMAX_ice': 5.504861, 'CWH': 0.12402764}
# Great KGE (0.85), great in diffMB=-0.02; great fit for validation period (0.8); little systematical shift in spring and autumn, weird peak in early summer
# parameters = {'BETA': 1.0345612, 'FC': 212.69034, 'K1': 0.025412053, 'K2': 0.0049677053, 'PERC': 2.1586323, 'UZL': 392.962, 'TT_snow': -1.4604422, 'CFMAX_ice': 5.3250813, 'CWH': 0.1916532}

parameters = {
    'BETA': 1.06986,
    'FC': 429.672,
    'K0': 0.167255,
    'K1': 0.0126635,
    'K2': 0.0972067,
    'LP': 0.918489,
    'MAXBAS': 2.9504,
    'PERC': 0.559362,
    'UZL': 359.547,
    'CFMAX_rel': 1.4054,
    'CWH': 0.0202685,
    'AG': 0.763657
}



# parameters = {    "FC": 93.4912,
#     "K1": 0.0123915,
#     "K2": 0.0568295,
#     "PERC": 0.0451047,
#     "UZL": 321.546,
#     "CWH": 0.0169412,
#     "AG": 0.532519
# }
##
results = matilda_simulation(**calibration_period, **settings, **parameters, **fix_val)

results[7].show()

# results[10].write_html('/home/phillip/Seafile/CLIMWATER/YSS/2024/Slides/figs/matilda_calib_annual_kyzylsuu.html')

# print(results[5].iloc[:, -3:])

## Compare mass balances

barandun = [-0.427142857142857, -0.171428571428571, -0.0957142857142857, -0.39, -0.0242857142857143, 0.07, -0.11,
            -0.0171428571428571, -0.105714285714286, 0.00714285714285714, 0.141428571428571, -0.194285714285714, -1.87
    , -0.0485714285714286, -1.21, -0.385714285714286, -1.32, -0.972857142857143, -0.977142857142857, np.NaN, np.NaN]
wgms = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
        -950, -880, -390, -1120, -810, -540, -240]


mass_balances = pd.DataFrame({'Simulation': results[5]['smb_water_year'][1:, ] / 1000,
                              'Barandun et.al.': barandun,
                              'WGMS (Karabatkak)': [x / 1000 for x in wgms]})
print(mass_balances.head(-3).mean())
print('diffMB:' + str(round(mass_balances.mean()[0] - mass_balances.mean()[1], 2)))

# Define accessible colors
colors = {
    'Simulation': '#0072B2',    # Blue
    'Barandun et.al.': '#D55E00',  # Orange
    'WGMS (Karabatkak)': '#009E73'  # Green
}

ax = mass_balances.plot(color=[colors['Simulation'], colors['Barandun et.al.'], colors['WGMS (Karabatkak)']])
plt.ylim(-2, 0.5)

# Add horizontal lines and shaded areas for mean values and uncertainties
shean_mean = -0.156
shean_unc = 0.324
miles_mean = -0.379
miles_std = 0.19

# Shean et al. (mean and uncertainty)
plt.axhline(y=shean_mean, color='#CC79A7', linestyle='-', label='Shean et.al. (mean)')  # Pink
plt.fill_between(
    x=mass_balances.index,
    y1=shean_mean - shean_unc,
    y2=shean_mean + shean_unc,
    color='#CC79A7',
    alpha=0.2,
    label='Shean et.al. (uncertainty)'
)

# Miles et al. (mean and uncertainty)
plt.axhline(y=miles_mean, color='#E69F00', linestyle='-', label='Miles et.al. (mean)')  # Yellow
plt.fill_between(
    x=mass_balances.index,
    y1=miles_mean - miles_std,
    y2=miles_mean + miles_std,
    color='#E69F00',
    alpha=0.2,
    label='Miles et.al. (std)'
)

plt.ylabel('m w.e.')
plt.xlabel('Year')

# Function to format the y-axis tick labels
def y_format(tick_val, pos):
    if tick_val == 0:
        return '0'
    elif tick_val > 0:
        return '{:.1f}'.format(tick_val)
    else:
        return '{:.1f}'.format(tick_val)

# Apply the formatter to the y-axis
ax.yaxis.set_major_formatter(FuncFormatter(y_format))

plt.legend()
plt.show()

## Compare SWE:

swe_obs = pd.read_csv(
    '/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/hmadsr/kyzylsuu_swe.csv',
    parse_dates=['Date'], index_col='Date')
swe_obs = swe_obs * 1000
swe_obs = swe_obs['2000-01-01':'2017-09-30']
swe_sim = results[0].snowpack_off_glaciers['2000-01-01':'2017-09-30'].to_frame(name="SWE_sim")
snow = results[0][['melt_off_glaciers', 'snow_off_glaciers']]['2000-01-01':'2017-09-30']
swe_sim.index = pd.to_datetime(swe_sim.index)
swe_obs.index = pd.to_datetime(swe_obs.index)
swe_df = pd.concat([swe_obs, swe_sim, snow], axis=1)
swe_df.columns = ['SWE_obs', 'SWE_sim', 'snow_melt', 'snow_fall']

swe_df_monthly = swe_df.resample('M').agg(
    {'SWE_obs': 'mean', 'SWE_sim': 'mean', 'snow_melt': 'sum', 'snow_fall': 'sum'})

# Objective function

swe_df.SWE_sim = swe_df.SWE_sim * 0.928
print('KGE: ' + str(he.kge_2012(swe_df.SWE_sim, swe_df.SWE_obs, remove_zero=False)))
print('MAE: ' + str(mae(swe_df.SWE_obs, swe_df.SWE_sim)))
print('RMSE: ' + str(rmse(swe_df.SWE_obs, swe_df.SWE_sim)))

# Plot timeseries

swe_df_monthly = swe_df.resample('M').agg(
    {'SWE_obs': 'mean', 'SWE_sim': 'mean', 'snow_melt': 'sum', 'snow_fall': 'sum'})
swe_df_monthly.plot()
plt.show()

swe_df_monthly[['SWE_obs', 'SWE_sim']].plot()
plt.show()

##
results[5]

## Annual cycle
swe_df = doy_col(swe_df)

swe_df = swe_df.reset_index()
swe_df['DOY'] = swe_df['index'].dt.dayofyear

# Group by 'DOY' and calculate the mean for each metric
swe_df_annual_cycle = swe_df.groupby('DOY').agg({
    'SWE_obs': 'mean',
    'SWE_sim': 'mean',
    'snow_melt': 'mean',
    'snow_fall': 'mean'
})
# Truncate to 365 days
swe_df_annual_cycle = swe_df_annual_cycle[:-1]

# Create a plot for the average annual cycle
plt.figure(figsize=(12, 6))
plt.plot(swe_df_annual_cycle.index, swe_df_annual_cycle['SWE_obs'], label='observed', linestyle='-')
plt.plot(swe_df_annual_cycle.index, swe_df_annual_cycle['SWE_sim'], label='simulated', linestyle='-')

plt.xlabel('Day of the Year')
plt.ylabel('mm')
plt.title('Average Annual Cycle of Observed and Simulated SWE')
plt.legend()
plt.grid(True)
plt.show()

## Write parameters into full dictionary

parameters.update(fix_val)
print(parameters)
write_yaml(parameters, home + '/EBA-CA/Repositories/matilda_edu/output/' + 'parameters.yml')

##
settings_resample = {
    'area_cat': 295.67484249904464,
    'area_glac': 31.829413146585885,
    'ele_cat': 3293.491688025922,
    'ele_dat': 3335.67,
    'ele_glac': 4001.8798828125,
    'elev_rescaling': True,
    'freq': 'D',
    'lat': 42.18280043250193,
    'glacier_profile': glacier_profile
}

# matilda_bulk = MatildaBulkSampler(df, obs, settings_resample, param_list, swe_obs, swe_scaling=0.928)
# matilda_resamples, summary = matilda_bulk.run_single_process()

# dict_to_pickle(matilda_resamples, home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/parameters/SWE/parameter_sets_final/DEMCz_SnowCal-update_Step3_chains10_loglikeKGE_2000-2017_top64.pickle')
matilda_resamples = pickle_to_dict(home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/parameters/SWE/parameter_sets_final/DEMCz_SnowCal-update_Step3_chains10_loglikeKGE_2000-2017_top64.pickle')
summary = extract_resamples(matilda_resamples)

## Pareto front

target_MB = -0.43
summary['MAE_SMB'] = abs(summary['SMB_mean18'] - target_MB)

# Normalize the objective functions using Min-Max normalization
objectives = ['KGE', 'MAE_SMB', 'SWE_KGE', 'KGE_summer', 'KGE_winter']
for obj in objectives:
    summary[f"{obj}_normalized"] = (summary[obj] - summary[obj].min()) / (summary[obj].max() - summary[obj].min())

summary["MAE_SMB_normalized"] = 1 - summary["MAE_SMB_normalized"]     # make MAE ascending

# Rank the normalized objective functions (ascending for MAE_SMB, descending for others)
pareto_columns = [col for col in summary.columns if col.endswith('_normalized')]

for norm in pareto_columns:
    summary[norm.replace('_normalized', '_rank')] = rankdata(-summary[norm], method="min")

# Define a function to determine the Pareto front

def is_pareto_efficient(points, maximize=True):
    num_points = points.shape[0]
    is_efficient = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        # If maximizing, a point is dominated if another point is greater in all objectives
        # If minimizing, a point is dominated if another point is lesser in all objectives
        if maximize:
            is_efficient[i] = not np.any(np.all(points > points[i], axis=1))
        else:
            is_efficient[i] = not np.any(np.all(points < points[i], axis=1))
    return is_efficient

# Check which points are on the Pareto front
pareto_points = summary[pareto_columns].values
pareto_front = is_pareto_efficient(pareto_points, maximize=True)

summary["on_pareto_front"] = pareto_front

## All but the seasons have been considered. Final decision based on seasonal KGE.

summary["season_ranks"] = rankdata((summary['KGE_summer_rank']*0.3 + summary['KGE_winter_rank']*0.7) / 2, method="min")
final = summary.loc[summary["season_ranks"] < 2, 'Parameters'].squeeze()

# 8826 ('P1') ist trotzdem besser....

## Run final set
results = matilda_simulation(**full_period, **settings, **final)
results[7].show()


## Water balance

balance = results[0]['total_precipitation'].resample('Y').sum() + results[0]['ice_melt_on_glaciers'].resample('Y').sum() - results[0]['actual_evaporation'].resample('Y').sum() - results[0]['total_runoff'].resample('Y').sum()
balance.sum()
