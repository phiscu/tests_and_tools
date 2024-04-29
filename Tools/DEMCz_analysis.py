## Imports
import numpy as np
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
from pathlib import Path
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'

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
data = pd.read_csv(data_path + 'normal/' + 'DEMCz_Paper_1_FAST1-5M-006_PCORR64_amount-param-normal_conlim-08_thin-1_burnIn500_400k_chains8_loglikeKGE_SMB430_2000-2017.csv')

## Filters

data = data[data['like1'] > 500]
data = data[data['like2'] < 100]
# data = data.tail(100)

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
custom_text = [f'Index: {index}<br>like1: {like1}<br>like2: {like2}' for (index, like1, like2) in zip(data.index, data['like1'], data['like2'])]

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

best = data[data.index == 24567]
# Filter columns with the prefix 'par'
par_columns = [col for col in best.columns if col.startswith('par')]

# Create a dictionary with keys as column names without the 'par' prefix
parameters = {col.replace('par', ''): best[col].values[0] for col in par_columns}

# Print the dictionary
print(parameters)

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
    'glacier_profile': glacier_profile
}

fix_val={'PCORR': 0.64, 'SFCF': 1, 'CET': 0}

# FAST 1M: good fit, shift in spring and autumn, MB perfect
# parameters = {'lr_temp': -0.0057516084, 'lr_prec': 0.0015256472, 'BETA': 5.6014814, 'FC': 323.61023, 'K0': 0.124523245, 'K1': 0.01791149, 'K2': 0.006872296, 'LP': 0.5467752, 'MAXBAS': 5.325173, 'PERC': 2.9256027, 'UZL': 354.0794, 'TT_snow': 0.5702063, 'TT_diff': 1.9629607, 'CFMAX_ice': 5.2739882, 'CFMAX_rel': 1.2821848, 'CWH': 0.05004947, 'AG': 0.5625456, 'RFS': 0.2245709}
# FAST005 (old FAST): perfect fit, MB 0.08 m lower (-0.5)
# parameters = {'lr_temp': -0.0065, 'BETA': 1.9256144, 'FC': 159.13263, 'K1': 0.015555069, 'K2': 0.00107, 'PERC': 1.814131, 'UZL': 411.4885, 'TT_snow': -1.5, 'CFMAX_ice': 5.47608, 'CWH': 0.11245408}
# FAST005 (old FAST) 2 : good fit, shift in spring and autumn, MB very good
# parameters = {'lr_temp': -0.0065, 'BETA': 2.2036834, 'FC': 165.63278, 'K1': 0.0103, 'K2': 0.00107, 'PERC': 1.8354536, 'UZL': 370.14172, 'TT_snow': 0.30567998, 'CFMAX_ice': 5.801351, 'CWH': 0.18327774}

# INTERNAL 10chains not converged: best fit yet, MB perfect
# parameters = {'lr_temp': -0.0065, 'lr_prec': 0.002, 'BETA': 5.923893, 'FC': 423.99445, 'K0': 0.34201974, 'K1': 0.0102, 'K2': 0.00106, 'LP': 0.4893323, 'MAXBAS': 2.3684337, 'PERC': 1.63444, 'UZL': 398.43973, 'TT_snow': 0.2792817, 'TT_diff': 2.5, 'CFMAX_ice': 6.499742, 'CFMAX_rel': 1.3407481, 'CWH': 0.2, 'AG': 0.6276681, 'RFS': 0.0995919}

# Random results from normal_DEMCz (DEMCz_Paper_1_timing-param-fix_PCORR64_amount-param-normal_conlim-08_thin-1_burnIn500_400k_chains8_loglikeKGE_SMB430_2000-2017.csv):
# superb in both likes (0.88, diffMB=-0.14; OK fit for validation period (0.7) --> lr_temp exceed bounds...:
# parameters = {'lr_temp': -0.00522, 'lr_prec': 0.00262, 'BETA': 1.7596115, 'TT_snow': -1.7349747, 'TT_diff': 2.8170776, 'CFMAX_ice': 3.6867228, 'CFMAX_rel': 0.7901475, 'FC': 127.04292, 'K0': 0.2840262, 'K1': 0.012222029, 'K2': 0.00119, 'LP': 0.92943203, 'MAXBAS': 2.0706086, 'PERC': 1.159848, 'UZL': 480.79425, 'CWH': 0.06322092, 'AG': 0.87717545, 'RFS': 0.21784203}
# superb in both likes (0.87, diffMB=0.002; OK fit for validation period (0.77) --> lr_temp and CFMAX_ice are out of your own bounds...:
parameters = {'lr_temp': -0.00522, 'lr_prec': 0.00262, 'BETA': 1.3601397, 'TT_snow': -1.7347374, 'TT_diff': 2.9, 'CFMAX_ice': 3.3, 'CFMAX_rel': 0.725, 'FC': 50.2, 'K0': 0.034036454, 'K1': 0.0101, 'K2': 0.00119, 'LP': 1.0, 'MAXBAS': 2.0, 'PERC': 0.87882555, 'UZL': 499.0, 'CWH': 0.112680875, 'AG': 0.999, 'RFS': 0.12816271}
# almost the same as above:
# parameters = {'lr_temp': -0.00522, 'lr_prec': 0.00262, 'BETA': 2.29, 'TT_snow': -1.7105813, 'TT_diff': 2.9, 'CFMAX_ice': 3.3, 'CFMAX_rel': 0.725, 'FC': 215.32507, 'K0': 0.29593968, 'K1': 0.0101, 'K2': 0.00119, 'LP': 1.0, 'MAXBAS': 2.0, 'PERC': 1.0314698, 'UZL': 499.0, 'CWH': 0.09771314, 'AG': 0.999, 'RFS': 0.12379003}

# Best results LHS workflow:
fix_val = {'PCORR': 0.64, 'SFCF': 1, 'CET': 0, 'lr_temp': -0.00605, 'lr_prec': 0.00117, 'TT_diff': 1.36708, 'CFMAX_rel': 1.81114}
# Great in both likes (0.85, diffMB=-0.005; great fit for validation period (0.81); systematical shift in spring and autumn
# parameters = {'BETA': 1.007054, 'FC': 302.78784, 'K1': 0.0130889015, 'K2': 0.0049547367, 'PERC': 0.8058457, 'UZL': 482.38788, 'TT_snow': -0.418914, 'CFMAX_ice': 5.592482, 'CWH': 0.10325227}
# Great KGE (0.86), good in diffMB=-0.056; great fit for validation period (0.79); fewer systematical shift in spring and autumn
# parameters = {'BETA': 4.469486, 'FC': 291.8014, 'K1': 0.013892675, 'K2': 0.0045123543, 'PERC': 1.4685482, 'UZL': 430.45685, 'TT_snow': -1.3098346, 'CFMAX_ice': 5.504861, 'CWH': 0.12402764}
# Great KGE (0.85), great in diffMB=-0.02; great fit for validation period (0.8); little systematical shift in spring and autumn, weird peak in early summer
parameters = {'BETA': 1.0345612, 'FC': 212.69034, 'K1': 0.025412053, 'K2': 0.0049677053, 'PERC': 2.1586323, 'UZL': 392.962, 'TT_snow': -1.4604422, 'CFMAX_ice': 5.3250813, 'CWH': 0.1916532}

results = matilda_simulation(**full_period, **settings, **parameters, **fix_val)

results[7].show()

# Compare mass balances
barandun = [-0.427142857142857, -0.171428571428571, -0.0957142857142857, -0.39, -0.0242857142857143, 0.07, -0.11, -0.0171428571428571, -0.105714285714286, 0.00714285714285714, 0.141428571428571, -0.194285714285714, -1.87
, -0.0485714285714286, -1.21, -0.385714285714286, -1.32, -0.972857142857143, -0.977142857142857, np.NaN,  np.NaN]
wgms = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,  np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,-950, -880, -390, -1120, -810, -540, -240]

mass_balances = pd.DataFrame({'matilda_mb': results[5]['smb_water_year'][1:,]/1000,
                              'barandun_mb': barandun,
                              'Kara-Batkak_wgms_mb': [x / 1000 for x in wgms]})
print(mass_balances.head(-2).mean())
print('diffMB:' + str(round(mass_balances.mean()[0]-mass_balances.mean()[1], 2)))
mass_balances.plot()
plt.ylim(-2,0.7)
plt.show()

# Compare SWE:
swe_obs = pd.read_csv('/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/hmadsr/kyzylsuu_swe.csv', parse_dates=['Date'], index_col='Date')
swe_obs = swe_obs * 1000
swe_obs = swe_obs['2000-01-01':'2017-09-30']
swe_sim = results[0].snowpack_off_glaciers['2000-01-01':'2017-09-30'].to_frame(name="SWE_sim")

swe_sim.index = pd.to_datetime(swe_sim.index)
swe_obs.index = pd.to_datetime(swe_obs.index)
swe_df = pd.concat([swe_obs, swe_sim], axis=1)
swe_df.columns = ['SWE_obs', 'SWE_sim']

swe_df.plot()
plt.show()
## Write parameters into full dictionary

parameters.update(fix_val)
print(parameters)



