## Imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import spotpy
import matplotlib.pyplot as plt
import seaborn as sns
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
data = pd.read_csv(data_path + 'converged/' + 'new_fast/' + 'DEMCz_Paper_1_NEW-FAST005_conlim-08_thin-1_burnIn500_100k_chains10_loglikeKGE_savesim_lr_temp55-65_SMB430_CFMAX35-8.csv')
## Filters

data = data[data['like1'] > 300]
data = data[data['like2'] < 100]
# data = data.tail(200)

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
    'freq': 'M',
    'lat': 42.18280043250193,
    'glacier_profile': glacier_profile
}

fix_val={'PCORR': 0.69, 'SFCF': 1, 'CET': 0}

# FAST 1M: good fit, shift in spring and autumn, MB perfect
parameters = {'lr_temp': -0.0057516084, 'lr_prec': 0.0015256472, 'BETA': 5.6014814, 'FC': 323.61023, 'K0': 0.124523245, 'K1': 0.01791149, 'K2': 0.006872296, 'LP': 0.5467752, 'MAXBAS': 5.325173, 'PERC': 2.9256027, 'UZL': 354.0794, 'TT_snow': 0.5702063, 'TT_diff': 1.9629607, 'CFMAX_ice': 5.2739882, 'CFMAX_rel': 1.2821848, 'CWH': 0.05004947, 'AG': 0.5625456, 'RFS': 0.2245709}
# FAST005 (old FAST): perfect fit, MB 0.08 m lower (-0.5)
# parameters = {'lr_temp': -0.0065, 'BETA': 1.9256144, 'FC': 159.13263, 'K1': 0.015555069, 'K2': 0.00107, 'PERC': 1.814131, 'UZL': 411.4885, 'TT_snow': -1.5, 'CFMAX_ice': 5.47608, 'CWH': 0.11245408}
# FAST005 (old FAST) 2 : good fit, shift in spring and autumn, MB very good
# parameters = {'lr_temp': -0.0065, 'BETA': 2.2036834, 'FC': 165.63278, 'K1': 0.0103, 'K2': 0.00107, 'PERC': 1.8354536, 'UZL': 370.14172, 'TT_snow': 0.30567998, 'CFMAX_ice': 5.801351, 'CWH': 0.18327774}

# INTERNAL 10chains not converged: best fit yet, MB perfect
# parameters = {'lr_temp': -0.0065, 'lr_prec': 0.002, 'BETA': 5.923893, 'FC': 423.99445, 'K0': 0.34201974, 'K1': 0.0102, 'K2': 0.00106, 'LP': 0.4893323, 'MAXBAS': 2.3684337, 'PERC': 1.63444, 'UZL': 398.43973, 'TT_snow': 0.2792817, 'TT_diff': 2.5, 'CFMAX_ice': 6.499742, 'CFMAX_rel': 1.3407481, 'CWH': 0.2, 'AG': 0.6276681, 'RFS': 0.0995919}


results = matilda_simulation(**full_period, **settings, **parameters, **fix_val)

results[7].show()

# Compare mass balances
barandun = [-0.427142857142857, -0.171428571428571, -0.0957142857142857, -0.39, -0.0242857142857143, 0.07, -0.11, -0.0171428571428571, -0.105714285714286, 0.00714285714285714, 0.141428571428571, -0.194285714285714, -1.87
, -0.0485714285714286, -1.21, -0.385714285714286, -1.32, -0.972857142857143, -0.977142857142857, np.NaN,  np.NaN]
wgms = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,  np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,-950, -880, -390, -1120, -810, -540, -240]

mass_balances = pd.DataFrame({'matilda_mb': results[5]['smb_water_year'][1:,]/1000,
                              'barandun_mb': barandun,
                              'Kara-Batkak_wgms_mb': [x / 1000 for x in wgms]})
print(mass_balances.mean())
mass_balances.plot()
plt.ylim(-2,0.7)
plt.show()


