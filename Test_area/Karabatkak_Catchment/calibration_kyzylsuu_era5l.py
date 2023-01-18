# -*- coding: UTF-8 -*-

## import
import os
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import socket
import salem
from datetime import date, datetime, timedelta
import spotpy
import matplotlib.pyplot as plt
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
sys.path.append(home + '/Ana-Lena_Phillip/data/tests_and_tools')
# from Preprocessing.Preprocessing_functions import dmod_score, load_cmip, cmip2df
from Test_area.SPOTPY import mspot, mspot_glacier
from matilda.core import matilda_simulation, matilda_parameter, matilda_preproc, input_scaling, calculate_glaciermelt, calculate_PDD, glacier_area_change, create_lookup_table, hbv_simulation, updated_glacier_melt, create_statistics


# Setting file paths and parameters

# Paths
wd = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data'
input_path = wd + "/input/kyzylsuu"
output_path = wd + "/output/kyzylsuu"

era_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/era5l'
mswx_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/mswx'
har_path = home + '/EBA-CA/Tianshan_data/HARv2/variables/all_variables_HARv2_daily_kyzylsuu_1980_2020.nc'

t2m_agg_path = '/met/temp_cat_agg_era5l_harv2_mswx_1982-2020.csv'
tp_agg_path = '/met/prec_cat_agg_era5l_harv2_mswx_1982-2020.csv'
runoff_obs = "/hyd/obs/Kyzylsuu_1982_2020_latest.csv"
cmip_path = '/met/cmip6/'
cal_path = '/validation/Compact/Kyzylsuu/'

# Files
# Obs
obs = pd.read_csv(input_path + runoff_obs)
obs_met = pd.read_csv(input_path + '/met/obs/met_data_full_kyzylsuu_2007-2015.csv', parse_dates=['time'], index_col='time')

# Initial glaciers
glacier_profile = pd.read_csv(wd + "/kyzulsuu_glacier_profile.csv")

# Catchment-wide aggregates
t2m_agg = pd.read_csv(input_path + t2m_agg_path)
tp_agg = pd.read_csv(input_path + tp_agg_path)
df_mswx = pd.concat([t2m_agg.time, t2m_agg.mswx, tp_agg.mswx], axis=1)
df_mswx.columns = ['TIMESTAMP', 'T2', 'RRR']
df_era = pd.concat([t2m_agg.time, t2m_agg.era, tp_agg.era], axis=1)
df_era.columns = ['TIMESTAMP', 'T2', 'RRR']
df_har = pd.concat([t2m_agg.time, t2m_agg.har, tp_agg.har], axis=1)
df_har.columns = ['TIMESTAMP', 'T2', 'RRR']

## Calibration data:
# 18y MBs (Shean et.al.)
mb_catch = -0.16
mb_catch_sigma = 0.32
mb_karab = -0.185
mb_karab_sigma = 0.139

# Annual MBs Karabatkak
mb_ann = pd.read_csv(wd + cal_path + 'wgms_mb_ela_aar_areachng_karabatkak_1957-2021.csv', index_col='YEAR', parse_dates=['BEGIN_PERIOD', 'END_WINTER', 'END_PERIOD'])
mb_ann = mb_ann[mb_ann.index.isin(range(2014, 2021))]
mb_ann.loc[mb_ann.index==2014,'BEGIN_PERIOD'] = datetime.fromisoformat('2013-09-15')    # Estimated start and end of winter as average of the existing dates
mb_ann.loc[mb_ann.index==2014,'END_WINTER'] = datetime.fromisoformat('2014-05-15')

## Step 1 - Limit temperature lapse rate

lr_temp_lo = -0.0065
lr_temp_up = -0.0055

# Prepare input dataframes
df_t = t2m_agg.set_index('time')                    # Set timeindex
df_t.index = pd.to_datetime(df_t.index)
df_t = df_t - 273.15                                # to °C
ele = [723, 621, 723]           # Elevation differences data - glaciers: MSWX/ER5L - 723m   HARv2 - 621m

## Step 2 - Limit the precipitation correction factor:
    # MSWX
PCORR_lo_mswx = 0.74; PCORR_up_mswx = 1.7
    # HARv2
PCORR_lo_har = 0.23; PCORR_up_har = 0.49
    # ERA5L
PCORR_lo_era = 0.29; PCORR_up_era = 0.67

## Step 3 - Calibrate SFCF, TT_snow, TT_diff, RFS, and lr_prec on Karabatkak winter mass balance:

## 3.3 Run SPOT with glacier routine only

glacier_profile_karab = pd.read_csv(wd + '/glacier_profile_karabatkak_farinotti_marie.csv') # inital glacier profile of Karabatkak
karab_area = 2.046              # Karabatkak area according to RGI 6
mb_ann_matilda = mb_ann.reset_index()

# Pass parameter bounds from step 1 and 2 as dict:
# lim_dict = {'lr_temp_lo': lr_temp_lo, 'lr_temp_up': lr_temp_up, 'PCORR_lo': PCORR_lo_era, 'PCORR_up': PCORR_up_era}
# **lim_dict, CFMAX_ice_up=6,
# >> Cirrus: era_glacier_only_step3_PCORRSD_cfmax6_sceua_10000

## Parameter sets - step 3:

# param_dict = {"lr_temp": -0.00554937, "lr_prec": 0.00193333, "PCORR": 0.586563, "TT_snow": -0.723249,
# "TT_diff": 0.513913, "CFMAX_ice": 2.0808, "CFMAX_rel": 1.20257, "SFCF": 0.99768, "RFS": 0.0500798}
# --> annual MB:    -0.06 m we
# --> winter MAE:   62.5 mm we

## Step 4 - Calibrate melt rates on Karabatkak annual (or summer) balance

# lim_dict = {'lr_temp_lo': -0.0058795563, 'lr_temp_up': -0.0055493726, 'lr_prec_lo': 0.0018089622,
# 'lr_prec_up': 0.0019422206, 'PCORR_lo': 0.5865627, 'PCORR_up': 0.6219727, 'TT_snow_lo': -0.9250981,
# 'TT_snow_up': -0.7232486, 'TT_diff_lo': 0.5000055, 'TT_diff_up': 0.5184937, 'SFCF_lo': 0.9741424,
# 'SFCF_up': 0.9989884}

# >> Cirrus: era_glacier_only_step4_best5_PCORRSD_cfmax6_sceua_10000 (accidentally limited CFMAX, repetition without it
# resulted in similar CFMAX bounds but higher (worse) obj)

## Parameters sets - step 4:
# Best 5% as bounds:
# param_dict = {"lr_temp": -0.00565182, "lr_prec": 0.00186596, "PCORR": 0.609555, "TT_snow": -0.866314,
# "TT_diff": 0.510023, "CFMAX_ice": 4.33581, "CFMAX_rel": 1.20012, "SFCF": 0.98971, "RFS": 0.050307}

# --> annual MB:  -0.81 m
# --> winter MAE: 74.5 mm
# --> summer MAE: 249.7 mm
# --> annual MAE: 291.9 mm

## Step 5 - Calibrate remaining parameters on runoff:

# param4 = {'lr_temp_lo': -0.005651822, 'lr_prec_lo': 0.0018659597, 'PCORR_lo': 0.6095553, 'TT_snow_lo': -0.8663144,
# 'TT_diff_lo': 0.51002336, 'CFMAX_ice_lo': 4.335807, 'CFMAX_rel_lo': 1.2001212, 'SFCF_lo': 0.98970973,
# 'RFS_lo': 0.05030702, 'lr_temp_up': -0.005651822, 'lr_prec_up': 0.0018659597, 'PCORR_up': 0.6095553,
# 'TT_snow_up': -0.8663144, 'TT_diff_up': 0.51002336, 'CFMAX_ice_up': 4.335807, 'CFMAX_rel_up': 1.2001212,
# 'SFCF_up': 0.98970973, 'RFS_up': 0.05030702}

# >> Cirrus: era_step5_best5_PCORRSD_cfmax6_demcz_10000

## Parameters sets - step 5:
# era_step5_best5_PCORRSD_cfmax6_demcz_10000:

param_dict = {'lr_temp': -0.00565, 'lr_prec': 0.00187, 'BETA': 2.022733, 'CET': 0.031154592, 'FC': 398.22998,
              'K0': 0.035520267, 'K1': 0.0148187075, 'K2': 0.0129077, 'LP': 0.8870321, 'MAXBAS': 3.2627285,
              'PERC': 2.9873798, 'UZL': 160.45435, 'PCORR': 0.61, 'TT_snow': -0.866, 'TT_diff': 0.51, 'CFMAX_ice': 4.34,
              'CFMAX_rel': 1.2, 'SFCF': 0.99, 'CWH': 0.021318546, 'AG': 0.60934013, 'RFS': 0.0503}

## Check results:
output_MATILDA = matilda_simulation(df_era, obs=obs, set_up_start='1997-01-01', set_up_end='1999-12-31', # output='/home/phillip/Seafile/Ana-Lena_Phillip/data/test',
                                    sim_start='2000-01-01', sim_end='2017-12-31', freq="M", glacier_profile=glacier_profile,
                                    area_cat=295.763, lat=42.33, warn=False, plot_type="all", plots=True, elev_rescaling=True,
                                    ele_dat=3273, ele_cat=3295, area_glac=32.51, ele_glac=4068, pfilter=0,
                                    parameter_set=param_dict)

print('Mean Annual MB: ' + str(round(output_MATILDA[5].smb_water_year.mean() / 1000, 2)) + ' (+-'
      + str(round(output_MATILDA[5].smb_water_year.std() / 1000, 2)) + ') m w.e.')

output_MATILDA[7].show()

# KGE coefficient: 0.84
# NSE coefficient: 0.8
# RMSE: 22.51
# Mean Annual MB: -0.28 (+-0.23) m w.e.

## Validation:

output_MATILDA = matilda_simulation(df_era, obs=obs, set_up_start='2015-01-01', set_up_end='2017-12-31', # output='/home/phillip/Seafile/Ana-Lena_Phillip/data/test',
                                    sim_start='2018-01-01', sim_end='2020-12-31', freq="M", glacier_profile=glacier_profile,
                                    area_cat=295.763, lat=42.33, warn=False, plot_type="all", plots=True, elev_rescaling=True,
                                    ele_dat=3273, ele_cat=3295, area_glac=32.51, ele_glac=4068, pfilter=0,
                                    parameter_set=param_dict)

print('Mean Annual MB: ' + str(round(output_MATILDA[5].smb_water_year.mean() / 1000, 2)) + ' (+-'
      + str(round(output_MATILDA[5].smb_water_year.std() / 1000, 2)) + ') m w.e.')

# KGE coefficient: 0.85
# NSE coefficient: 0.87
# RMSE: 18.41
# Mean Annual MB: -0.32 (+-0.09) m w.e.


