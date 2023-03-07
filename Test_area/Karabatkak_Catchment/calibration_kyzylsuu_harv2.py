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
ele = [780, 695, 780]           # Elevation differences data - glaciers: MSWX/ER5L - 780   HARv2 - 695

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
# >> Cirrus: har_glacier_only_step3_PCORRSD_cfmax6_sceua_10000

## Parameter sets - step 3:

param_dict = {"lr_temp": -0.00649637, "lr_prec": 0.00199965, "PCORR": 0.489982, "TT_snow": 1.48491,
"TT_diff": 1.98706, "CFMAX_ice": 3.40165, "CFMAX_rel": 1.20019, "SFCF": 0.999927, "RFS": 0.155885}
# --> annual MB:    -0.24 m we
# --> winter MAE:   92.1 mm we

## Step 4 - Calibrate melt rates on Karabatkak annual (or summer) balance

# lim_dict = {'lr_temp_lo': -0.0064999675, 'lr_temp_up': -0.0064825267, 'lr_prec_lo': 0.001998146,
# 'lr_prec_up': 0.001999993, 'PCORR_lo': 0.4895411, 'PCORR_up': 0.48999646, 'TT_snow_lo': 1.4373187,
# 'TT_snow_up': 1.4992715, 'TT_diff_lo': 1.8972002, 'TT_diff_up': 2.1292267, 'SFCF_lo': 0.9995677,
# 'SFCF_up': 0.99999976}

# >> Cirrus: har_glacier_only_step4_best5_sceua_10000

## Parameters sets - step 4:
# Best 10% as bounds:       ?????????????????????
param_dict = {"lr_temp": -0.00649287, "lr_prec": 0.00199906, "PCORR": 0.489875, "TT_snow": 1.46076,
"TT_diff": 2.07356, "CFMAX_ice": 4.89683, "CFMAX_rel": 1.25909, "SFCF": 0.999755, "RFS": 0.104154}

# --> annual MB:  -0.69 m
# --> winter MAE: 117.84 mm
# --> summer MAE: 222.76 mm
# --> annual MAE: 234.74 mm

# --> leads to 100% positive MB_cum for full catchment!
# --> passed best10% bounds for step 5 instead (forgot to limit CFMAX, RFS etc.):

## Step 5 - Calibrate remaining parameters on runoff:

# param4 = {'lr_temp_lo': -0.0064999284, 'lr_temp_up': -0.006482626, 'lr_prec_lo': 0.0019981468,
# # 'lr_prec_up': 0.001999989, 'PCORR_lo': 0.48954123, 'PCORR_up': 0.48999646, 'TT_snow_lo': 1.4373997,
# # 'TT_snow_up': 1.4990101, 'TT_diff_lo': 1.9036899, 'TT_diff_up': 2.1261978, 'SFCF_lo': 0.9995679,
# # 'SFCF_up': 0.99999964}

# >> Cirrus

## Parameters sets - step 5:
# har_step5_best5_PCORRSD_cfmax6_best10-of-step4_demcz_10000:
# param_dict = {'lr_temp': -0.00648808, 'lr_prec': 0.002, 'BETA': 1.5065616, 'CET': 0.018232437, 'FC': 368.49615,
#               'K0': 0.24859273, 'K1': 0.018363932, 'K2': 0.0102617275, 'LP': 0.91912854, 'MAXBAS': 5.603384,
#               'PERC': 2.2507815, 'UZL': 440.29865, 'PCORR': 0.49, 'TT_snow': 1.4965256, 'TT_diff': 2.0546567,
#               'CFMAX_ice': 6.881965, 'CFMAX_rel': 1.4422476, 'SFCF': 1.0, 'CWH': 0.09515469, 'AG': 0.38603407,
#               'RFS': 0.07024568}

# har_step5_best5_PCORR-1_CFMAX8_best-of-step4_demcz_10000
param_dict = {'lr_temp': -0.00605, 'lr_prec': 0.000476, 'BETA': 1.2063544, 'CET': 0.16037333, 'FC': 446.18478,
              'K0': 0.23083933, 'K1': 0.012752315, 'K2': 0.123080276, 'LP': 0.979405, 'MAXBAS': 3.186689,
              'PERC': 0.012815826, 'UZL': 294.05124, 'PCORR': 0.676, 'TT_snow': 1.48, 'TT_diff': 2.22,
              'CFMAX_ice': 4.34, 'CFMAX_rel': 1.32, 'SFCF': 0.816, 'CWH': 0.03950917, 'AG': 0.19969909, 'RFS': 0.211}

# har_step5_best5_PCORR-1_CFMAX8_best-5-of-step4_demcz_10000
param_dict = {'lr_temp': -0.00551, 'lr_prec': 0.0011019271, 'BETA': 1.5167366, 'CET': 0.08388762, 'FC': 312.6267,
              'K0': 0.057340793, 'K1': 0.014489007, 'K2': 0.011614603, 'LP': 0.62394303, 'MAXBAS': 4.3378434,
              'PERC': 1.9138637, 'UZL': 480.3686, 'PCORR': 0.82988787, 'TT_snow': -0.7955638, 'TT_diff': 1.8914235,
              'CFMAX_ice': 6.3914685, 'CFMAX_rel': 2.163788, 'SFCF': 0.81514865, 'CWH': 0.10640457, 'AG': 0.3912738,
              'RFS': 0.13966435}

# har_step5_PCORRfix1_summerMBfirst_best-05-of-step4_demcz_10000
param_dict = {'lr_temp': -0.00575854, 'lr_prec': 0.0017964235, 'BETA': 3.093238, 'CET': 0.119686745, 'FC': 260.40015, 'K0': 0.11771713, 'K1': 0.028871195, 'K2': 0.007295351, 'LP': 0.73932546, 'MAXBAS': 3.9770935, 'PERC': 1.947096, 'UZL': 441.1273, 'PCORR': 0.9, 'TT_snow': -1.2201041, 'TT_diff': 1.1371124, 'CFMAX_ice': 5.6754417, 'CFMAX_rel': 1.4699625, 'SFCF': 0.8310065, 'CWH': 0.0075647812, 'AG': 0.826539, 'RFS': 0.123936825}
# --> PCORR is necessary!

# Multi-objective: har_no_lim_lhs_multiobj_mb160_10000 -
param_dict = {'lr_temp': -0.0051079653, 'lr_prec': 0.000844849, 'BETA': 3.4739997, 'CET': 0.094185196, 'FC': 313.67563, 'K0': 0.10679094, 'K1': 0.018912481, 'K2': 0.0032495812, 'LP': 0.9779223, 'MAXBAS': 5.897662, 'PERC': 0.8597642, 'UZL': 358.6113, 'PCORR': 0.9487066, 'TT_snow': 0.84052354, 'TT_diff': 1.3764166, 'CFMAX_ice': 3.210709, 'CFMAX_rel': 1.8540443, 'SFCF': 0.50506264, 'CWH': 0.113060504, 'AG': 0.7944547, 'RFS': 0.19186838}
# har_no_lim_lhs_multiobj_mb160_30000
param_dict = {'lr_temp': -0.008754168, 'lr_prec': 0.0007020797, 'BETA': 4.6194205, 'CET': 0.23149519, 'FC': 487.8178, 'K0': 0.3046358, 'K1': 0.011214738, 'K2': 0.0102247195, 'LP': 0.42217332, 'MAXBAS': 4.782967, 'PERC': 2.1861486, 'UZL': 93.953156, 'PCORR': 1.0329615, 'TT_snow': -1.0262012, 'TT_diff': 1.0059689, 'CFMAX_ice': 10.075248, 'CFMAX_rel': 1.7703471, 'SFCF': 0.41434342, 'CWH': 0.03910887, 'AG': 0.44393122, 'RFS': 0.056584343}
# KGE coefficient: 0.77
# NSE coefficient: 0.7
# Mean Annual MB: -0.29 (+-0.29) m w.e.

# --> Riesige lapse rate und CFMAX: Brauche wohl mindestens temp limits!

# Multi-objective: har_lrtemp007-005_PCORR08-12_lhs_multiobj_mb160_50000 - Limit lr_temp and PCORR
param_dict = {'lr_temp': -0.006077369, 'lr_prec': 0.0013269137, 'BETA': 5.654754, 'CET': 0.08080378, 'FC': 365.68375, 'K0': 0.36890236, 'K1': 0.022955153, 'K2': 0.060069658, 'LP': 0.63395154, 'MAXBAS': 5.094901, 'PERC': 0.39491335, 'UZL': 348.0978, 'PCORR': 1.0702422, 'TT_snow': -1.1521467, 'TT_diff': 1.5895765, 'CFMAX_ice': 3.6518102, 'CFMAX_rel': 1.8089349, 'SFCF': 0.42293832, 'CWH': 0.11234668, 'AG': 0.9618855, 'RFS': 0.11432563}
# KGE coefficient: 0.8
# NSE coefficient: 0.71
# Mean Annual MB: -0.18 (+-0.19) m w.e.

param_dict = {'lr_temp': -0.0056301677, 'lr_prec': 0.0019705386, 'BETA': 1.2099667, 'CET': 0.26492256, 'FC': 218.59938, 'K0': 0.18391511, 'K1': 0.022600068, 'K2': 0.123332605, 'LP': 0.59431714, 'MAXBAS': 3.8761072, 'PERC': 0.24085149, 'UZL': 333.58185, 'PCORR': 0.9268524, 'TT_snow': -1.2417217, 'TT_diff': 0.5462903, 'CFMAX_ice': 2.9941382, 'CFMAX_rel': 1.554281, 'SFCF': 0.47420865, 'CWH': 0.10106187, 'AG': 0.63298637, 'RFS': 0.14237927}
# KGE coefficient: 0.76
# NSE coefficient: 0.76
# Mean Annual MB: -0.16 (+-0.17) m w.e.

# Best result in uncertainty bounds
param_dict = {'lr_temp': -0.0057281437, 'lr_prec': 0.0008959501, 'BETA': 3.2377098, 'CET': 0.07599049, 'FC': 64.35039, 'K0': 0.3286151, 'K1': 0.010778711, 'K2': 0.08118787, 'LP': 0.512051, 'MAXBAS': 6.219792, 'PERC': 1.1484363, 'UZL': 417.24612, 'PCORR': 0.8649721, 'TT_snow': -0.95179623, 'TT_diff': 2.0308511, 'CFMAX_ice': 4.0415125, 'CFMAX_rel': 1.255553, 'SFCF': 0.80183876, 'CWH': 0.051763818, 'AG': 0.85030746, 'RFS': 0.2012494}
# KGE coefficient: 0.86
# NSE coefficient: 0.76
# Mean Annual MB: -0.27 (+-0.24) m w.e.



## Check results:
output_MATILDA = matilda_simulation(df_har, obs=obs, set_up_start='1997-01-01', set_up_end='1999-12-31', # output='/home/phillip/Seafile/Ana-Lena_Phillip/data/test',
                                    sim_start='2000-01-01', sim_end='2017-12-31', freq="M", glacier_profile=glacier_profile,
                                    area_cat=295.763, lat=42.33, warn=False, plot_type="all", plots=True, elev_rescaling=True,
                                    ele_dat=3256, ele_cat=3295, area_glac=32.51, ele_glac=4068, pfilter=0,
                                    parameter_set=param_dict)

print('Mean Annual MB: ' + str(round(output_MATILDA[5].smb_water_year.mean() / 1000, 2)) + ' (+-'
      + str(round(output_MATILDA[5].smb_water_year.std() / 1000, 2)) + ') m w.e.')

abs(160 - output_MATILDA[5].smb_water_year.mean())

# output_MATILDA[9].show()
# > PCORR_SD, CFMAX6:
# KGE coefficient: 0.62
# NSE coefficient: 0.64
# RMSE: 30.14
# Mean Annual MB: -0.25 (+-0.3) m w.e.

# > PCORR1, CFMAX8:
# KGE coefficient: 0.72
# NSE coefficient: 0.69
# RMSE: 28.1
# Mean Annual MB: -0.26 (+-0.22) m w.e.

# > PCORR1, CFMAX8 - best5 of step4:
# KGE coefficient: 0.9
# NSE coefficient: 0.81
# RMSE: 22.11
# Mean Annual MB: -0.52 (+-0.35) m w.e.

# No limits - multi-obj:
# KGE coefficient: 0.73
# NSE coefficient: 0.65
# RMSE: 1.02
# Mean Annual MB: -0.19 (+-0.2) m w.e.

output_MATILDA[7].show()

## Validation:

output_MATILDA = matilda_simulation(df_har, obs=obs, set_up_start='2015-01-01', set_up_end='2017-12-31', # output='/home/phillip/Seafile/Ana-Lena_Phillip/data/test',
                                    sim_start='2018-01-01', sim_end='2019-12-31', freq="M", glacier_profile=glacier_profile,
                                    area_cat=295.763, lat=42.33, warn=False, plot_type="all", plots=True, elev_rescaling=True,
                                    ele_dat=3256, ele_cat=3295, area_glac=32.51, ele_glac=4068, pfilter=0,
                                    parameter_set=param_dict)

print('Mean Annual MB: ' + str(round(output_MATILDA[5].smb_water_year.mean() / 1000, 2)) + ' (+-'
      + str(round(output_MATILDA[5].smb_water_year.std() / 1000, 2)) + ') m w.e.')

##
# # param_dict = best_summary['best_param']
#
DATAFRAME = df_har
SEASON = 'winter'
if DATAFRAME is df_har:
    ele_dat = 3256
else:
    ele_dat = 3341

parameter = matilda_parameter(DATAFRAME, set_up_start='1997-01-01', set_up_end='1999-12-31',
                              sim_start='2000-01-01', sim_end='2020-12-31', freq="D", lat=42.33,
                              area_cat=karab_area, area_glac=karab_area, ele_dat=ele_dat, ele_cat=None, ele_glac=3830,
                              parameter_set=param_dict)

df_preproc = matilda_preproc(DATAFRAME, parameter)
lookup_table = create_lookup_table(glacier_profile_karab, parameter)
output_DDM, glacier_change, input_df_catchment = updated_glacier_melt(df_preproc, lookup_table, glacier_profile_karab, parameter)

sim = []
mb_obs = []
evaluation = mb_ann
simulation = output_DDM.DDM_smb
for i in evaluation.index:
    if SEASON is 'winter':
        mb = simulation[mspot_glacier.winter(i, evaluation)].sum()
        part = evaluation[evaluation.index == i].WINTER_BALANCE.squeeze()
    elif SEASON is 'summer':
        mb = simulation[mspot_glacier.summer(i, evaluation)].sum()
        part = evaluation[evaluation.index == i].SUMMER_BALANCE.squeeze()
    else:
        mb = simulation[mspot_glacier.annual(i, evaluation)].sum()
        part = evaluation[evaluation.index == i].ANNUAL_BALANCE.squeeze()
    sim.append(mb)

    mb_obs.append(part)
sim_new = pd.DataFrame()
sim_new['mod'] = pd.DataFrame(sim)
sim_new['mb_obs'] = pd.DataFrame(mb_obs)
clean = sim_new.dropna()
simulation_clean = clean['mod']
evaluation_clean = clean['mb_obs']

print(spotpy.objectivefunctions.mae(evaluation_clean, simulation_clean))
scores = spotpy.objectivefunctions.calculate_all_functions(evaluation_clean, simulation_clean)
print(pd.DataFrame(scores))
print(clean)
print('Mean Annual MB (2000-2018): ' + str(round(glacier_change[glacier_change.time.isin(evaluation.index)].
                                                 smb_water_year.mean() / 1000, 2)) + ' m w.e.')


## Remarks:

# PCORR is at the very limit with very narrow limits through the bench! We probably need higher values!
