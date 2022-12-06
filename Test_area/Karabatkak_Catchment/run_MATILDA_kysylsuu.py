# -*- coding: UTF-8 -*-

## import of necessary packages
import os
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import socket
import HydroErr as he
import hydroeval
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

t2m_path = "/met/era5l/t2m_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
tp_path = "/met/era5l/tp_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
t2m_agg_path = '/met/temp_cat_agg_era5l_harv2_mswx_1982-2020.csv'
tp_agg_path = '/met/prec_cat_agg_era5l_harv2_mswx_1982-2020.csv'
runoff_obs = "/hyd/obs/Kyzylsuu_1982_2021_latest.csv"
cmip_path = '/met/cmip6/'

    # Calibration period
t2m = pd.read_csv(input_path + t2m_path)
tp = pd.read_csv(input_path + tp_path)
df = pd.concat([t2m, tp.tp], axis=1)
df.rename(columns={'time': 'TIMESTAMP', 't2m': 'T2','tp':'RRR'}, inplace=True)
obs = pd.read_csv(input_path + runoff_obs)

t2m_agg = pd.read_csv(input_path + t2m_agg_path)
tp_agg = pd.read_csv(input_path + tp_agg_path)
df_mswx = pd.concat([t2m_agg.time, t2m_agg.mswx, tp_agg.mswx], axis=1)
df_mswx.columns = ['TIMESTAMP', 'T2', 'RRR']
df_era = pd.concat([t2m_agg.time, t2m_agg.era, tp_agg.era], axis=1)
df_era.columns = ['TIMESTAMP', 'T2', 'RRR']
df_har = pd.concat([t2m_agg.time, t2m_agg.har, tp_agg.har], axis=1)
df_har.columns = ['TIMESTAMP', 'T2', 'RRR']

    # Scenarios
# cmipT = load_cmip(input_path + cmip_path, 't2m_CMIP6_all_models_adjusted_42.516-79.0167_1982-01-01-2100-12-31_')
# cmipP = load_cmip(input_path + cmip_path, 'tp_CMIP6_all_models_adjusted_42.516-79.0167_1982-01-01-2100-12-31_')
glacier_profile = pd.read_csv(wd + "/kyzulsuu_glacier_profile.csv")

## Elevations: # ERA5L (and MSWX): 3273m (diff: 723m -> 4.34K)          HARv2: 3172m (diff: 621m -> 3,73K)


# # Basic overview plot
# obs_fig = obs.copy()
# obs_fig.set_index('Date', inplace=True)
# obs_fig.index = pd.to_datetime(obs_fig.index)
# obs_fig = obs_fig[slice('1998-01-01','2020-01-31')]
# plt.figure()
# ax = obs_fig.resample('M').agg(pd.Series.sum, min_count=1).plot(label='Kyzylsuu (Hydromet)')
# ax.set_ylabel('Discharge [m³/s]')
# plt.show()

#> Data from 1982-01-01 to 1989-12-31 [8y], 1992-01-01 to 2007-12-31 [16y], 2010-01-01 to 2014-12-31 [5y], 2017-05-04 to 2021-07-30 [4y]

# interf=4
# freqst=2
# k=21
# par_iter = (1 + 4 * interf ** 2 * (1 + (k - 2) * freqst)) * k
# print(par_iter)

##
# Shean et.al. 2000-2018 - catchment-wide mean: -0.16	+-0.32
# Shean et.al. 2000-2018 - Karabatkak:          -0.185  +-0.139 --> well within this range!

## HARv2:
param_dict = mspot.load_parameters(output_path + '/matilda_par_smpl_har_cat_agg_2000_2020_1000rep.csv')      # From csv
param_dict['CFMAX_ice'] = param_dict.pop('CFMAX_snow')                          # Switched CFMAX_ice and CFMAX_snow in the source code
param_dict['CFMAX_ice'] = param_dict['CFMAX_ice'] * param_dict['CFMAX_rel']

output_MATILDA = matilda_simulation(df_har, obs=obs, set_up_start='1997-01-01', set_up_end='1999-12-31',# output='/home/phillip/Seafile/Ana-Lena_Phillip/data/test',
                                      sim_start='2000-01-01', sim_end='2018-12-31', freq="M", glacier_profile=glacier_profile,
                                      area_cat=295.763, lat=42.33, warn=False, plot_type="print", plots=True, elev_rescaling=True,
                                      ele_dat=3172, ele_cat=3295, area_glac=32.51, ele_glac=4068, pfilter=0,
                                      parameter_set=param_dict)

print('Mean Annual MB: ' + str(round(output_MATILDA[5].smb_water_year.mean() / 1000, 2)) + ' m w.e.')
# --> mean MB ok, score (0.74) gut (mit 'falscher' elevation)
# --> mean MB sehr gut, score (0.66) OK (mit richtiger elevation) --> scheinbar sinnvolles parameter set

## MSWX:
param_dict = mspot.load_parameters(output_path + '/matilda_par_smpl_mswx_cat_agg_2000_2020_1000rep.csv')      # From csv
param_dict['CFMAX_ice'] = param_dict.pop('CFMAX_snow')                          # Switched CFMAX_ice and CFMAX_snow in the source code
param_dict['CFMAX_ice'] = param_dict['CFMAX_ice'] * param_dict['CFMAX_rel']

output_MATILDA = matilda_simulation(df_mswx, obs=obs, set_up_start='1997-01-01', set_up_end='1999-12-31', # output='/home/phillip/Seafile/Ana-Lena_Phillip/data/test',
                                      sim_start='2000-01-01', sim_end='2018-12-31', freq="M", glacier_profile=glacier_profile,
                                      area_cat=295.763, lat=42.33, warn=False, plot_type="print", plots=True, elev_rescaling=True,
                                      ele_dat=3273, ele_cat=3295, area_glac=32.51, ele_glac=4068, pfilter=0,
                                    parameter_set=param_dict)

print('Mean Annual MB: ' + str(round(output_MATILDA[5].smb_water_year.mean() / 1000, 2)) + ' m w.e.')
# --> MB ok, score nicht so

## ERA5L:
# param_dict = mspot.load_parameters(output_path + '/matilda_par_smpl_era_cat_agg_2000_2020_1000rep.csv')      # From csv
# param_dict['CFMAX_ice'] = param_dict.pop('CFMAX_snow')                          # Switched CFMAX_ice and CFMAX_snow in the source code
# param_dict['CFMAX_ice'] = param_dict['CFMAX_ice'] * param_dict['CFMAX_rel']
param_dict = {"lr_temp": -0.00483, "lr_prec": 0.0006614, "BETA": 2.328, "CET": 0.2166, "FC": 436.5, "K0": 0.07947,
              "K1": 0.1713, "K2": 0.00596, "LP": 0.571, "MAXBAS": 3.023, "PERC": 1.355, "UZL": 474.0, "PCORR": 1.034,
              "TT_snow": -0.599, "TT_diff": 3.752, "CFMAX_ice": 2.736352, "CFMAX_rel": 1.468, "SFCF": 0.443,
              "CWH": 0.0009346, "AG": 0.4846, "RFS": 0.2327}


output_MATILDA = matilda_simulation(df_era, obs=obs, set_up_start='1997-01-01', set_up_end='1999-12-31',# output='/home/phillip/Seafile/Ana-Lena_Phillip/data/test',
                                      sim_start='2000-01-01', sim_end='2018-12-31', freq="M", glacier_profile=glacier_profile,
                                      area_cat=295.763, lat=42.33, warn=False, plot_type="print", plots=True, elev_rescaling=True,
                                      ele_dat=3273, ele_cat=3295, area_glac=32.51, ele_glac=4068, pfilter=0,
                                    parameter_set=param_dict)

print('Mean Annual MB: ' + str(round(output_MATILDA[5].smb_water_year.mean() / 1000, 2)) + ' m w.e.')

# --> MB ok, score (0.74) gut

## Load parameter from GloH2O
params = pd.read_csv(input_path + '/hyd/HBV_params_gloh2o_kyzylsuu_cat_agg.csv', index_col='realization')
params = round(params.drop('time', axis=1), 2)
params = params.iloc[0].to_dict()



##
output_MATILDA[7].show()
output_MATILDA[0].SMB.resample('Y').sum()
print(output_MATILDA[0].SMB.resample('Y').sum().mean())
print(output_MATILDA[1].DDM_smb.resample('Y').sum().mean())



##

glacier_profile_karab = pd.read_csv(wd + '/glacier_profile_karabatkak_farinotti_marie.csv')

karab_area = 2.046          # RGI 6

param_dict = {"lr_temp": -0.00483, "lr_prec": 0.0006614, "BETA": 2.328, "CET": 0.2166, "FC": 436.5, "K0": 0.07947,
              "K1": 0.1713, "K2": 0.00596, "LP": 0.571, "MAXBAS": 3.023, "PERC": 1.355, "UZL": 474.0, "PCORR": 1.034,
              "TT_snow": -0.599, "TT_diff": 3.752, "CFMAX_ice": 2.736352, "CFMAX_rel": 1.468, "SFCF": 0.443,
              "CWH": 0.0009346, "AG": 0.4846, "RFS": 0.2327}

parameter = matilda_parameter(df_era, set_up_start='1997-01-01', set_up_end='1999-12-31',
                                      sim_start='2000-01-01', sim_end='2020-12-31', freq="M", lat=42.33,
                                      area_cat=karab_area, area_glac=karab_area, ele_dat=3273, ele_cat=None, ele_glac=3830,
                                      parameter_set=param_dict)

df_preproc = matilda_preproc(df_era, parameter)
lookup_table = create_lookup_table(glacier_profile_karab, parameter)
output_DDM, glacier_change, input_df_catchment = updated_glacier_melt(df_preproc, lookup_table, glacier_profile_karab, parameter)

print('Mean Annual MB: ' + str(round(glacier_change.smb_water_year.mean() / 1000, 2)) + ' m w.e.')



## Run SPOTPY:

best_summary = mspot_glacier.psample(df=df_era, obs=obs, rep=10, #output=output_path,
                             set_up_start='1997-01-01', set_up_end='1999-12-31',
                             sim_start='2000-01-01', sim_end='2020-12-31',
                             freq="D", lat=42.33,
                             area_cat=295.763, area_glac=32.51,
                             ele_dat=3273, ele_cat=3295, ele_glac=4068,
                             glacier_profile=glacier_profile, elev_rescaling=True,

                            parallel=False, dbformat='csv', algorithm='sceua', #cores=20,
                            obj_dir="minimize",
                            dbname='mspot_glacier_test')

param_dict = best_summary['best_param']

output_MATILDA = matilda_simulation(df_era, obs=obs,
                                    set_up_start='1997-01-01', set_up_end='1999-12-31',
                                    sim_start='2000-01-01', sim_end='2020-12-31',
                                    freq="D", lat=42.33,
                                    area_cat=295.763, area_glac=32.51,
                                    ele_dat=3273, ele_cat=3295, ele_glac=4068,
                                    glacier_profile=glacier_profile, elev_rescaling=True,

                                    warn=False, plots=False,
                                    parameter_set=param_dict)


## Run SPOTPY - glacier only:

best_summary = mspot.psample(df=df_era, obs=obs, rep=10, #output=output_path,
                             set_up_start='1997-01-01', set_up_end='1999-12-31',
                             sim_start='2000-01-01', sim_end='2020-12-31', freq="D",
                             glacier_profile=glacier_profile, area_cat=karab_area, lat=42.33,
                             elev_rescaling=True, ele_dat=3273,
                             ele_cat=None, area_glac=karab_area, ele_glac=4068, #pfilter = 0.2,



                            parallel=False, dbformat='csv', algorithm='sceua', #cores=20,
                            dbname='mspot_glacier_test')

param_dict = best_summary['best_param']

## Load external mspot results
results = mspot.analyze_results(output_path + '/matilda_par_smpl_mswx_cat_agg_2000_2020_1000rep.csv',
                      output_path + '/matilda_par_smpl_mswx_cat_agg_2000_2020_1000rep_observations.csv')

results['best_run_plot'].show()

sampling_csv = output_path + '/matilda_par_smpl_test2.csv'
param_dict = mspot.load_parameters(sampling_csv)      # From csv
# param_dict = best_summary['best_param']             # From result dict


## With glacier change

df_scen1 = cmip2df(cmipT, cmipP, 'ssp1', 'mean')
out_scen = '/home/phillip/Seafile/Alex_Phillip/Scenarios/'

for i in cmipT.keys():
    df_scen = cmip2df(cmipT, cmipP, i, 'mean')
    output_MATILDA = matilda_simulation(df_scen, output=out_scen, set_up_start='2018-01-01 00:00:00', set_up_end='2020-12-31 23:00:00',
                                      sim_start='2021-01-01 00:00:00', sim_end='2100-12-31 23:00:00', freq="D",
                                      area_cat=315.694, area_glac=32.51, lat=42.33, glacier_profile=glacier_profile,
                                      ele_dat=2550, ele_glac=4074, ele_cat=3225, lr_temp=-0.0059, lr_prec=-0.0002,
                                      TT_snow=0.354, TT_diff=0.228, CFMAX_snow=4, CFMAX_rel=2,
                                      BETA=2.03, CET=0.0471, FC=462.5, K0=0.03467, K1=0.0544, K2=0.1277,
                                      LP=0.4917, MAXBAS=2.494, PERC=1.723, UZL=413.0, PCORR=1.19, SFCF=0.874, CWH=0.011765,
                                      AG=0.7, RHO_snow=500)



output_MATILDA[6].show()


## Validation

# Adapt when parametrization is set up:

# dmod_score(bc_check['sdm'], bc_check['y_predict'], bc_check['x_predict'], ylabel="Temperature [C]")
