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

# NetCDFs
    # MSWX:
mswx_p = salem.open_xr_dataset(mswx_path + '/P_MSWX_daily_kyzylsuu_19792022.nc')
mswx_pev = salem.open_xr_dataset(mswx_path + '/PEV_fao56_MSWX_daily_kyzylsuu_1980-2022.nc')
    #   HARv2:
har_ds = salem.open_wrf_dataset(har_path)
har_ds = mswx_p.salem.transform(har_ds)    # Transform HAR dataset to MSWEX grid (xagg requires lat/lon coordinates)
har_ds['prcp'] = har_ds['prcp'] * 24      # mm h^⁻1 --> mm d^-1
har_p = har_ds.prcp
    # ERA5L:
era_ds = salem.open_xr_dataset(era_path + '/t2m_tp_pev_Kysylsuu_ERA5L_1982_2021.nc')
era_p = era_ds.tp
era_p = era_p.isel(time=era_p.time.dt.hour == 0) * 1000       # ERA5L saves cumulative values since 00h. Values at 00h represent the daily sum of the previous day. From m to mm.
era_p = era_p.shift(time=-1, fill_value=0)               # Data starts at 00h and ends at 23h --> Shift data to assign correct dates to daily sums. Fill last day with 0 (need additional day!).
era_p = era_p.where(era_p >= 0, 0)

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
summer_mb = mb_ann[mb_ann.index.isin(range(2014, 2021))].SUMMER_BALANCE       # Only 2014 to 2021 have winter end dates
winter_mb = mb_ann[mb_ann.index.isin(range(2014, 2021))].WINTER_BALANCE
annual_mb = mb_ann[mb_ann.index.isin(range(2014, 2021))].ANNUAL_BALANCE


## Step 1 - Limit temperature lapse rate

lr_temp_lo = -0.0065
lr_temp_up = -0.0055

# Prepare input dataframes
df_t = t2m_agg.set_index('time')                    # Set timeindex
df_t.index = pd.to_datetime(df_t.index)
df_t = df_t - 273.15                                # to °C
ele = [780, 695, 780]           # Elevation differences data - glaciers: MSWX/ER5L - 780   HARv2 - 695

## 1.1 Check the influence of melt on the winter mass balance

# Upper lapse rate limit:
df_t_up_pdd = pd.DataFrame(np.column_stack([mspot_glacier.scaled_pdd(df_t[d], e, lr_temp_up) for d, e in zip(df_t.columns, ele)]),
             columns=[df_t.columns], index=df_t.index)

# Lower lapse rate limit:
df_t_lo_pdd = pd.DataFrame(np.column_stack([mspot_glacier.scaled_pdd(df_t[d], e, lr_temp_lo) for d, e in zip(df_t.columns, ele)]),
             columns=[df_t.columns], index=df_t.index)

# Sum up PDDs:
pdd_up = pd.DataFrame()
for i in mb_ann.index:
    pdd = df_t_up_pdd[mspot_glacier.winter(i, mb_ann)].sum()
    pdd_up = pd.concat([pdd_up, pdd], axis=1)
pdd_up.columns = mb_ann.index

pdd_lo = pd.DataFrame()
for i in mb_ann.index:
    pdd = df_t_lo_pdd[mspot_glacier.winter(i, mb_ann)].sum()
    pdd_lo = pd.concat([pdd_lo, pdd], axis=1)
pdd_lo.columns = mb_ann.index

print(pd.DataFrame({'pdd_lo': pdd_lo.mean(axis=1), '+/-': pdd_lo.std(axis=1)}))
print(pd.DataFrame({'pdd_up': pdd_up.mean(axis=1), '+/-': pdd_up.std(axis=1)}))

# Depending on the melt rates, melt has substantial influence on the winter balance: e.g. har_up has 57 pdd on average!

## 1.2 Check influence of accumulation on summer mass balance

df_t_up_ndd = pd.DataFrame(np.column_stack([mspot_glacier.scaled_ndd(df_t[d], e, lr_temp_up) for d, e in zip(df_t.columns, ele)]),
             columns=[df_t.columns], index=df_t.index)

df_t_lo_ndd = pd.DataFrame(np.column_stack([mspot_glacier.scaled_ndd(df_t[d], e, lr_temp_lo) for d, e in zip(df_t.columns, ele)]),
             columns=[df_t.columns], index=df_t.index)

ndd_up = pd.DataFrame()
for i in mb_ann.index:
    ndd = df_t_up_ndd[mspot_glacier.summer(i, mb_ann)].sum()
    days = len(df_t_up_ndd[mspot_glacier.winter(i, mb_ann)])
    ndd_up = pd.concat([ndd_up, ndd/days], axis=1)
ndd_up.columns = mb_ann.index

ndd_lo = pd.DataFrame()
for i in mb_ann.index:
    ndd = df_t_lo_ndd[mspot_glacier.winter(i, mb_ann)].sum()
    days = len(df_t_lo_ndd[mspot_glacier.winter(i, mb_ann)])
    ndd_lo = pd.concat([ndd_lo, ndd/days], axis=1)
ndd_lo.columns = mb_ann.index

print(pd.DataFrame({'ndd_lo (frac)': ndd_lo.mean(axis=1), '+/-': ndd_lo.std(axis=1)}))
print(pd.DataFrame({'ndd_up (frac)': ndd_up.mean(axis=1), '+/-': ndd_up.std(axis=1)}))

# --> Accumulation very sensitive to lr_temp. Lower bound results in >90% of freezing days.
# --> Better to fix the accumulation parameters and lapse rates first


## Step 2 - Limit the precipitation correction factor:

period_p = slice(obs_met.index[0], "2014-12-31")                        # Period of precipitation obs
# df_p = tp_agg.set_index('time')                                         # Set timeindex to preciptation data
# df_p.index = pd.to_datetime(df_p.index)
# df_p_obs = pd.concat([df_p[period_p], obs_met.tp[period_p]], axis=1)    # Obs and reananlysis in one df
# summer_p = df_p_obs[df_p_obs.index.month.isin(range(4, 9))]             # Select summer months only

# Compare closest grid cells to derive PCORR:
aws_lat = 42.191433; aws_lon = 78.200253
har_closest = har_p.sel(lon=aws_lon, lat=aws_lat, method='nearest').to_dataframe().prcp
era_closest = era_p.sel(longitude=aws_lon, latitude=aws_lat, method='nearest').to_dataframe().tp
mswx_closest = mswx_p.precipitation.sel(lon=aws_lon, lat=aws_lat, method='nearest').to_dataframe().precipitation
# Select summer months (Apr-Sept)
summer_p_closest = pd.concat([mswx_closest, har_closest, era_closest], axis=1)[period_p]
summer_p_closest = pd.concat([summer_p_closest, obs_met.tp[period_p]], axis=1)
summer_p_closest = summer_p_closest[summer_p_closest.index.month.isin(range(4, 10))]
summer_p_closest.columns = ['mswx', 'har', 'era', 'aws']

# Calculate factor of monthly renanalysis data to obs
summer_mon = summer_p_closest.resample('M').sum()
pcorrs = pd.concat([summer_mon.aws/summer_mon[i] for i in summer_mon.columns], axis=1)
pcorrs.columns = ['mswx', 'har', 'era', 'aws']
pcorrs.replace([np.inf, -np.inf], np.nan, inplace=True)

# Write correction factors and standard deviations to variables
mswx_pcorr, har_pcorr, era_pcorr = [round(pcorrs[i].dropna().mean(), 2) for i in pcorrs.columns[0:-1]]
mswx_pcorr_sigma, har_pcorr_sigma, era_pcorr_sigma = [round(pcorrs[i].dropna().std(), 2) for i in pcorrs.columns[0:-1]]

# Use the SD as range:
    # MSWX
PCORR_lo_mswx = mswx_pcorr - mswx_pcorr_sigma; PCORR_up_mswx = mswx_pcorr + mswx_pcorr_sigma
    # HARv2
PCORR_lo_har = har_pcorr - har_pcorr_sigma; PCORR_up_har = har_pcorr + har_pcorr_sigma
    # ERA5L
PCORR_lo_era = era_pcorr - era_pcorr_sigma; PCORR_up_era = era_pcorr + era_pcorr_sigma

for i in ["PCORR_mswx", "PCORR_har", "PCORR_era"]:
    print(i + ": [" +
          str(round(globals()[i.split("_")[0] + "_lo_" + i.split("_")[1]], 2)) + ", " +
          str(round(globals()[i.split("_")[0] + "_up_" + i.split("_")[1]], 2))
          + "]")

## Step 3 - Calibrate SFCF, TT_snow, TT_diff, RFS, and lr_prec on Karabatkak winter mass balance:

## 3.3 Run SPOT with glacier routine only

glacier_profile_karab = pd.read_csv(wd + '/glacier_profile_karabatkak_farinotti_marie.csv') # inital glacier profile of Karabatkak
karab_area = 2.046              # Karabatkak area according to RGI 6
mb_ann_matilda = mb_ann.reset_index()

# Pass parameter bounds from step 1 and 2 as dict:
# lim_dict = {'lr_temp_lo': lr_temp_lo, 'lr_temp_up': lr_temp_up, 'PCORR_lo': PCORR_lo_mswx, 'PCORR_up': PCORR_up_mswx}
#
# best_summary = mspot_glacier.psample(df=df_mswx, obs=mb_obs, rep=6, output=output_path + '/glacier_only',
#                                      set_up_start='1997-01-01', set_up_end='1999-12-31',
#                                      sim_start='2000-01-01', sim_end='2020-12-31', freq="D",
#                                      glacier_profile=glacier_profile_karab, area_cat=karab_area, lat=42.33,
#                                      elev_rescaling=True, ele_dat=3341,
#                                      ele_cat=None, area_glac=karab_area, ele_glac=3830,
#                                      glacier_only=True, obs_type="winter",
#                                      obj_dir="minimize",
#
#                                      **lim_dict,
#
#                                      CFMAX_ice_up=4,
#
#
#                                      parallel=False, dbformat='csv', algorithm='sceua', #cores=20,
#                                      dbname='mswx_glacier_only_step3_cfmaxice4_sceua_2000')

## Parameter sets - step 3:

# MSWX
# Step 3:
# param_dict = {"lr_temp": -0.00604341, "lr_prec": 0.00199707, "PCORR": 1.45612, "TT_snow": 0.880102, "TT_diff": 0.571304, "CFMAX_ice": 1.82167, "CFMAX_rel": 1.20094, "SFCF": 0.971164, "RFS": 0.050841}
# --> annual MB -0.01 m
# --> winter MAE: 55.6 mm

# --> lr_temp_lo_up, PCORR + SD, CFMAX_ice = 6, SCEUA, 10k

## Step 4 - Calibrate melt rates on Karabatkak annual (or summer) balance

## Pass results as fix parameters or pass bounds for next step:

# param3 = mspot_glacier.dict2bounds(param_dict, drop=['CFMAX_ice', 'CFMAX_rel', 'RFS'])
# OR
# lim_dict = mspot_glacier.get_par_bounds(output_path + '/glacier_only' + '/mswx_glacier_only_step3_PCORRSD_cfmax6_sceua_10000',
#                                  threshold=5, percentage=True, drop=['CFMAX_ice', 'CFMAX_rel', 'RFS'])
## Run mspot with parameter limits from step 3

# best_summary = mspot_glacier.psample(df=df_mswx, obs=mb_ann_matilda, rep=2000, output=output_path + '/glacier_only',
#                                      set_up_start='1997-01-01', set_up_end='1999-12-31',
#                                      sim_start='2000-01-01', sim_end='2020-12-31', freq="D",
#                                      glacier_profile=glacier_profile_karab, area_cat=karab_area, lat=42.33,
#                                      elev_rescaling=True, ele_dat=3341,
#                                      ele_cat=None, area_glac=karab_area, ele_glac=3830,
#                                      glacier_only=True, obs_type="annual",
#                                      obj_dir="minimize",
#
#                                      **lim_dict,    # param3
#
#                                      parallel=False, dbformat='csv', algorithm='sceua', #cores=20,
#                                      dbname='era_glacier_only_step3_cfmaxice4_sceua_2000')

## Parameters sets - step 4:
# Fixed parameters:
# Best 5% as bounds - BEST CHOICE!:
# param_dict = {"lr_temp": -0.00615121, "lr_prec": 0.00198416, "PCORR": 1.46401, "TT_snow": 0.901088, "TT_diff": 0.570674, "CFMAX_ice": 5.25168, "CFMAX_rel": 1.7523, "SFCF": 0.969817, "RFS": 0.0832019}

# --> annual MB:  -0.85 m
# --> winter MAE: 108 mm
# --> summer MAE: 199.8 mm
# --> annual MAE: 286.3 mm


# param_dict = {"lr_temp": -0.00604341, "lr_prec": 0.00199707, "PCORR": 1.45612, "TT_snow": 0.880102, "TT_diff": 0.571304, "CFMAX_ice": 5.2711, "CFMAX_rel": 1.77628, "SFCF": 0.971164, "RFS": 0.147832}
# --> annual MB -0.85 m
# --> winter MAE: 109.6 mm
# --> summer MAE: 203.7 mm
# --> annual MAE: 288.9 mm

# Best 1% as bounds:
# param_dict = {"lr_temp": -0.00611364, "lr_prec": 0.00199376, "PCORR": 1.4562, "TT_snow": 0.905665, "TT_diff": 0.56885, "CFMAX_ice": 5.16905, "CFMAX_rel": 1.70474, "SFCF": 0.969577, "RFS": 0.108975}
# --> annual MB:  -0.86 m
# --> winter MAE: 108.8 mm
# --> summer MAE: 201.7 mm
# --> annual MAE: 289.8 mm

# Best 5% calibrated on annual MB:
# param_dict = {"lr_temp": -0.00611322, "lr_prec": 0.00198087, "PCORR": 1.46418, "TT_snow": 0.834553, "TT_diff": 0.525529, "CFMAX_ice": 5.85729, "CFMAX_rel": 2.42333, "SFCF": 0.971802, "RFS": 0.209699}
# --> annual MB:  -0.68 m
# --> winter MAE: 115.8 mm
# --> summer MAE: 262.7 mm
# --> annual MAE: 209.8 mm
# --> POSITIVE MB FOR CATCHMENT MEAN! - DISSMISS


## Step 5 - Calibrate remaining parameters on runoff:

##
# param4 = mspot_glacier.dict2bounds(param_dict)
# best_summary = mspot_glacier.psample(df=df_mswx, obs=obs, rep=10, #output=output_path + '/glacier_only',
#                                      set_up_start='1997-01-01', set_up_end='1999-12-31',
#                                      sim_start='2000-01-01', sim_end='2020-12-31', freq="D",
#                                      area_cat=295.763, area_glac=32.51, lat=42.33,
#                                      ele_dat=3341, ele_cat=3295, ele_glac=4068,
#                                      glacier_profile=glacier_profile, elev_rescaling=True,
#                                      glacier_only=False,
#                                      obj_dir="maximize",
#
#                                      **param4,
#
#                                      parallel=False, dbformat='csv', algorithm='demcz', #cores=20,
#                                      dbname='mswx_glacier_only_step5_best5_summer_demcz')

## Parameters sets - step 5:
# mswx_step5_best5-summer_2000-2017_demcz_10000:

param_dict = {"lr_temp": -0.00615, "lr_prec": 0.00198, "BETA": 1, "CET": 0.000176, "FC": 50.5, "K0": 0.20239,
              "K1": 0.0101, "K2": 0.15, "LP": 1, "MAXBAS": 2.58113, "PERC": 0.00256, "UZL": 472.01, "PCORR": 1.46,
              "TT_snow": 0.901, "TT_diff": 0.571, "CFMAX_ice": 5.25, "CFMAX_rel": 1.75, "SFCF": 0.97, "CWH": 0.109155,
              "AG": 0.671963, "RFS": 0.0832}

# --> Mean Annual MB (2000-2017): -0.24 m w.e.
# KGE coefficient: 0.84 (monthly)
# NSE coefficient: 0.77
# RMSE: 23.9


## Check results:
output_MATILDA = matilda_simulation(df_mswx, obs=obs, set_up_start='1997-01-01', set_up_end='1999-12-31', # output='/home/phillip/Seafile/Ana-Lena_Phillip/data/test',
                                    sim_start='2000-01-01', sim_end='2017-12-31', freq="M", glacier_profile=glacier_profile,
                                    area_cat=295.763, lat=42.33, warn=False, plot_type="all", plots=True, elev_rescaling=True,
                                    ele_dat=3341, ele_cat=3295, area_glac=32.51, ele_glac=4068, pfilter=0,
                                    parameter_set=param_dict)

print('Mean Annual MB (2000-2017): ' + str(round(output_MATILDA[5].smb_water_year.mean() / 1000, 2)) + ' m w.e.')

# output_MATILDA[9].show()


## Validation:

output_MATILDA = matilda_simulation(df_mswx, obs=obs, set_up_start='2015-01-01', set_up_end='2017-12-31', # output='/home/phillip/Seafile/Ana-Lena_Phillip/data/test',
                                    sim_start='2018-01-01', sim_end='2019-12-31', freq="M", glacier_profile=glacier_profile,
                                    area_cat=295.763, lat=42.33, warn=False, plot_type="all", plots=True, elev_rescaling=True,
                                    ele_dat=3341, ele_cat=3295, area_glac=32.51, ele_glac=4068, pfilter=0,
                                    parameter_set=param_dict)

print('Mean Annual MB: ' + str(round(output_MATILDA[5].smb_water_year.mean() / 1000, 2)) + ' m w.e.')

##
# # param_dict = best_summary['best_param']
#
DATAFRAME = df_mswx
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
obs = []
evaluation = mb_ann
simulation = output_DDM.DDM_smb
for i in evaluation.index:
    if SEASON is 'winter':
        mb = simulation[mspot_glacier.winter(i, evaluation)].sum()
        mb_obs = evaluation[evaluation.index == i].WINTER_BALANCE.squeeze()
    elif SEASON is 'summer':
        mb = simulation[mspot_glacier.summer(i, evaluation)].sum()
        mb_obs = evaluation[evaluation.index == i].SUMMER_BALANCE.squeeze()
    else:
        mb = simulation[mspot_glacier.annual(i, evaluation)].sum()
        mb_obs = evaluation[evaluation.index == i].ANNUAL_BALANCE.squeeze()
    sim.append(mb)

    obs.append(mb_obs)
sim_new = pd.DataFrame()
sim_new['mod'] = pd.DataFrame(sim)
sim_new['obs'] = pd.DataFrame(obs)
clean = sim_new.dropna()
simulation_clean = clean['mod']
evaluation_clean = clean['obs']

print(spotpy.objectivefunctions.mae(evaluation_clean, simulation_clean))
scores = spotpy.objectivefunctions.calculate_all_functions(evaluation_clean, simulation_clean)
print(pd.DataFrame(scores))
print(clean)
print('Mean Annual MB (2000-2018): ' + str(round(glacier_change[glacier_change.time.isin(evaluation.index)].
                                                 smb_water_year.mean() / 1000, 2)) + ' m w.e.')


## Remarks:

# Positive MBcum are excluded by setting the SMB to 9999. Always run mspot on full dataset (e.g. 2000-2020) even if
# obs are only available for shorter periods. Final dataset might cause positive MBcum in other years!

# ground measurements disagree with remote sensing results: mean annual MB -0.7 mwea not in range of -0.185 +- 0.139 mwea
# --> remote sensing more exact because not point based? See final validation.

# Validation:
#   - Runoff in validation period (2018-2020)
#   - 18y MB 2000-2018
#   - Area changes

# Calibration:
#   - winter MB 2014-2020
#   - summer MB 2014-2020
#   - Runoff 2000-2017

# Other quality criteria:
#   - Timing of peak runoff
#   - start of melting season (runoff onset in spring)

## Filter results

# likes = results['like1']
# minimum = np.nanmin(likes)
# index = np.where(likes == minimum)
#
# best_param = trues[index]
# best_param_values = spotpy.analyser.get_parameters(trues[index])[0]
# par_names = spotpy.analyser.get_parameternames(trues)
# param_zip = zip(par_names, best_param_values)
# best_param = dict(param_zip)


## Resample Matilda to filter for mass balance values

dir = 'mswx_step5_best5-summer_demcz_6000_2022-12-13_13-44-03'

db_name = dir.split('/')[-1]
db_name_short = db_name[:len(db_name)-20]
sampling_file = dir + '/' + db_name_short
output_file = dir + '/' + db_name_short + '_kge_mb_matrix.csv'

percentage = 1
maximize = True

output_file = sampling_file + '_kge_mb_matrix.csv'
results = spotpy.analyser.load_csv_results(sampling_file)
best = spotpy.analyser.get_posterior(results, maximize=maximize, percentage=percentage)  # get best xx% runs
par_names = spotpy.analyser.get_parameter_fields(best)

par = []
mean_mb = []
kge = []
for i in range(0, len(best)):
    params = spotpy.analyser.get_parameters(best)[i]
    param_dict = dict(zip([i.split('par')[1] for i in par_names], params))
    par.append(param_dict)
    with mspot_glacier.HiddenPrints():
        output_MATILDA = matilda_simulation(df_mswx, obs=obs, set_up_start='1997-01-01', set_up_end='1999-12-31',
                    sim_start='2000-01-01', sim_end='2017-12-31', freq="M", glacier_profile=glacier_profile,
                    area_cat=295.763, lat=42.33, warn=False, plot_type="all", plots=True, elev_rescaling=True,
                    ele_dat=3273, ele_cat=3295, area_glac=32.51, ele_glac=4068, pfilter=0,
                    parameter_set=param_dict)
    mean_mb.append(round(output_MATILDA[5].smb_water_year.mean() / 1000, 6))
    kge.append(round(output_MATILDA[2], 4))

kge_mb = pd.DataFrame({'kge': kge, 'mb': mean_mb, 'par': [str(i) for i in par]})
kge_mb = kge_mb.sort_values('mb', ascending=False)
kge_mb.to_csv(output_file)


