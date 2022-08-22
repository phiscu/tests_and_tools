# -*- coding: UTF-8 -*-

## import of necessary packages
import os
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import socket
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
from Preprocessing.Preprocessing_functions import dmod_score, load_cmip, cmip2df
from Test_area.SPOTPY import mspot
from matilda.core import matilda_simulation

## Setting file paths and parameters
    # Paths
wd = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data'
input_path = wd + "/input/kyzylsuu"
output_path = wd + "/output/kyzylsuu"

t2m_path = "/met/era5l/t2m_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
tp_path = "/met/era5l/tp_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
runoff_obs = "/hyd/obs/Kyzylsuu_1982_2021_latest.csv"
cmip_path = '/met/cmip6/'

    # Calibration period
t2m = pd.read_csv(input_path + t2m_path)
tp = pd.read_csv(input_path + tp_path)
df = pd.concat([t2m, tp.tp], axis=1)
df.rename(columns={'time': 'TIMESTAMP', 't2m': 'T2','tp':'RRR'}, inplace=True)
obs = pd.read_csv(input_path + runoff_obs)

    # Scenarios
cmipT = load_cmip(input_path + cmip_path, 't2m_CMIP6_all_models_adjusted_42.516-79.0167_1982-01-01-2100-12-31_')
cmipP = load_cmip(input_path + cmip_path, 'tp_CMIP6_all_models_adjusted_42.516-79.0167_1982-01-01-2100-12-31_')
glacier_profile = pd.read_csv(wd + "/kyzulsuu_glacier_profile.csv")


# Basic overview plot
obs_fig = obs.copy()
obs_fig.set_index('Date', inplace=True)
obs_fig.index = pd.to_datetime(obs_fig.index)
# obs_fig = obs_fig[slice('1984-10-01','1985-01-31')]
plt.figure()
ax = obs_fig.resample('M').sum().plot(label='Kyzylsuu (Hydromet)')
ax.set_ylabel('Discharge [m³/s]')

plt.show()

##> Data from 1982-01-01 to 1989-12-31 [8y], 1992-01-01 to 2007-12-31 [16y], 2010-01-01 to 2014-12-31 [5y], 2017-05-04 to 2021-07-30 [4y]

##
output_MATILDA = matilda_simulation(df, obs=obs, set_up_start='1982-01-01 00:00:00', set_up_end='1984-12-31 23:00:00', #output=output_path,
                                      sim_start='1982-01-01 00:00:00', sim_end='2020-12-31 23:00:00', freq="D", #glacier_profile=glacier_profile,
                                      area_cat=315.694, area_glac=32.51, lat=42.33, warn=True, plot_type="all", plots=False,
                                      ele_dat=2550, ele_glac=4074, ele_cat=3225,

                                      lr_temp=-0.0059, lr_prec=0, TT_snow=0.354, TT_diff=0.228, CFMAX_snow=4, CFMAX_rel=2,
                                      BETA=2.03, CET=0.0471, FC=462.5, K0=0.03467, K1=0.0544, K2=0.1277,
                                      LP=0.4917, MAXBAS=2.494, PERC=1.723, UZL=413.0, PCORR=1.19, SFCF=0.874, CWH=0.011765,
                                      AG=0.7, RHO_snow=500)

output_MATILDA[6].show()

print(output_MATILDA[2].Q_Total)
print(output_MATILDA[2].DDM_refreezing_snow)



## Run SPOTPY:

best_summary = mspot.psample(df=df, obs=obs, rep=5,# output= output_path,
                            set_up_start='1982-01-01 00:00:00', set_up_end='1984-12-31 23:00:00',
                            sim_start='1985-01-01 00:00:00', sim_end='2020-12-31 23:00:00', freq="D",# soi=[5, 10],
                            area_cat=315.694, area_glac=32.51, lat=42.33, parallel=False, algorithm='mcmc', cores=20,
                            dbname='matilda_par_smpl_test', dbformat='csv')



# results = spotpy.analyser.load_csv_results('matilda_par_smpl_test')
# sampling_csv = output_path + '/matilda_par_smpl_test.csv'
# sampling_obs = output_path + '/matilda_par_smpl_test_observations.csv'
#
# results = mspot.analyze_results(sampling_csv, sampling_obs)
# results['sampling_plot'].show()
# results['best_run_plot'].show()
# results['par_uncertain_plot'].show()

# setup = mspot.spot_setup(set_up_start='1982-01-01 00:00:00', set_up_end='1984-12-31 23:00:00', #output=output_path,
#                             sim_start='1985-01-01 00:00:00', sim_end='1987-12-31 23:00:00', freq="D",# soi=[5, 10],
#                             area_cat=315.694, area_glac=32.51, lat=42.33)
#
# psample_setup = setup(df, obs)  # Define objective function using obj_func=, otherwise KGE is used.
#
# sampler = spotpy.algorithms.rope(psample_setup, dbname="dbname", dbformat=None, parallel='mpi')
#
# sampler.sample(10)


# best_summary['par_uncertain_plot'].show()
# best_summary['best_run_plot'].show()
# best_summary['sampling_plot'].show()
#
# best_param = pd.DataFrame(best_summary['best_param'], index=[0]).transpose()
# best_param.to_csv(output_path + "best_param_85-89_sceau300.csv")
#
#
# output_MATILDA = MATILDA.MATILDA_simulation(df, obs=obs, set_up_start='1982-01-01 00:00:00', set_up_end='1984-12-31 23:00:00', #output=output_path,
#                                       sim_start='1985-01-01 00:00:00', sim_end='1987-12-31 23:00:00', freq="D",
#                                       area_cat=315.694, area_glac=32.51, lat=42.33, warn=True, # soi=[5, 10],
#                                       ele_dat=2550, ele_glac=4074, ele_cat=3225, parameter_df=best_param)

# Irgendwie wird der parameter_df zwar gelesen aber nicht akzeptiert. KGE sollte 0.8555 betragen. Oder wird der anders berechnet?
# Ergebnis von psample sollte direkt in MATILDA überführbar sein.


# Glacier profile mit rein?!

## Mit default parameters:

# output_MATILDA = MATILDA.MATILDA_simulation(df, obs=obs,  output=output_path, set_up_start='1982-01-01 00:00:00', set_up_end='1984-12-31 23:00:00',
#                                       sim_start='1985-01-01 00:00:00', sim_end='1989-12-31 23:00:00', freq="D",
#                                       area_cat=315.694, area_glac=32.51, lat=42.33,# soi=[5, 10],
#                                       ele_dat=2550, ele_glac=4074, ele_cat=3225)
#
# output_MATILDA[6].show()


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
#
# glac_area = output_MATILDA[4].iloc[:,:-1]
# glac_area = glac_area.set_index('time')
# glac_area.plot()
# plt.show()

## Validation

# Adapt when parametrization is set up:

# dmod_score(bc_check['sdm'], bc_check['y_predict'], bc_check['x_predict'], ylabel="Temperature [C]")



## Running MATILDA
# parameter = MATILDA.MATILDA_parameter(df, set_up_start='1987-01-01 00:00:00', set_up_end='1988-12-31 23:00:00',
#                                       sim_start='1992-01-01 00:00:00', sim_end='1995-07-30 23:00:00', freq="D",
#                                       area_cat=315.694, area_glac=32.51, lat=42.33,
#                                       ele_dat=2550, ele_glac=4074, ele_cat=3225, lr_temp=-0.005936, lr_prec=-0.0002503,
#                                       TT_snow=0.354, TT_rain=0.5815, CFMAX_snow=4.824, CFMAX_ice=5.574, CFR_snow=0.08765,
#                                       CFR_ice=0.01132, BETA=2.03, CET=0.0471, FC=462.5, K0=0.03467, K1=0.0544, K2=0.1277,
#                                       LP=0.4917, MAXBAS=2.494, PERC=1.723, UZL=413.0, PCORR=1.19, SFCF=0.874, CWH=0.011765)
#
# df_preproc, obs_preproc = MATILDA.MATILDA_preproc(df, parameter, obs=obs)
# #
# output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter, obs_preproc)
# #
# output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)
# output_MATILDA[6].show()


# MATILDA.MATILDA_save_output(output_MATILDA, parameter, output_path)
