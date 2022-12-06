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
# sys.path.append(home + '/Ana-Lena_Phillip/data/matilda/MATILDA')
sys.path.append(home + '/Ana-Lena_Phillip/data/tests_and_tools')
from Test_area.SPOTPY import mspot


## Settings

rep = 10
alg = 'sceua'
dbname='MATILDA_har_1000'

output_path = '/data/projects/ebaca/Ana-Lena_Phillip/data/matilda/Cirrus/hotfix/MATILDA_har_1000_2022-11-21_17-29-50'
set_up_start='1997' + '-01-01 00:00:00'; set_up_end='1999' + '-12-31 23:00:00'
sim_start='2000' + '-01-01 00:00:00'; sim_end='2020' + '-12-31 23:00:00' 
freq="D"
dbformat='csv'
save_sim = True


## Setting file paths and parameters
    # Paths
wd = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data'
input_path = wd + "/input/kyzylsuu"


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

glacier_profile = pd.read_csv(wd + "/kyzulsuu_glacier_profile.csv")


## Run SPOTPY:

best_summary = mspot.psample(df=df_har, obs=obs, rep=rep, #output=output_path,
                            set_up_start=set_up_start, set_up_end=set_up_end,
                            sim_start=sim_start, sim_end=sim_end, freq=freq,
                            area_cat=295.763, area_glac=32.51, lat=42.33,
                            glacier_profile = glacier_profile, elev_rescaling = True,
                            ele_dat = 3172, ele_cat = 3295, ele_glac = 4068,
                            parallel=False, algorithm=alg,
                            dbname=dbname, dbformat=dbformat, save_sim=save_sim
                            
                            , CFMAX_snow_up=3, CFMAX_rel_up=2
				)




