# -*- coding: UTF-8 -*-

## import of necessary packages
import pandas as pd
from pathlib import Path
import sys
import socket

host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
sys.path.append(home + '/Ana-Lena_Phillip/data/tests_and_tools')
from Test_area.SPOTPY import mspot

# Setting file paths and parameters
    # Paths
wd = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data'
input_path = wd + "/input/kyzylsuu"
output_path = wd + "/output/kyzylsuu"

t2m_path = "/met/era5l/t2m_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
tp_path = "/met/era5l/tp_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
runoff_obs = "/hyd/obs/Kyzylsuu_1982_2021_latest.csv"

    # Calibration period
t2m = pd.read_csv(input_path + t2m_path)
tp = pd.read_csv(input_path + tp_path)
df = pd.concat([t2m, tp.tp], axis=1)
df.rename(columns={'time': 'TIMESTAMP', 't2m': 'T2','tp':'RRR'}, inplace=True)
obs = pd.read_csv(input_path + runoff_obs)

glacier_profile = pd.read_csv(wd + "/kyzulsuu_glacier_profile.csv")

## Run SPOTPY:

best_summary = mspot.psample(df=df, obs=obs, rep=1000, output=output_path,
                            set_up_start='1997-01-01 00:00:00', set_up_end='1999-12-31 23:00:00',
                            sim_start='2000-01-01 00:00:00', sim_end='2020-12-31 23:00:00', freq="D",
                            glacier_profile = glacier_profile, area_cat = 295.763, lat = 42.33,
                            elev_rescaling = True, ele_dat = 2550,
                            ele_cat = 3295, area_glac = 32.51, ele_glac = 4068, #pfilter = 0.2,

                            CFMAX_snow_up=3, CFMAX_rel_up=2,

                            parallel=False, dbformat='csv', algorithm='sceua', #cores=20,
                            dbname='matilda_par_smpl_test_update')