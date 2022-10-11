## Run in conda-base env!
import socket
from pathlib import Path
import sys
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
sys.path.append(home + '/Ana-Lena_Phillip/data/tests_and_tools/Preprocessing')
from Preprocessing_functions_conda import weighted_avg
import salem
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

aws_lat = 42.191433; aws_lon = 78.200253
# start_date = '2007-01-01'; end_date = '2014-12-31'

## Paths:
era_path = home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Kysylsuu/t2m_tp_kysylsuu_ERA5L_1982_2020.nc'
msw_path = home + '/EBA-CA/Tianshan_data/GloH2O'
har_path = home + '/EBA-CA/Tianshan_data/HARv2/variables/all_variables_HARv2_daily_kyzylsuu_1980_2020.nc'
obs_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/hyd/obs/Kyzylsuu_1982_2021_latest.csv'
static_har_path = home + "/EBA-CA/Tianshan_data/HARv2/static/all_static_kyzylsuu_HARv2.nc"
static_era_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/static/ERA5_Tien-Shan_land_Z_geopotential.nc'
catchment_path = home + "/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/shapefile_hydro_kyzylsuu.shp"

## Preprocessing:

# MSWEP/MSWX:
mswep_p = salem.open_xr_dataset(msw_path +  '/MSWEP/MSWEP_daily_past_kyzylsuu_1979-2020.nc')
mswx_p = salem.open_xr_dataset(msw_path +  '/MSWX/MSWX_P_daily_past_kyzylsuu_1979_2022.nc')
mswx_t = salem.open_xr_dataset(msw_path +  '/MSWX/MSWX_Temp_daily_past_kyzylsuu_1979_2022.nc')

# HARv2:
har_ds = salem.open_wrf_dataset(har_path)
static_har = salem.open_xr_dataset(static_har_path)
har_ds = mswep_p.salem.transform(har_ds)    # Transform HAR dataset to MSWEP grid (xagg requires lat/lon coordinates)
har_ds[['et', 'potevap', 'prcp']] = har_ds[['et', 'potevap', 'prcp']] * 24      # mm h^⁻1 >> mm d^-1
har_p = har_ds.prcp
har_t = har_ds.t2

# ERA5L:
era_ds = salem.open_xr_dataset(era_path)
static_era = salem.open_xr_dataset(static_era_path)
era_p = era_ds.tp
era_t = era_ds.t2m

era_p = era_p.isel(time=era_p.time.dt.hour == 0) * 1000       # ERA5L saves cumulative values since 00h. Values at 00h represent the daily sum of the previous day. From m to mm.
era_p = era_p.shift(time=1, fill_value=0)               # Shift data to assign correct data to daily sums. Fill day 1 with 0 (need previous day!).
era_p = era_p.where(era_p >= 0, 0)
era_t = era_t.resample(time="D").mean(dim='time')                   # Daily temperature means. BOTTLENECK!!!

    # see: https://confluence.ecmwf.int/display/CUSF/Total+Precipitation+%5BValues+not+in+range%5D+-+ERA5-Land+hourly+data+from+1981+to+present

# Observations:
obs = pd.read_csv(obs_path)
# Load catchment outline
catchment = gpd.read_file(catchment_path)


## Catchment-wide aggregates:

# MSWEP:
mswep_p_cat, mswep_p_clip = weighted_avg(mswep_p, catchment, return_clip=True)

# MSWX:
mswx_p_cat, mswx_p_clip = weighted_avg(mswx_p, catchment, return_clip=True)
mswx_t_cat, mswx_t_clip = weighted_avg(mswx_t, catchment, return_clip=True)

# HARv2:
har_p_cat, har_p_clip = weighted_avg(har_p, catchment, return_clip=True)
har_t_cat, har_t_clip = weighted_avg(har_t, catchment, return_clip=True)

# ERA5L:
era_p_cat, era_p_clip = weighted_avg(era_p, catchment, return_clip=True)
era_t_cat, era_t_clip = weighted_avg(era_t, catchment, return_clip=True)


# mswx SIND DATASETS, HAR UND ERA SIND DATAARRAYS. MIT LETZTEREN FUNKTIONIERT DIE PLOTFUNKTION NICHT, WEIL SIE KEINE KEYS HABEN!!!



## Plot and compare:

df = pd.concat([mswep_p_cat,mswx_p_cat,har_p_cat,era_p_cat], axis=1, join='inner')
df.columns = ['mswep','mswx','har','era']
df = df.resample('Y').sum()

df.plot()
plt.legend()
plt.show()

# HAR UND ERA HABEN CA. DEN DOPPELTEN NIEDERSCHLAG. PREPROCESSING CHECKEN.