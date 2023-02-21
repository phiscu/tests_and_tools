## Run in conda-base env!
import socket
from pathlib import Path
import sys
import salem
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray

host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
sys.path.append(home + '/Ana-Lena_Phillip/data/tests_and_tools/Preprocessing')
from Preprocessing_functions_conda import weighted_avg

# start_date = '2007-01-01'; end_date = '2014-12-31'

## Paths:
era_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/era5l'
mswx_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/mswx'
har_path = home + '/EBA-CA/Tianshan_data/HARv2/variables/all_variables_HARv2_daily_kyzylsuu_1980_2020.nc'
obs_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu'
static_har_path = home + "/EBA-CA/Tianshan_data/HARv2/static/all_static_kyzylsuu_HARv2.nc"
static_era_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/static/ERA5_land_Z_geopotential.nc'
catchment_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/GIS/Kysylsuu/Catchment_shapefile_new.shp'

## Preprocessing:

# MSWX:
mswx_p = salem.open_xr_dataset(mswx_path + '/P_MSWX_daily_kyzylsuu_19792022.nc')
mswx_t = salem.open_xr_dataset(mswx_path + '/Temp_MSWX_daily_kyzylsuu_19792022.nc')
mswx_t = mswx_t + 273.15
mswx_pev = salem.open_xr_dataset(mswx_path + '/PEV_fao56_MSWX_daily_kyzylsuu_1980-2022.nc')

# HARv2:
har_ds = salem.open_wrf_dataset(har_path)
har_ds = mswx_p.salem.transform(har_ds)    # Transform HAR dataset to MSWEP grid (xagg requires lat/lon coordinates)
har_ds[['et', 'potevap', 'prcp']] = har_ds[['et', 'potevap', 'prcp']] * 24      # mm h^⁻1 --> mm d^-1
har_p = har_ds.prcp
har_t = har_ds.t2

# ERA5L:
era_ds = salem.open_xr_dataset(era_path + '/t2m_tp_pev_Kysylsuu_ERA5L_1982_2021.nc')
era_t = era_ds.t2m
era_p = era_ds.tp
era_pev = era_ds.pev

era_p = era_p.isel(time=era_p.time.dt.hour == 0) * 1000       # ERA5L saves cumulative values since 00h. Values at 00h represent the daily sum of the previous day. From m to mm.
era_p = era_p.shift(time=-1, fill_value=0)               # Data starts at 00h and ends at 23h --> Shift data to assign correct dates to daily sums. Fill last day with 0 (need additional day!).
era_p = era_p.where(era_p >= 0, 0)
era_pev = era_pev.isel(time=era_pev.time.dt.hour == 0) * 1000   # Same applies for PEV...
era_pev = era_pev.shift(time=-1, fill_value=0)
era_pev = era_pev.where(era_pev >= 0, 0)
era_t = era_t.resample(time="D").mean(dim='time')                   # Daily temperature means. BOTTLENECK!!!
    # see: https://confluence.ecmwf.int/display/CUSF/Total+Precipitation+%5BValues+not+in+range%5D+-+ERA5-Land+hourly+data+from+1981+to+present

# Observations:
obs_hyd = pd.read_csv(obs_path + '/hyd/obs/Kyzylsuu_1982_2020_latest.csv')
obs_met = pd.read_csv(obs_path + '/met/obs/met_data_full_kyzylsuu_2007-2015.csv', parse_dates=['time'], index_col='time')
obs_met2 = pd.read_csv(obs_path + '/met/obs/kyzylsuu_hydromet_aws_2019-2020.csv', parse_dates=['time'], index_col='time')

# Load catchment outline
catchment = gpd.read_file(catchment_path)

# Static files:
static_era = salem.open_xr_dataset(static_era_path)
static_har = salem.open_xr_dataset(static_har_path)
static_har = mswx_p.salem.transform(static_har)
static_era = static_era.salem.roi(shape=catchment, all_touched=True).z/9.80665  # from geopotential to m.a.s.l.
static_era = static_era[~np.isnan(static_era).all(axis=1), ~np.isnan(static_era).all(axis=0)]    # remove all NaN rows/cols
static_har = static_har.salem.roi(shape=catchment, all_touched=True).hgt

static_era_cat, era_clip = weighted_avg(static_era, catchment, return_clip=True)
static_har_cat, har_clip = weighted_avg(static_har, catchment, return_clip=True)
alt_era = round(static_era_cat.values[0][0])
alt_har = round(static_har_cat.values[0][0])
print('Average altitude ERA5L: ' + str(alt_era) + 'm')
print('Average altitude HARv2: ' + str(alt_har) + 'm')

## Time periods

# MSWEP -   1979 - 2021-12-30
# MSWX -    1979 - 2022-10-04
# ERA5L -   1982 - 2021
# HARv2 -   1980 - 2020

period = slice("1982", "2021")
data_sets = [mswx_p, mswx_t, era_p, era_t, har_p, har_t]
mswx_p, mswx_t, era_p, era_t, har_p, har_t = [i.sel(time=period) for i in data_sets]

## Compare values:

# fig, ax = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
# catchment.plot(ax=ax[0, 0], zorder=3, facecolor="none", edgecolor='white', lw=1)
# mswx_p.precipitation.mean(dim='time').plot(ax=ax[0, 0], zorder=-1, vmin=0.8, vmax=5.5)
# ax[0, 0].set_title('MSWX')
# catchment.plot(ax=ax[1, 0], zorder=3, facecolor="none", edgecolor='white', lw=1)
# har_p.mean(dim='time').plot(ax=ax[1, 0], zorder=-1, vmin=0.8, vmax=5.5)
# ax[1, 0].set_title('HARv2')
# catchment.plot(ax=ax[1, 1], zorder=3, facecolor="none", edgecolor='white', lw=1)
# mswx_p.salem.transform(era_p).mean(dim='time').plot(ax=ax[1, 1], zorder=-1, vmin=0.8, vmax=5.5)
# ax[1, 1].set_title('ERA5L')
# fig.tight_layout()
#
# plt.show()

## Catchment-wide aggregates:

# MSWX:
# mswx_p_cat, mswx_p_clip, fig_mswx_p = weighted_avg(mswx_p, catchment, return_clip=True, plot=True)
mswx_p_cat, mswx_p_clip = weighted_avg(mswx_p, catchment, return_clip=True)
mswx_t_cat, mswx_t_clip = weighted_avg(mswx_t, catchment, return_clip=True)

# HARv2:
# har_p_cat, har_p_clip, fig_har_p = weighted_avg(har_p, catchment, return_clip=True, plot=True)
har_p_cat, har_p_clip = weighted_avg(har_p, catchment, return_clip=True)
har_t_cat, har_t_clip = weighted_avg(har_t, catchment, return_clip=True)

# ERA5L:
# era_p_cat, era_p_clip, fig_era_p = weighted_avg(era_p, catchment, return_clip=True, plot=True)
era_p_cat, era_p_clip = weighted_avg(era_p, catchment, return_clip=True)
era_t_cat, era_t_clip = weighted_avg(era_t, catchment, return_clip=True)

# plt.show()

## Dataframes for plotting

df_t = pd.concat([mswx_t_cat, har_t_cat, era_t_cat], axis=1, join='inner')
df_t.columns = ['mswx', 'har', 'era']
# df_t = df_t.resample('Y').mean()

df_p = pd.concat([mswx_p_cat, har_p_cat, era_p_cat], axis=1, join='inner')
df_p.columns = ['mswx', 'har', 'era']
# df_p = df_p.resample('Y').sum()

## Write to files
# df_t.to_csv(home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/temp_cat_agg_era5l_harv2_mswx_1982-2020.csv')
# df_p.to_csv(home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/prec_cat_agg_era5l_harv2_mswx_1982-2020.csv')

## Stats

print(df_t.mean())
print('AWS: ' + str(obs_met.t2m.mean()))

print(df_p.sum())
print('AWS: ' + str(obs_met.tp.sum()))

# Number of days with no precipitation:
print('MSWX: ' + str(mswx_p_cat[mswx_p_cat!=0].isna().sum()))
print('HAR: ' + str(har_p_cat[har_p_cat!=0].isna().sum()))
print('ERA: ' + str(era_p_cat[era_p_cat!=0].isna().sum()))


## Compare AWS period:

period_t = slice(obs_met.index[0], obs_met.index[-1])
period_p = slice(obs_met.index[0], "2014-12-31")
df_p_obs = pd.concat([df_p[period_p], obs_met.tp[period_p]], axis=1)

df_t[period_t].resample('M').mean().plot()
obs_met.t2m.resample('M').mean().plot()
plt.legend()
plt.title('Temperature in observation period')
plt.show()

df_p_obs.resample('3M').sum().plot()
plt.legend()
plt.title('Precipitation in observation period')
plt.show()


## Focus on summer months:

# Catchment averages
summer_p = df_p_obs[df_p_obs.index.month.isin(range(4, 9))]
summer_p.sum()

# Compare closest grid cells to derive PCORR:
aws_lat = 42.191433; aws_lon = 78.200253
har_closest = har_p.sel(lon=aws_lon, lat=aws_lat, method='nearest').to_dataframe().prcp
era_closest = era_p.sel(longitude=aws_lon, latitude=aws_lat, method='nearest').to_dataframe().tp
mswx_closest = mswx_p.precipitation.sel(lon=aws_lon, lat=aws_lat, method='nearest').to_dataframe().precipitation

summer_p_closest = pd.concat([mswx_closest, har_closest, era_closest], axis=1)[period_p]
summer_p_closest = pd.concat([summer_p_closest, obs_met.tp[period_p]], axis=1)
summer_p_closest = summer_p_closest[summer_p_closest.index.month.isin(range(4, 10))]
summer_p_closest.columns = ['mswx', 'har', 'era', 'aws']

print(summer_p_closest.sum())

# Compare to grid cell with most similar elevation:
aws_alt = 2561

altitude_differences_har = np.abs(static_har - aws_alt)
closest_har_lat = float(static_har.where(altitude_differences_har == np.nanmin(altitude_differences_har), drop=True).lat)
closest_har_lon = float(static_har.where(altitude_differences_har == np.nanmin(altitude_differences_har), drop=True).lon)

altitude_differences_era = np.abs(static_era - aws_alt)
closest_era_lat = float(static_era.where(altitude_differences_era == np.nanmin(altitude_differences_era), drop=True).lat)
closest_era_lon = float(static_era.where(altitude_differences_era == np.nanmin(altitude_differences_era), drop=True).lon)

har_clos_elev = har_p.sel(lon=closest_har_lon, lat=closest_har_lat, method='nearest').to_dataframe().prcp
era_clos_elev = era_p.sel(longitude=closest_era_lon, latitude=closest_era_lat, method='nearest').to_dataframe().tp
mswx_clos_elev = mswx_p.precipitation.sel(lon=closest_era_lon, lat=closest_era_lat, method='nearest').to_dataframe().precipitation

summer_p_clos_elev = pd.concat([mswx_clos_elev, har_clos_elev, era_clos_elev], axis=1)[period_p]
summer_p_clos_elev = pd.concat([summer_p_clos_elev, obs_met.tp[period_p]], axis=1)
summer_p_clos_elev = summer_p_clos_elev[summer_p_clos_elev.index.month.isin(range(4, 10))]
summer_p_clos_elev.columns = ['mswx', 'har', 'era', 'aws']

print(summer_p_clos_elev.sum())

## Calculate individual PCORR values for all datasets
# Proximity:
summer_mon = summer_p_closest.resample('M').sum()
pcorrs = pd.concat([summer_mon.aws/summer_mon[i] for i in summer_mon.columns], axis=1)
pcorrs.columns = ['mswx', 'har', 'era', 'aws']
pcorrs.replace([np.inf, -np.inf], np.nan, inplace=True)
print(pcorrs.columns)
print([pcorrs[i].dropna().mean() for i in pcorrs.columns])
print([pcorrs[i].dropna().std() for i in pcorrs.columns])


# Elevation:
summer_mon = summer_p_clos_elev.resample('M').sum()
pcorrs = pd.concat([summer_mon.aws/summer_mon[i] for i in summer_mon.columns], axis=1)
pcorrs.columns = ['mswx', 'har', 'era', 'aws']
pcorrs.replace([np.inf, -np.inf], np.nan, inplace=True)
print(pcorrs.columns)
print([pcorrs[i].dropna().mean() for i in pcorrs.columns])
print([pcorrs[i].dropna().std() for i in pcorrs.columns])


# Results link the relation of a single grid cell and one weather station with the whole catchment...
# --> HAR wird gedrittelt, Era nur halbiert obwohl era im Gesamtcatchment viel mehr NS hat
# --> Try different PCORRS (average and closest) and take the best
# OR
# --> Just take proximity because HAR can't be increased even more!

## REMARKS:

# ADAPTING THE ERA5 GRID TO FIT MSWX REDUCES THE AVERAGE TEMPERATURE BY ~2k AND THE MEAN ALTITUDE BY 115m
# --> IS THE TRANSFORMATION OF HAR DATA CHANGING THE VALUES IN SIMILAR MANNER?

# ele_dat=2561 (AWS), ele_cat=3287.1672 (Merit)
#   --> Average altitude ERA5L: 3341 --> diff2AWS: 780 -> 4.68K, diff2catchment: 53.83
#   --> Average altitude HARv2: 3256 --> diff2AWS: 695 -> 4.17K, diff2catchment: -31.17
# Mean temps:
# mswx    269.438239 + 4.68?? = 274.118
# har     270.280027 + 4.17   = 274.450
# era     268.641005 + 4.68   = 273.321
# aws     274.00

# Precipitation sums (1982-2020):
# mswx    21227.937761
# har     47363.199615
# era     50809.670629

# Precipitation sums (obs period):
# mswx    3902.743760
# har     9024.597155
# era     9454.323305
# aws      4330.10

# Number of days without precipitation (of 14245):
# mswx      2522 (17.7%)
# har       1637 (11.5%)
# era       578  (4.1%)

# Number of days without precipitation in obs period (of 2710):
# mswx      498 (18.4%)
# har       320 (11.8%)
# era       117  (4.3%)
# aws       1956 (72.2%)

# All datasets agree very well in temperature when applying a lapse rate of -0.006K/m
# MSWX prec agrees best with obs but shows even less although obs have undercatch and reference altitude is much higher
#   --> underestimate likely --> undercatch correction??
# HAR and ERA have twice the precipitation or more, HAR less than ERA
# all datasets have far more precipitation events than obs, ERA5 is far off

# The spatial precipitation distribution is completely different in HARv2 vs ERA5/MSWEP/MSWX


## Load parameter from GloH2O
params = salem.open_xr_dataset(home + '/EBA-CA/Tianshan_data/GloH2O/HBV/HBV_params_gloh2o_kyzylsuu.nc')
params_cat = weighted_avg(params, catchment, return_clip=False)
params_cat.to_csv(home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/hyd/'
                         'HBV_params_gloh2o_kyzylsuu_cat_agg.csv')
