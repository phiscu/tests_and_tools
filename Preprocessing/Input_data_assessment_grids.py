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
wd = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data'

era_path = wd + '/input/kyzylsuu/met/era5l'
mswx_path = wd + '/input/kyzylsuu/met/mswx'
har_path = wd + '/input/kyzylsuu/met/harv2/all_variables_kyzylsuu_HARv2_daily_1980_2021.nc'
static_har_path = wd + '/input/static/all_static_kyzylsuu_HARv2.nc'
static_era_path = wd + '/input/static/ERA5_land_Z_geopotential.nc'
catchment_path = wd + '/GIS/Kysylsuu/Catchment_shapefile_new.shp'

## Preprocessing:

# MSWX:
mswx_p = salem.open_xr_dataset(mswx_path + '/P_MSWX_daily_kyzylsuu_19792022.nc')
mswx_t = salem.open_xr_dataset(mswx_path + '/Temp_MSWX_daily_kyzylsuu_19792022.nc')
mswx_t = mswx_t + 273.15
mswx_pev = salem.open_xr_dataset(mswx_path + '/PEV_fao56_MSWX_daily_kyzylsuu_1980-2022.nc')

# HARv2:
har_ds = salem.open_wrf_dataset(har_path, decode_coords=True)
har_ds_re = mswx_p.salem.transform(har_ds, interp='linear')    # Transform HAR dataset to MSWEP grid (xagg requires lat/lon coordinates)
har_ds[['et', 'potevap', 'prcp']] = har_ds[['et', 'potevap', 'prcp']] * 24      # mm h^⁻1 --> mm d^-1
har_ds_re[['et', 'potevap', 'prcp']] = har_ds_re[['et', 'potevap', 'prcp']] * 24      # mm h^⁻1 --> mm d^-1
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

# Load catchment outline
catchment = gpd.read_file(catchment_path)

# Static files:
static_era = salem.open_xr_dataset(static_era_path) / 9.80665
static_har = salem.open_xr_dataset(static_har_path)

static_mswx2 = salem.open_xr_dataset('/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/static/mswx_elevation_GMTED2010_nearest_kyzylsuu.tif')
static_mswx = salem.open_xr_dataset('/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/static/mswx_elevation_GMTED2010_average_kyzylsuu.tif')


## Transform:
static_har_re = mswx_p.salem.transform(static_har, interp='linear')
static_era_re = mswx_p.salem.transform(static_era, interp='linear')

har_t_re = har_ds_re.t2
har_p_re = har_ds_re.prcp

era_t_re = mswx_t.salem.transform(era_t, interp='linear')
era_p_re = mswx_p.salem.transform(era_p, interp='linear')

catchment_har = catchment.to_crs(har_ds.pyproj_srs)

# mswx_t_re = mswx_t.salem.transform(mswx_t, interp='linear')
# mswx_p_re = mswx_p.salem.transform(mswx_p, interp='linear')

## Plot:

fig, ax = plt.subplots(3, 2, figsize=(10, 14), dpi=300)

catchment.plot(ax=ax[0, 0], zorder=3, facecolor="none", edgecolor='white', lw=1)
mswx_p.precipitation.mean(dim='time').plot(ax=ax[0, 0], zorder=-1, vmin=0.8, vmax=5.5)
for i, j in np.ndindex(mswx_p.precipitation.mean(dim='time').shape):
    ax[0, 0].text(j+0.5, i+0.5, f'{mswx_p.precipitation.mean(dim="time").values[i,j]:.1f}', ha='center', va='center', color='white')
ax[0, 0].set_title('MSWX')
catchment_har.plot(ax=ax[1, 0], zorder=3, facecolor="none", edgecolor='white', lw=1)
har_p.mean(dim='time').plot(ax=ax[1, 0], zorder=-1, vmin=0.8, vmax=5.5)
ax[1, 0].set_title('HARv2')
catchment.plot(ax=ax[2, 0], zorder=3, facecolor="none", edgecolor='white', lw=1)
era_p.mean(dim='time').plot(ax=ax[2, 0], zorder=-1, vmin=0.8, vmax=5.5)
ax[2, 0].set_title('ERA5L')

catchment.plot(ax=ax[0, 1], zorder=3, facecolor="none", edgecolor='white', lw=1)
mswx_p.precipitation.mean(dim='time').plot(ax=ax[0, 1], zorder=-1, vmin=0.8, vmax=5.5)
ax[0, 1].set_title('MSWX')
catchment.plot(ax=ax[1, 1], zorder=3, facecolor="none", edgecolor='white', lw=1)
har_p_re.mean(dim='time').plot(ax=ax[1, 1], zorder=-1, vmin=0.8, vmax=5.5)
ax[1, 1].set_title('HARv2')
catchment.plot(ax=ax[2, 1], zorder=3, facecolor="none", edgecolor='white', lw=1)
era_p_re.mean(dim='time').plot(ax=ax[2, 1], zorder=-1, vmin=0.8, vmax=5.5)
ax[2, 1].set_title('ERA5L')

# fig.tight_layout()
plt.show()


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
era_p_cat, era_p_clip = weighted_avg(era_p_re, catchment, return_clip=True)
era_t_cat, era_t_clip = weighted_avg(era_t_re, catchment, return_clip=True)

    # Differences of regridding (40y prec sums):
    # orig      - 52009.760311
    # linear    - 51818.684568
    # nearest   - 46699.256862
    # spline    - 52258.182089

    # Differences of regridding (40y mean temp):
    # orig      - 268.658411
    # linear    - 268.676267
    # nearest   - 266.725814
    # spline    - 268.610407

    #> Nearest simply shifts the grid by 0.5° --> alters the catchment values a lot but might just fit with the shifted elevation?!
    #> Linear and spline change the cell values but have little influence on the catchment values
    # --> Linear is the simplest method with the least impact


#  Static:
static_era_cat, era_clip = weighted_avg(static_era_re, catchment, return_clip=True)
static_har_cat, har_clip = weighted_avg(static_har_re, catchment, return_clip=True)
static_mswx_cat, mswx_clip = weighted_avg(static_mswx, catchment, return_clip=True)

alt_era = round(static_era_cat.values[0][0])
alt_har = round(static_har_cat.values[0][0])
alt_mswx = round(static_mswx_cat.values[0][0])
print('Average altitude ERA5L: ' + str(alt_era) + 'm')
print('Average altitude HARv2: ' + str(alt_har) + 'm')
print('Average altitude MSWX: ' + str(alt_mswx) + 'm')

##

static_mswx = salem.open_xr_dataset('/home/phillip/Seafile/EBA-CA/Papers/' +
                                    'No1_Kysylsuu_Bash-Kaingdy/data/input/static/' +
                                    'mswx_elevation_worldclim2_30arcsec_aggr-from-global_kyzylsuu.tif')
static_mswx_cat, mswx_clip = weighted_avg(static_mswx, catchment, return_clip=True)

print('Average altitude MSWX: ' + str(round(static_mswx_cat.values[0][0])) + 'm')

static_mswx = mswx_p.salem.transform(static_mswx, interp='linear')
catchment.plot(zorder=3, facecolor="none", edgecolor='white', lw=1)
static_mswx.data.plot()
mswx_clip.data.plot()
plt.show()

# average, bilinear, cubic 3711
# spline 3633
# nearest 3548
# orig
# 30arcsec aggr from global - 3593
# 30arcsec aggr from subset - 3717
# original 30arsec resolution - 3291
# worldclim2 30arsec res    - 3295
# worldclim2 aggr from global - 3639

# --> the problem is the area-weighted aggregation!

##
# Differences of regridding (ERA5L):
    # orig      - 3341
    # linear    - 3325
    # nearest   - 3677
    # spline    - 3350



## Time periods

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



# plt.show()

## Dataframes for plotting

df_t = pd.concat([mswx_t_cat, har_t_cat, era_t_cat], axis=1, join='inner')
df_t.columns = ['mswx', 'har', 'era']
# df_t = df_t.resample('Y').mean()

df_p = pd.concat([mswx_p_cat, har_p_cat, era_p_cat], axis=1, join='inner')
df_p.columns = ['mswx', 'har', 'era']
# df_p = df_p.resample('Y').sum()



