##
import sys
import os
import xarray as xr
import pandas as pd
from shapely.geometry import Point, mapping
import matplotlib.pyplot as plt
import rioxarray as rxr
import geopandas as gpd
import numpy as np
import salem
import cartopy.crs as ccrs
from pathlib import Path; home = str(Path.home())
from MATILDA_slim import MATILDA
from bias_correction import BiasCorrection



lat = 42.19; lon = 78.2
start_date = '2007-01-01'; end_date = '2014-12-31'

## Paths
shdf = salem.read_shapefile(home+ "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/shapefile_hydro_kyzylsuu.shp")
har_ds = salem.open_wrf_dataset(home +"/Seafile/EBA-CA/Tianshan_data/HARv2/variables/all_variables_HARv2_daily_kyzylsuu_1980_2020.nc")
wd = home + '/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu'
t2m_bc_path = wd + "/met/era5l/t2m_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
tp_bc_path = wd + "/met/era5l/tp_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
t2m_path = wd + "/met/era5l/t2m_era5l_42.516-79.0167_1982-01-01-2020-12-31.csv"
tp_path = wd + "/met/era5l/tp_era5l_42.516-79.0167_1982-01-01-2020-12-31.csv"
obs_path = wd + "/hyd/obs/Kyzylsuu_1982_2021_latest.csv"
static_path = home + "/Seafile/EBA-CA/Tianshan_data/HARv2/static/all_static_kyzylsuu_HARv2.nc"

crop_extent = shdf.to_crs(har_ds.pyproj_srs)
crop_extent.crs = har_ds.pyproj_srs

## Read input files
obs = pd.read_csv(obs_path)
static_har = xr.open_dataset(static_path)

t2m = pd.read_csv(t2m_path, index_col='time', parse_dates=['time'])
tp = pd.read_csv(tp_path, index_col='time', parse_dates=['time'])
era = pd.concat([t2m, tp.tp], axis=1)
era = era["2007-01-01T00:00:00":"2014-12-31T00:00:00"]

t2m_bc = pd.read_csv(t2m_bc_path, index_col='time', parse_dates=['time'])
tp_bc = pd.read_csv(tp_bc_path, index_col='time', parse_dates=['time'])
era_bc = pd.concat([t2m_bc, tp_bc], axis=1)
era_bc = era_bc["2007-01-01T00:00:00":"2014-12-31T00:00:00"]

har_ds = xr.open_dataset(home + "/Seafile/EBA-CA/Tianshan_data/HARv2/variables/all_variables_HARv2_daily_kyzylsuu_1980_2020.nc")

# Select grid cell based on elevation:
aws_alt = 2561
catch_alt = 3225
altitude_differences_gp = np.abs(static_har.hgt - catch_alt)
closest_lat = float(static_har.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lat)
closest_lon = float(static_har.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lon)

# ...or based on proximity to the AWS:
aws_lat = 42.191433; aws_lon = 78.200253

# Transform coordinates to CRS of the data (Lambert Conformal, arguments retrieved from NCDF-Metadata)
data_crs = ccrs.LambertConformal(central_longitude=83, central_latitude=31.99998856, standard_parallels=(32, 38))
x, y = data_crs.transform_point(closest_lon, closest_lat, src_crs=ccrs.PlateCarree())


pick = har_ds.sel(south_north=y, west_east=x, method='nearest')
har = pick.to_dataframe().filter(['t2', 'prcp'])
har.rename(columns={'t2':'t2m_har','prcp':'tp_har'}, inplace=True)
har.tp_har = har.tp_har * 24            # Precipitation is in mm h^⁻1
har = har["2007-01-01T00:00:00":"2014-12-31T00:00:00"]


## Compare
print(har.describe())
print(har.sum())
print(era.describe())
print(era.sum())

era.t2m.plot()
har.t2m_har.plot()
plt.legend()
plt.show()

har.tp_har.resample("M").sum().plot()
era.tp.resample("M").sum().plot()
plt.legend()
plt.show()


## Bias adjustment

aws_temp_int = pd.read_csv(wd + '/met/obs/t2m-with-gap_aws_2007-08-10-2016-01-01.csv', parse_dates=["time"], index_col="time")
aws_tp = pd.read_csv(wd + '/met/obs/tp_aws_2007-08-01-2016-01-01.csv', parse_dates=["time"], index_col="time")

har_temp_int1 = har[slice('2007-08-10', '2011-10-11')].t2m_har
har_temp_int2 = har[slice('2011-11-01', '2014-12-31')].t2m_har
har_temp_int = pd.concat([har_temp_int1, har_temp_int2], axis=0)  # Data gap of 18 days in October 2011

train_slice = slice('2007-08-10', '2014-12-31')
predict_slice = slice('2007-01-01', '2014-12-31')

# Temperature:

x_train = har_temp_int[train_slice].squeeze()
y_train = aws_temp_int[train_slice].squeeze()
x_predict = har.t2m_har[predict_slice].squeeze()
bc_har = BiasCorrection(y_train, x_train, x_predict)
har_corrT = pd.DataFrame(bc_har.correct(method='normal_mapping'))
# har_corrT.to_csv(har_path + 't2m_har5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv')


# Precipitation:

x_train = har.tp_har[train_slice].squeeze()
y_train = aws_tp[train_slice].squeeze()
x_predict = har.tp_har[predict_slice].squeeze()
bc_har = BiasCorrection(y_train, x_train, x_predict)
har_corrP = pd.DataFrame(bc_har.correct(method='normal_mapping'))
har_corrP[har_corrP < 0] = 0  # only needed when using normal mapping for precipitation
# har_corrP.to_csv(har_path + 'tp_har5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv')

## Matilda
era = era_bc.reset_index()
har = pd.concat([har_corrT, har_corrP], axis=1)
har = har.reset_index()
era.rename(columns={'time': 'TIMESTAMP', 't2m': 'T2', 'tp': 'RRR'}, inplace=True)
har.rename(columns={'time': 'TIMESTAMP', 't2m_har': 'T2', 'tp_har': 'RRR'}, inplace=True)
# elev_har = static_har.sel(south_north=y, west_east=x, method='nearest').hgt.values[0]


# ERA5
output_MATILDA_era = MATILDA.MATILDA_simulation(era, obs=obs, set_up_start='2007-01-01 00:00:00', set_up_end='2009-12-31 23:00:00', #output=output_path,
                                      sim_start='2010-01-01 00:00:00', sim_end='2014-12-31 23:00:00', freq="D",
                                      area_cat=315.694, area_glac=32.51, lat=42.33, warn=True, # soi=[5, 10],
                                      ele_dat=2550, ele_glac=4074, ele_cat=3225, lr_temp=-0.0059, lr_prec=0,
                                      TT_snow=0.354, TT_rain=0.5815, CFMAX_snow=4, CFMAX_ice=6,
                                      BETA=2.03, CET=0.0471, FC=462.5, K0=0.03467, K1=0.0544, K2=0.1277,
                                      LP=0.4917, MAXBAS=2.494, PERC=1.723, UZL=413.0, PCORR=1.19, SFCF=0.874, CWH=0.011765,
                                      AG=1, RHO_snow=300)

output_MATILDA_era[6].show()
print(output_MATILDA_era[2].Q_Total)

# HAR
output_MATILDA_har = MATILDA.MATILDA_simulation(har, obs=obs, set_up_start='2007-01-01 00:00:00', set_up_end='2009-12-31 23:00:00', #output=output_path,
                                      sim_start='2010-01-01 00:00:00', sim_end='2014-12-31 23:00:00', freq="D",
                                      area_cat=315.694, area_glac=32.51, lat=42.33, warn=True, # soi=[5, 10],
                                      ele_dat=2561, ele_glac=4074, ele_cat=3225, lr_temp=-0.0059, lr_prec=0,
                                      TT_snow=0.354, TT_rain=0.5815, CFMAX_snow=4, CFMAX_ice=6,
                                      BETA=2.03, CET=0.0471, FC=462.5, K0=0.03467, K1=0.0544, K2=0.1277,
                                      LP=0.4917, MAXBAS=2.494, PERC=1.723, UZL=413.0, PCORR=1.19, SFCF=0.874, CWH=0.011765,
                                      AG=1, RHO_snow=300)

output_MATILDA_har[6].show()
print(output_MATILDA_har[2].Q_Total)



## To-tries:

# - Compare average precipitation of all grid cells (HAR & ERA) in the catchment to the observed
# - perform bias adjustment for whole HAR dataset and store for later use


## Results

# - grid cell most similar to mean catchment altitude contains the AWS




##
new_shdf = crop_extent.copy()
#add buffer to geom, otherwise not detected within bounhar_ds of 10km spacing
new_shdf.geometry = new_shdf.geometry.buffer(7000)

clipped_har_ds = har_ds.salem.subset(shape=new_shdf)

har_ds_static = salem.open_wrf_dataset(home + '/Seafile/EBA-CA/Tianshan_data/HARv2/static/all_static_kyzylsuu_HARv2.nc')
clipped_static = har_ds_static.salem.subset(shape=new_shdf)

fig = plt.figure(figsize=(16,12), dpi=300)
ax = fig.add_subplot(111)
shdf.plot(ax=ax, zorder=3)
new_shdf.plot(alpha=0.2, zorder=2, ax=ax)
clipped_har_ds.prcp.mean(dim='time').plot(ax=ax, zorder=-1)
plt.scatter(x, y, color='r')
plt.text(x,y, 'AWS')
plt.show()







### Convert shapefile to wrf projection ###
wrf_har_ds = salem.open_wrf_dataset(home +"/Desktop/HAR/HARv2_d10km_d_2d_prcp_2000.nc")
crop_extent_lat_lon = gpd.read_file(home+ "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/shapefile_hydro_kyzylsuu.shp")
crop_extent_lat_lon = crop_extent_lat_lon.to_crs(wrf_har_ds.pyproj_srs)
crop_extent_lat_lon.crs = wrf_har_ds.pyproj_srs
crop_extent_lat_lon.to_file(har_path+'abramov_har_proj.shp')



### The same thing can be done using salem ###
salem_xhar_ds = salem.open_xr_dataset(home +"/Seafile/EBA-CA/Tianshan_data/HARv2/variables/all_variables_kyzylsuu_HARv2_2007_2014.nc")

new_test = salem_xhar_ds.salem.subset(shape=new_shdf)
ax=new_shdf.plot(alpha=0.2)
shdf.plot(ax=ax, zorder=6)
new_test.prcp_nc.mean(dim='time').plot(ax=ax, zorder=-1)
plt.show()

new_test_v2 = new_test.where(new_test.south_north > 900000, drop=True)
new_test_v2
ax=new_shdf.plot(alpha=0.2)
shdf.plot(ax=ax, zorder=6)
new_test_v2.prcp_nc.mean(dim='time').plot(ax=ax, zorder=-1)
plt.show()
