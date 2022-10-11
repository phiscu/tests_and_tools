## USE conda-base environment!!
import sys
import os
import xarray as xr
import pandas as pd
from shapely.geometry import Point, mapping
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import salem
import xagg
import cartopy.crs as ccrs
from pathlib import Path; home = str(Path.home())
from matilda.core import matilda_simulation
from bias_correction import BiasCorrection

# lat = 42.19; lon = 78.2
aws_lat = 42.191433; aws_lon = 78.200253
start_date = '2007-01-01'; end_date = '2014-12-31'

## Paths
wd = home + '/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu'
t2m_bc_path = wd + "/met/era5l/t2m_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
tp_bc_path = wd + "/met/era5l/tp_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
t2m_path = wd + "/met/era5l/t2m_era5l_42.516-79.0167_1982-01-01-2020-12-31.csv"
tp_path = wd + "/met/era5l/tp_era5l_42.516-79.0167_1982-01-01-2020-12-31.csv"
obs_path = wd + "/hyd/obs/Kyzylsuu_1982_2021_latest.csv"
static_path = home + "/Seafile/EBA-CA/Tianshan_data/HARv2/static/all_static_kyzylsuu_HARv2.nc"

# crop_extent = shdf.to_crs(har_ds.pyproj_srs)
# crop_extent.crs = har_ds.pyproj_srs

## Read input files
catchment = gpd.read_file(home+ "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/shapefile_hydro_kyzylsuu.shp")
# shdf = salem.read_shapefile(home+ "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/shapefile_hydro_kyzylsuu.shp")
har_ds = salem.open_wrf_dataset(home +"/Seafile/EBA-CA/Tianshan_data/HARv2/variables/all_variables_HARv2_daily_kyzylsuu_1980_2020.nc")

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

## Select single grid cell:
# Select grid cell based on elevation:
# aws_alt = 2561
# catch_alt = 3225
# altitude_differences_gp = np.abs(static_har.hgt - catch_alt)
# closest_lat = float(static_har.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lat)
# closest_lon = float(static_har.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lon)
# # ...or based on proximity to the AWS.
#
# # Transform coordinates to CRS of the data (Lambert Conformal, arguments retrieved from NCDF-Metadata)
# print(har_ds.PROJ_NAME)  # Check original CRS
# data_crs = ccrs.LambertConformal(central_longitude=float(har_ds.PROJ_CENTRAL_LON), central_latitude=float(har_ds.PROJ_CENTRAL_LAT), standard_parallels=(float(har_ds.PROJ_STANDARD_PAR1), float(har_ds.PROJ_STANDARD_PAR1)))
# x, y = data_crs.transform_point(closest_lon, closest_lat, src_crs=ccrs.PlateCarree())
#
# pick = har_ds.sel(south_north=y, west_east=x, method='nearest')
# har = pick.to_dataframe().filter(['t2', 'prcp'])
# har.rename(columns={'t2':'t2m_har','prcp':'tp_har'}, inplace=True)
# har.tp_har = har.tp_har * 24            # Precipitation is in mm h^⁻1
# har = har["2007-01-01T00:00:00":"2014-12-31T00:00:00"]

## Convert shapefile to wrf projection (only needed if data is not transformed)
# catchment = catchment.to_crs(har_ds.pyproj_srs)
# catchment.crs = har_ds.pyproj_srs

## Transform HAR dataset to MSWEP grid (xagg requires lat/lon coordinates)
mswep_ds = salem.open_xr_dataset(home + "/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWEP/MSWEP_daily_past_kyzylsuu_1979-2020.nc")
har_ds = mswep_ds.salem.transform(har_ds)

## Area weighted average of gridcells in the catchment
# Clip to overlapping grid cells (for plotting)
clip = har_ds.salem.roi(shape=catchment, all_touched=True)

# Calculate overlaps:
weightmap = xagg.pixel_overlaps(clip, catchment)
# Aggregate
aggregated = xagg.aggregate(clip, weightmap)
# Produce dataframe
har_df = aggregated.to_dataframe()
har_df = har_df.reset_index(level='poly_idx', drop=True).drop('LABEL', axis=1)

## Adapt units: mm h^⁻1 >> mm d^-1

har_df[['et', 'potevap', 'prcp']] = har_df[['et', 'potevap', 'prcp']] * 24

## Plot
# fig = plt.figure(figsize=(16, 12), dpi=300)
# ax = fig.add_subplot(111)
# catchment.plot(ax=ax, zorder=3)
# clip.prcp.mean(dim='time').plot(ax=ax, zorder=-1)
# plt.scatter(aws_lon, aws_lat, color='r')
# plt.text(aws_lon, aws_lat, 'AWS')
# # plt.scatter(x, y, color='r')
# # plt.text(x,y, 'AWS')
# plt.show()


## To-tries:

# - Compare average precipitation of all grid cells (HAR & ERA) in the catchment to the observed
# - run matilda with averaged catchment gridcells
# - perform bias adjustment for whole HAR dataset and store for later use           - check!
# - run matilda including the et series


## Results

# - grid cell most similar to mean catchment altitude contains the AWS

##
# new_shdf = crop_extent.copy()
# #add buffer to geom, otherwise not detected within bounhar_ds of 10km spacing
# new_shdf.geometry = new_shdf.geometry.buffer(7000)
#
# clipped_har_ds = har_ds.salem.subset(shape=new_shdf)
#
# har_ds_static = salem.open_wrf_dataset(home + '/Seafile/EBA-CA/Tianshan_data/HARv2/static/all_static_kyzylsuu_HARv2.nc')
# clipped_static = har_ds_static.salem.subset(shape=new_shdf)
#
# fig = plt.figure(figsize=(16,12), dpi=300)
# ax = fig.add_subplot(111)
# shdf.plot(ax=ax, zorder=3)
# new_shdf.plot(alpha=0.2, zorder=2, ax=ax)
# clipped_har_ds.prcp.mean(dim='time').plot(ax=ax, zorder=-1)
# plt.scatter(x, y, color='r')
# plt.text(x,y, 'AWS')
# plt.show()
