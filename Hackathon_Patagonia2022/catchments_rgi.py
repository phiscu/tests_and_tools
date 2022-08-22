## Packages
from pathlib import Path; home = str(Path.home())
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysheds.grid import Grid
import fiona
import geopandas as gpd
import subprocess
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

## Paths
working_directory = home + "/Seafile/Hackathon_Patagonia2022/"
data_dir = working_directory + "CAMELS_CL_v202201/"
# input_DEM = home + "/Seafile/EBA-CA/Azamat_AvH/workflow/data/Jyrgalang/static/jyrgalang_dem_alos.tif"
RGI_files = home + "/Seafile/Hackathon_Patagonia2022/17_rgi60_SouthernAndes/17_rgi60_SouthernAndes.shp"
catchments = home + "/Seafile/Hackathon_Patagonia2022/CAMELS_CL_v202201/camels_cl_boundaries/camels_cl_boundaries.shp"
output_path = working_directory + "outputs/"


## Cutting all glaciers within the catchment from the RGI shapefile
rgi_shp = gpd.GeoDataFrame.from_file(RGI_files)
rgi_shp.drop(rgi_shp.columns.difference(['RGIId', 'GLIMSId', 'CenLon', 'CenLat', 'Area', 'geometry']), 1, inplace=True)

catchment_shp = gpd.GeoDataFrame.from_file(catchments)
catchment_shp['catchment_center_lon'] = catchment_shp.centroid.x
catchment_shp['catchment_center_lat'] = catchment_shp.centroid.y
catchment_shp['catchm_lon_min'] = catchment_shp.bounds['minx']
catchment_shp['catchm_lat_min'] = catchment_shp.bounds['miny']
catchment_shp['catchm_lon_max'] = catchment_shp.bounds['maxx']
catchment_shp['catchm_lat_max'] = catchment_shp.bounds['maxy']

glaciers_catchment = gpd.overlay(rgi_shp, catchment_shp, how='intersection')

# Reproject GDF to CRS with unit Meter:
glac_copy = glaciers_catchment.copy()
glac_copy = glac_copy.to_crs({'proj': 'cea'})
glac_copy['glacier_area_km2'] = glac_copy.area / 10**6


#
glac_copy = glac_copy.iloc[:, 4:len(glac_copy.columns)]
glac_copy = glac_copy.sort_values(by='gauge_id')
glac_area = glac_copy.groupby(['gauge_id'])['glacier_area_km2'].sum().reset_index()  # viele catchments haben exakt die gleiche gletscherfläche!?
glac_area = glac_area.sort_values(by='gauge_id')

glac_copy = glac_copy.groupby(['gauge_id']).first().reset_index().drop(['Area', 'glacier_area_km2'], axis=1)
glac_copy['glaciated_area_km2'] = glac_area['glacier_area_km2']
glac_copy['glaciated_fraction'] = glac_copy['glaciated_area_km2'] / glac_copy['area_km2']
glac_copy.drop('geometry', axis=1).to_csv(output_path + 'catchments_with_glaciers_compact.csv')

# fig, ax = plt.subplots(1, 1)
# base = catchment_shp.plot(color='white', edgecolor='black')
# glaciers_catchment.plot(ax=base, column="RGIId", legend=True)
# plt.show()

# glaciers_catchment.to_file(driver = 'ESRI Shapefile', filename= output_path + "glaciers_in_catchments.shp")

glaciers_catchment.drop('geometry', axis=1).to_csv(output_path + 'catchments_with_glaciers.csv', index=False)




## Data analysis

# data = xr.open_dataset(output_path + "camels_cl_v202201.nc")

prec = pd.read_csv(data_dir + "precip_mswep_mm_mon.csv", index_col='date', parse_dates=['date'])
pet = pd.read_csv(data_dir + "pet_hargreaves_mm_mon.csv", index_col='date', parse_dates=['date'])
runoff = pd.read_csv(data_dir + "q_mm_mon.csv", index_col='date', parse_dates=['date'])
# swe = pd.read_csv(data_dir + "cons_sf_wr_Ls_year.csv", index_col='date', parse_dates=['date'])

catchm_attr = pd.read_csv(data_dir + "catchment_attributes.csv")

for i in [prec, pet, runoff]: i.drop(i.columns[[0,1,2]], axis=1, inplace=True)      # Delete additional date columns

excess = runoff - prec - pet

excess.mean().to_csv(output_path + "excess_runoff_potential.csv")
runoff.mean().to_csv(output_path + "runoff_mon_mean.csv")

