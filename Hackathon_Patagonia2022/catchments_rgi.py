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
import warnings
warnings.filterwarnings('ignore')

## Paths
working_directory = home + "/Seafile/Hackathon_Patagonia2022/"
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


# glaciers_catchment_compact = glaciers_catchment.iloc[:, 4:len(glaciers_catchment.columns)-1]
# glaciers_catchment_compact = glaciers_catchment_compact.sort_values(by='gauge_id')
# glac_area = glaciers_catchment_compact.groupby(['gauge_id'])['Area'].sum().reset_index()
# glac_area = glac_area.sort_values(by='gauge_id')
#
# glaciers_catchment_compact = glaciers_catchment_compact.groupby(['gauge_id']).first().reset_index().drop(['Area'], axis=1)
# glaciers_catchment_compact['glaciated_area'] = glac_area['Area']
# glaciers_catchment_compact['glaciated_fraction'] = glaciers_catchment_compact['glaciated_area'] /glaciers_catchment_compact['area_km2']
# glaciers_catchment_compact.to_csv(output_path + 'catchments_with_glaciers_compact.csv')