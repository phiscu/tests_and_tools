# -*- coding: UTF-8 -*-

## import
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import transform
from functools import partial
import pyproj
import numpy as np
import warnings
from shapely.errors import ShapelyDeprecationWarning
import geopandas as gpd
import ee
import geemap
import utm
from pyproj import CRS
import sys
wd = os.getcwd()
sys.path.append(wd + '/tests_and_tools/Tools/CLIMWATER')
import matilda_functions
import geopandas as gpd
from shapely.geometry import Point
import utm
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# Paths
directory_path = '/home/phillip/Seafile/CLIMWATER/Data/Hydrometeorology/Meteo'
output = '/home/phillip/Seafile/CLIMWATER/Data/Hydrometeorology/CMIP6/'

# Read data
region_data, station_coords = matilda_functions.read_station_data(directory_path)

# Create buffers around station coordinates and save them to a GeoPackage file
gis_dir = output + 'GIS/'
gis_file = gis_dir + 'station_gis.gpkg'
if not os.path.exists(gis_dir):
    os.makedirs(gis_dir)
buffered_stations = matilda_functions.create_buffer(station_coords, gis_file, buffer_radius=1000, write_files=False)

## Data checks and plots

# matilda_functions.plot_region_data(region_data, show=True, output=output + 'Plots/aws_data_raw.png')

# Remove temperature outliers
matilda_functions.process_nested_dict(region_data, matilda_functions.remove_outliers, sd_factor=2)

# Remove years with an annual precipitation of 0
matilda_functions.process_nested_dict(region_data, matilda_functions.remove_annual_zeros)

# Plot again
# matilda_functions.plot_region_data(region_data, show=True, output=output + 'Plots/aws_data_filtered.png')


######
# Projection class (input: )

# Download CMIP6 for buffers
# Bias adjust CMIP6 with station data
# Store data
# Plot data


## GEE

# Example
buffer_file = gis_file
station = 'Karshi'
starty = 1979
endy = 2100
cmip_dir = '/home/phillip/Seafile/CLIMWATER/Data/Hydrometeorology/Meteo/Kashkadarya/cmip6/'

aws = matilda_functions.search_dict(region_data, station)

# cmip6_station = matilda_functions.CMIP6DataProcessor(buffer_file, station, starty, endy, cmip_dir)

# cmip6_station.download_cmip6_data()

# cmip6_station.bias_adjustment(region_data)
# temp_cmip = cmip6_station.ssp_tas_dict
# prec_cmip = cmip6_station.ssp_pr_dict

## Data checks

# temp_cmip, prec_cmip = matilda_functions.apply_filters(temp_cmip, prec_cmip, zscore_threshold=3, jump_threshold=5, resampling_rate='Y')

## Back-up files

# matilda_functions.dict_to_pickle(temp_cmip, output + 'adjusted/temp_' + station + '_adjusted.pickle')
# matilda_functions.dict_to_pickle(prec_cmip, output + 'adjusted/prec_' + station + '_adjusted.pickle')

temp_cmip = matilda_functions.pickle_to_dict(output + 'adjusted/temp_' + station + '_adjusted.pickle')
prec_cmip = matilda_functions.pickle_to_dict(output + 'adjusted/prec_' + station + '_adjusted.pickle')

## Plots
# Lineplots with all models
matilda_functions.cmip_plot_combined(data=temp_cmip, target=aws,
                                     title=f'"{station}" - 5y Rolling Mean of Annual Air Temperature', target_label='Observations',
                  filename=f'cmip6_bias_adjustment_{station}_temperature.png', show=True,
                                     intv_mean='Y', rolling=5, out_dir=output + 'Plots/')
matilda_functions.cmip_plot_combined(data=prec_cmip, target=aws.dropna(),
                                     title=f'"{station}" - 5y Rolling Mean of Annual Precipitation', precip=True,
                   target_label='Observations', filename=f'cmip6_bias_adjustment_{station}_precipitation.png', show=True,
                                     intv_sum='Y', rolling=5, out_dir=output + 'Plots/')
print('Figures for CMIP6 bias adjustment created.')

# Ensemble mean plots
matilda_functions.cmip_plot_ensemble(temp_cmip, aws['temp'], intv_mean='Y', show=True, out_dir=output + 'Plots/',
                                     target_label="Observations", filename=f'cmip6_ensemble_{station}',
                                     site_label=station)
matilda_functions.cmip_plot_ensemble(prec_cmip, aws['prec'].dropna(), precip=True, intv_sum='Y',
                                     show=True, out_dir=output + 'Plots/', target_label="Observations",
                                     site_label=station, filename=f'cmip6_ensemble_{station}')
print('Figures for CMIP6 ensembles created.')

## Probability plots:

start_temp = aws['temp'].first_valid_index().year
end_temp = aws['temp'].last_valid_index().year
start_prec = aws['prec'].dropna().first_valid_index().year
end_prec = aws['prec'].dropna().last_valid_index().year

matilda_functions.pp_matrix(temp_cmip['SSP2_raw'], aws['temp'], temp_cmip['SSP2_adjusted'], scenario='SSP2',
                            starty=start_temp, endy=end_temp, target_label='Observed',
                            out_dir=output + 'Plots/', show=True, site=station)
matilda_functions.pp_matrix(temp_cmip['SSP5_raw'], aws['temp'], temp_cmip['SSP5_adjusted'], scenario='SSP5',
                            starty=start_temp, endy=end_temp, target_label='Observed',
                            out_dir=output + 'Plots/', show=True, site=station)

matilda_functions.pp_matrix(prec_cmip['SSP2_raw'], aws['prec'].dropna().astype(float), prec_cmip['SSP2_adjusted'],
                            precip=True, starty=start_prec, endy=end_prec, target_label='Observed',
                            scenario='SSP2', out_dir=output + 'Plots/', show=True, site=station)
matilda_functions.pp_matrix(prec_cmip['SSP5_raw'], aws['prec'].dropna().astype(float), prec_cmip['SSP5_adjusted'],
                            precip=True, starty=start_prec, endy=end_prec, target_label='Observed',
                            scenario='SSP5', out_dir=output + 'Plots/', show=True, site=station)


print('Figures for CMIP6 bias adjustment performance created.')

# Close all open figures
plt.close('all')

# clean up function in the end to delete annual cmip files

