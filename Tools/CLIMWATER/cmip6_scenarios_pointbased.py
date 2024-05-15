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
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


# Directory path
directory_path = '/home/phillip/Seafile/CLIMWATER/Data/Hydrometeorology/Meteo'

# Dictionaries to store data for each region
region_data = {}
station_coords = {}

# Iterate over the subdirectories
for subdir in os.listdir(directory_path):
    subdir_path = os.path.join(directory_path, subdir)

    # Check if the subdirectory is a directory
    if os.path.isdir(subdir_path):
        # Read the AWS coordinates CSV file
        coords_file = os.path.join(subdir_path, 'aws_coords.csv')
        coords_df = pd.read_csv(coords_file)

        # Read precipitation data
        prec_file = os.path.join(subdir_path, f'{subdir}_prec.xlsx')
        prec_df = pd.read_excel(prec_file)

        # Read temperature data
        temp_file = os.path.join(subdir_path, f'{subdir}_temp.xlsx')
        temp_df = pd.read_excel(temp_file)

        # Create datetime index from ['day', 'month', 'year'] columns
        prec_df['date'] = pd.to_datetime(prec_df[['year', 'month', 'day']])
        temp_df['date'] = pd.to_datetime(temp_df[['year', 'month', 'day']])

        # Set the datetime index and drop the original time columns
        prec_df.set_index('date', inplace=True)
        prec_df.drop(columns=['day', 'month', 'year'], inplace=True)
        temp_df.set_index('date', inplace=True)
        temp_df.drop(columns=['day', 'month', 'year'], inplace=True)

        # Convert Celsius to Kelvin for consistency with CMIP6
        temp_df = temp_df + 273.15

        # All values not of type float to NaN
        prec_df = prec_df.apply(pd.to_numeric, errors='coerce')
        # All NaN to 0 (only three cells and bias adjustment can't handle NaN)
        prec_df = prec_df.fillna(0)

        # Initialize a dictionary to store station DataFrames for the region
        region_stations = {}

        # Iterate over each station in the region
        for station in temp_df.columns:
            station_df = pd.DataFrame({'temp': temp_df[station], 'prec': prec_df[station]})
            region_stations[station] = station_df  # Store station DataFrame in the dictionary

        # Store the dictionary of station DataFrames in the region_data dictionary
        region_data[subdir] = region_stations

        # Create a dictionary to store station coordinates
        station_dict = {}
        # Iterate over each row in the coordinates DataFrame
        for index, row in coords_df.iterrows():
            station_name = row['Name']
            latitude = row['Latitude']
            longitude = row['Longitude']

            # Store station coordinates as a tuple in the dictionary
            if station_name in region_stations.keys():
                station_dict[station_name] = (longitude, latitude)  # Note: Changed order to (lon, lat)

        # Store the station coordinates dictionary in the main dictionary
        station_coords[subdir] = station_dict

# Create buffers around station coordinates and save them to a GeoPackage file
output = '/home/phillip/Seafile/CLIMWATER/Data/Hydrometeorology/GIS/station_gis.gpkg'
buffered_stations = matilda_functions.create_buffer(station_coords, output, buffer_radius=1000, write_files=False)

## Data checks and plots

######
# Projection class (input: )

# Download CMIP6 for buffers
# Bias adjust CMIP6 with station data
# Store data
# Plot data


## GEE


# Example:
buffer_file = output
station = 'Karshi'
starty = 1979
endy = 2100
cmip_dir = '/home/phillip/Seafile/CLIMWATER/Data/Hydrometeorology/Meteo/Kashkadarya/cmip6/'                        # Loop through dicts to maintain regional folder structure


# Set up GEE
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Define target polygon
all_buffers = gpd.read_file(buffer_file, layer='station_buffers')
buffer = all_buffers.loc[all_buffers['Station_Name'] == station]
buffer_ee = geemap.geopandas_to_ee(buffer)

# Download CMIP6 data
# downloader_t = matilda_functions.CMIPDownloader(var='tas', starty=starty, endy=endy, shape=buffer_ee, processes=5, dir=cmip_dir)
# downloader_t.download()
# downloader_p = matilda_functions.CMIPDownloader(var='pr', starty=starty, endy=endy, shape=buffer_ee, processes=5, dir=cmip_dir)
# downloader_p.download()

# Process CMIP6 data
processor_t = matilda_functions.CMIPProcessor(file_dir=cmip_dir, var='tas', start=starty, end=endy)
ssp2_tas_raw, ssp5_tas_raw = processor_t.get_results()

processor_p = matilda_functions.CMIPProcessor(file_dir=cmip_dir, var='pr', start=starty, end=endy)
ssp2_pr_raw, ssp5_pr_raw = processor_p.get_results()

print(ssp2_tas_raw.info())
print('Models that failed the consistency checks:\n')
print(processor_t.dropped_models)


# Bias adjustment
print('Running bias adjustment routine...')
aws = matilda_functions.search_dict(region_data, station)
train_start = str(aws.index.min())
train_end = str(aws.index.max())
ssp2_tas = matilda_functions.adjust_bias(predictand=ssp2_tas_raw, predictor=aws, era5=False, train_start=train_start, train_end=train_end)
ssp5_tas = matilda_functions.adjust_bias(predictand=ssp5_tas_raw, predictor=aws, era5=False, train_start=train_start, train_end=train_end)
ssp2_pr = matilda_functions.adjust_bias(predictand=ssp2_pr_raw, predictor=aws, era5=False, train_start=train_start, train_end=train_end)
ssp5_pr = matilda_functions.adjust_bias(predictand=ssp5_pr_raw, predictor=aws, era5=False, train_start=train_start, train_end=train_end)
print('Done!')

# store our raw and adjusted data in dictionaries.
ssp_tas_dict = {'SSP2_raw': ssp2_tas_raw, 'SSP2_adjusted': ssp2_tas, 'SSP5_raw': ssp5_tas_raw,
                'SSP5_adjusted': ssp5_tas}
ssp_pr_dict = {'SSP2_raw': ssp2_pr_raw, 'SSP2_adjusted': ssp2_pr, 'SSP5_raw': ssp5_pr_raw, 'SSP5_adjusted': ssp5_pr}

