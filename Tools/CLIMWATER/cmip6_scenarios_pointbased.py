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
import utm
from pyproj import CRS
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import geopandas as gpd
from shapely.geometry import Point
import utm


def custom_buffer(point, buffer_radius_meters):
    """
    Create a circular buffer around a point with a given radius in meters.

    Args:
    - point (tuple): Coordinate (longitude, latitude) of the point.
    - buffer_radius_meters (float): Radius of the buffer in meters.

    Returns:
    - buffer (shapely.geometry.Polygon): Circular buffer around the point.
    """
    lon, lat = point

    point_geom = Point(lon, lat)

    # Projected coordinate system (EPSG:3857) for meters-based calculations
    project_meters = pyproj.CRS('EPSG:3857')

    # Original coordinate system (EPSG:4326) for latitude and longitude
    project_latlon = pyproj.CRS('EPSG:4326')

    # Create transformers
    project = pyproj.Transformer.from_crs(project_latlon, project_meters, always_xy=True).transform
    reproject = pyproj.Transformer.from_crs(project_meters, project_latlon, always_xy=True).transform

    # Convert buffer radius from meters to degrees (approximate)
    lon_m, lat_m = transform(project, point_geom).x, transform(project, point_geom).y
    point_buffer = Point(lon_m + buffer_radius_meters, lat_m)
    lon_buffer, lat_buffer = transform(reproject, point_buffer).x, transform(reproject, point_buffer).y

    buffer_radius_degrees = np.abs(lon_buffer - lon)

    # Create a buffer around the station coordinates
    buffer = point_geom.buffer(buffer_radius_degrees)

    return buffer


def create_buffer(station_coords, output_dir, buffer_radius=1000, write_files=True):
    """
    Create spatial buffers around station coordinates and save them in a GeoPackage (.gpkg) file.

    Args:
    - station_coords (dict): Dictionary containing station coordinates for each region.
    - output_dir (str): Directory path where the GeoPackage file will be saved.
    - buffer_radius (float): Radius of the buffer in degrees (or any appropriate unit).
    """
    # Create an empty GeoDataFrame to store the buffers
    buffer_gdf = gpd.GeoDataFrame(columns=['Station_Name', 'geometry'])

    # Create an empty GeoDataFrame to store the station locations
    locations_gdf = gpd.GeoDataFrame(columns=['Station_Name', 'geometry'])

    buffer_dict = {}

    # Iterate over each region
    for region, stations in station_coords.items():
        region_buffers = {}
        # Iterate over each station in the region
        for station_name, coordinates in stations.items():
            # Create a buffer around the station coordinates
            buffer_geom = custom_buffer(coordinates, buffer_radius)
            region_buffers[station_name] = buffer_geom

            # Add the buffer geometry to the GeoDataFrame
            buffer_gdf = buffer_gdf.append({'Station_Name': station_name,
                                            'geometry': buffer_geom}, ignore_index=True)

            # Create a point geometry for station location
            point_geom = Point(coordinates)

            # Add the point geometry to the GeoDataFrame
            locations_gdf = locations_gdf.append({'Station_Name': station_name,
                                                  'geometry': point_geom}, ignore_index=True)
        buffer_dict[region] = region_buffers

    # Save the GeoDataFrames to a GeoPackage file
    if write_files:
        output_file = os.path.join(output_dir, 'station_data.gpkg')
        buffer_gdf.to_file(output_file, driver='GPKG', layer='station_buffers')
        locations_gdf.to_file(output_file, driver='GPKG', layer='station_locations')

    return buffer_dict


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
output_directory = '/home/phillip/Seafile/CLIMWATER/Data/Hydrometeorology/GIS'
buffered_stations = create_buffer(station_coords, output_directory)

## Data checks and plots

######
# Projection class (input: Region)

# Download CMIP6 for buffers
# Bias adjust CMIP6 with station data
# Store data
# Plot data


##
