import ee
import geemap
import configparser
import ast
import geopandas as gpd
import concurrent.futures
import os
import requests
from retry import retry
from tqdm import tqdm
from bias_correction import BiasCorrection
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from matplotlib.legend import Legend
import probscale
import matplotlib.pyplot as plt
import sys
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
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


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


def create_buffer(station_coords, output, buffer_radius=1000, write_files=True):
    """
    Create spatial buffers around station coordinates and save them in a GeoPackage (.gpkg) file.

    Args:
    - station_coords (dict): Dictionary containing station coordinates for each region.
    - output (str): Path where the GeoPackage file will be saved.
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
        buffer_gdf.to_file(output, driver='GPKG', layer='station_buffers')
        locations_gdf.to_file(output, driver='GPKG', layer='station_locations')

    return buffer_dict


def search_dict(input_dict, key):
    """
    Search for a specific key in a nested dictionary and return the corresponding value.
    Parameters
    ----------
    input_dict : dict
        The input dictionary to search for the key.
    key : str
        The key to search for in the nested dictionary.
    Returns
    -------
    value
        The value corresponding to the specified key in the nested dictionary. Returns None if the key is not found.
    """
    if key in input_dict:
        return input_dict[key]

    for k, v in input_dict.items():
        if isinstance(v, dict):
            result = search_dict(v, key)
            if result is not None:
                return result

    return None


class CMIPDownloader:
    """Class to download spatially averaged CMIP6 data for a given period, variable, and spatial subset."""

    def __init__(self, var, starty, endy, shape, processes=10, dir='./'):
        self.var = var
        self.starty = starty
        self.endy = endy
        self.shape = shape
        self.processes = processes
        self.directory = dir

        # create the download directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def download(self):
        """Runs a subset routine for CMIP6 data on GEE servers to create ee.FeatureCollections for all years in
        the requested period. Downloads individual years in parallel processes to increase the download time."""

        print('Initiating download request for NEX-GDDP-CMIP6 data from ' +
              str(self.starty) + ' to ' + str(self.endy) + '.')

        def getRequests(starty, endy):
            """Generates a list of years to be downloaded. [Client side]"""

            return [i for i in range(starty, endy + 1)]

        @retry(tries=10, delay=1, backoff=2)
        def getResult(index, year):
            """Handle the HTTP requests to download one year of CMIP6 data. [Server side]"""

            start = str(year) + '-01-01'
            end = str(year + 1) + '-01-01'
            startDate = ee.Date(start)
            endDate = ee.Date(end)
            n = endDate.difference(startDate, 'day').subtract(1)

            def getImageCollection(var):
                """Create and image collection of CMIP6 data for the requested variable, period, and region.
                [Server side]"""

                collection = ee.ImageCollection('NASA/GDDP-CMIP6') \
                    .select(var) \
                    .filterDate(startDate, endDate) \
                    .filterBounds(self.shape)
                return collection

            def renameBandName(b):
                """Edit variable names for better readability. [Server side]"""

                split = ee.String(b).split('_')
                return ee.String(split.splice(split.length().subtract(2), 1).join("_"))

            def buildFeature(i):
                """Create an area weighted average of the defined region for every day in the given year.
                [Server side]"""

                t1 = startDate.advance(i, 'day')
                t2 = t1.advance(1, 'day')
                # feature = ee.Feature(point)
                dailyColl = collection.filterDate(t1, t2)
                dailyImg = dailyColl.toBands()
                # renaming and handling names
                bands = dailyImg.bandNames()
                renamed = bands.map(renameBandName)
                # Daily extraction and adding time information
                dict = dailyImg.rename(renamed).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=self.shape,
                ).combine(
                    ee.Dictionary({'system:time_start': t1.millis(), 'isodate': t1.format('YYYY-MM-dd')})
                )
                return ee.Feature(None, dict)

            # Create features for all days in the respective year. [Server side]
            collection = getImageCollection(self.var)
            year_feature = ee.FeatureCollection(ee.List.sequence(0, n).map(buildFeature))

            # Create a download URL for a CSV containing the feature collection. [Server side]
            url = year_feature.getDownloadURL()

            # Handle downloading the actual csv for one year. [Client side]
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                r.raise_for_status()
            filename = os.path.join(self.directory, 'cmip6_' + self.var + '_' + str(year) + '.csv')
            with open(filename, 'w') as f:
                f.write(r.text)

            return index

        # Create a list of years to be downloaded. [Client side]
        items = getRequests(self.starty, self.endy)

        # Launch download requests in parallel processes and display a status bar. [Client side]
        with tqdm(total=len(items), desc="Downloading CMIP6 data for variable '" + self.var + "'") as pbar:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.processes) as executor:
                for i, year in enumerate(items):
                    results.append(executor.submit(getResult, i, year))
                for future in concurrent.futures.as_completed(results):
                    index = future.result()
                    pbar.update(1)

        print("All downloads complete.")


class CMIPProcessor:
    """Class to read and pre-process CSV files downloaded by the CMIPDownloader class."""

    def __init__(self, var, file_dir='.', start=1979, end=2100):
        self.file_dir = file_dir
        self.var = var
        self.start = start
        self.end = end
        self.df_hist = self.append_df(self.var, self.start, self.end, self.file_dir, hist=True)
        self.df_ssp = self.append_df(self.var, self.start, self.end, self.file_dir, hist=False)
        self.ssp2_common, self.ssp5_common, self.hist_common, \
            self.common_models, self.dropped_models = self.process_dataframes()
        self.ssp2, self.ssp5 = self.get_results()

    def read_cmip(self, filename):
        """Reads CMIP6 CSV files and drops redundant columns."""

        df = pd.read_csv(filename, index_col='isodate', parse_dates=['isodate'])
        df = df.drop(['system:index', '.geo', 'system:time_start'], axis=1)
        return df

    def append_df(self, var, start, end, file_dir='.', hist=True):
        """Reads CMIP6 CSV files of individual years and concatenates them into dataframes for the full downloaded
        period. Historical and scenario datasets are treated separately. Converts precipitation unit to mm."""

        df_list = []
        if hist:
            starty = start
            endy = 2014
        else:
            starty = 2015
            endy = end
        for i in range(starty, endy + 1):
            filename = file_dir + 'cmip6_' + var + '_' + str(i) + '.csv'
            df_list.append(self.read_cmip(filename))
        if hist:
            hist_df = pd.concat(df_list)
            if var == 'pr':
                hist_df = hist_df * 86400  # from kg/(m^2*s) to mm/day
            return hist_df
        else:
            ssp_df = pd.concat(df_list)
            if var == 'pr':
                ssp_df = ssp_df * 86400  # from kg/(m^2*s) to mm/day
            return ssp_df

    def process_dataframes(self):
        """Separates the two scenarios and drops models not available for both scenarios and the historical period."""

        ssp2 = self.df_ssp.loc[:, self.df_ssp.columns.str.startswith('ssp245')]
        ssp5 = self.df_ssp.loc[:, self.df_ssp.columns.str.startswith('ssp585')]
        hist = self.df_hist.loc[:, self.df_hist.columns.str.startswith('historical')]

        ssp2.columns = ssp2.columns.str.lstrip('ssp245_').str.rstrip('_' + self.var)
        ssp5.columns = ssp5.columns.str.lstrip('ssp585_').str.rstrip('_' + self.var)
        hist.columns = hist.columns.str.lstrip('historical_').str.rstrip('_' + self.var)

        # Get all the models the three datasets have in common
        common_models = set(ssp2.columns).intersection(ssp5.columns).intersection(hist.columns)

        # Get the model names that contain NaN values
        nan_models_list = [df.columns[df.isna().any()].tolist() for df in [ssp2, ssp5, hist]]
        # flatten the list
        nan_models = [col for sublist in nan_models_list for col in sublist]
        # remove duplicates
        nan_models = list(set(nan_models))

        # Remove models with NaN values from the list of common models
        common_models = [x for x in common_models if x not in nan_models]

        ssp2_common = ssp2.loc[:, common_models]
        ssp5_common = ssp5.loc[:, common_models]
        hist_common = hist.loc[:, common_models]

        dropped_models = list(set([mod for mod in ssp2.columns if mod not in common_models] +
                                  [mod for mod in ssp5.columns if mod not in common_models] +
                                  [mod for mod in hist.columns if mod not in common_models]))

        return ssp2_common, ssp5_common, hist_common, common_models, dropped_models

    def get_results(self):
        """Concatenates historical and scenario data to combined dataframes of the full downloaded period.
        Arranges the models in alphabetical order."""

        ssp2_full = pd.concat([self.hist_common, self.ssp2_common])
        ssp2_full.index.names = ['TIMESTAMP']
        ssp5_full = pd.concat([self.hist_common, self.ssp5_common])
        ssp5_full.index.names = ['TIMESTAMP']

        ssp2_full = ssp2_full.reindex(sorted(ssp2_full.columns), axis=1)
        ssp5_full = ssp5_full.reindex(sorted(ssp5_full.columns), axis=1)

        return ssp2_full, ssp5_full


def read_era5l(file):
    """Reads ERA5-Land data, drops redundant columns, and adds DatetimeIndex.
    Resamples the dataframe to reduce the DatetimeIndex to daily resolution."""

    return pd.read_csv(file, **{
        'usecols': ['temp', 'prec', 'dt'],
        'index_col': 'dt',
        'parse_dates': ['dt']}).resample('D').agg({'temp': 'mean', 'prec': 'sum'})


def adjust_bias(predictand, predictor, era5=True, train_start='1979-01-01', train_end='2022-12-31', method='normal_mapping'):
    """Applies bias correction to specified periods individually."""

    # Read predictor data
    if era5:
        predictor = read_era5l(predictor)

    # Determine variable type based on the mean value
    var = 'temp' if predictand.mean().mean() > 100 else 'prec'

    # Adjust bias in discrete blocks as suggested by Switanek et.al. (2017) (https://doi.org/10.5194/hess-21-2649-2017)
    # Initialize periods dict
    correction_periods = [
        {'correction_range': ('1979-01-01', '2010-12-31'), 'extraction_range': ('1979-01-01', '1990-12-31')},
    ]
    # Add decades from 1991 to 2090
    for decade_start in range(1991, 2090, 10):
        correction_start = f"{decade_start - 10}-01-01"
        correction_end = f"{decade_start + 19}-12-31"
        extraction_start = f"{decade_start}-01-01"
        extraction_end = f"{decade_start + 9}-12-31"

        correction_periods.append({
            'correction_range': (correction_start, correction_end),
            'extraction_range': (extraction_start, extraction_end)
        })

    # Add the last decade of the century (2091-2100)
    correction_periods.append({
        'correction_range': ('2081-01-01', '2100-12-31'),
        'extraction_range': ('2091-01-01', '2100-12-31')
    })

    # Prepare a dataframe to hold the corrected results
    corrected_data = pd.DataFrame()
    # Define one common training period
    training_period = slice(train_start, train_end)
    # Loop through each correction period
    for period in correction_periods:
        correction_start, correction_end = period['correction_range']
        extraction_start, extraction_end = period['extraction_range']

        correction_slice = slice(correction_start, correction_end)
        extraction_slice = slice(extraction_start, extraction_end)

        # Perform bias correction for each period
        data_corr = pd.DataFrame()
        for col in predictand.columns:
            x_train = predictand[col][training_period].squeeze()
            y_train = predictor[training_period][var].squeeze()
            x_predict = predictand[col][correction_slice].squeeze()
            bc_corr = BiasCorrection(y_train, x_train, x_predict)
            corrected_col = pd.DataFrame(bc_corr.correct(method=method))

            # Extract the desired years after correction
            data_corr[col] = corrected_col.loc[extraction_slice]

        # Append the corrected data to the main dataframe
        corrected_data = corrected_data.append(data_corr, ignore_index=False)

    return corrected_data