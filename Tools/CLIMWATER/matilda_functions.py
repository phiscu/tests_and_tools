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
import pickle

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def read_station_data(directory_path):
    region_data = {}
    station_coords = {}

    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)

        if os.path.isdir(subdir_path):
            coords_file = os.path.join(subdir_path, 'aws_coords.csv')
            coords_df = pd.read_csv(coords_file)

            prec_file = os.path.join(subdir_path, f'{subdir}_prec.xlsx')
            prec_df = pd.read_excel(prec_file)

            temp_file = os.path.join(subdir_path, f'{subdir}_temp.xlsx')
            temp_df = pd.read_excel(temp_file)

            prec_df['date'] = pd.to_datetime(prec_df[['year', 'month', 'day']])
            temp_df['date'] = pd.to_datetime(temp_df[['year', 'month', 'day']])

            prec_df.set_index('date', inplace=True)
            prec_df.drop(columns=['day', 'month', 'year'], inplace=True)
            temp_df.set_index('date', inplace=True)
            temp_df.drop(columns=['day', 'month', 'year'], inplace=True)

            temp_df = temp_df + 273.15
            prec_df = prec_df.apply(pd.to_numeric, errors='coerce')
            prec_df = prec_df.fillna(0)

            region_stations = {}
            for station in temp_df.columns:
                station_df = pd.DataFrame({'temp': temp_df[station], 'prec': prec_df[station]})
                region_stations[station] = station_df

            region_data[subdir] = region_stations

            station_dict = {}
            for index, row in coords_df.iterrows():
                station_name = row['Name']
                latitude = row['Latitude']
                longitude = row['Longitude']

                if station_name in region_stations.keys():
                    station_dict[station_name] = (longitude, latitude)

            station_coords[subdir] = station_dict

    return region_data, station_coords


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


def plot_meteo(ax, aws, aws_name):
    color = 'tab:red'
    # Drop NaN values before plotting
    aws_temp = aws.dropna(subset=['temp'])
    ax.plot(aws_temp.index, aws_temp['temp'], color=color, alpha=0.7)
    ax.set_ylabel('Daily Temperature [K]', color=color)
    ax.tick_params(axis='y', labelcolor=color)

    ax2 = ax.twinx()
    color = 'tab:blue'
    # Drop NaN values before plotting
    aws_prec = aws.dropna(subset=['prec'])
    ax2.plot(aws_prec.index, aws_prec['prec'], color=color, alpha=0.7)
    ax2.set_ylabel('Daily Precipitation [mm]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax.set_title(aws_name)


def plot_region_data(region_data, show=True, output=None):
    aws_names = [aws for region in region_data.keys() for aws in region_data[region].keys()]
    aws_list = [search_dict(region_data, station) for station in aws_names]

    fig, axs = plt.subplots(len(aws_names), 1, figsize=(10, 20), sharex=True)

    for i, (aws_data, aws_name) in enumerate(zip(aws_list, aws_names)):
        ax = axs[i]
        plot_meteo(ax, aws_data, aws_name)

    plt.xlabel('Date')
    plt.tight_layout()

    if output:
        # Create the directory if it doesn't exist
        output_dir = os.path.dirname(output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(output, format='png')

    if show:
        plt.show()


def remove_outliers(series, sd_factor=2):
    """
    Replace outliers in a time series with NaN values and interpolate the missing values using linear interpolation.
    Parameters
    ----------
    series : pandas.Series
        The input time series data with outliers.
    sd_factor : int, optional
        The factor to determine outliers based on standard deviations. Default is 2.
    Returns
    -------
    pandas.Series
        The time series data with outliers replaced by NaN values and interpolated missing values.
    """
    mean = series.mean()
    std = series.std()
    outliers = np.abs(series - mean) > sd_factor * std
    series[outliers] = np.nan
    series.interpolate(method='linear', inplace=True)
    return series


def process_nested_dict(d, func, *args, **kwargs):
    for key, value in d.items():
        if isinstance(value, pd.DataFrame):
            d[key] = func(value, *args, **kwargs)
        elif isinstance(value, dict):
            process_nested_dict(value, func, *args, **kwargs)


def remove_annual_zeros(df):
    """
    Sets all days to NaN that are in years, where the annual precipitation sum is 0.
    Parameters:
        df (DataFrame): Input dataframe with a datetime index and 'prec' column.
    Returns:
        DataFrame: Processed dataframe with days in zero-annual-precipitation years set to NaN.
    """
    # Group by year and calculate annual precipitation sum
    annual_precipitation = df.groupby(df.index.year)['prec'].sum()

    # Identify years with zero annual precipitation
    zero_precipitation_years = annual_precipitation[annual_precipitation == 0].index

    # Set days in zero-precipitation years to NaN
    for year in zero_precipitation_years:
        df.loc[df.index.year == year, 'prec'] = pd.NA

    return df


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
    """
    Adjusts for the bias between target data and the historic CMIP6 model runs. Loops through discrete blocks of
    data as suggested by Switanek et.al. (2017) (https://doi.org/10.5194/hess-21-2649-2017).
    """

    # Read predictor data
    if era5:
        predictor = read_era5l(predictor)

    # Determine variable type based on the mean value
    var = 'temp' if predictand.mean().mean() > 100 else 'prec'

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


class CMIP6DataProcessor:
    """
    Class to download, process, and bias adjust CMIP6 climate data for buffer zone around a weather station.

    Parameters
    ----------
    buffer_file : str
        Path to the buffer file containing station information.
    station : str
        Name of the station for which data processing is performed.
    starty : int
        Start year for data processing.
    endy : int
        End year for data processing.
    cmip_dir : str
        Directory to save CMIP6 data.

    Returns
    -------
    None
    """

    def __init__(self, buffer_file, station, starty, endy, cmip_dir, processes):
        self.buffer_file = buffer_file
        self.station = station
        self.starty = starty
        self.endy = endy
        self.cmip_dir = cmip_dir
        self.processes = processes

    def initialize_target(self):
        """
        Initialize Google Earth Engine (GEE), select station buffer from buffer file, and make it GEE-readable.
        """
        try:
            ee.Initialize()
        except Exception as e:
            ee.Authenticate()
            ee.Initialize()

        all_buffers = gpd.read_file(self.buffer_file, layer='station_buffers')
        buffer = all_buffers.loc[all_buffers['Station_Name'] == self.station]
        self.buffer_ee = geemap.geopandas_to_ee(buffer)

    def download_cmip6_data(self):
        """
        Download area-weighted aggregates of CMIP6 temperature and precipitation data for the selected buffer zone.
        """
        self.initialize_target()

        downloader_t = CMIPDownloader(var='tas', starty=self.starty, endy=self.endy, shape=self.buffer_ee,
                                      processes=self.processes, dir=self.cmip_dir)
        downloader_t.download()
        downloader_p = CMIPDownloader(var='pr', starty=self.starty, endy=self.endy, shape=self.buffer_ee,
                                      processes=self.processes, dir=self.cmip_dir)
        downloader_p.download()

    def process_cmip6_data(self):
        """
        Combine individual years of CMIP6 data to consistent timeseries and drops models with missing data.
        """
        processor_t = CMIPProcessor(file_dir=self.cmip_dir, var='tas', start=self.starty, end=self.endy)
        ssp2_tas_raw, ssp5_tas_raw = processor_t.get_results()

        processor_p = CMIPProcessor(file_dir=self.cmip_dir, var='pr', start=self.starty, end=self.endy)
        ssp2_pr_raw, ssp5_pr_raw = processor_p.get_results()

        self.ssp2_tas_raw = ssp2_tas_raw
        self.ssp5_tas_raw = ssp5_tas_raw
        self.ssp2_pr_raw = ssp2_pr_raw
        self.ssp5_pr_raw = ssp5_pr_raw

    def bias_adjustment(self, region_data):
        """
        Adjust for the bias between AWS data and the historic CMIP6 model runs.
        """
        self.initialize_target()
        self.process_cmip6_data()

        print('Running bias adjustment routine...')
        aws = search_dict(region_data, self.station)

        # Find the first and last non-NaN timestamp for each variable
        first_valid_temp = aws['temp'].first_valid_index()
        last_valid_temp = aws['temp'].last_valid_index()
        first_valid_prec = aws['prec'].first_valid_index()
        last_valid_prec = aws['prec'].last_valid_index()

        # Adjust training start and end dates for temperature
        train_start_temp = str(first_valid_temp)
        train_end_temp = str(last_valid_temp)

        # Adjust training start and end dates for precipitation
        train_start_prec = str(first_valid_prec)
        train_end_prec = str(last_valid_prec)

        # Adjust bias for temperature and precipitation
        self.ssp2_tas = adjust_bias(predictand=self.ssp2_tas_raw, predictor=aws, era5=False,
                                    train_start=train_start_temp, train_end=train_end_temp)
        self.ssp5_tas = adjust_bias(predictand=self.ssp5_tas_raw, predictor=aws, era5=False,
                                    train_start=train_start_temp, train_end=train_end_temp)
        self.ssp2_pr = adjust_bias(predictand=self.ssp2_pr_raw, predictor=aws, era5=False, train_start=train_start_prec,
                                   train_end=train_end_prec)
        self.ssp5_pr = adjust_bias(predictand=self.ssp5_pr_raw, predictor=aws, era5=False, train_start=train_start_prec,
                                   train_end=train_end_prec)

        self.ssp_tas_dict = {'SSP2_raw': self.ssp2_tas_raw, 'SSP2_adjusted': self.ssp2_tas,
                             'SSP5_raw': self.ssp5_tas_raw, 'SSP5_adjusted': self.ssp5_tas}
        self.ssp_pr_dict = {'SSP2_raw': self.ssp2_pr_raw, 'SSP2_adjusted': self.ssp2_pr, 'SSP5_raw': self.ssp5_pr_raw,
                            'SSP5_adjusted': self.ssp5_pr}

        print('Done!')


def dict_filter(dictionary, filter_string):
    """Returns a dict with all elements of the input dict that contain a filter string in their keys."""
    return {key.split('_')[0]: value for key, value in dictionary.items() if filter_string in key}


class DataFilter:
    def __init__(self, df, zscore_threshold=3, resampling_rate=None, prec=False, jump_threshold=5):
        self.df = df
        self.zscore_threshold = zscore_threshold
        self.resampling_rate = resampling_rate
        self.prec = prec
        self.jump_threshold = jump_threshold
        self.filter_all()

    def check_outliers(self):
        """
        A function for filtering a pandas dataframe for columns with obvious outliers
        and dropping them based on a z-score threshold.

        Returns
        -------
        models : list
            A list of columns identified as having outliers.
        """
        # Resample if rate specified
        if self.resampling_rate is not None:
            if self.prec:
                self.df = self.df.resample(self.resampling_rate).sum()
            else:
                self.df = self.df.resample(self.resampling_rate).mean()

        # Calculate z-scores for each column
        z_scores = pd.DataFrame((self.df - self.df.mean()) / self.df.std())

        # Identify columns with at least one outlier (|z-score| > threshold)
        cols_with_outliers = z_scores.abs().apply(lambda x: any(x > self.zscore_threshold))
        self.outliers = list(self.df.columns[cols_with_outliers])

        # Return the list of columns with outliers
        return self.outliers

    def check_jumps(self):
        """
        A function for checking a pandas dataframe for columns with sudden jumps or drops
        and returning a list of the columns that have them.

        Returns
        -------
        jumps : list
            A list of columns identified as having sudden jumps or drops.
        """
        cols = self.df.columns
        jumps = []

        for col in cols:
            diff = self.df[col].diff()
            if (abs(diff) > self.jump_threshold).any():
                jumps.append(col)

        self.jumps = jumps
        return self.jumps

    def filter_all(self):
        """
        A function for filtering a dataframe for columns with obvious outliers
        or sudden jumps or drops in temperature, and returning a list of the
        columns that have been filtered using either or both methods.

        Returns
        -------
        filtered_models : list
            A list of columns identified as having outliers or sudden jumps/drops in temperature.
        """
        self.check_outliers()
        self.check_jumps()
        self.filtered_models = list(set(self.outliers) | set(self.jumps))
        return self.filtered_models


def loop_checks(ssp_dict, **kwargs):
    """
    Wrapper for class DataFilter to iterate over all scenarios.
    """
    outliers = []
    jumps = []
    both_checks = []
    for scenario in ssp_dict.keys():
        filter = DataFilter(ssp_dict[scenario], **kwargs)
        outliers.extend(set(filter.outliers))
        jumps.extend(set(filter.jumps))
        both_checks.extend(set(filter.filtered_models))

    return outliers, jumps, both_checks


def drop_model(col_names, dict_or_df):
    """
    Drop columns with given names from either a dictionary of dataframes
    or a single dataframe.
    Parameters
    ----------
    col_names : list of str
        The list of model names to drop.
    dict_or_df : dict of pandas.DataFrame or pandas.DataFrame
        If a dict of dataframes, all dataframes in the dict will be edited.
        If a single dataframe, only that dataframe will be edited.
    Returns
    -------
    dict_of_dfs : dict of pandas.DataFrame or pandas.DataFrame
        The updated dictionary of dataframes or dataframe with dropped columns.
    """
    if isinstance(dict_or_df, dict):
        # loop through the dictionary and edit each dataframe
        for key in dict_or_df.keys():
            if all(col_name in dict_or_df[key].columns for col_name in col_names):
                dict_or_df[key] = dict_or_df[key].drop(columns=col_names)
        return dict_or_df
    elif isinstance(dict_or_df, pd.DataFrame):
        # edit the single dataframe
        if all(col_name in dict_or_df.columns for col_name in col_names):
            return dict_or_df.drop(columns=col_names)
    else:
        raise TypeError('Input must be a dictionary or a dataframe')


def apply_filters(temp_dict, prec_dict, **kwargs):
    """
    Applies data filters to temperature and precipitation dictionaries.
    Parameters
    ----------
    temp_dict : dict
        Dictionary containing temperature data.
    prec_dict : dict
        Dictionary containing precipitation data.
    **kwargs
        Additional keyword arguments for loop_checks function.
    Returns
    -------
    tuple of pandas.DataFrame
        Tuple containing filtered temperature and precipitation data.
    """
    tas_raw = dict_filter(temp_dict, 'raw')
    outliers, jumps, both_checks = loop_checks(tas_raw, **kwargs)
    print('Applying data filters...')
    print('Models with temperature outliers: ' + str(outliers))
    print('Models with temperature jumps: ' + str(jumps))
    print('Models excluded: ' + str(both_checks))

    return drop_model(both_checks, temp_dict), drop_model(both_checks, prec_dict)


def pickle_to_dict(file_path):
    """
    Loads a dictionary from a pickle file at a specified file path.
    Parameters
    ----------
    file_path : str
        The path of the pickle file to load.
    Returns
    -------
    dict
        The dictionary loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        dic = pickle.load(f)
    return dic


def dict_to_pickle(dic, target_path):
    """
    Saves a dictionary to a pickle file at the specified target path.
    Creates target directory if not existing.
    Parameters
    ----------
    dic : dict
        The dictionary to save to a pickle file.
    target_path : str
        The path of the file where the dictionary shall be stored.
    Returns
    -------
    None
    """
    target_dir = os.path.dirname(target_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(target_path, 'wb') as f:
        pickle.dump(dic, f)


def cmip_plot(ax, df, target, title=None, precip=False, intv_sum='M', intv_mean='10Y',
              target_label='Target', show_target_label=False, rolling=None):
    """Resamples and plots climate model and target data."""
    if intv_mean == '10Y' or intv_mean == '5Y' or intv_mean == '20Y':
        closure = 'left'
    else:
        closure = 'right'
    if not precip:
        if rolling is not None:
            ax.plot(df.resample(intv_mean, closed=closure, label='left').mean().iloc[:, :].rolling(rolling).mean(), linewidth=1.2)
        else:
            ax.plot(df.resample(intv_mean, closed=closure, label='left').mean().iloc[:, :], linewidth=1.2)
        era_plot, = ax.plot(target['temp'].resample(intv_mean).mean(), linewidth=1.5, c='red', label=target_label,
                            linestyle='dashed')
    else:
        if rolling is not None:
            ax.plot(df.resample(intv_sum, closed=closure, label='left').sum().iloc[:, :].rolling(rolling).mean(), linewidth=1.2)
        else:
            ax.plot(df.resample(intv_sum, closed=closure, label='left').sum().iloc[:, :], linewidth=1.2)
        era_plot, = ax.plot(target['prec'].resample(intv_sum).sum(), linewidth=1.5,
                            c='red', label=target_label, linestyle='dashed')
    if show_target_label:
        ax.legend(handles=[era_plot], loc='upper left')
    ax.set_title(title)
    ax.grid(True)


def cmip_plot_combined(data, target, title=None, precip=False, intv_sum='M', intv_mean='10Y',
                       target_label='Target', show=False, filename=None, out_dir='./', rolling=None):
    """Combines multiple subplots of climate data in different scenarios before and after bias adjustment.
    Shows target data for comparison"""
    figure, axis = plt.subplots(2, 2, figsize=(12, 12), sharex="col", sharey="all")
    t_kwargs = {'target': target, 'intv_mean': intv_mean, 'target_label': target_label, 'rolling': rolling}
    p_kwargs = {'target': target, 'intv_mean': intv_mean, 'target_label': target_label,
                'intv_sum': intv_sum, 'precip': True, 'rolling': rolling}
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not precip:
        cmip_plot(axis[0, 0], data['SSP2_raw'], show_target_label=True, title='SSP2 raw', **t_kwargs)
        cmip_plot(axis[0, 1], data['SSP2_adjusted'], title='SSP2 adjusted', **t_kwargs)
        cmip_plot(axis[1, 0], data['SSP5_raw'], title='SSP5 raw', **t_kwargs)
        cmip_plot(axis[1, 1], data['SSP5_adjusted'], title='SSP5 adjusted', **t_kwargs)
        figure.legend(data['SSP5_adjusted'].columns, loc='lower right', ncol=6, mode="expand")
        figure.tight_layout()
        figure.subplots_adjust(bottom=0.15, top=0.92)
        figure.suptitle(title, fontweight='bold')
        plt.savefig(out_dir + filename)
        if show:
            plt.show()
    else:
        cmip_plot(axis[0, 0], data['SSP2_raw'], show_target_label=True, title='SSP2 raw', **p_kwargs)
        cmip_plot(axis[0, 1], data['SSP2_adjusted'], title='SSP2 adjusted', **p_kwargs)
        cmip_plot(axis[1, 0], data['SSP5_raw'], title='SSP5 raw', **p_kwargs)
        cmip_plot(axis[1, 1], data['SSP5_adjusted'], title='SSP5 adjusted', **p_kwargs)
        figure.legend(data['SSP5_adjusted'].columns, loc='lower right', ncol=6, mode="expand")
        figure.tight_layout()
        figure.subplots_adjust(bottom=0.15, top=0.92)
        figure.suptitle(title, fontweight='bold')
        plt.savefig(out_dir + filename)
        if show:
            plt.show()


def df2long(df, intv_sum='M', intv_mean='Y', precip=False):
    """Resamples dataframes and converts them into long format to be passed to seaborn.lineplot()."""

    if precip:
        df = df.resample(intv_sum).sum()
        df = df.reset_index()
        df = df.melt('TIMESTAMP', var_name='model', value_name='prec')
    else:
        df = df.resample(intv_mean).mean()
        df = df.reset_index()
        df = df.melt('TIMESTAMP', var_name='model', value_name='temp')
    return df


def cmip_plot_ensemble(cmip, target, precip=False, intv_sum='M', intv_mean='Y', figsize=(10, 6), site_label:str=None,
                       target_label='ERA5L', show=True, out_dir='./', filename='cmip6_ensemble'):
    """
    Plots the multi-model mean of climate scenarios including the 90% confidence interval.
    Parameters
    ----------
    cmip: dict
        A dictionary with keys representing the different CMIP6 models and scenarios as pandas dataframes
        containing data of temperature and/or precipitation.
    target: pandas.DataFrame
        Dataframe containing the historical reanalysis data.
    precip: bool
        If True, plot the mean precipitation. If False, plot the mean temperature. Default is False.
    intv_sum: str
        Interval for precipitation sums. Default is monthly ('M').
    intv_mean: str
        Interval for the mean of temperature data or precipitation sums. Default is annual ('Y').
    figsize: tuple
        Figure size for the plot. Default is (10,6).
    show: bool
        If True, show the resulting plot. If False, do not show it. Default is True.
    out_dir: str
        Target directory to save figure
    """

    warnings.filterwarnings(action='ignore')
    figure, axis = plt.subplots(figsize=figsize)

    # Define color palette
    colors = ['darkorange', 'orange', 'darkblue', 'dodgerblue']
    # create a new dictionary with the same keys but new values from the list
    col_dict = {key: value for key, value in zip(cmip.keys(), colors)}

    if site_label is None:
        site_label = str()
    else:
        site_label = f'"{site_label}" - '

    if precip:
        for i in cmip.keys():
            df = df2long(cmip[i], intv_sum=intv_sum, intv_mean=intv_mean, precip=True)
            sns.lineplot(data=df, x='TIMESTAMP', y='prec', color=col_dict[i])
        axis.set(xlabel='Year', ylabel='Precipitation [mm]')
        if intv_sum == 'M':
            figure.suptitle(site_label + 'Ensemble Mean of Monthly Precipitation', fontweight='bold')
        elif intv_sum == 'Y':
            figure.suptitle(site_label + 'Ensemble Mean of Annual Precipitation', fontweight='bold')
        target_plot = axis.plot(target.resample(intv_sum).sum(), linewidth=1.5, c='black',
                                label=target_label, linestyle='dashed')
    else:
        for i in cmip.keys():
            df = df2long(cmip[i], intv_mean=intv_mean)
            sns.lineplot(data=df, x='TIMESTAMP', y='temp', color=col_dict[i])
        axis.set(xlabel='Year', ylabel='Air Temperature [K]')
        if intv_mean == '10Y':
            figure.suptitle(site_label + 'Ensemble Mean of 10y Air Temperature', fontweight='bold')
        elif intv_mean == 'Y':
            figure.suptitle(site_label + 'Ensemble Mean of Annual Air Temperature', fontweight='bold')
        elif intv_mean == 'M':
            figure.suptitle(site_label + 'Ensemble Mean of Monthly Air Temperature', fontweight='bold')
        target_plot = axis.plot(target.resample(intv_mean).mean(), linewidth=1.5, c='black',
                                label=target_label, linestyle='dashed')
    axis.legend(['SSP2_raw', '_ci1', 'SSP2_adjusted', '_ci2', 'SSP5_raw', '_ci3', 'SSP5_adjusted', '_ci4'],
                loc="upper center", bbox_to_anchor=(0.43, -0.15), ncol=4,
                frameon=False)  # First legend --> Workaround as seaborn lists CIs in legend
    leg = Legend(axis, target_plot, [target_label], loc='upper center', bbox_to_anchor=(0.83, -0.15), ncol=1,
                 frameon=False)  # Second legend (Target)
    axis.add_artist(leg)
    plt.grid()

    figure.tight_layout(rect=[0, 0.02, 1, 1])  # Make some room at the bottom
    if not precip:
        plt.savefig(out_dir + f'{filename}_temperature.png')
    else:
        plt.savefig(out_dir + f'{filename}_precipitation.png')

    if show:
        plt.show()
    warnings.filterwarnings(action='always')


def prob_plot(original, target, corrected, ax, title=None, ylabel="Temperature [K]", **kwargs):
    """
    Combines probability plots of climate model data before and after bias adjustment
    and the target data.

    Parameters
    ----------
    original : pandas.DataFrame
        The original climate model data.
    target : pandas.DataFrame
        The target data.
    corrected : pandas.DataFrame
        The climate model data after bias adjustment.
    ax : matplotlib.axes.Axes
        The axes on which to plot the probability plot.
    title : str, optional
        The title of the plot. Default is None.
    ylabel : str, optional
        The label for the y-axis. Default is "Temperature [K]".
    **kwargs : dict, optional
        Additional keyword arguments passed to the probscale.probplot() function.

    Returns
    -------
    fig : matplotlib Figure
        The generated figure.
    """

    scatter_kws = dict(label="", marker=None, linestyle="-")
    common_opts = dict(plottype="qq", problabel="", datalabel="", **kwargs)

    scatter_kws["label"] = "original"
    fig = probscale.probplot(original, ax=ax, scatter_kws=scatter_kws, **common_opts)

    scatter_kws["label"] = "target"
    fig = probscale.probplot(target, ax=ax, scatter_kws=scatter_kws, **common_opts)

    scatter_kws["label"] = "adjusted"
    fig = probscale.probplot(corrected, ax=ax, scatter_kws=scatter_kws, **common_opts)

    ax.set_title(title)

    ax.set_xlabel("Standard Normal Quantiles")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    score = round(target.corr(corrected), 2)
    ax.text(0.05, 0.8, f"R² = {score}", transform=ax.transAxes, fontsize=15)

    return fig


def pp_matrix(original, target, corrected, scenario=None, nrow=7, ncol=5, precip=False,
              starty=1979, endy=2022, show=False, out_dir='./', target_label='ERA5-Land', site:str=None):
    """
    Arranges the prob_plots of all CMIP6 models in a matrix and adds the R² score.
    Parameters
    ----------
    original : pandas.DataFrame
        The original climate model data.
    target : pandas.DataFrame
        The target data.
    corrected : pandas.DataFrame
        The climate model data after bias adjustment.
    scenario : str, optional
        The climate scenario to be added to the plot title.
    nrow : int, optional
        The number of rows in the plot matrix. Default is 7.
    ncol : int, optional
        The number of columns in the plot matrix. Default is 5.
    precip : bool, optional
        Indicates whether the data is precipitation data. Default is False.
    show : bool, optional
        Indicates whether to display the plot. Default is False.
    out_dir: str
        Target directory to save figure
    Returns
    -------
    None
    """

    starty = f'{str(starty)}-01-01'
    endy = f'{str(endy)}-12-31'
    period = slice(starty, endy)

    if site is None:
        site_label = str()
    else:
        site_label = f'"{site}" - '

    if precip:
        var = 'Precipitation'
        var_label = 'Monthly ' + var
        unit = ' [mm]'
        original = original.resample('M').sum()
        target = target.resample('M').sum()
        corrected = corrected.resample('M').sum()
    else:
        var = 'Temperature'
        var_label = 'Daily Mean ' + var
        unit = ' [K]'

    fig = plt.figure(figsize=(16, 16))

    for i, col in enumerate(original.columns):
        ax = plt.subplot(nrow, ncol, i + 1)
        prob_plot(original[col][period], target[period],
                  corrected[col][period], ax=ax, ylabel=var + unit)
        ax.set_title(col, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, ['original (CMIP6 raw)', f'target ({target_label})', 'adjusted (CMIP6 after SDM)'], loc='lower right',
               bbox_to_anchor=(0.96, 0.024), fontsize=20)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.7, wspace=0.4)
    starty = period.start.split('-')[0]
    endy = period.stop.split('-')[0]
    if scenario is None:
        fig.suptitle(site_label + f'Probability Plots of CMIP6 and {target_label} ' + var_label + ' (' + starty + '-' + endy + ')',
                     fontweight='bold', fontsize=20)
    else:
        fig.suptitle(site_label + 'Probability Plots of CMIP6 (' + scenario + f') and {target_label} ' + var_label +
                     ' (' + starty + '-' + endy + ')', fontweight='bold', fontsize=20)
    plt.subplots_adjust(top=0.93)
    if precip:
        plt.savefig(out_dir + f'cmip6_ensemble_probplots_{site}_precipitation_{scenario}.png')
    else:
        plt.savefig(out_dir + f'cmip6_ensemble_probplots_{site}_temperature_{scenario}.png')

    if show:
        plt.show()


def write_output(adj_dict: dict, output: str, station: str, starty: str, endy: str, type: str, ndigits: int=3):
    """
    Writes the full output of the adjusted dictionary to CSV files. Variable name is determined based on the mean value
    of the first DataFrame.
    Parameters
    ----------
    adj_dict : dict
        A dictionary containing DataFrames to be processed.
    output : str
        Output directory path.
    station : str
        Station name for naming convention.
    starty : str
        Start year for naming convention.
    endy : str
        End year for naming convention.
    type : str
        Type of the file written. Usually 'full' or 'summary'.
    ndigits : int
        Number of digits the dataframes will be rounded to before writing them to file.
    Returns
    -------
    None
    """
    if adj_dict['SSP2_raw'].mean().mean() > 100:
        var = 'temp'
    else:
        var = 'prec'
    final_output = f'{output}posterior/{type}/'
    if not os.path.exists(final_output):
        os.makedirs(final_output)
    for name, data in adj_dict.items():
        round(data, ndigits).to_csv(f'{final_output}{name}_{station}_{type}_{var}_{starty}-{endy}.csv')


def ensemble_summary(df):
    """
    Calculate ensemble summary statistics for the input DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with ensemble member data.
    Returns
    -------
    pandas.DataFrame
        DataFrame with ensemble summary statistics including mean, median, min, max, standard deviation,
        and 90% confidence interval bounds.
    """
    summary_df = pd.DataFrame(index=df.index)
    # Calculate ensemble mean
    summary_df['ensemble_mean'] = df.mean(axis=1)
    # Calculate ensemble median
    summary_df['ensemble_median'] = df.median(axis=1)
    # Calculate ensemble min
    summary_df['ensemble_min'] = df.min(axis=1)
    # Calculate ensemble max
    summary_df['ensemble_max'] = df.max(axis=1)
    # Calculate ensemble standard deviation
    summary_df['ensemble_sd'] = df.std(axis=1)
    # Calculate ensemble 90% confidence interval lower bound
    summary_df['ensemble_90perc_CI_lower'] = df.quantile(0.05, axis=1)
    # Calculate ensemble 90% confidence interval upper bound
    summary_df['ensemble_90perc_CI_upper'] = df.quantile(0.95, axis=1)

    return summary_df


def summary_dict(results_dict: dict):
    """
    Generate a dictionary of ensemble summary DataFrames for each value in the input dictionary.
    Parameters
    ----------
    results_dict : dict
        Input dictionary containing ensemble results.
    Returns
    -------
    dict
        A summary dictionary with keys as original keys and values as ensemble summaries.
    """
    return {key: ensemble_summary(value) for key, value in results_dict.items()}


class StationPreprocessor:
    def __init__(self, input_dir, output_dir, buffer_radius=1000, show=True, sd_factor=2):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.buffer_radius = buffer_radius
        self.show = show
        self.sd_factor = sd_factor
        self.gis_dir = self.output_dir + 'GIS/'
        self.gis_file = self.gis_dir + 'station_gis.gpkg'

    def read_data_and_create_buffers(self):
        self.region_data, self.station_coords = read_station_data(self.input_dir)
        if not os.path.exists(self.gis_dir):
            os.makedirs(self.gis_dir)
        self.buffered_stations = create_buffer(self.station_coords, self.gis_file, buffer_radius=self.buffer_radius,
                                                  write_files=True)

    def data_checks_and_plots(self):
        # Plots before filters
        plot_region_data(self.region_data, show=self.show,
                            output=self.output_dir + 'overview_plots/aws_data_raw.png')

        # Remove temperature outliers
        process_nested_dict(self.region_data, remove_outliers, sd_factor=self.sd_factor)

        # Remove years with an annual precipitation of 0
        process_nested_dict(self.region_data, remove_annual_zeros)

        # Plot after filters
        plot_region_data(self.region_data, show=self.show,
                            output=self.output_dir + 'overview_plots/aws_data_filtered.png')

    def full_preprocessing(self):
        self.read_data_and_create_buffers()
        self.data_checks_and_plots()


class ClimateScenarios:
    def __init__(self, output, region_data, station, buffer_file, download=False, load_backup=True, show=True,
                 starty=1979, endy=2100, processes=5):
        self.output = output
        self.download = download
        self.load_backup = load_backup
        self.show = show
        self.buffer_file = buffer_file
        self.station = station
        self.starty = starty
        self.endy = endy
        self.region_data = region_data
        self.processes = processes
        self.aws = search_dict(self.region_data, self.station)

    def cmip6_data_processing(self):
        cmip_dir = f'{self.output}raw/'
        self.cmip6_station = CMIP6DataProcessor(self.buffer_file, self.station, self.starty, self.endy,
                                                   cmip_dir, self.processes)
        print(f'CMIP6DataProcessor instance for station "{self.station}" configured.')

        if self.download:
            self.cmip6_station.download_cmip6_data()

        self.cmip6_station.bias_adjustment(self.region_data)
        self.temp_cmip = self.cmip6_station.ssp_tas_dict
        self.prec_cmip = self.cmip6_station.ssp_pr_dict

    def data_checks(self):
        self.temp_cmip, self.prec_cmip = apply_filters(self.temp_cmip, self.prec_cmip, zscore_threshold=3,
                                                          jump_threshold=5, resampling_rate='Y')
        print(f'Consistency-checks applied to adjusted data for "{self.station}".')

        process_nested_dict(self.temp_cmip, round, ndigits=3)
        process_nested_dict(self.prec_cmip, round, ndigits=3)
        print(f'Data for "{self.station}" rounded to save storage space.')

    def backup_files(self):
        if not self.load_backup:
            dict_to_pickle(self.temp_cmip, self.output + 'back_ups/temp_' + self.station + '_adjusted.pickle')
            dict_to_pickle(self.prec_cmip, self.output + 'back_ups/prec_' + self.station + '_adjusted.pickle')
        else:
            self.temp_cmip = pickle_to_dict(self.output + 'back_ups/temp_' + self.station + '_adjusted.pickle')
            self.prec_cmip = pickle_to_dict(self.output + 'back_ups/prec_' + self.station + '_adjusted.pickle')
        print(f'Back-up of adjusted data for "{self.station}" written.')

    def plots(self):
        cmip_plot_combined(data=self.temp_cmip, target=self.aws,
                              title=f'"{self.station}" - 5y Rolling Mean of Annual Air Temperature',
                              target_label='Observations',
                              filename=f'cmip6_bias_adjustment_{self.station}_temperature.png', show=self.show,
                              intv_mean='Y', rolling=5, out_dir=self.output + 'Plots/')
        cmip_plot_combined(data=self.prec_cmip, target=self.aws.dropna(),
                              title=f'"{self.station}" - 5y Rolling Mean of Annual Precipitation', precip=True,
                              target_label='Observations',
                              filename=f'cmip6_bias_adjustment_{self.station}_precipitation.png', show=self.show,
                              intv_sum='Y', rolling=5, out_dir=self.output + 'Plots/')
        print(f'Figures for CMIP6 bias adjustment for "{self.station}" created.')

        cmip_plot_ensemble(self.temp_cmip, self.aws['temp'], intv_mean='Y', show=self.show,
                              out_dir=self.output + 'Plots/', target_label="Observations",
                              filename=f'cmip6_ensemble_{self.station}', site_label=self.station)
        cmip_plot_ensemble(self.prec_cmip, self.aws['prec'].dropna(), precip=True, intv_sum='Y', show=self.show,
                              out_dir=self.output + 'Plots/', target_label="Observations", site_label=self.station,
                              filename=f'cmip6_ensemble_{self.station}')
        print(f'Figures for CMIP6 ensembles for "{self.station}" created.')

        start_temp = self.aws['temp'].first_valid_index().year
        end_temp = self.aws['temp'].last_valid_index().year
        start_prec = self.aws['prec'].dropna().first_valid_index().year
        end_prec = self.aws['prec'].dropna().last_valid_index().year

        pp_matrix(self.temp_cmip['SSP2_raw'], self.aws['temp'], self.temp_cmip['SSP2_adjusted'], scenario='SSP2',
                     starty=start_temp, endy=end_temp, target_label='Observed', out_dir=self.output + 'Plots/',
                     show=self.show, site=self.station)
        pp_matrix(self.temp_cmip['SSP5_raw'], self.aws['temp'], self.temp_cmip['SSP5_adjusted'], scenario='SSP5',
                     starty=start_temp, endy=end_temp, target_label='Observed', out_dir=self.output + 'Plots/',
                     show=self.show, site=self.station)

        pp_matrix(self.prec_cmip['SSP2_raw'], self.aws['prec'].dropna().astype(float), self.prec_cmip['SSP2_adjusted'],
                     precip=True, starty=start_prec, endy=end_prec, target_label='Observed', scenario='SSP2',
                     out_dir=self.output + 'Plots/', show=self.show, site=self.station)
        pp_matrix(self.prec_cmip['SSP5_raw'], self.aws['prec'].dropna().astype(float), self.prec_cmip['SSP5_adjusted'],
                     precip=True, starty=start_prec, endy=end_prec, target_label='Observed', scenario='SSP5',
                     out_dir=self.output + 'Plots/', show=self.show, site=self.station)
        print(f'Figures for CMIP6 bias adjustment performance for "{self.station}" created.')

        plt.close('all')

    def write_output_files(self):
        write_output(self.prec_cmip, self.output, self.station, self.starty, self.endy, type='full')
        write_output(self.temp_cmip, self.output, self.station, self.starty, self.endy, type='full')

        temp_summary = summary_dict(self.temp_cmip)
        prec_summary = summary_dict(self.prec_cmip)
        write_output(temp_summary, self.output, self.station, self.starty, self.endy, type='summary')
        write_output(prec_summary, self.output, self.station, self.starty, self.endy, type='summary')
        print(f'Output files for "{self.station}" written.')

    def complete_workflow(self):
        self.cmip6_data_processing()
        self.data_checks()
        self.backup_files()
        self.plots()
        self.write_output_files()
        print(f'Finished workflow for station "{self.station}".\n--------------------------------------')


def count_dataframes(dictionary):
    count = 0
    for value in dictionary.values():
        if isinstance(value, dict):
            count += count_dataframes(value)
        elif isinstance(value, pd.DataFrame):  # Assuming pandas DataFrame
            count += 1
    return count