##
import ee
import geemap
import logging
import multiprocessing
import geopandas as gpd
import concurrent.futures
import os
import requests
from retry import retry
from tqdm import tqdm


try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()         # authenticate when using GEE for the first time
    ee.Initialize()
os.chdir("/home/phillip/Seafile/EBA-CA/Repositories/ee_download_test")

"""
This tool downloads daily CMIP6 data from Google Earth Engine using parallel requests.
It applies an area-weighted averaging of all overlapping grid cells within a provided
polygon and stores individual years in separate CSV files.
"""

## CMIPDownloader as a function

def download_cmip(var, starty, endy, shape, processes=10):

    def getRequests(starty, endy):
        """Generates a list of years to be downloaded."""
        return [i for i in range(starty, endy+1)]

    global getResult

    @retry(tries=10, delay=1, backoff=2)
    def getResult(index, year):
        """Handle the HTTP requests to download one year of CMIP6 data."""

        start = str(year) + '-01-01'
        end = str(year + 1) + '-01-01'

        startDate = ee.Date(start)
        endDate = ee.Date(end)
        n = endDate.difference(startDate, 'day').subtract(1)

        def getImageCollection(var):
            collection = ee.ImageCollection('NASA/GDDP-CMIP6') \
                .select(var) \
                .filterDate(startDate, endDate) \
                .filterBounds(shape)
            return collection

        def renameBandName(b):
            split = ee.String(b).split('_')
            return ee.String(split.splice(split.length().subtract(2), 1).join("_"))

        def buildFeature(i):
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
                geometry=shape,
            ).combine(
                ee.Dictionary({'system:time_start': t1.millis(), 'isodate': t1.format('YYYY-MM-dd')})
            )
            return ee.Feature(None, dict)

        # Create features for all days in the respective year
        collection = getImageCollection(var)
        year_feature = ee.FeatureCollection(ee.List.sequence(0, n).map(buildFeature))

        # Create a download URL for a CSV containing the feature collection
        url = year_feature.getDownloadURL()

        # Handle downloading the actual annual csv
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            r.raise_for_status()

        filename = 'cmip6_' + var + '_' + str(year) + '.csv'
        with open(filename, 'w') as f:
            f.write(r.text)

        print("Done: ", index)

    logging.basicConfig()
    items = getRequests(starty, endy)

    pool = multiprocessing.Pool(processes)
    pool.starmap(getResult, enumerate(items))

    pool.close()
    pool.join()


## CMIPDownloader as a class including a status bar
class CMIPDownloader:
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

        def getRequests(starty, endy):
            """Generates a list of years to be downloaded."""
            return [i for i in range(starty, endy+1)]

        @retry(tries=10, delay=1, backoff=2)
        def getResult(index, year):
            """Handle the HTTP requests to download one year of CMIP6 data."""
            start = str(year) + '-01-01'
            end = str(year + 1) + '-01-01'
            startDate = ee.Date(start)
            endDate = ee.Date(end)
            n = endDate.difference(startDate, 'day').subtract(1)

            def getImageCollection(var):
                collection = ee.ImageCollection('NASA/GDDP-CMIP6') \
                    .select(var) \
                    .filterDate(startDate, endDate) \
                    .filterBounds(self.shape)
                return collection

            def renameBandName(b):
                split = ee.String(b).split('_')
                return ee.String(split.splice(split.length().subtract(2), 1).join("_"))

            def buildFeature(i):
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

            # Create features for all days in the respective year
            collection = getImageCollection(self.var)
            year_feature = ee.FeatureCollection(ee.List.sequence(0, n).map(buildFeature))

            # Create a download URL for a CSV containing the feature collection
            url = year_feature.getDownloadURL()

            # Handle downloading the actual annual csv
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                r.raise_for_status()
            filename = os.path.join(self.directory, 'cmip6_' + self.var + '_' + str(year) + '.csv')
            with open(filename, 'w') as f:
                f.write(r.text)

            return index

        items = getRequests(self.starty, self.endy)

        with tqdm(total=len(items), desc="Downloading CMIP6 data") as pbar:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.processes) as executor:
                for i, year in enumerate(items):
                    results.append(executor.submit(getResult, i, year))
                for future in concurrent.futures.as_completed(results):
                    index = future.result()
                    pbar.update(1)

        print("All downloads complete.")

## Application example

output_gpkg = '/home/phillip/Seafile/EBA-CA/Repositories/matilda_edu/output/catchment_data.gpkg'
catchment_new = gpd.read_file(output_gpkg, layer='catchment_new')
catchment = geemap.geopandas_to_ee(catchment_new)

downloader = CMIPDownloader('tas', 1979, 2100, catchment, processes=25, dir='./new_test')
downloader.download()

##
import pandas as pd

cols = list(pd.read_csv('cmip6_pr_2022.csv', nrows=1))


def read_cmip(filename):
    """Read files downloaded by download_cmip() and drop obsolete columns"""
    df = pd.read_csv(filename)
    df = df.drop(['system:index', '.geo', 'system:time_start'], axis=1)
    df = df.rename(columns={'isodate': 'TIMESTAMP'})
    return df

def append_df(var, dir_path='.', hist=True):
    df_list = []
    if hist:
        starty = 1979
        endy = 2014
    else:
        starty = 2015
        endy = 2100

    for i in range(starty, endy+1):
        filename = dir_path + '/cmip6_' + var + '_' + str(i) + '.csv'
        df_list.append(read_cmip(filename))
    if hist:
        hist_df = pd.concat(df_list, ignore_index=True).drop('historical_GFDL-CM4_' + var, axis=1)
        hist_df.columns = hist_df.columns.str.lstrip('historical_')
        return hist_df
    else:
        ssp_df = pd.concat(df_list, ignore_index=True).drop(['ssp585_GFDL-CM4_' + var, 'ssp245_GFDL-CM4_' + var], axis=1)
        return ssp_df


append_df('tas', hist=False)



# --> append respective scenario data with their historic data:
    # - append_df
    # - separate scenarios
    # - remove ssp prefix
    # - join both scenarios indidvidually with hist
    # - save two dfs


