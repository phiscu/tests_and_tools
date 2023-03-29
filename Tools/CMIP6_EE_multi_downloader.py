import ee
import geemap
import logging
import multiprocessing
from retry import retry
import geopandas as gpd
import requests

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()         # authenticate when using GEE for the first time
    ee.Initialize()
import os
os.chdir("/home/phillip/Seafile/EBA-CA/Repositories/ee_download_test")

"""
This tool downloads data from Earth Engine using parallel requests.
It extracts the timeseries of 8-day max LST from MODIS MOD11A2 per GAUL level-2 region
for all regions in South America, with each time-series written to its own file.
"""

##

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

# --> als class() mit den funktionen start() (startet den download) und to_array() (lädt die csv)

##

output_gpkg = '/home/phillip/Seafile/EBA-CA/Repositories/matilda_edu/output/catchment_data.gpkg'
catchment_new = gpd.read_file(output_gpkg, layer='catchment_new')
catchment = geemap.geopandas_to_ee(catchment_new)

download_cmip('tas', 1979, 2100, catchment, 25)
download_cmip('pr', 1979, 2100, catchment, 25)

##
import pandas as pd

cols = list(pd.read_csv('cmip6_pr_2022.csv', nrows=1))


def read_cmip(filename):
    df = pd.read_csv(filename)
    df = df.drop(['system:index', '.geo', 'system:time_start'], axis=1)
    df = df.rename(columns={'isodate': 'TIMESTAMP'})
    return df

def append_df(dir_path, var, hist=True):
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


append_df('.', 'tas', hist=False)

# --> append respective scenario data with their historic data

