##
import ee
import geemap
import pandas as pd
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

"""
This tool applies an approach by [Noel Gorelick et. al.](https://gorelick.medium.com/fast-er-downloads-a2abd512aa26)
to download daily CMIP6 data from Google Earth Engine using parallel requests. It applies an area-weighted averaging 
to all overlapping grid cells within a provided polygon and stores individual years in separate CSV files.
The files are then loaded, pre-processed, and combined in dataframes.

@author: Phillip Schuster and Alexander Georgi
"""

## Path

wd = '/home/phillip/Seafile/EBA-CA/Repositories'


## Class to download spatially averaged CMIP6 data vor a given period, variable, and spatial subset.


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

        def getRequests(starty, endy):
            """Generates a list of years to be downloaded. [Client side]"""

            return [i for i in range(starty, endy+1)]

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


## Usage example

output_gpkg = wd + '/matilda_edu/output/catchment_data.gpkg'
catchment_new = gpd.read_file(output_gpkg, layer='catchment_new')
catchment = geemap.geopandas_to_ee(catchment_new)

downloader = CMIPDownloader('tas', 2000, 2020, catchment, processes=25, dir=wd + '/ee_download_test/new_test')
downloader.download()


## Class to pre-process the downloaded files

class CMIPProcessor:
    """Class to read and pre-process CSV files downloaded by the CMIPDownloader class."""
    def __init__(self, var, file_dir='.'):
        self.file_dir = file_dir
        self.var = var
        self.df_hist = self.append_df(self.var, self.file_dir, hist=True)
        self.df_ssp = self.append_df(self.var, self.file_dir, hist=False)
        self.ssp2_common, self.ssp5_common, self.hist_common,\
            self.common_models, self.dropped_models = self.process_dataframes()
        self.ssp2, self.ssp5 = self.get_results()

    def read_cmip(self, filename):
        """Reads CMIP6 CSV files and drops redundant columns."""

        df = pd.read_csv(filename, index_col='isodate', parse_dates=['isodate'])
        df = df.drop(['system:index', '.geo', 'system:time_start'], axis=1)
        return df

    def append_df(self, var, file_dir='.', hist=True):
        """Reads CMIP6 CSV files of individual years and concatenates them into dataframes for the full downloaded
        period. Historical and scenario datasets are treated separately. Converts precipitation unit to mm."""

        df_list = []
        if hist:
            starty = 1979
            endy = 2014
        else:
            starty = 2015
            endy = 2100
        for i in range(starty, endy + 1):
            filename = file_dir + 'cmip6_' + var + '_' + str(i) + '.csv'
            df_list.append(self.read_cmip(filename))
        if hist:
            hist_df = pd.concat(df_list)
            if var == 'pr':
                hist_df = hist_df * 86400       # from kg/(m^2*s) to mm/day
            return hist_df
        else:
            ssp_df = pd.concat(df_list)
            if var == 'pr':
                ssp_df = ssp_df * 86400       # from kg/(m^2*s) to mm/day
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


## Usage example
processor = CMIPProcessor(file_dir=wd + '/ee_download_test/', var='pr')
ssp2_pr_raw, ssp5_pr_raw = processor.get_results()
processor = CMIPProcessor(file_dir=wd + '/ee_download_test/', var='tas')
ssp2_tas_raw, ssp5_tas_raw = processor.get_results()


## Bias-adjustment of CMIP6 data using ERA5-Land

from bias_correction import BiasCorrection

era5_file = wd + '/matilda_edu/output/ERA5L.csv'


def read_era5l(file):
    return pd.read_csv(file, **{
        'usecols':      ['temp', 'prec', 'dt'],
        'index_col':    'dt',
        'parse_dates':  ['dt']})


def adjust_bias(predictand, predictor, method='normal_mapping'):
    predictor = read_era5l(predictor)
    if predictand.mean().mean() > 100:
        var = 'temp'
    else:
        var = 'prec'
    training_period = slice('1979-01-01', '2014-12-31')
    prediction_period = slice('1979-01-01', '2100-12-31')
    corr = pd.DataFrame()
    for m in predictand.columns:
        x_train = predictand[m][training_period].squeeze()
        y_train = predictor[training_period][var].squeeze()
        x_predict = predictand[m][prediction_period].squeeze()
        bc_corr = BiasCorrection(y_train, x_train, x_predict)
        corr[m] = pd.DataFrame(bc_corr.correct(method=method))

    return corr

ssp2_tas = adjust_bias(ssp2_tas_raw, era5_file)
ssp5_tas = adjust_bias(ssp5_tas_raw, era5_file)
ssp2_pr = adjust_bias(ssp2_pr_raw, era5_file)
ssp5_pr = adjust_bias(ssp5_pr_raw, era5_file)

era5 = read_era5l(era5_file)

## Plots
import matplotlib.pyplot as plt


def cmip_plot(ax, df, title, precip=False, intv_sum='M', intv_mean='10Y',  era_label=False):
    if not precip:
        ax.plot(df.resample(intv_mean).mean().iloc[:, :-1], linewidth=0.6)
        ax.plot(df.resample(intv_mean).mean().iloc[:, -1], linewidth=1, c='black')
        era_plot, = ax.plot(era5['temp'].resample(intv_mean).mean(), linewidth=1.5, c='red', label='ERA5L',
                            linestyle='dashed')
    else:
        ax.plot(df.resample(intv_sum).sum().resample(intv_mean).mean().iloc[:, :-1],
                linewidth=0.6)
        ax.plot(df.resample(intv_sum).sum().resample(intv_mean).mean().iloc[:, -1],
                linewidth=1, c='black')
        era_plot, = ax.plot(era5['prec'].resample(intv_sum).sum().resample(intv_mean).mean(), linewidth=1.5, c='red',
                    label='ERA5L', linestyle='dashed')
    if era_label:
        ax.legend(handles=[era_plot], loc='upper left')
    ax.set_title(title)
    ax.grid(True)


# Temperature:
interval = '10Y'
figure, axis = plt.subplots(2, 2, figsize=(12, 12), sharex="col", sharey="all")
cmip_plot(axis[0, 0], ssp2_tas_raw, era_label=True, title='SSP2 raw', intv_mean=interval)
cmip_plot(axis[0, 1], ssp2_tas, title='SSP2 adjusted', intv_mean=interval)
cmip_plot(axis[1, 0], ssp5_tas_raw, title='SSP5 raw', intv_mean=interval)
cmip_plot(axis[1, 1], ssp5_tas, title='SSP5 adjusted', intv_mean=interval)
figure.legend(ssp5_tas.columns, loc='lower right', ncol=6, mode="expand")
figure.tight_layout()
figure.subplots_adjust(bottom=0.15, top=0.92)
figure.suptitle('10y Mean of Air Temperature', fontweight='bold')
plt.show()

# Precipitation:
interval = '10Y'
figure, axis = plt.subplots(2, 2, figsize=(12, 12), sharex="col", sharey="all")
cmip_plot(axis[0, 0], ssp2_pr_raw, era_label=True, title='SSP2 raw', precip=True)
cmip_plot(axis[0, 1], ssp2_pr, title='SSP2 adjusted', precip=True)
cmip_plot(axis[1, 0], ssp5_pr_raw, title='SSP5 raw', precip=True)
cmip_plot(axis[1, 1], ssp5_pr, title='SSP5 adjusted', precip=True)
figure.legend(ssp5_pr.columns, loc='lower right', ncol=6, mode="expand")
figure.tight_layout()
figure.subplots_adjust(bottom=0.15, top=0.92)
figure.suptitle('10y Mean of Monthly Precipitation', fontweight='bold')
plt.show()


# ZWEI AUSREIẞER BEI DEN TEMPERATUREN
# ERA5 NIEDERSCHLAG CA. 4 MAL SO HOCH WIE CMIP. FEHLER? BEI ECMWF DATEN GENAUSO?

############# CONTINUE ###################
