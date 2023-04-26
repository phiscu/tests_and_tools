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

# downloader = CMIPDownloader('tas', 2000, 2020, catchment, processes=25, dir=wd + '/ee_download_test/new_test')
# downloader.download()


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
    """Reads ERA5-Land data, drops redundant columns, and adds DatetimeIndex.
    Resamples the dataframe to reduce the DatetimeIndex to daily resolution."""
    return pd.read_csv(file, **{
        'usecols':      ['temp', 'prec', 'dt'],
        'index_col':    'dt',
        'parse_dates':  ['dt']}).resample('D').agg({'temp': 'mean', 'prec': 'sum'})


def adjust_bias(predictand, predictor, method='normal_mapping'):
    """Applies scaled distribution mapping to all passed climate projections (predictand)
     based on a predictor timeseries."""
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
ssp2_pr = adjust_bias(ssp2_pr_raw, era5_file)       # method='gamma_mapping' results in a strange output
ssp5_pr = adjust_bias(ssp5_pr_raw, era5_file)

era5 = read_era5l(era5_file)

## Timeseries plots
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

##
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from matplotlib.legend import Legend

def df2long(df, intv_sum='M', intv_mean='Y', precip=False):       # Convert dataframes to long format for use in seaborn-lineplots.
    if precip:
        df = df.resample(intv_sum).sum().resample(intv_mean).mean()
        df = df.reset_index()
        df = df.melt('TIMESTAMP', var_name='model', value_name='tp')
    else:
        df = df.resample(intv_mean).mean()
        df = df.reset_index()
        df = df.melt('TIMESTAMP', var_name='model', value_name='t2m')
    return df


def cmip_plot_ensemble(cmip, era, precip=False, intv_sum='M', intv_mean='Y', figsize=(10, 6), show=True):
    warnings.filterwarnings(action='ignore')
    figure, axis = plt.subplots(figsize=figsize)

    # Define color palette
    colors = ['darkorange', 'orange', 'darkblue', 'dodgerblue']
    # create a new dictionary with the same keys but new values from the list
    col_dict = {key: value for key, value in zip(cmip.keys(), colors)}

    if precip:
        for i in cmip.keys():
            df = df2long(cmip[i], intv_sum=intv_sum, intv_mean=intv_mean, precip=True)
            sns.lineplot(data=df, x='TIMESTAMP', y='tp', color=col_dict[i])
        axis.set(xlabel='Year', ylabel='Mean Precipitation [mm]')
        if intv_sum=='M':
            figure.suptitle('Mean Monthly Precipitation [mm]', fontweight='bold')
        elif intv_sum=='Y':
            figure.suptitle('Mean Annual Precipitation [mm]', fontweight='bold')
        era_plot = axis.plot(era.resample(intv_sum).sum().resample(intv_mean).mean(), linewidth=1.5, c='black',
                             label='ERA5', linestyle='dashed')
    else:
        for i in cmip.keys():
            df = df2long(cmip[i], intv_mean=intv_mean)
            sns.lineplot(data=df, x='TIMESTAMP', y='t2m', color=col_dict[i])
        axis.set(xlabel='Year', ylabel='Mean Air Temperature [K]')
        if intv_mean=='10Y':
            figure.suptitle('Mean 10y Air Temperature [K]', fontweight='bold')
        elif intv_mean == 'Y':
            figure.suptitle('Mean Annual Air Temperature [K]', fontweight='bold')
        elif intv_mean == 'M':
            figure.suptitle('Mean Monthly Air Temperature [K]', fontweight='bold')
        era_plot = axis.plot(era.resample(intv_mean).mean(), linewidth=1.5, c='black',
                         label='ERA5', linestyle='dashed')
    axis.legend(['SSP2_raw', '_ci1', 'SSP2_adjusted', '_ci2', 'SSP5_raw', '_ci3', 'SSP5_adjusted', '_ci4'], loc="upper center", bbox_to_anchor=(0.43, -0.15), ncol=4,
                frameon=False)  # First legend --> Workaround as new seaborn version listed CIs in legend
    leg = Legend(axis, era_plot, ['ERA5L'], loc='upper center', bbox_to_anchor=(0.83, -0.15), ncol=1,
                 frameon=False)  # Second legend (ERA5)
    axis.add_artist(leg)
    plt.grid()

    figure.tight_layout(rect=[0, 0.02, 1, 1]) # Make some room at the bottom

    if show:
        plt.show()
    warnings.filterwarnings(action='always')


ssp_tas_dict = {'SSP2_raw': ssp2_tas_raw, 'SSP2_adjusted': ssp2_tas, 'SPP5_raw': ssp5_tas_raw, 'SSP5_adjusted': ssp5_tas}
ssp_pr_dict = {'SSP2_raw': ssp2_pr_raw, 'SSP2_adjusted': ssp2_pr, 'SPP5_raw': ssp5_pr_raw, 'SSP5_adjusted': ssp5_pr}

# Temperature:
cmip_plot_ensemble(ssp_tas_dict, era5['temp'], intv_mean='Y')

# Precipitation:
cmip_plot_ensemble(ssp_pr_dict, era5['prec'], precip=True, intv_sum='Y', intv_mean='Y')

## Violin plots

# Rearrange dictionaries for two-column display
tas_raw = {'SSP2': ssp2_tas_raw, 'SSP5': ssp5_tas_raw}
tas_adjusted = {'SSP2': ssp2_tas, 'SSP5': ssp5_tas}

for i in tas_raw.keys():
    tas_raw[i] = tas_raw[i].loc[slice('1979-01-01', '2015-12-31')].copy()
    tas_raw[i]['ERA5L'] = era5['temp'][slice('1979-01-01', '2015-12-31')]

for i in tas_adjusted.keys():
    tas_adjusted[i] = tas_adjusted[i].loc[slice('1979-01-01', '2015-12-31')].copy()
    tas_adjusted[i]['ERA5L'] = era5['temp'][slice('1979-01-01', '2015-12-31')]

fig = plt.figure(figsize=(20, 20))
outer = fig.add_gridspec(1, 2)

inner = outer[0].subgridspec(2, 1)
axis = inner.subplots(sharex='col')

# Calculate total range of all input datasets
all_data = pd.concat([df2long(tas_raw[i]) for i in tas_raw.keys()] + [df2long(tas_adjusted[i]) for i in tas_adjusted.keys()])
xmin, xmax = all_data['t2m'].min(), all_data['t2m'].max()

for (i, k) in zip(tas_raw.keys(), range(0, 4, 1)):
    df = df2long(tas_raw[i])
    axis[k].grid()
    sns.violinplot(ax=axis[k], x='t2m', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlim=(xmin-1, xmax+1))
    axis[k].set_ylabel(i, fontsize=18, fontweight='bold')
    if k == 0:
        axis[k].set_title('Before Scaled Distribution Mapping', fontweight='bold', fontsize=18)
plt.xlabel('Air Temperature [K]')

inner = outer[1].subgridspec(2, 1)
axis = inner.subplots(sharex='col')
for (i, k) in zip(tas_adjusted.keys(), range(0, 4, 1)):
    df = df2long(tas_adjusted[i])
    axis[k].grid()
    sns.violinplot(ax=axis[k], x='t2m', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlim=(xmin-1, xmax+1))
    axis[k].set_ylabel(i, fontsize=18, fontweight='bold')
    axis[k].get_yaxis().set_visible(False)
    if k == 0:
        axis[k].set_title('After Scaled Distribution Mapping', fontweight='bold', fontsize=18)
plt.xlabel('Air Temperature [K]')

fig.suptitle('Kernel Density Estimation of Mean Annual Air Temperature (1982-2020)', fontweight='bold', fontsize=20)
fig.tight_layout()
fig.subplots_adjust(top=0.93)
plt.show()

## Probability plots
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

fig, axs = plt.subplots(7, 5, figsize=(20, 20))

for i, col in enumerate(ssp2_tas_raw.columns):
    row, colax = divmod(i, 5)
    ax = axs[row, colax]

    raw_data = ssp2_tas_raw[col].dropna()
    adjusted_data = ssp2_tas[col].dropna()

    stats.probplot(raw_data, plot=ax, rvalue=True)
    ax.get_lines()[1].set_markerfacecolor('b')          # Seems to have no effect
    ax.get_lines()[0].set_markersize(4.0)
    stats.probplot(adjusted_data, plot=ax, rvalue=True)

    ax.set_title(col)

plt.tight_layout()
plt.show()

## BESSER ALS DIE DAVOR ABER BRUAHCT NOCH LINIEN, SCORES (R2) UND MUSS LOOPEN.
import probscale

def prob_plot(original, target, corrected, title=None, ylabel="Temperature [C]", **kwargs):
    fig, ax = plt.subplots(sharex=True, sharey=True)
    scatter_kws = dict(label="", marker=None, linestyle="-")
    common_opts = dict(plottype="qq", problabel="", datalabel="", **kwargs)

    scatter_kws["label"] = "original"
    fig = probscale.probplot(original, ax=ax, scatter_kws=scatter_kws, **common_opts)

    scatter_kws["label"] = "target"
    fig = probscale.probplot(target, ax=ax, scatter_kws=scatter_kws, **common_opts)

    scatter_kws["label"] = "corrected"
    fig = probscale.probplot(corrected, ax=ax, scatter_kws=scatter_kws, **common_opts)
    ax.set_title(title)
    ax.legend()

    ax.set_xlabel("Standard Normal Quantiles")
    ax.set_ylabel(ylabel)
    fig.tight_layout()



original = ssp2_tas_raw['ACCESS-CM2']['1979-01-01': '2015-12-31']
target = era5['1979-01-01': '2015-12-31']['temp']
adjusted = ssp2_tas['ACCESS-CM2']['1979-01-01': '2015-12-31']

prob_plot(original, target, adjusted)

plt.show()