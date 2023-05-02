##
import ee
import geemap
import pandas as pd
import geopandas as gpd
import concurrent.futures
import os
import requests
from retry import retry
from tqdm import tqdm

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
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

        print('Initiating download request for NEX-GDDP-CMIP6 data from ' +
              str(self.starty) + ' to ' + str(self.endy) + '.')

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

cmip_dir = wd + '/matilda_edu/output/' + 'cmip6/'
downloader_t = CMIPDownloader(var='tas', starty=1979, endy=2100, shape=catchment, processes=30, dir=cmip_dir)
downloader_t.download()
downloader_p = CMIPDownloader(var='pr', starty=1979, endy=2100, shape=catchment, processes=30, dir=cmip_dir)
downloader_p.download()


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
processor = CMIPProcessor(file_dir=cmip_dir, var='pr')
ssp2_pr_raw, ssp5_pr_raw = processor.get_results()
processor = CMIPProcessor(file_dir=cmip_dir, var='tas')
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
    training_period = slice('1979-01-01', '2022-12-31')
    prediction_period = slice('1979-01-01', '2100-12-31')
    corr = pd.DataFrame()
    for m in predictand.columns:
        x_train = predictand[m][training_period].squeeze()
        y_train = predictor[training_period][var].squeeze()
        x_predict = predictand[m][prediction_period].squeeze()
        bc_corr = BiasCorrection(y_train, x_train, x_predict)
        corr[m] = pd.DataFrame(bc_corr.correct(method=method))

    return corr

##
ssp2_tas = adjust_bias(predictand=ssp2_tas_raw, predictor=era5_file)
ssp5_tas = adjust_bias(predictand=ssp5_tas_raw, predictor=era5_file)
ssp2_pr = adjust_bias(predictand=ssp2_pr_raw, predictor=era5_file)
ssp5_pr = adjust_bias(predictand=ssp5_pr_raw, predictor=era5_file)

era5 = read_era5l(era5_file)


## Create input for plotting functions

ssp_tas_dict = {'SSP2_raw': ssp2_tas_raw, 'SSP2_adjusted': ssp2_tas, 'SSP5_raw': ssp5_tas_raw, 'SSP5_adjusted': ssp5_tas}
ssp_pr_dict = {'SSP2_raw': ssp2_pr_raw, 'SSP2_adjusted': ssp2_pr, 'SSP5_raw': ssp5_pr_raw, 'SSP5_adjusted': ssp5_pr}


## Timeseries plots
import matplotlib.pyplot as plt


def cmip_plot(ax, df, title, target, precip=False, intv_sum='M', intv_mean='10Y',
              target_label='Target', show_target_label=False):
    """Resamples and plots climate model and target data."""

    if not precip:
        ax.plot(df.resample(intv_mean).mean().iloc[:, :-1], linewidth=0.6)
        ax.plot(df.resample(intv_mean).mean().iloc[:, -1], linewidth=1, c='black')
        era_plot, = ax.plot(target['temp'].resample(intv_mean).mean(), linewidth=1.5, c='red', label=target_label,
                            linestyle='dashed')
    else:
        ax.plot(df.resample(intv_sum).sum().resample(intv_mean).mean().iloc[:, :-1],
                linewidth=0.6)
        ax.plot(df.resample(intv_sum).sum().resample(intv_mean).mean().iloc[:, -1],
                linewidth=1, c='black')
        era_plot, = ax.plot(target['prec'].resample(intv_sum).sum().resample(intv_mean).mean(), linewidth=1.5,
                            c='red', label=target_label, linestyle='dashed')
    if show_target_label:
        ax.legend(handles=[era_plot], loc='upper left')
    ax.set_title(title)
    ax.grid(True)
    

def cmip_plot_combined(data, target, title, precip=False, intv_sum='M', intv_mean='10Y',
                       target_label='Target', show=False):
    """Combines multiple subplots of climate data in different scenarios before and after bias adjustment.
    Shows target data for comparison"""

    figure, axis = plt.subplots(2, 2, figsize=(12, 12), sharex="col", sharey="all")
    
    t_kwargs = {'target': target, 'intv_mean': intv_mean, 'target_label': target_label}
    p_kwargs = {'target': target, 'intv_mean': intv_mean, 'target_label': target_label,
                'intv_sum': intv_sum, 'precip': True}

    if not precip:
        cmip_plot(axis[0, 0], data['SSP2_raw'], show_target_label=True, title='SSP2 raw', **t_kwargs)
        cmip_plot(axis[0, 1], data['SSP2_adjusted'], title='SSP2 adjusted', **t_kwargs)
        cmip_plot(axis[1, 0], data['SSP5_raw'], title='SSP5 raw', **t_kwargs)
        cmip_plot(axis[1, 1], data['SSP5_adjusted'], title='SSP5 adjusted', **t_kwargs)
        figure.legend(data['SSP5_adjusted'].columns, loc='lower right', ncol=6, mode="expand")
        figure.tight_layout()
        figure.subplots_adjust(bottom=0.15, top=0.92)
        figure.suptitle(title, fontweight='bold')
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
        if show:
            plt.show()

# cmip_plot_combined(data=ssp_tas_dict, target=era5, title='10y Mean of Air Temperature', target_label='ERA5-Land', show=True)
# cmip_plot_combined(data=ssp_pr_dict, target=era5, title='Mean of Monthly Precipitation', precip=True, target_label='ERA5-Land', show=True)

# ZWEI AUSREIẞER BEI DEN TEMPERATUREN
# ERA5 NIEDERSCHLAG CA. 4 MAL SO HOCH WIE CMIP. FEHLER? BEI ECMWF DATEN GENAUSO?

##
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from matplotlib.legend import Legend

def df2long(df, intv_sum='M', intv_mean='Y', precip=False):
    """Resamples dataframes and converts them into long format to be passed to seaborn.lineplot()."""

    if precip:
        df = df.resample(intv_sum).sum().resample(intv_mean).mean()
        df = df.reset_index()
        df = df.melt('TIMESTAMP', var_name='model', value_name='prec')
    else:
        df = df.resample(intv_mean).mean()
        df = df.reset_index()
        df = df.melt('TIMESTAMP', var_name='model', value_name='temp')
    return df


def cmip_plot_ensemble(cmip, era, precip=False, intv_sum='M', intv_mean='Y', figsize=(10, 6), show=True):
    """Plots the multi-model mean of climate scenarios including the 90% confidence interval."""

    warnings.filterwarnings(action='ignore')
    figure, axis = plt.subplots(figsize=figsize)

    # Define color palette
    colors = ['darkorange', 'orange', 'darkblue', 'dodgerblue']
    # create a new dictionary with the same keys but new values from the list
    col_dict = {key: value for key, value in zip(cmip.keys(), colors)}

    if precip:
        for i in cmip.keys():
            df = df2long(cmip[i], intv_sum=intv_sum, intv_mean=intv_mean, precip=True)
            sns.lineplot(data=df, x='TIMESTAMP', y='prec', color=col_dict[i])
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
            sns.lineplot(data=df, x='TIMESTAMP', y='temp', color=col_dict[i])
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

# cmip_plot_ensemble(ssp_tas_dict, era5['temp'], intv_mean='Y')
# cmip_plot_ensemble(ssp_pr_dict, era5['prec'], precip=True, intv_sum='Y', intv_mean='Y')

## Violin plots

# Rearrange dictionaries for two-column display
tas_raw = {'SSP2': ssp2_tas_raw, 'SSP5': ssp5_tas_raw}
tas_adjusted = {'SSP2': ssp2_tas, 'SSP5': ssp5_tas}
pr_raw = {'SSP2': ssp2_pr_raw, 'SSP5': ssp5_pr_raw}
pr_adjusted = {'SSP2': ssp2_pr, 'SSP5': ssp5_pr}

def vplots(before, after, target, target_label='Target', precip=False, show=False):
    """Creates violin plots of the kernel density estimation for all models before and after bias adjustment."""

    period = slice('1979-01-01', '2022-12-31')
    if precip:
        var = 'prec'
        var_label = 'Annual Precipitation'
        unit = ' [mm]'
    else:
        var = 'temp'
        var_label = 'Mean Annual Air Temperature'
        unit = ' [K]'
    for i in before.keys():
        before[i] = before[i].loc[period].copy()
        before[i][target_label] = target[var][period]

    for i in after.keys():
        after[i] = after[i].loc[period].copy()
        after[i][target_label] = target[var][period]

    fig = plt.figure(figsize=(20, 20))
    outer = fig.add_gridspec(1, 2)
    inner = outer[0].subgridspec(2, 1)
    axis = inner.subplots(sharex='col')

    all_data = pd.concat([df2long(before[i], precip=precip, intv_sum='Y') for i in before.keys()] +
                         [df2long(after[i], precip=precip, intv_sum='Y') for i in after.keys()])
    xmin, xmax = all_data[var].min(), all_data[var].max()

    if precip:
        xlim = (xmin * 0.95, xmax * 1.05)
    else:
        xlim = (xmin - 1, xmax +1 )

    for (i, k) in zip(before.keys(), range(0, 4, 1)):
        df = df2long(before[i], precip=precip, intv_sum='Y')
        axis[k].grid()
        sns.violinplot(ax=axis[k], x=var, y='model', data=df, scale="count", bw=.2)
        axis[k].set(xlim=xlim)
        axis[k].set_ylabel(i, fontsize=18, fontweight='bold')
        if k == 0:
            axis[k].set_title('Before Scaled Distribution Mapping', fontweight='bold', fontsize=18)
    plt.xlabel(var_label + unit)

    inner = outer[1].subgridspec(2, 1)
    axis = inner.subplots(sharex='col')
    for (i, k) in zip(after.keys(), range(0, 4, 1)):
        df = df2long(after[i], precip=precip, intv_sum='Y')
        axis[k].grid()
        sns.violinplot(ax=axis[k], x=var, y='model', data=df, scale="count", bw=.2)
        axis[k].set(xlim=xlim)
        axis[k].set_ylabel(i, fontsize=18, fontweight='bold')
        axis[k].get_yaxis().set_visible(False)
        if k == 0:
            axis[k].set_title('After Scaled Distribution Mapping', fontweight='bold', fontsize=18)
    plt.xlabel(var_label + unit)

    starty = period.start.split('-')[0]
    endy = period.stop.split('-')[0]
    fig.suptitle('Kernel Density Estimation of ' + var_label + ' (' + starty + '-' + endy + ')',
                 fontweight='bold', fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    if show:
        plt.show()


# vplots(tas_raw, tas_adjusted, era5, target_label='ERA5-Land', show=True)
# vplots(pr_raw, pr_adjusted, era5, target_label='ERA5-Land', precip=True, show=True)


## Probability plots

import probscale
import matplotlib.pyplot as plt


def prob_plot(original, target, corrected, ax, title=None, ylabel="Temperature [K]", **kwargs):
    """Combines probability plots of climate model data before and after bias adjustment and the target data. """

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


def pp_matrix(original, target, corrected, nrow=7, ncol=5, precip=False, show=False):
    """Arranges the prob_plots of all CMIP6 models in a matrix and adds the R² score."""

    period = slice('1979-01-01', '2022-12-31')
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

    for i, col in enumerate(ssp2_tas.columns):
        ax = plt.subplot(nrow, ncol, i + 1)
        prob_plot(original[col][period], target[period],
                  corrected[col][period], ax=ax, ylabel=var + unit)
        ax.set_title(col, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, ['original (CMIP6 raw)', 'target (ERA5-Land)', 'adjusted (CMIP6 after SDM)'], loc='lower right', bbox_to_anchor=(0.96, 0.024), fontsize=20)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.7, wspace=0.4)
    starty = period.start.split('-')[0]
    endy = period.stop.split('-')[0]
    fig.suptitle('Probability Plots of CMIP6 ' + var_label + ' (' + starty + '-' + endy + ')',
                 fontweight='bold', fontsize=20)
    plt.subplots_adjust(top=0.93)
    if show:
        plt.show()

pp_matrix(ssp2_tas_raw, era5['temp'], ssp2_tas, show=True)
pp_matrix(ssp2_pr_raw, era5['prec'], ssp2_pr, precip=True, show=True)
