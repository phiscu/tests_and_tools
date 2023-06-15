import pickle
import numpy as np
import spei
import spotpy.hydrology.signatures as sig
from climate_indices.indices import spei, spi, Distribution
from climate_indices import compute, utils
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
from fastparquet import write
from tqdm import tqdm
test_dir = '/home/phillip/Seafile/EBA-CA/Repositories/matilda_edu/output/cmip6/'


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


matilda_scenarios = pickle_to_dict(test_dir + 'adjusted/matilda_scenarios.pickle')   # pickle for speed/parquet for size

df = matilda_scenarios['SSP2']['EC-Earth3']['model_output']
df2 = matilda_scenarios['SSP2']['EC-Earth3']['glacier_rescaling']
test = df['total_runoff']

# print(df.columns)
# print(df2.columns)


## What to analyze?

# Annual stats
    # Month with highest precipitation
    # DoY with highest Runoff
    # start, end, and length of melting season:
        # melt variables depend on availability of snow/ice AND temperature!
        # Temperature might be better --> three consecutive days above 0°C
    # length an frequency of dry periods
        # precipitation >/< potential evap

# Time series to integrate in MATILDA
    # frozen water storage
    # total runoff ratio (runoff/precipitation)
    # Standardized Precipitation Evapotranspiration Index (SPEI)
## Functions

# DIE MEISTEN FUNKTIONEN AUF HYDROLOGISCHE JAHRE UMSTELLEN!!

# Convert to hydrological time series

## Helper functions
def water_year(df, begin=10):
    """
    Calculates the water year for each date in the index of the input DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a DatetimeIndex.
    begin : int, optional
        The month (1-12) that marks the beginning of the water year. Default is 10.
    Returns
    -------
    numpy.ndarray
        An array of integers representing the water year for each date in the input DataFrame index.
    """
    return np.where(df.index.month < begin, df.index.year, df.index.year + 1)


def crop2wy(df, begin=10):
    """
    Crops a DataFrame to include only the rows that fall within a complete water year.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a DatetimeIndex and a 'water_year' column.
    begin : int, optional
        The month (1-12) that marks the beginning of the water year. Default is 10.
    Returns
    -------
    pandas.DataFrame or None
        A new DataFrame containing only the rows that fall within a complete water year.
    """
    cut_begin = pd.to_datetime(f'{begin}-{df.water_year[0]}', format='%m-%Y')
    cut_end = pd.to_datetime(f'{begin}-{df.water_year[-1]-1}', format='%m-%Y') - pd.DateOffset(days=1)
    return df[cut_begin:cut_end].copy()


def hydrologicalize(df, begin_of_water_year=10):
    """
    Adds a 'water_year' column to a DataFrame and crops it to include only complete water years.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a DatetimeIndex.
    begin_of_water_year : int, optional
        The month (1-12) that marks the beginning of the water year. Default is 10.
    Returns
    -------
    pandas.DataFrame or None
        A new DataFrame with a 'water_year' column and only rows that fall within complete water years.
    """
    df_new = df.copy()
    df_new['water_year'] = water_year(df_new, begin_of_water_year)
    return crop2wy(df_new, begin_of_water_year)


## Annual stats:

# Month with maximum precipitation
def prec_minmax(df):
    """
    Compute the months of extreme precipitation for each year.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of daily precipitation data with a datetime index and a 'total_precipitation' column.
    Returns
    -------
    pandas.DataFrame
        A DataFrame with the months of extreme precipitation as a number for every calendar year.
    """
    # Use water years
    df = hydrologicalize(df)
    # group the data by year and month and sum the precipitation values
    grouped = df.groupby([df.water_year, df.index.month]).sum()
    # get the month with extreme precipitation for each year
    max_month = grouped.groupby(level=0)['total_precipitation'].idxmax()
    min_month = grouped.groupby(level=0)['total_precipitation'].idxmin()
    max_month = [p[1] for p in max_month]
    min_month = [p[1] for p in min_month]
    # create a new dataframe
    result = pd.DataFrame({'max_prec_month': max_month, 'min_prec_month': min_month},
                          index=pd.to_datetime(df.water_year.unique(), format='%Y'))
    return result


# Day of the Year with maximum flow
def peak_doy(df, smoothing_window_peakdoy=7):
    """
    Compute the day of the calendar year with the peak value for each hydrological year.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of daily data with a datetime index.
    smoothing_window_peakdoy : int, optional
        The window size of the rolling mean used for smoothing the data.
        Default is 7.
    Returns
    -------
    pandas.DataFrame
        A DataFrame with the day of the year with the peak value for each hydrological year.
    """
    # Use water years
    df = hydrologicalize(df)

    # find peak day for each hydrological year
    peak_dates = []
    for year in df.water_year.unique():
        # slice data for hydrological year
        hy_data = df.loc[df.water_year == year, 'total_runoff']

        # smooth data using rolling mean with window of 7 days
        smoothed_data = hy_data.rolling(smoothing_window_peakdoy, center=True).mean()

        # find day of peak value
        peak_day = smoothed_data.idxmax().strftime('%j')

        # append peak day to list
        peak_dates.append(peak_day)

    # create output dataframe with DatetimeIndex
    output_df = pd.DataFrame({'Hydrological Year': df.water_year.unique(),
                              'peak_day': pd.to_numeric(peak_dates)})
    output_df.index = pd.to_datetime(output_df['Hydrological Year'], format='%Y')
    output_df = output_df.drop('Hydrological Year', axis=1)

    return output_df


# Melting season
def melting_season(df, smoothing_window_meltseas=14, min_weeks=10):
    """
    Compute the start, end, and length of the melting season for each calendar year based on the daily
    the rolling mean of the temperature.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of daily mean temperature data with a datetime index.

    smoothing_window_meltseas : int, optional
        The size of the rolling window in days used to smooth the temperature data. Default is 14.

    min_weeks : int, optional
        The minimum number of weeks that the melting season must last for it to be considered valid. Default is 10.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the start, end, and length of the melting season for each calendar year, with a DatetimeIndex.
    """

    # Compute rolling mean of temperature data
    rolling_mean = df['avg_temp_catchment'].rolling(window=smoothing_window_meltseas).mean()

    # Find start of melting season for each year (first day above 0°C)
    start_mask = rolling_mean > 0
    start_mask = start_mask.groupby(df.index.year).apply(lambda x: x.index[np.nanargmax(x)])
    start_dates = start_mask - pd.Timedelta(days=smoothing_window_meltseas - 1)      # rolling() selects last day, we want the first

    # Add minimum length of melting season to start dates
    earliest_end_dates = start_dates + pd.Timedelta(weeks=min_weeks)

    # group rolling_mean by year and apply boolean indexing to replace values before start_dates with 999 in every year
    rolling_mean = rolling_mean.groupby(rolling_mean.index.year).\
        apply(lambda x: np.where(x.index < earliest_end_dates.loc[x.index.year], 999, x))

    # Transform the resulting array back to a time series with DatetimeIndex
    rolling_mean = pd.Series(rolling_mean.values.flatten())
    rolling_mean = rolling_mean.explode()
    rolling_mean = pd.DataFrame({'rolling_mean': rolling_mean}).set_index(df.index)

    # Filter values below 0 (including 999!)
    end_mask = rolling_mean < 0
    # Find end of melting season
    end_mask = end_mask.groupby(df.index.year).apply(lambda x: x.index[np.nanargmax(x)])
    end_dates = end_mask - pd.Timedelta(days=smoothing_window_meltseas - 1)

    # Compute length of melting season for each year
    lengths = (end_dates - start_dates).dt.days

    # Assemble output dataframe
    output_df = pd.DataFrame({'melt_season_start': [d.timetuple().tm_yday for d in start_dates],
                              'melt_season_end': [d.timetuple().tm_yday for d in end_dates]},
                             index=pd.to_datetime(df.index.year.unique(), format='%Y'))
    output_df['melt_season_length'] = output_df.melt_season_end - output_df.melt_season_start

    return output_df


# Dry periods
def dry_periods(df, dry_period_length=30):
    """
    Compute the number of days for which the rolling mean of evaporation exceeds precipitation for each hydrological
    year in the input DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns 'evap_off_glaciers' and 'prec_off_glaciers' with daily evaporation and
        precipitation data, respectively.
    dry_period_length : int, optional
        Length of the rolling window in days. Default is 30.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the number of days for which the rolling mean of evaporation exceeds precipitation for each
        year in the input DataFrame.
    """
    # Use hydrological years
    df = hydrologicalize(df)
    # Find number of days when the rolling mean of evaporation exceeds precipitation
    periods = []
    for year in df.water_year.unique():
        year_data = df.loc[df.water_year == year]
        evap_roll = year_data['evap_off_glaciers'].rolling(window=dry_period_length).mean()
        prec_roll = year_data['prec_off_glaciers'].rolling(window=dry_period_length).mean()

        dry = evap_roll[evap_roll - prec_roll > 0]
        periods.append(len(dry))

    # Assemble the output dataframe
    output_df = pd.DataFrame(
        {'dry_period_days': periods},
        index=pd.to_datetime(df.water_year.unique(), format='%Y'))

    return output_df


# Hydrological signatures
def get_qhf(data, global_median, measurements_per_day=1):
    """
    Variation of spotpy.hydrology.signatures.get_qhf() that allows definition of a global
    median to investigate long-term trends.
    Calculates the frequency of high flow events defined as :math:`Q > 9 \\cdot Q_{50}`
    cf. [CLBGS2000]_, [WESMCM2015]_. The frequency is given as :math: :math:`yr^{-1}`
    :param data: the timeseries
    :param measurements_per_day: the measurements_per_day of the timeseries
    :return: Q_{HF}, Q_{HD}
    """

    def highflow(value, median):
        return value > 9 * median

    fq, md = sig.flow_event(data, highflow, global_median)

    return fq * measurements_per_day * 365, md / measurements_per_day


def get_qlf(data, global_mean, measurements_per_day=1):
    """
    Variation of spotpy.hydrology.signatures.get_qlf() that allows comparison of
    individual years with a global mean to investigate long-term trends.
    Calculates the frequency of low flow events defined as
    :math:`Q < 0.2 \\cdot \\overline{Q_{mean}}`
    cf. [CLBGS2000]_, [WESMCM2015]_. The frequency is given
    in :math:`yr^{-1}` and for the whole timeseries
    :param data: the timeseries
    :param measurements_per_day: the measurements_per_day of the timeseries
    :return: Q_{LF}, Q_{LD}
    """

    def lowflow(value, mean):
        return value < 0.2 * mean

    fq, md = sig.flow_event(data, lowflow, global_mean)
    return fq * measurements_per_day * 365, md / measurements_per_day


def hydrological_signatures(df):
    """
    Calculate hydrological signatures for a given input dataframe.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a column 'total_runoff' and a DatetimeIndex.
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the calculated hydrological signatures for each year in the input dataframe.
        The columns of the output dataframe are as follows:
         - 'q5': the 5th percentile of total runoff for each year
        - 'q50': the 50th percentile of total runoff for each year
        - 'q95': the 95th percentile of total runoff for each year
        - 'qlf_freq': the frequency of low flow events (defined as Q < 2*Qmean_global) for each year, in yr^⁻1
        - 'qlf_dur': the mean duration of low flow events (defined as Q < 2*Qmean_global) for each year, in days
        - 'qhf_freq': the frequency of high flow events (defined as Q > 9*Q50_global) for each year, in yr^⁻1
        - 'qhf_dur': the mean duration of high flow events (defined as Q > 9*Q50_global) for each year, in days
    """
    # Create lists of quantile functions to apply and column names
    functions = [sig.get_q5, sig.get_q50, sig.get_q95]
    cols = ['q5', 'q50', 'q95']

    # Create an empty dataframe to store the results
    results_df = pd.DataFrame()

    # Use water_year
    df = hydrologicalize(df)

    # Loop through each year in the input dataframe
    for year in df.water_year.unique():
        # Select the data for the current year
        year_data = df[df.water_year == year].total_runoff
        # Apply each quantile function to the year data and store the results in a dictionary
        year_results = {}
        for i, func in enumerate(functions):
            year_results[cols[i]] = func(year_data)
        # Calculate frequency and duration of global low flows
        qlf_freq, qlf_dur = get_qlf(year_data, np.mean(df.total_runoff))
        year_results['qlf_freq'] = qlf_freq
        year_results['qlf_dur'] = qlf_dur
        # Calculate frequency and duration of global high flows
        qhf_freq, qhf_dur = get_qhf(year_data, np.median(df.total_runoff))
        year_results['qhf_freq'] = qhf_freq
        year_results['qhf_dur'] = qhf_dur
        # Convert the dictionary to a dataframe and append it to the results dataframe
        year_results_df = pd.DataFrame(year_results, index=[year])
        results_df = pd.concat([results_df, year_results_df])

    results_df.set_index(pd.to_datetime(df.water_year.unique(), format='%Y'), inplace=True)

    return results_df


# Wrapper function
import inspect
def cc_indicators(df, **kwargs):
    """
    Apply a list of climate change indicator functions to output DataFrame of MATILDA and concatenate
    the output columns into a single DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns 'A' and 'B'.
    **kwargs : optional
        Optional arguments to be passed to the functions in the list. Possible arguments are 'smoothing_window_peakdoy',
        'smoothing_window_meltseas', 'min_weeks', and 'dry_period_length'.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the output columns of all functions applied to the input DataFrame.
    Notes
    -----
    The list of functions to apply is hard-coded into the function and cannot be modified from outside.
     The optional arguments are passed to the respective functions only if they are relevant for the respective
     function.
     If no optional arguments are passed, the function is applied to the input DataFrame with default arguments.
    """
    # List of all functions to apply
    functions = [prec_minmax, peak_doy, melting_season, dry_periods, hydrological_signatures]
    # Empty result dataframe
    indicator_df = pd.DataFrame()
    # Loop through all functions
    for func in functions:
        func_kwargs = {}
        # Apply only those optional kwargs relevant for the respective function
        for kwarg in kwargs:
            if kwarg in inspect.getfullargspec(func)[0]:
                func_kwargs.update({kwarg: kwargs.get(kwarg)})
        result = func(df, **func_kwargs)
        # Concat all output columns in one dataframe
        indicator_df = pd.concat([indicator_df, result], axis=1)

    return indicator_df


## Drought indicators
def drought_indicators(df, freq='M', period=1, dist='pearson'):
    """
    Calculate the climatic water balance, SPI (Standardized Precipitation Index) and
    SPEI (Standardized Precipitation Evapotranspiration Index)
    Parameters
    ----------
    df : pandas.DataFrame
         Input DataFrame containing columns 'prec_off_glaciers' and 'evap_off_glaciers'.
    freq : str, optional
         Resampling frequency for precipitation and evaporation data. Default is 'M' for monthly.
    period : int, optional
         Period for SPI and SPEI calculation. Default is 1.
    dist : str, optional
         Distribution type for SPI and SPEI calculation. Choose either 'pearson' or 'gamma'. Default is 'pearson'.
    Returns
    -------
    pandas.DataFrame
         DataFrame containing the calculated indicators: 'clim_water_balance', 'spi', and 'spei'.
         Index is based on the resampled frequency of the input DataFrame.
    Raises
    ------
    ValueError
         If 'freq' is not 'D' or 'M'.
         If 'dist' is not 'pearson' or 'gamma'.
     Notes
    -----
    SPI (Standardized Precipitation Index) and SPEI (Standardized Precipitation Evapotranspiration Index)
    are drought indicators that are used to quantify drought severity and duration.
     'clim_water_balance' is the difference between total precipitation and total evapotranspiration.
     If 'freq' is 'D', the input data is transformed from Gregorian to a 366-day format for SPI and SPEI calculation,
    and then transformed back to Gregorian format for output.
     The default period for SPI and SPEI calculation is 1 month.
     The default distribution for SPI and SPEI calculation is Pearson Type III.
     The calibration period for SPI and SPEI calculation is from 1981 to 2020.
    """

    # Check if frequency is valid
    if freq != 'D' and freq != 'M':
        raise ValueError("Invalid value for 'freq'. Choose either 'D' or 'M'.")

    # Resample precipitation and evaporation data based on frequency
    prec = df.prec_off_glaciers.resample(freq).sum().values
    evap = df.evap_off_glaciers.resample(freq).sum().values

    # Calculate water balance
    water_balance = prec - evap

    # If frequency is daily, transform data to 366-day format
    if freq == 'D':
        prec = utils.transform_to_366day(prec, year_start=df.index.year[0],
                                         total_years=len(df.index.year.unique()))
        evap = utils.transform_to_366day(evap, year_start=df.index.year[0],
                                         total_years=len(df.index.year.unique()))

    # Set distribution based on input
    if dist == 'pearson':
        distribution = Distribution.pearson
    elif dist == 'gamma':
        distribution = Distribution.pearson
    else:
        raise ValueError("Invalid value for 'dist'. Choose either 'pearson' or 'gamma'.")

    # Set periodicity based on frequency
    if freq == 'D':
        periodicity = compute.Periodicity.daily
    elif freq == 'M':
        periodicity = compute.Periodicity.monthly

    # Set common parameters
    common_params = {'scale': period,
                     'distribution': distribution,
                     'periodicity': periodicity,
                     'data_start_year': 1981,
                     'calibration_year_initial': 1981,
                     'calibration_year_final': 2020}

    # Set parameters for SPEI calculation
    spei_params = {'precips_mm': prec,
                   'pet_mm': evap,
                   **common_params}

    # Set parameters for SPI calculation
    spi_params = {'values': prec,
                  **common_params}

    # Calculate SPI and SPEI
    spi_arr = spi(**spi_params)
    spei_arr = spei(**spei_params)

    # If frequency is daily, transform data back to Gregorian format
    if freq == 'D':
        spi_arr = utils.transform_to_gregorian(spi_arr, df.index.year[0])
        spei_arr = utils.transform_to_gregorian(spei_arr, df.index.year[0])

    # Return a DataFrame containing the calculated indicators
    return pd.DataFrame({'clim_water_balance': water_balance,
                         'spi': spi_arr,
                         'spei': spei_arr
                         }, index=df.resample(freq).sum().index)


## Performance check
%%time
print('prec_minmax')
%time prec_minmax(df)
print('peak_doy')
%time peak_doy(df)
print('melting_season')
%time melting_season(df)
print('dry_periods')
%time dry_periods(df)
print('hydrological_signatures')
%time hydrological_signatures(df)
print('drought_indicators')
%time drought_indicators(df, 'D')        # very slow in the first iteration, fast afterwards
print()


##

matilda_scenarios = pickle_to_dict(test_dir + 'adjusted/matilda_scenarios.pickle')


## Long-term annual cycle of evaporation and precipitation for every decade



# Compute the rolling mean of evaporation and precipitation
# df_avg = df[['prec_off_glaciers', 'evap_off_glaciers']].rolling(window=30).mean()
#
# # Split the data into decades
# decades = range(df.index.year.min(), df.index.year.max() + 1, 10)
#
# # Iterate over each decade to get global maximum
# decade_max = []
# for i, decade in enumerate(decades):
#     # Filter the data for the current decade
#     decade_data = df_avg.loc[(df_avg.index.year >= decade) & (df_avg.index.year < decade + 10)]
#     # Compute the mean value for each day of the year for the current decade
#     decade_data = decade_data.groupby([decade_data.index.month, decade_data.index.day]).mean()
#     # Get maximum value
#     decade_max.append(decade_data.max().max())
#
# global_max = max(decade_max)
#
# # Create a new figure with a 4x3 subplot grid
# fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12, 12))
#
# # Iterate over each decade and create a plot for each
# for i, decade in enumerate(decades):
#     # Compute the row and column indices of the current subplot
#     row = i // 3
#     col = i % 3
#     # Filter the data for the current decade
#     decade_data = df_avg.loc[(df_avg.index.year >= decade) & (df_avg.index.year < decade + 10)]
#     # Compute the mean value for each day of the year for the current decade
#     decade_data = decade_data.groupby([decade_data.index.month, decade_data.index.day]).mean()
#     # Create a new subplot for the current decade
#     ax = axs[row, col]
#     # Plot the data for the current decade
#     decade_data.plot(ax=ax, legend=False)
#     # Set the tick formatter of the x-axis to only show the month name
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
#     # Make sure every month is labeled
#     ax.xaxis.set_tick_params(rotation=0, which='major')
#
#     # Set the y-axis limit to the maximum range of the whole dataset
#     ax.set_ylim(-0.3, global_max*1.1)
#
#     # Set the plot title and labels
#     ax.set(title=f'{decade}-{decade+9}',
#            xlabel=None, ylabel='mm')
#
# # Create a common legend for all subplots at the bottom of the figure
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=2)
# # Add title
# fig.suptitle('Average annual cycle of Evaporation and Precipitation', fontsize=16)
# # Make sure the subplots don't overlap
# plt.tight_layout()
# # Add some space at the bottom of the figure for the legend
# fig.subplots_adjust(bottom=0.06, top=0.92)
# # Show the plot
# plt.show()
#
#
#
# ##
# import plotly.express as px
# import pandas as pd
# import plotly.io as pio
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
#
# pio.renderers.default = "browser"
#
# df_avg = df[['prec_off_glaciers', 'evap_off_glaciers']].rolling(window=30).mean()
#
# # Split the data into decades
# decades = range(df.index.year.min(), df.index.year.max() + 1, 10)
#
# # Iterate over each decade to get global maximum
# decade_max = []
# for i, decade in enumerate(decades):
#     # Filter the data for the current decade
#     decade_data = df_avg.loc[(df_avg.index.year >= decade) & (df_avg.index.year < decade + 10)]
#     # Compute the mean value for each day of the year for the current decade
#     decade_data = decade_data.groupby([decade_data.index.month, decade_data.index.day]).mean()
#     # Get maximum value
#     decade_max.append(decade_data.max().max())
#
# global_max = max(decade_max)
#
#
#
# # Create a new figure with a 4x3 subplot grid
# fig = make_subplots(rows=4, cols=3,
#                     shared_xaxes=True,
#                     vertical_spacing=0.04,
#                     subplot_titles=[f'{decade}-{decade+9}' for decade in decades])
#
# # Iterate over each decade and create a plot for each
# for i, decade in enumerate(decades):
#     # Compute the row and column indices of the current subplot
#     row = i // 3 + 1
#     col = i % 3 + 1
#     # Filter the data for the current decade
#     decade_data = df_avg.loc[(df_avg.index.year >= decade) & (df_avg.index.year < decade + 10)]
#     # Compute the mean value for each day of the year for the current decade
#     decade_data = decade_data.groupby([decade_data.index.month, decade_data.index.day]).mean()
#     # Rename and reset MultiIndex
#     decade_data.index = decade_data.index.set_names(['Month', 'Day'])
#     decade_data = decade_data.reset_index()
#     # Create dummy datetime index (year irrelevant)
#     decade_data['datetime'] = pd.to_datetime(
#         '2000-' + decade_data['Month'].astype(str) + '-' + decade_data['Day'].astype(str))
#     # Create a new subplot for the current decade
#     fig.add_trace(go.Scatter(x=decade_data.datetime, y=decade_data.prec_off_glaciers, name='Precipitation',
#                             line = dict(color='blue'),
#                             showlegend=False,
#                                                       # fill='tonext',  # fill to the next trace (Evaporation)
#                                                       # fillcolor='lightblue',  # transparent blue
#                              ),
#                   row=row, col=col
#
#                   )
#     fig.add_trace(go.Scatter(x=decade_data.datetime, y=decade_data.evap_off_glaciers, name='Evaporation',
#                              line=dict(color='orange'),
#                              showlegend=False,
#                              # fill='tonexty',  # fill to the next trace (Precipitation)
#                              # fillcolor='honeydew'  # transparent orange
#                              ),
#                   row=row, col=col)
#
#
# fig.update_traces(selector=-1, showlegend=True)         # Show legend for the last trace (Evaporation)
# fig.update_traces(selector=-2, showlegend=True)         # Show legend for the trace before the last (Precipitation)
# fig.update_yaxes(
#     range=[0, global_max],
#     showgrid=True, gridcolor='lightgrey')
# fig.update_xaxes(
#     dtick="M1",
#     tickformat="%b",
#     hoverformat='%b %d',
#     showgrid=True, gridcolor='lightgrey')
# fig.update_layout(
#     hovermode='x',
#     margin=dict(l=10, r=10, t=90, b=10),  # Adjust the margins to remove space around the plot
#     plot_bgcolor='white',  # set the background color of the plot to white
#     xaxis_title='Month',
#     yaxis_title=output_vars[var][0] + ' [' + output_vars[var][1] + ']',
#     title={'text': '<b>' + output_vars[var][0] + '</b>', 'font': {'size': 28, 'color': 'darkblue', 'family': 'Arial'}},
#     legend={'font': {'size': 18, 'family': 'Arial'}},
#
#
# )
#
# fig.show()
#

##

# Design plotting function that runs every indicator function depending on the chosen var_name --> Might result in processing time when choosing certain vars
# OR
# Run all functions for every model and create an indicator-df --> will result in a long processing time! ~ 6s * 31 models = 3min

var_name = ['max_prec_month', 'min_prec_month',
            'peak_day',
            'melt_season_start', 'melt_season_end', 'melt_season_length',
            'dry_period_days',
            'qlf_freq', 'qlf_dur', 'qhf_freq', 'qhf_dur',
            'water_balance', 'spi', 'spei']

title = ['Month with Maximum Precipitation', 'Month with Minimum Precipitation',
         'Timing of Peak Runoff',
         'Beginning of Melt Season', 'End of Melt Season', 'Length of Melt Season',
         'Total Length of Dry Periods',
         'Frequency of Low-flow events', 'Mean Duration of Low-flow events',
         'Frequency of High-flow events', 'Mean Duration of High-flow events',
         'Climatic Water Balance', 'Standardized Precipitation Index', 'Standardized Precipitation Evaporation Index']

unit = ['-', '-',
        'DoY',
        'DoY', 'DoY', 'days',
        'days',
        'yr^-1', 'days', 'yr^-1', 'days',
        'mm w.e.', '-', '-']

output_vars = {key: (val1, val2) for key, val1, val2 in zip(var_name, title, unit)}


def custom_df(dic, scenario, var, **kwargs):

    out1_cols = ['max_prec_month', 'min_prec_month',
                'peak_day',
                'melt_season_start', 'melt_season_end', 'melt_season_length',
                'dry_period_days',
                'qlf_freq', 'qlf_dur', 'qhf_freq', 'qhf_dur']
    out2_cols = ['water_balance', 'spi', 'spei']

    if var in out1_cols:
        func = cc_indicators
    elif var in out2_cols:
        func = drought_indicators
    else:
        raise ValueError("var needs to be one of the following strings: " +
                         str([i for i in [out1_cols, out2_cols]]))

    # Create an empty list to store the dataframes
    dfs = []
    # Loop over the models in the selected scenario
    for model in dic[scenario].keys():
        # Get the dataframe for the current model
        if var in out1_cols:
            df = func(dic[scenario][model]['model_output'], **kwargs)
        else:
            df = func(dic[scenario][model]['model_output'])






            # SO WIRD DAS NICHT GEHEN, WEIL JEDE ITERATION 14 SPALTEN HAT
            # --> ERSTMAL DURCH ALLE MODELLE LOOPEN UND ZWEI ZUSÄTZLICHE OUTPUT_DF HINZUFÜGEN.






        # Append the dataframe to the list of dataframes
        dfs.append(df[var])
    # Concatenate the dataframes into a single dataframe
    combined_df = pd.concat(dfs, axis=1)
    # Set the column names of the combined dataframe to the model names
    combined_df.columns = dic[scenario].keys()






# To-dos:

    # Add shading humid/arid
    # Add units
    # generalize to fit to other variables as well
    # add climate-indices to requirements