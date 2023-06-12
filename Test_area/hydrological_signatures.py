import pickle
import numpy as np
import spei
import spotpy.hydrology.signatures as sig
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

print(df.columns)
print(df2.columns)


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
def peak_doy(df, smoothing_window=7):
    """
    Compute the day of the calendar year with the peak value for each hydrological year.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of daily data with a datetime index.
    smoothing_window : int, optional
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
        smoothed_data = hy_data.rolling(smoothing_window, center=True).mean()

        # find day of peak value
        peak_day = smoothed_data.idxmax().strftime('%j')

        # append peak day to list
        peak_dates.append(peak_day)

    # create output dataframe with DatetimeIndex
    output_df = pd.DataFrame({'Hydrological Year': df.water_year.unique(),
                              'Peak Day of Year': pd.to_numeric(peak_dates)})
    output_df.index = pd.to_datetime(output_df['Hydrological Year'], format='%Y')
    output_df = output_df.drop('Hydrological Year', axis=1)

    return output_df


# Melting season
def melting_season(df, smoothing_window=14, min_weeks=10):
    """
    Compute the start, end, and length of the melting season for each calendar year based on the daily mean temperature data provided in the input dataframe.
    The start and end of the melting season are the first day of the first two week period with mean temperatures above and below 0°C, respectively.
    The length of the melting season is the number of days between the start and end of the melting season.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of daily mean temperature data with a datetime index.
    Returns
    -------
    pandas.DataFrame
        A DataFrame with the start, end, and length of the melting season for each calendar year, with a DatetimeIndex.
    """

    # Find the start of the melting season for each year
    start_dates = []
    for year in df.index.year.unique():
        year_data = df.loc[df.index.year == year, 'avg_temp_catchment']
        year_roll = year_data.rolling(window=smoothing_window).mean()
        start_index = year_roll[year_roll > 0].index[0]
        start_index = start_index - pd.Timedelta(days=smoothing_window-1)  # rolling selects last day of window, we want the first
        start_dates.append(start_index)

    # Find the end of the melting season for each year
    end_dates = []
    for year in df.index.year.unique():
        year_data = df.loc[df.index.year == year, 'avg_temp_catchment']
        year_roll = year_data.rolling(window=smoothing_window).mean()
        start_index = start_dates[year - df.index.year.min()]
        year_roll = year_roll.loc[start_index + pd.Timedelta(weeks=min_weeks):]  # add min season duration of 10 weeks
        end_index = year_roll[year_roll < 0].index[0]
        end_index = end_index - pd.Timedelta(days=smoothing_window-1)  # rolling selects last day of window, we want the first
        end_dates.append(end_index)

    # Compute the length of the melting season for each year
    lengths = [(end_dates[i] - start_dates[i]).days for i in range(len(start_dates))]

    # Assemble the output dataframe
    output_df = pd.DataFrame({'melt_season_start': [d.timetuple().tm_yday for d in start_dates],
                              'melt_season_end': [d.timetuple().tm_yday for d in end_dates],
                              'melt_season_length': lengths},
                             index=pd.to_datetime(df.index.year.unique(), format='%Y'))
    return output_df


# Dry periods
def dry_periods(df, period_length=30):
    """
    Compute the number of days for which the rolling mean of evaporation exceeds precipitation for each hydrological
    year in the input DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns 'evap_off_glaciers' and 'prec_off_glaciers' with daily evaporation and
        precipitation data, respectively.
    period_length : int, optional
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
        evap_roll = year_data['evap_off_glaciers'].rolling(window=period_length).mean()
        prec_roll = year_data['prec_off_glaciers'].rolling(window=period_length).mean()

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


## Daily stats:

# drought indicators
def drought_indicators(df, freq='30D'):
    """
    Calculates climatic water balance, Standardized (Evaporation) Precipitation Index (SPI/SPEI), and
    Standardized Streamflow Index (SSFI) for a given DataFrame with a DatetimeIndex and a given frequency.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a datetime index containing columns for 'prec_off_glaciers',
        'evap_off_glaciers', 'total_precipitation', and 'total_runoff'.
    freq : str
        Rolling window frequency in pandas format (e.g. '30D' for 30 days).
    Returns
    -------
    pandas.DataFrame
        DataFrame with the datetime index of the input DataFrame and columns for
        'water_balance', 'spi', 'spei', and 'ssfi'.
    """
    water_balance = df.prec_off_glaciers.rolling(freq).sum() - df.evap_off_glaciers.rolling(freq).sum()
    df_spei = spei.spei(water_balance)  # Input: climatic water balance (prec - pet)
    df_spi = spei.spi(df.total_precipitation.rolling(freq).sum())  # Input: precipitation
    df_ssfi = spei.ssfi(df.total_runoff.rolling(freq).sum())  # Input: streamflow data

    # Combine the results into a single DataFrame with the same datetime index as the input DataFrame
    result_df = pd.concat([water_balance, df_spi, df_spei, df_ssfi], axis=1)
    result_df.columns = ['water_balance', 'spi', 'spei', 'ssfi']

    return result_df


# Total runoff ratio
def runoff_ratio(df):
    """
    Calculates the proportion of precipitation that does not infiltrate, evapotranspirate or is stored in ice.
    Parameters
    -----------
        df : pandas.DataFrame
            DataFrame containing the variables 'total_runoff' and 'total_precipitation'.
    Returns
    --------
        pandas.DataFrame
            DataFrame containing the runoff ratio for each observation in the input DataFrame,
            with the same datetime index as the input DataFrame.
            If 'total_precipitation' is 0, the corresponding runoff ratio is set to 0.
    """
    runoff_ratio = np.where(df['total_precipitation'] == 0, 0, df['total_runoff'] / df['total_precipitation'])
    return pd.DataFrame(data=runoff_ratio, index=df.index, columns=['runoff_ratio'])


## Long-term annual cycle of evaporation and precipitation for every decade



# Compute the rolling mean of evaporation and precipitation
df_avg = df[['prec_off_glaciers', 'evap_off_glaciers']].rolling(window=30).mean()

# Split the data into decades
decades = range(df.index.year.min(), df.index.year.max() + 1, 10)

# Iterate over each decade to get global maximum
decade_max = []
for i, decade in enumerate(decades):
    # Filter the data for the current decade
    decade_data = df_avg.loc[(df_avg.index.year >= decade) & (df_avg.index.year < decade + 10)]
    # Compute the mean value for each day of the year for the current decade
    decade_data = decade_data.groupby([decade_data.index.month, decade_data.index.day]).mean()
    # Get maximum value
    decade_max.append(decade_data.max().max())

global_max = max(decade_max)

# Create a new figure with a 4x3 subplot grid
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12, 12))

# Iterate over each decade and create a plot for each
for i, decade in enumerate(decades):
    # Compute the row and column indices of the current subplot
    row = i // 3
    col = i % 3
    # Filter the data for the current decade
    decade_data = df_avg.loc[(df_avg.index.year >= decade) & (df_avg.index.year < decade + 10)]
    # Compute the mean value for each day of the year for the current decade
    decade_data = decade_data.groupby([decade_data.index.month, decade_data.index.day]).mean()
    # Create a new subplot for the current decade
    ax = axs[row, col]
    # Plot the data for the current decade
    decade_data.plot(ax=ax, legend=False)
    # Set the tick formatter of the x-axis to only show the month name
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    # Make sure every month is labeled
    ax.xaxis.set_tick_params(rotation=0, which='major')

    # Set the y-axis limit to the maximum range of the whole dataset
    ax.set_ylim(-0.3, global_max*1.1)

    # Set the plot title and labels
    ax.set(title=f'{decade}-{decade+9}',
           xlabel=None, ylabel='mm')

# Create a common legend for all subplots at the bottom of the figure
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2)
# Add title
fig.suptitle('Average annual cycle of Evaporation and Precipitation', fontsize=16)
# Make sure the subplots don't overlap
plt.tight_layout()
# Add some space at the bottom of the figure for the legend
fig.subplots_adjust(bottom=0.06, top=0.92)
# Show the plot
plt.show()



##
import plotly.express as px
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

pio.renderers.default = "browser"

df_avg = df[['prec_off_glaciers', 'evap_off_glaciers']].rolling(window=30).mean()

# Split the data into decades
decades = range(df.index.year.min(), df.index.year.max() + 1, 10)

# Iterate over each decade to get global maximum
decade_max = []
for i, decade in enumerate(decades):
    # Filter the data for the current decade
    decade_data = df_avg.loc[(df_avg.index.year >= decade) & (df_avg.index.year < decade + 10)]
    # Compute the mean value for each day of the year for the current decade
    decade_data = decade_data.groupby([decade_data.index.month, decade_data.index.day]).mean()
    # Get maximum value
    decade_max.append(decade_data.max().max())

global_max = max(decade_max)



# Create a new figure with a 4x3 subplot grid
fig = make_subplots(rows=4, cols=3,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    subplot_titles=[f'{decade}-{decade+9}' for decade in decades])

# Iterate over each decade and create a plot for each
for i, decade in enumerate(decades):
    # Compute the row and column indices of the current subplot
    row = i // 3 + 1
    col = i % 3 + 1
    # Filter the data for the current decade
    decade_data = df_avg.loc[(df_avg.index.year >= decade) & (df_avg.index.year < decade + 10)]
    # Compute the mean value for each day of the year for the current decade
    decade_data = decade_data.groupby([decade_data.index.month, decade_data.index.day]).mean()
    # Rename and reset MultiIndex
    decade_data.index = decade_data.index.set_names(['Month', 'Day'])
    decade_data = decade_data.reset_index()
    # Create dummy datetime index (year irrelevant)
    decade_data['datetime'] = pd.to_datetime(
        '2000-' + decade_data['Month'].astype(str) + '-' + decade_data['Day'].astype(str))
    # Create a new subplot for the current decade
    fig.add_trace(go.Scatter(x=decade_data.datetime, y=decade_data.prec_off_glaciers, name='Precipitation',
                            line = dict(color='blue'),
                            showlegend=False,
                                                      # fill='tonext',  # fill to the next trace (Evaporation)
                                                      # fillcolor='lightblue',  # transparent blue
                             ),
                  row=row, col=col

                  )
    fig.add_trace(go.Scatter(x=decade_data.datetime, y=decade_data.evap_off_glaciers, name='Evaporation',
                             line=dict(color='orange'),
                             showlegend=False,
                             # fill='tonexty',  # fill to the next trace (Precipitation)
                             # fillcolor='honeydew'  # transparent orange
                             ),
                  row=row, col=col)


fig.update_traces(selector=-1, showlegend=True)         # Show legend for the last trace (Evaporation)
fig.update_traces(selector=-2, showlegend=True)         # Show legend for the trace before the last (Precipitation)
fig.update_yaxes(
    range=[0, global_max],
    showgrid=True, gridcolor='lightgrey')
fig.update_xaxes(
    dtick="M1",
    tickformat="%b",
    hoverformat='%b %d',
    showgrid=True, gridcolor='lightgrey')
fig.update_layout(
    hovermode='x',
    margin=dict(l=10, r=10, t=90, b=10),  # Adjust the margins to remove space around the plot
    plot_bgcolor='white',  # set the background color of the plot to white


)

fig.show()


# To-dos:

    # Add shading humid/arid
    # Add units
    # generalize to fit to other variables as well