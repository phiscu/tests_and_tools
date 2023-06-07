import pickle
import numpy as np
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
    # Percentage of Normal Precipitation (PNP)

## Functions

# Month with maximum precipitation

def prec_minmax(df, max=True):
    """
    Compute the month(s) of extreme precipitation for each year.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of daily precipitation data with a datetime index and a 'total_precipitation' column.
    max : bool, optional
        If True, return the month(s) of maximum precipitation. Otherwise, return the month(s) of minimum precipitation.
        Default is True.
    Returns
    -------
    pandas.DataFrame
        A DataFrame with the month(s) of extreme precipitation as a number for every calendar year.
    """
    # group the data by year and month and sum the precipitation values
    grouped = df.groupby([df.index.year, df.index.month]).sum()
    # get the month with extreme precipitation for each year
    if max:
        extreme_month = grouped.groupby(level=0)['total_precipitation'].idxmax()
    else:
        extreme_month = grouped.groupby(level=0)['total_precipitation'].idxmin()
    extreme_month = [p[1] for p in extreme_month]
    # create a new dataframe
    if max:
        result = pd.DataFrame({'max_prec_month': extreme_month}, index=grouped.index.levels[0])
    else:
        result = pd.DataFrame({'min_prec_month': extreme_month}, index=grouped.index.levels[0])
    return result


# Day of the Year with maximum flow

def peak_doy(df, smoothing_window=7):
    """
    Compute the day of the year with the peak value for each hydrological year.
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
    # resample data to get daily values, forward fill missing values
    daily_data = df.resample('D').ffill()

    # create hydrological years starting from October 1
    hy_start_dates = pd.date_range(start=daily_data.index[0], end=daily_data.index[-1], freq='AS-OCT')
    hy_end_dates = hy_start_dates + pd.DateOffset(months=9, days=30)
    hydro_years = [(start, end) for start, end in zip(hy_start_dates, hy_end_dates)]

    # find peak day for each hydrological year
    peak_dates = []
    for start, end in hydro_years:
        # slice data for hydrological year
        hy_data = daily_data[(daily_data.index >= start) & (daily_data.index <= end)]

        # smooth data using rolling mean with window of 7 days
        smoothed_data = hy_data.rolling(smoothing_window, center=True).mean()

        # find day of peak value
        peak_day = smoothed_data.idxmax().strftime('%j')

        # append peak day to list
        peak_dates.append(peak_day)

    # create output dataframe with DatetimeIndex
    output_df = pd.DataFrame({'Hydrological Year': [start.year for start, end in hydro_years],
                              'Peak Day of Year': pd.to_numeric(peak_dates)})
    output_df.index = pd.to_datetime(output_df['Hydrological Year'], format='%Y')
    output_df = output_df.drop('Hydrological Year', axis=1)

    # Delete the last (incomplete) hydrological year
    output_df = output_df[:-1]

    return output_df


# Melting season

def melting_season(df):
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
        year_roll = year_data.rolling(window=14).mean()
        start_index = year_roll[year_roll > 0].index[0]
        start_index = start_index - pd.Timedelta(days=13)  # rolling selects last day of window, we want the first
        start_dates.append(start_index)

    # Find the end of the melting season for each year
    end_dates = []
    for year in df.index.year.unique():
        year_data = df.loc[df.index.year == year, 'avg_temp_catchment']
        year_roll = year_data.rolling(window=14).mean()
        start_index = start_dates[year - df.index.year.min()]
        year_roll = year_roll.loc[start_index + pd.Timedelta(weeks=10):]  # add min season duration of 10 weeks
        end_index = year_roll[year_roll < 0].index[0]
        end_index = end_index - pd.Timedelta(days=13)  # rolling selects last day of window, we want the first
        end_dates.append(end_index)

    # Compute the length of the melting season for each year
    lengths = [(end_dates[i] - start_dates[i]).days for i in range(len(start_dates))]

    # Assemble the output dataframe
    output_df = pd.DataFrame({'melt_season_start': [d.timetuple().tm_yday for d in start_dates],
                              'melt_season_end': [d.timetuple().tm_yday for d in end_dates],
                              'melt_season_length': lengths},
                             index=pd.to_datetime(df.index.year.unique(), format='%Y'))
    return output_df


# Total runoff ratio

def runoff_ratio(df):
    """
    Calculates the proportion of precipitation that does not infiltrate and or evapotranspirate.
     Parameters:
    -----------
        df: pandas.DataFrame
            DataFrame containing the variables 'total_runoff' and 'total_precipitation'.
     Returns:
    --------
        pandas.DataFrame
            DataFrame containing the runoff ratio for each observation in the input DataFrame,
            with the same datetime index as the input DataFrame.
            If 'total_precipitation' is 0, the corresponding runoff ratio is set to 0.
    """
    runoff_ratio = np.where(df['total_precipitation'] == 0, 0, df['total_runoff'] / df['total_precipitation'])
    return pd.DataFrame(data=runoff_ratio, index=df.index, columns=['runoff_ratio'])




## Long-term annual cycle of evaporation and precipitation for every decade

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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


## Hydrological signatures

import spotpy.hydrology.signatures as sig

# Time series:
sig.calc_baseflow(test)         # 5-day baseflow
# Two outputs:
sig.get_qlf(test)               # frequency and mean duration of low flow events defined as Q < 2⋅Qmean per year
sig.get_qhf(test)               # frequency and mean duration of high flow events defined as Q > 9⋅Q50 per year
# one ouput:
sig.get_bfi(test)               # baseflow index
sig.get_q5(test)                # Quantiles....
sig.get_q50(test)
sig.get_q95(test)
sig.get_qcv(test)               # variation coeff
sig.get_qhv(test)               # high flow variability
sig.get_qlv(test)               # low flow variability
sig.get_sfdc(test)              # slope in the middle part of the flow duration curve


##
def dry_periods(df, period_length=30):
    """
    Compute the number of days for which the rolling mean of evaporation exceeds precipitation for each year in the
    input DataFrame.
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

    # Find number of days when the rolling mean of evaporation exceeds precipitation
    periods = []
    for year in df.index.year.unique():
        year_data = df.loc[df.index.year == year]
        evap_roll = year_data['evap_off_glaciers'].rolling(window=period_length).mean()
        prec_roll = year_data['prec_off_glaciers'].rolling(window=period_length).mean()

        dry = evap_roll[evap_roll - prec_roll > 0]
        periods.append(len(dry))

    # Assemble the output dataframe
    output_df = pd.DataFrame(
        {'dry_period_days': periods},
        index=pd.to_datetime(df.index.year.unique(), format='%Y'))

    return output_df


dry_periods(df).plot()
plt.show()