import pickle
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
test = df['total_runoff']

print(df.columns)

## What to analyze?

# Annual stats
    # Month with highest precipitation
    # DoY with highest Runoff
    # start, end, and length of melting season:
        # melt variables depend on availability of snow/ice AND temperature!
        # Temperature might be better --> three consecutive days above 0°C
    # start, end, and length of wet/dry season:
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




## Long-term annual cycle of evaporation and precipitation
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


df_avg = df[['evap_off_glaciers', 'prec_off_glaciers']]

df_avg = df[['evap_off_glaciers', 'prec_off_glaciers']].groupby([df.index.month, df.index.day]).mean()
df_avg["date"] = pd.date_range(df.index[0], freq='D', periods=len(df_avg)).strftime(
            '%Y-%m-%d')
df_avg.index = pd.to_datetime(df_avg["date"])

# plot the data
fig, ax = plt.subplots()
df_avg.plot(ax=ax)

# set the tick formatter of the x-axis to only show the month name
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# make sure every month is labeled
ax.xaxis.set_tick_params(rotation=0, which='major')

# set the plot title and labels
ax.set(title='10-year average annual cycle of Evaporation and Precipitation',
       xlabel=None, ylabel='mm')

# show the plot
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

##  PROBLEM: EVAPORATION IS MUCH LOWER FOR MOST YEARS. NO DISTINCT DRY PERIOD. PARAMETER PROBLEM?
def dry_season(df):
    """
    Compute the start, end, and length of the dry season for each calendar year based on the daily mean "total_precipitation" and "evapotranspiration" data provided in the input dataframe.
    The start of the dry season is defined as the first day of the first two-week period where the average "evapotranspiration" is larger than the average "total_precipitation".
    The end of the dry season is the first day of the first two-week period where the average "evapotranspiration" is smaller than the average "total_precipitation".
    The length of the dry season is the number of days between the start and end of the dry season.
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of daily mean "prec_off_glaciers" and "evap_off_glaciers" data with a datetime index.
    Returns
    -------
    pandas.DataFrame
        A DataFrame with the start, end, and length of the dry season for each calendar year, with a DatetimeIndex.
    """

    # Find the start of the dry season for each year
    start_dates = []
    for year in df.index.year.unique():
        year_data = df.loc[df.index.year == year]
        year_roll = year_data['evap_off_glaciers'].rolling(window=30).mean()
        prec_roll = year_data['prec_off_glaciers'].rolling(window=30).mean()
        if (year_roll > prec_roll).any():
            start_index = year_roll[year_roll > prec_roll].index[0]
            start_index = start_index - pd.Timedelta(days=29)
            start_dates.append(start_index)
        else:
            start_dates.append(pd.NaT)

            # Find the end of the dry season for each year
    end_dates = []
    for year in df.index.year.unique():
        year_data = df.loc[df.index.year == year]
        year_roll = year_data['evap_off_glaciers'].rolling(window=30).mean()
        prec_roll = year_data['prec_off_glaciers'].rolling(window=30).mean()
        start_index = start_dates[year - df.index.year.min()]
        if not start_index:
            end_dates.append(pd.NaT)
        else:
            year_roll = year_roll.loc[start_index + pd.Timedelta(weeks=6):]
            prec_roll = prec_roll.loc[start_index + pd.Timedelta(weeks=6):]
            if (year_roll < prec_roll).any():
                end_index = year_roll[year_roll < prec_roll].index[0]
                end_index = end_index - pd.Timedelta(days=29)
                end_dates.append(end_index)
            else:
                end_dates.append(pd.NaT)

                # Compute the length of the dry season for each year
    lengths = [pd.NaT if not start_dates[i] or not end_dates[i] else (end_dates[i] - start_dates[i]).days for i in
               range(len(start_dates))]

    # Assemble the output dataframe
    output_df = pd.DataFrame(
        {'dry_season_start': [d.timetuple().tm_yday if not pd.isnull(d) else pd.NaT for d in start_dates],
         'dry_season_end': [d.timetuple().tm_yday if not pd.isnull(d) else pd.NaT for d in end_dates],
         'dry_season_length': lengths},
        index=pd.to_datetime(df.index.year.unique(), format='%Y'))
    return output_df

dry = dry_season(df)
