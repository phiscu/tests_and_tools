import pickle
import numpy as np
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


def dict_to_parquet(dictionary: dict, directory_path: str, pbar: bool = True) -> None:
    """
    Recursively stores the dataframes in the input dictionary as parquet files in the specified directory.
    Nested dictionaries are supported. If the specified directory does not exist, it will be created.
    Parameters
    ----------
    dictionary : dict
        A nested dictionary containing pandas dataframes.
    directory_path : str
        The directory path to store the parquet files.
    pbar : bool, optional
        A flag indicating whether to display a progress bar. Default is True.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if pbar:
        bar_iter = tqdm(dictionary.items(), desc='Writing parquet files: ')
    else:
        bar_iter = dictionary.items()
    for k, v in bar_iter:
        if isinstance(v, dict):
            dict_to_parquet(v, os.path.join(directory_path, k), pbar=False)
        else:
            file_path = os.path.join(directory_path, k + ".parquet")
            write(file_path, v, compression='GZIP')


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


matilda_scenarios = pickle_to_dict(test_dir + 'adjusted/matilda_scenarios.pickle')   # pickle for speed/parquet for size
df = matilda_scenarios['SSP2']['EC-Earth3']['model_output']
# print(df.columns)

## Functions

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


# Aridity
def aridity(df, hist_starty=1986, hist_endy=2015):
    """
    Calculates aridity indexes from precipitation, and potential and actual evaporation respectively. Aridity is defined
    as mean annual ratio of potential/actual evapotranspiration and precipitation. The indexes are defined as the
    relative change of a 30 years period compared to a given historical period. Uses hydrological years (Oct - Sep).
    Inspired by climateinformation.org (https://doi.org/10.5194/egusphere-egu23-16216).
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns 'evap_off_glaciers', 'actual_evaporation', and 'prec_off_glaciers'.
    hist_starty : int, optional
        Start year of the historical period in YYYY format. Default is 1986.
    hist_endy : int, optional
        End year of the historical period in YYYY format. Default is 2015.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the relative change in aridity over time.
        Columns:
            - 'actual_aridity': Relative change in actual aridity.
            - 'potential_aridity': Relative change in potential aridity.
    """
    # Use water years
    df = hydrologicalize(df)
    # Potential evapotranspiration (PET)
    pet = df['evap_off_glaciers']
    # Actual evapotranspiration (AET)
    aet = df['actual_evaporation']
    # Precipitation
    prec = df['prec_off_glaciers']
    # Calculate the potential aridity as ratio of AET/PET to precipitation
    aridity_pot = pet.groupby(df['water_year']).sum() / prec.groupby(df['water_year']).sum()
    aridity_act = aet.groupby(df['water_year']).sum() / prec.groupby(df['water_year']).sum()
    # Filter historical period
    hist_pot = aridity_pot[(aridity_pot.index >= hist_starty) & (aridity_pot.index <= hist_endy)].mean()
    hist_act = aridity_act[(aridity_act.index >= hist_starty) & (aridity_act.index <= hist_endy)].mean()
    # Calculate rolling mean with a 30y period
    aridity_pot_rolling = aridity_pot.rolling(window=30).mean()
    aridity_act_rolling = aridity_act.rolling(window=30).mean()
    # Calculate the relative change in the aridity indexes
    pot_rel = 100 * (aridity_pot_rolling - hist_pot) / hist_pot
    act_rel = 100 * (aridity_act_rolling - hist_act) / hist_act
    # Concat in one dataframe
    aridity = pd.DataFrame({'actual_aridity': act_rel, 'potential_aridity': pot_rel})
    aridity.set_index(pd.to_datetime(df.water_year.unique(), format='%Y'), inplace=True)
    aridity = aridity.dropna()


    return aridity


# Dry spells
def dry_spells(df, dry_spell_length=5):
    """
    Compute the total length of dry spells in days per year. A dry spell is defined as a period for which the rolling
    mean of evaporation in a given window exceeds precipitation. Uses hydrological years (Oct - Sep).
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns 'evap_off_glaciers' and 'prec_off_glaciers' with daily evaporation and
        precipitation data, respectively.
    dry_spell_length : int, optional
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
        evap_roll = year_data['evap_off_glaciers'].rolling(window=dry_spell_length).mean()
        prec_roll = year_data['prec_off_glaciers'].rolling(window=dry_spell_length).mean()

        dry = evap_roll[evap_roll - prec_roll > 0]
        periods.append(len(dry))

    # Assemble the output dataframe
    output_df = pd.DataFrame(
        {'dry_spell_days': periods},
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


# Drought indicators
def drought_indicators(df, freq='M', dist='gamma'):
    """
    Calculate the climatic water balance, SPI (Standardized Precipitation Index), and
    SPEI (Standardized Precipitation Evapotranspiration Index) for 1, 3, 6, 12, and 24 months..
    Parameters
    ----------
    df : pandas.DataFrame
         Input DataFrame containing columns 'prec_off_glaciers' and 'evap_off_glaciers'.
    freq : str, optional
         Resampling frequency for precipitation and evaporation data. Default is 'M' for monthly.
    dist : str, optional
         Distribution for SPI and SPEI calculation. Either Pearson-Type III ('pearson') or
         Gamma distribution ('gamma'). Default is 'gamma'.
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
    The default distribution for SPI and SPEI calculation is Gamma.
    The calibration period for SPI and SPEI calculation is th full data range from 1981 to 2100.
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
        distribution = Distribution.gamma
    else:
        raise ValueError("Invalid value for 'dist'. Choose either 'pearson' or 'gamma'.")

    # Set periodicity based on frequency
    if freq == 'D':
        periodicity = compute.Periodicity.daily
    elif freq == 'M':
        periodicity = compute.Periodicity.monthly

    # Set common parameters
    common_params = {'distribution': distribution,
                     'periodicity': periodicity,
                     'data_start_year': 1981,
                     'calibration_year_initial': 1981,
                     'calibration_year_final': 2100}

    # Set parameters for SPEI calculation
    spei_params = {'precips_mm': prec,
                   'pet_mm': evap,
                   **common_params}

    # Set parameters for SPI calculation
    spi_params = {'values': prec,
                  **common_params}

    # Calculate SPI and SPEI for various periods
    drought_df = pd.DataFrame()
    for s in [1, 3, 6, 12, 24]:
        spi_arr = spi(**spi_params, scale=s)
        spei_arr = spei(**spei_params, scale=s)
        # If frequency is daily, transform data back to Gregorian format
        if freq == 'D':
            spi_arr = utils.transform_to_gregorian(spi_arr, df.index.year[0])
            spei_arr = utils.transform_to_gregorian(spei_arr, df.index.year[0])
        drought_df['spi' + str(s)] = spi_arr
        drought_df['spei' + str(s)] = spei_arr
    drought_df.set_index(df.resample(freq).mean().index, inplace=True)

    # DataFrame resample Dataframe
    out_df = pd.DataFrame({'clim_water_balance': water_balance}, index=df.resample(freq).sum().index)
    out_df = pd.concat([out_df.resample('YS').sum(), drought_df.resample('YS').mean()], axis=1).rename(lambda x: x.replace(day=1))

    return out_df


# Wrapper function
import inspect
def cc_indicators(df, **kwargs):
    """
    Apply a list of climate change indicator functions to output DataFrame of MATILDA and concatenate
    the output columns into a single DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    **kwargs : optional
        Optional arguments to be passed to the functions in the list. Possible arguments are 'smoothing_window_peakdoy',
        'smoothing_window_meltseas', 'min_weeks', and 'dry_spell_length'.
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
    functions = [prec_minmax, peak_doy, melting_season, aridity, dry_spells, hydrological_signatures, drought_indicators]
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


## Performance check
# %%time
# print('prec_minmax')
# %time prec_minmax(df)
# print('peak_doy')
# %time peak_doy(df)
# print('melting_season')
# %time melting_season(df)
# print('aridity')
# %time aridity(df)
# print('dry_spells')
# %time dry_spells(df)
# print('hydrological_signatures')
# %time hydrological_signatures(df)
# print('drought_indicators')
# %time drought_indicators(df, 'D')        # very slow in the first iteration, fast afterwards
# print()


## Loop indicator function over all models

def calculate_indicators(dic, **kwargs):
    """
    Calculate climate change indicators for all scenarios and models.
    Parameters
    ----------
    dic : dict
        Dictionary containing MATILDA outputs for all scenarios and models.
    **kwargs : optional
        Optional keyword arguments to be passed to the cc_indicators() function.
    Returns
    -------
    dict
        Dictionary with the same structure as the input but containing climate change indicators in annual resolution.
    """
    # Create an empty dictionary to store the outputs
    out_dict = {}
    # Loop over the scenarios with progress bar
    for scenario in dic.keys():
        model_dict = {}  # Create an empty dictionary to store the model outputs
        # Loop over the models with progress bar
        for model in tqdm(dic[scenario].keys(), desc=scenario):
            # Get the dataframe for the current scenario and model
            df = dic[scenario][model]['model_output']
            # Run the indicator function
            indicators = cc_indicators(df, **kwargs)
            # Store indicator time series in the model dictionary
            model_dict[model] = indicators
        # Store the model dictionary in the scenario dictionary
        out_dict[scenario] = model_dict

    return out_dict


# matilda_indicators = calculate_indicators(matilda_scenarios)
# dict_to_parquet(matilda_indicators, test_dir + 'adjusted/matilda_indicators.parquet')
# dict_to_pickle(matilda_indicators, test_dir + 'adjusted/matilda_indicators.pickle')
matilda_indicators = pickle_to_dict(test_dir + 'adjusted/matilda_indicators.pickle')

## Store variable names and respective plot labels
# var_name = ['max_prec_month', 'min_prec_month',
#             'peak_day',
#             'melt_season_start', 'melt_season_end', 'melt_season_length',
#             'actual_aridity', 'potential_aridity',
#             'dry_spell_days',
#             'qlf_freq', 'qlf_dur', 'qhf_freq', 'qhf_dur',
#             'clim_water_balance', 'spi1', 'spei1', 'spi3', 'spei3', 'spi6', 'spei6', 'spi12', 'spei12', 'spi24', 'spei24']
#
# title = ['Month with Maximum Precipitation', 'Month with Minimum Precipitation',
#          'Timing of Peak Runoff',
#          'Beginning of Melting Season', 'End of Melting Season', 'Length of Melting Season',
#          'Relative Change of Actual Aridity', 'Relative Change of Potential Aridity',
#          'Total Length of Dry Spells per year',
#          'Frequency of Low-flow events', 'Mean Duration of Low-flow events',
#          'Frequency of High-flow events', 'Mean Duration of High-flow events',
#          'Climatic Water Balance',
#          'Standardized Precipitation Index (1 month)', 'Standardized Precipitation Evaporation Index (1 month)',
#          'Standardized Precipitation Index (3 months)', 'Standardized Precipitation Evaporation Index (3 months)',
#          'Standardized Precipitation Index (6 months)', 'Standardized Precipitation Evaporation Index (6 months)',
#          'Standardized Precipitation Index (12 months)', 'Standardized Precipitation Evaporation Index (12 months)',
#          'Standardized Precipitation Index (24 months)', 'Standardized Precipitation Evaporation Index (24 months)']
#
# unit = ['-', '-',
#         'DoY',
#         'DoY', 'DoY', 'd',
#         '%', '%',
#         'd/a',
#         'yr^-1', 'd', 'yr^-1', 'd',
#         'mm w.e.', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
#
# indicator_vars = {key: (val1, val2) for key, val1, val2 in zip(var_name, title, unit)}

indicator_vars = {'max_prec_month': ('Month with Maximum Precipitation', '-'),
 'min_prec_month': ('Month with Minimum Precipitation', '-'),
 'peak_day': ('Timing of Peak Runoff', 'DoY'),
 'melt_season_start': ('Beginning of Melting Season', 'DoY'),
 'melt_season_end': ('End of Melting Season', 'DoY'),
 'melt_season_length': ('Length of Melting Season', 'd'),
 'actual_aridity': ('Relative Change of Actual Aridity', '%'),
 'potential_aridity': ('Relative Change of Potential Aridity', '%'),
 'dry_spell_days': ('Total Length of Dry Spells per year', 'd/a'),
 'qlf_freq': ('Frequency of Low-flow events', 'yr^-1'),
 'qlf_dur': ('Mean Duration of Low-flow events', 'd'),
 'qhf_freq': ('Frequency of High-flow events', 'yr^-1'),
 'qhf_dur': ('Mean Duration of High-flow events', 'd'),
 'clim_water_balance': ('Climatic Water Balance', 'mm w.e.'),
 'spi1': ('Standardized Precipitation Index (1 month)', '-'),
 'spei1': ('Standardized Precipitation Evaporation Index (1 month)', '-'),
 'spi3': ('Standardized Precipitation Index (3 months)', '-'),
 'spei3': ('Standardized Precipitation Evaporation Index (3 months)', '-'),
 'spi6': ('Standardized Precipitation Index (6 months)', '-'),
 'spei6': ('Standardized Precipitation Evaporation Index (6 months)', '-'),
 'spi12': ('Standardized Precipitation Index (12 months)', '-'),
 'spei12': ('Standardized Precipitation Evaporation Index (12 months)', '-'),
 'spi24': ('Standardized Precipitation Index (24 months)', '-'),
 'spei24': ('Standardized Precipitation Evaporation Index (24 months)', '-')}


## Create custom df for individual plot
def custom_df_indicators(dic, scenario, var):
    """
    Takes a dictionary of climate change indicators and returns a combined dataframe of a specific variable for
    a given scenario.
    Parameters
    ----------
    dic : dict
        Dictionary containing the outputs of calculate_indicators() for different scenarios and models.
    scenario : str
        Name of the selected scenario.
    var : str
        Name of the variable to extract from the DataFrame.
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the selected variable from different models within the specified scenario.
    Raises
    ------
    ValueError
        If the provided variable is not one of the function outputs.
    """

    out_cols = ['max_prec_month', 'min_prec_month',
                'peak_day',
                'melt_season_start', 'melt_season_end', 'melt_season_length',
                'actual_aridity', 'potential_aridity',
                'dry_spell_days',
                'qlf_freq', 'qlf_dur', 'qhf_freq', 'qhf_dur',
                'clim_water_balance', 'spi1', 'spei1', 'spi3', 'spei3',
                'spi6', 'spei6', 'spi12', 'spei12', 'spi24', 'spei24']

    if var not in out_cols:
        raise ValueError("var needs to be one of the following strings: " +
                         str([i for i in out_cols]))

    # Create an empty list to store the dataframes
    dfs = []
    # Loop over the models in the selected scenario
    for model in dic[scenario].keys():
        # Get the dataframe for the current model
        df = dic[scenario][model]
        # Append the dataframe to the list of dataframes
        dfs.append(df[var])
    # Concatenate the dataframes into a single dataframe
    combined_df = pd.concat(dfs, axis=1)
    # Set the column names of the combined dataframe to the model names
    combined_df.columns = dic[scenario].keys()

    return combined_df


## Plot functions

import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
pio.renderers.default = "browser"

def confidence_interval(df):
    """
    Calculate the mean and 95% confidence interval for each row in a dataframe.
    Parameters:
    -----------
        df (pandas.DataFrame): The input dataframe.
    Returns:
    --------
        pandas.DataFrame: A dataframe with the mean and confidence intervals for each row.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    count = df.count(axis=1)
    ci = 1.96 * std / np.sqrt(count)
    ci_lower = mean - ci
    ci_upper = mean + ci
    df_ci = pd.DataFrame({'mean': mean, 'ci_lower': ci_lower, 'ci_upper': ci_upper})
    return df_ci

def plot_ci_indicators(var, dic, plot_type='line', show=False):
    """
    A function to plot multi-model mean and confidence intervals of a given variable for two different scenarios.
    Parameters:
    -----------
    var: str
        The variable to plot.
    dic: dict, optional (default=matilda_scenarios)
        A dictionary containing the scenarios as keys and the dataframes as values.
    plot_type: str, optional (default='line')
        Whether the plot should be a line or a bar plot.
    show: bool, optional (default=False)
        Whether to show the resulting plot or not.
    Returns:
    --------
    go.Figure
        A plotly figure object containing the mean and confidence intervals for the given variable in the two selected scenarios.
    """

    if var is None:
        var = 'total_runoff'       # Default if nothing selected

    # SSP2
    df1 = custom_df_indicators(dic, scenario='SSP2', var=var)
    df1_ci = confidence_interval(df1)
    # SSP5
    df2 = custom_df_indicators(dic, scenario='SSP5', var=var)
    df2_ci = confidence_interval(df2)

    if plot_type == 'line':
        fig = go.Figure([
        # SSP2
        go.Scatter(
            name='SSP2',
            x=df1_ci.index,
            y=round(df1_ci['mean'], 2),
            mode='lines',
            line=dict(color='darkorange'),
        ),
        go.Scatter(
            name='95% CI Upper',
            x=df1_ci.index,
            y=round(df1_ci['ci_upper'], 2),
            mode='lines',
            marker=dict(color='#444'),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='95% CI Lower',
            x=df1_ci.index,
            y=round(df1_ci['ci_lower'], 2),
            marker=dict(color='#444'),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(255, 165, 0, 0.3)',
            fill='tonexty',
            showlegend=False
        ),

        # SSP5
        go.Scatter(
            name='SSP5',
            x=df2_ci.index,
            y=round(df2_ci['mean'], 2),
            mode='lines',
            line=dict(color='darkblue'),
        ),
        go.Scatter(
            name='95% CI Upper',
            x=df2_ci.index,
            y=round(df2_ci['ci_upper'], 2),
            mode='lines',
            marker=dict(color='#444'),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='95% CI Lower',
            x=df2_ci.index,
            y=round(df2_ci['ci_lower'], 2),
            marker=dict(color='#444'),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(0, 0, 255, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    elif plot_type == 'bar':
        fig = go.Figure([
            # SSP2
            go.Bar(
                name='SSP2',
                x=df1_ci.index,
                y=round(df1_ci['mean'], 2),
                marker=dict(color='darkorange'),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=round(df1_ci['mean'] - df1_ci['ci_lower'], 2),
                    arrayminus=round(df1_ci['ci_upper'] - df1_ci['mean'], 2),
                    color='grey'
                )
            ),
            # SSP5
            go.Bar(
                name='SSP5',
                x=df2_ci.index,
                y=round(df2_ci['mean'], 2),
                marker=dict(color='darkblue'),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=round(df2_ci['mean'] - df2_ci['ci_lower'], 2),
                    arrayminus=round(df2_ci['ci_upper'] - df2_ci['mean'], 2),
                    color='grey'
                )
            )
        ])
    else:
        raise ValueError("Invalid property specified for 'plot_type'. Choose either 'line' or 'bar'")

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=indicator_vars[var][0] + ' [' + indicator_vars[var][1] + ']',
        title={'text': '<b>' + indicator_vars[var][0] + '</b>', 'font': {'size': 28, 'color': 'darkblue', 'family': 'Arial'}},
        legend={'font': {'size': 18, 'family': 'Arial'}},
        hovermode='x',
        plot_bgcolor='rgba(255, 255, 255, 1)',  # Set the background color to white
        margin=dict(l=10, r=10, t=90, b=10),  # Adjust the margins to remove space around the plot
        xaxis=dict(gridcolor='lightgrey'),  # set the grid color of x-axis to lightgrey
        yaxis=dict(gridcolor='lightgrey'),  # set the grid color of y-axis to lightgrey
    )
    fig.update_yaxes(rangemode='tozero')

    # show figure
    if show:
        fig.show()
    else:
        return fig


##
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.io as pio
pio.renderers.default = "browser"
app = dash.Dash()

# Create default variables for every figure
default_vars = ['peak_day', 'melt_season_length', 'potential_aridity', 'spei12']
default_types = ['line', 'line', 'line', 'bar']

default_vars = ['melt_season_length', 'potential_aridity', 'spei12']
default_types = ['line', 'line', 'bar']

# Create separate callback functions for each dropdown menu and graph combination
for i in range(3):
    @app.callback(
        Output(f'line-plot-{i}', 'figure'),
        Input(f'arg-dropdown-{i}', 'value'),
        Input(f'type-dropdown-{i}', 'value')
    )
    def update_figure(selected_arg, selected_type, i=i):
        fig = plot_ci_indicators(selected_arg, matilda_indicators, selected_type)
        return fig

# Define the dropdown menus and figures
dropdowns_and_figures = []
for i in range(3):
    arg_dropdown = dcc.Dropdown(
        id=f'arg-dropdown-{i}',
        options=[{'label': indicator_vars[var][0], 'value': var} for var in indicator_vars.keys()],
        value=default_vars[i],
        clearable=False,
        style={'width': '400px', 'fontFamily': 'Arial', 'fontSize': 15}
    )
    type_dropdown = dcc.Dropdown(
        id=f'type-dropdown-{i}',
        options=[{'label': lab, 'value': val} for lab, val in [('Line', 'line'), ('Bar', 'bar')]],
        value=default_types[i],
        clearable=False,
        style={'width': '150px'}
    )
    dropdowns_and_figures.append(
        html.Div([
            html.Div([
                html.Label("Variable:"),
                arg_dropdown,
            ], style={'display': 'inline-block', 'margin-right': '30px'}),
            html.Div([
                html.Label("Plot Type:"),
                type_dropdown,
            ], style={'display': 'inline-block'}),
            dcc.Graph(id=f'line-plot-{i}'),
        ])
    )
# Combine the dropdown menus and figures into a single layout
app.layout = html.Div(dropdowns_and_figures)
# Run the app
app.run_server(debug=True, use_reloader=False)  # Turn off reloader inside Jupyter

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
#     yaxis_title=indicator_vars[var][0] + ' [' + indicator_vars[var][1] + ']',
#     title={'text': '<b>' + indicator_vars[var][0] + '</b>', 'font': {'size': 28, 'color': 'darkblue', 'family': 'Arial'}},
#     legend={'font': {'size': 18, 'family': 'Arial'}},
#
#
# )
#
# fig.show()
#




# To-dos:

    # Add shading humid/arid
    # Add units
    # generalize to fit to other variables as well
    # add climate-indices to requirements
