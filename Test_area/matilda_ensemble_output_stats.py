##
import matplotlib.pyplot as plt
import pickle
import configparser
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

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

dir_output = "/home/phillip/Seafile/EBA-CA/Repositories/matilda_edu/output/"

print("Importing MATILDA scenarios...")
# For size:
# matilda_scenarios = parquet_to_dict(f"{dir_output}cmip6/adjusted/matilda_scenarios_parquet")

# For speed:
matilda_scenarios = pickle_to_dict(f"{dir_output}cmip6/adjusted/matilda_scenarios.pickle")

for m in matilda_scenarios['SSP2'].keys():
    print(m)

##

def custom_df_matilda(dic, scenario, var, resample_freq=None):
    """
    Takes a dictionary of model outputs and returns a combined dataframe of a specific variable for a given scenario.
    Parameters
    -------
    dic : dict
        A nested dictionary of model outputs.
        The outer keys are scenario names and the inner keys are model names.
        The corresponding values are dictionaries containing two keys:
        'model_output' (DataFrame): containing model outputs for a given scenario and model
        'glacier_rescaling' (DataFrame): containing glacier properties for a given scenario and model
    scenario : str
        The name of the scenario to select from the dictionary.
    var : str
        The name of the variable to extract from the model output DataFrame.
    resample_freq : str, optional
        The frequency of the resulting time series data.
        Defaults to None (i.e. no resampling).
        If provided, should be in pandas resample frequency string format.
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the combined data of the specified variable for the selected scenario
        and models. The DataFrame is indexed by the time steps of the original models.
        The columns are the names of the models in the selected scenario.
    Raises
    -------
    ValueError
        If the provided  var  string is not one of the following: ['avg_temp_catchment', 'avg_temp_glaciers',
        'evap_off_glaciers', 'prec_off_glaciers', 'prec_on_glaciers', 'rain_off_glaciers', 'snow_off_glaciers',
        'rain_on_glaciers', 'snow_on_glaciers', 'snowpack_off_glaciers', 'soil_moisture', 'upper_groundwater',
        'lower_groundwater', 'melt_off_glaciers', 'melt_on_glaciers', 'ice_melt_on_glaciers', 'snow_melt_on_glaciers',
        'refreezing_ice', 'refreezing_snow', 'total_refreezing', 'SMB', 'actual_evaporation', 'total_precipitation',
        'total_melt', 'runoff_without_glaciers', 'runoff_from_glaciers', 'runoff_ratio', 'total_runoff', 'glacier_area',
        'glacier_elev', 'smb_water_year', 'smb_scaled', 'smb_scaled_capped', 'smb_scaled_capped_cum', 'surplus',
        'glacier_melt_perc', 'glacier_mass_mmwe', 'glacier_vol_m3', 'glacier_vol_perc']
    """
    out1_cols = ['avg_temp_catchment',
                 'avg_temp_glaciers',
                 'evap_off_glaciers',
                 'prec_off_glaciers',
                 'prec_on_glaciers',
                 'rain_off_glaciers',
                 'snow_off_glaciers',
                 'rain_on_glaciers',
                 'snow_on_glaciers',
                 'snowpack_off_glaciers',
                 'soil_moisture',
                 'upper_groundwater',
                 'lower_groundwater',
                 'melt_off_glaciers',
                 'melt_on_glaciers',
                 'ice_melt_on_glaciers',
                 'snow_melt_on_glaciers',
                 'refreezing_ice',
                 'refreezing_snow',
                 'total_refreezing',
                 'SMB',
                 'actual_evaporation',
                 'total_precipitation',
                 'total_melt',
                 'runoff_without_glaciers',
                 'runoff_from_glaciers',
                 'runoff_ratio',
                 'total_runoff']

    out2_cols = ['glacier_area',
                 'glacier_elev',
                 'smb_water_year',
                 'smb_scaled',
                 'smb_scaled_capped',
                 'smb_scaled_capped_cum',
                 'surplus',
                 'glacier_melt_perc',
                 'glacier_mass_mmwe',
                 'glacier_vol_m3',
                 'glacier_vol_perc']

    if var in out1_cols:
        output_df = 'model_output'
    elif var in out2_cols:
        output_df = 'glacier_rescaling'
    else:
        raise ValueError("var needs to be one of the following strings: " +
                         str([i for i in [out1_cols, out2_cols]]))

    # Create an empty list to store the dataframes
    dfs = []
    # Loop over the models in the selected scenario
    for model in dic[scenario].keys():
        # Get the dataframe for the current model
        df = dic[scenario][model][output_df]
        # Append the dataframe to the list of dataframes
        dfs.append(df[var])
    # Concatenate the dataframes into a single dataframe
    combined_df = pd.concat(dfs, axis=1)
    # Set the column names of the combined dataframe to the model names
    combined_df.columns = dic[scenario].keys()
    # Resample time series
    if resample_freq is not None:
        if output_df == 'glacier_rescaling':
            if resample_freq == '10Y':
                if var in ['glacier_area', 'glacier_elev']:
                    combined_df = combined_df.resample(resample_freq).mean()
                else:
                    combined_df = combined_df.resample(resample_freq).sum()
        else:
            if var in ['avg_temp_catchment', 'avg_temp_glaciers']:
                combined_df = combined_df.resample(resample_freq).mean()
            else:
                combined_df = combined_df.resample(resample_freq).sum()

    return combined_df


# [c for c in matilda_scenarios['SSP2']['CESM2']['model_output'].columns]
# [c for c in matilda_scenarios['SSP2']['CESM2']['glacier_rescaling'].columns]

var_glacier = ['glacier_area',
               'glacier_elev',
               'smb_water_year',
               'glacier_melt_perc',
               'glacier_mass_mmwe',
               'glacier_vol_m3',
               'glacier_vol_perc']

var_output = ['avg_temp_catchment',
              'avg_temp_glaciers',
              'evap_off_glaciers',
              'actual_evaporation',
              'total_precipitation',
              'total_melt',
              'runoff_without_glaciers',
              'runoff_from_glaciers',
              'total_runoff',
              'snow_off_glaciers',
              'snow_on_glaciers',
              'melt_off_glaciers',
              'melt_on_glaciers',
              'ice_melt_on_glaciers',
              'snow_melt_on_glaciers']


def calculate_statistics(df, start_year, end_year):
    """
    Calculate the ensemble mean and standard deviation for each decade, and the linear trends.
    """
    # Define the decades
    decades = [(year, year + 10) for year in range(2000, 2100, 10)]

    # Calculate the mean and standard deviation for each decade
    mean_decades = []
    std_decades = []

    for start, end in decades:
        mask = (df.index.year >= start) & (df.index.year <= end)
        decade_data = df[mask]
        mean_decades.append(decade_data.mean().mean())  # Mean across time and models
        std_decades.append(decade_data.stack().std())  # Standard deviation across time and models

    # Create a DataFrame for mean and standard deviation for each decade
    mean_decades_df = pd.DataFrame(mean_decades, index=[f"{start}-{end}" for start, end in decades], columns=['mean'])
    std_decades_df = pd.DataFrame(std_decades, index=[f"{start}-{end}" for start, end in decades], columns=['std'])

    # Calculate the linear trends
    def linear_trend(y, start, end):
        x = np.arange(start, end + 1)
        y_mean = y.resample('Y').mean()
        mask = (y_mean.index.year >= start) & (y_mean.index.year <= end)
        y_mean = y_mean[mask]
        x = x[:len(y_mean)]
        model = LinearRegression().fit(x.reshape(-1, 1), y_mean)
        return model.coef_[0]

    trend_2001_2020 = linear_trend(df.mean(axis=1), 2001, 2020)
    trend_2021_2060 = linear_trend(df.mean(axis=1), 2021, 2060)
    trend_2061_2100 = linear_trend(df.mean(axis=1), 2061, 2100)

    # Calculate relative change in % per year for each trend period
    mean_value_2001_2020 = df[(df.index.year >= 2001) & (df.index.year <= 2020)].mean().mean()
    mean_value_2021_2060 = df[(df.index.year >= 2021) & (df.index.year <= 2060)].mean().mean()
    mean_value_2061_2100 = df[(df.index.year >= 2061) & (df.index.year <= 2100)].mean().mean()

    relative_change_2001_2020 = (trend_2001_2020 / mean_value_2001_2020) * 100
    relative_change_2021_2060 = (trend_2021_2060 / mean_value_2021_2060) * 100
    relative_change_2061_2100 = (trend_2061_2100 / mean_value_2061_2100) * 100

    return (mean_decades_df, std_decades_df,
            trend_2001_2020, trend_2021_2060, trend_2061_2100,
            relative_change_2001_2020, relative_change_2021_2060, relative_change_2061_2100)


def summary_statistics(scenario_dict, scenario_name, variables, resampling_resolution):
    summary_data = []

    for var in variables:
        # Get the DataFrame for the given variable
        df = custom_df_matilda(scenario_dict, scenario_name, var, resampling_resolution)

        # Calculate statistics
        (mean_decades_df, std_decades_df, trend_2001_2020, trend_2021_2060,
         trend_2061_2100, relative_change_2001_2020, relative_change_2021_2060,
         relative_change_2061_2100) = calculate_statistics(df, 2001, 2100)

        # Format trends with relative changes
        trend_2001_2020_str = f"{round(trend_2001_2020, 2)} ({round(relative_change_2001_2020, 2)}%)"
        trend_2021_2060_str = f"{round(trend_2021_2060, 2)} ({round(relative_change_2021_2060, 2)}%)"
        trend_2061_2100_str = f"{round(trend_2061_2100, 2)} ({round(relative_change_2061_2100, 2)}%)"

        # Create a summary row
        summary_row = {
            'Variable': var,
            'Trend 2001-2020': trend_2001_2020_str,
            'Trend 2021-2060': trend_2021_2060_str,
            'Trend 2061-2100': trend_2061_2100_str,
        }

        # Add the mean and standard deviation for each decade to the summary row
        for decade in mean_decades_df.index:
            summary_row[f'mean_{decade}'] = mean_decades_df.loc[decade, 'mean']
            summary_row[f'std_{decade}'] = std_decades_df.loc[decade, 'std']

        summary_data.append(summary_row)

    # Convert the summary data to a DataFrame
    summary_df = pd.DataFrame(summary_data)

    return summary_df


## Output summary
scenario = 'SSP2'
resampling_resolution = 'Y'
summary_df = summary_statistics(matilda_scenarios, scenario, var_output, resampling_resolution)

# Dictionary mapping old names to new formatted names
name_mapping = {
    'avg_temp_catchment': 'Mean Catchment Temperature',
    'avg_temp_glaciers': 'Mean Temperature of Glacierized Area',
    'evap_off_glaciers': 'Potential Evaporation',
    'actual_evaporation': 'Actual Evaporation',
    'total_precipitation': 'Total Precipitation',
    'total_melt': 'Total Melt',
    'runoff_without_glaciers': 'Runoff Without Glaciers',
    'runoff_from_glaciers': 'Runoff From Glaciers',
    'runoff_ratio': 'Runoff Ratio',
    'total_runoff': 'Total Runoff',
    'snow_off_glaciers': 'Snow Off Glaciers',
    'snow_on_glaciers': 'Snow On Glaciers',
    'melt_off_glaciers': 'Snow Melt Off Glaciers',
    'melt_on_glaciers': 'Total Melt On Glaciers',
    'ice_melt_on_glaciers': 'Glacier Melt',
    'snow_melt_on_glaciers': 'Snow Melt On Glaciers'
}

# Replace the variable names with the new names
summary_df['Variable'] = summary_df['Variable'].replace(name_mapping)

decades = ['2000-2010', '2010-2020', '2020-2030',
           '2030-2040', '2040-2050', '2050-2060', '2060-2070', '2070-2080',
           '2080-2090', '2090-2100']

# Initialize a new DataFrame with the trends
compact_df = summary_df[['Variable', 'Trend 2001-2020', 'Trend 2021-2060', 'Trend 2061-2100']].copy()

# Combine mean and standard deviation for each decade
for decade in decades:
    mean_col = f'mean_{decade}'
    std_col = f'std_{decade}'
    compact_df[decade] = summary_df.apply(lambda row: f"{round(row[mean_col], 1)} (± {round(row[std_col], 1)})", axis=1)

## Glacier summary
summary_glac = summary_statistics(matilda_scenarios, scenario, var_glacier, resampling_resolution)
compact_glac = summary_glac[['Variable', 'Trend 2001-2020', 'Trend 2021-2060', 'Trend 2061-2100']].copy()
for decade in decades:
    mean_col = f'mean_{decade}'
    std_col = f'std_{decade}'
    compact_glac[decade] = summary_glac.apply(lambda row: f"{round(row[mean_col], 1)} (± {round(row[std_col], 1)})",
                                              axis=1)

## Calculate absolute values
col_compare = ["Actual Evaporation",
               "Total Precipitation",
               "Snow Melt Off Glaciers",
               "Snow Melt On Glaciers",
               "Glacier Melt",
               "Total Runoff"]

col_add = ['Mean Catchment Temperature', 'Total Precipitation', 'Potential Evaporation']

# Filter and make a copy of the relevant subset of the DataFrame
summary_compare = summary_df.loc[summary_df['Variable'].isin(col_compare)].copy()

# Calculate km³ from mm
summary_compare.loc[:, summary_compare.columns[4:]] *= (295.28 / 1e6)

# Add " (m^3)" to the variables in col_compare
summary_compare['Variable'] = summary_compare['Variable'].apply(
    lambda x: f"{x} (km^3)" if x in col_compare else x)

# Add other variables to compare
additional_variables = summary_df.loc[summary_df['Variable'].isin(col_add)].copy()
summary_compare = pd.concat([additional_variables, summary_compare], ignore_index=True)

# Compact comparison DataFrame
compact_compare = summary_compare[['Variable']].copy()

less_decades = decades[1:6]

for decade in less_decades:
    mean_col = f'mean_{decade}'
    std_col = f'std_{decade}'
    compact_compare[decade] = summary_compare.apply(lambda row: row[mean_col], axis=1)

# Sum up snow melt
snow_melt_sum = compact_compare[compact_compare['Variable'].str.contains('Snow Melt')].iloc[:, 1:].sum()
new_row = pd.DataFrame([['Total Snow Melt (km^3)'] + snow_melt_sum.tolist()], columns=compact_compare.columns)
compact_compare = pd.concat([compact_compare, new_row], ignore_index=True)
compact_compare = compact_compare[~compact_compare['Variable'].str.contains('Snow Melt (On|Off) Glaciers')]
compact_compare.reset_index(drop=True, inplace=True)

# Reorder
order = [
    'Mean Catchment Temperature', 'Total Precipitation', 'Potential Evaporation',
    'Actual Evaporation (km^3)', 'Total Snow Melt (km^3)', 'Glacier Melt (km^3)', 'Total Runoff (km^3)'
]
compact_compare = compact_compare.set_index('Variable').reindex(order).reset_index()

# Add diff
compact_compare['diff'] = compact_compare['2050-2060'] - compact_compare['2010-2020']

# Round
compact_compare = round(compact_compare, 3)

print(compact_compare)


## Write files
compact_df.to_csv(
    f'/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/output/kyzylsuu/results_stats/output_summary_{scenario}_start2000.csv',
    index=False)
compact_glac.to_csv(
    f'/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/output/kyzylsuu/results_stats/glacier_summary_{scenario}_start2000.csv',
    index=False)

compact_compare.to_csv(
    f'/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/output/kyzylsuu/results_stats/compare_summary_single-model_{scenario}_start2000.csv',
    index=False)



## Full series linear trends

# Rename variables
name_mapping = {
    'avg_temp_catchment': 'Mean Catchment Temperature',
    'avg_temp_glaciers': 'Mean Temperature of Glacierized Area',
    'evap_off_glaciers': 'Potential Evaporation',
    'actual_evaporation': 'Actual Evaporation',
    'total_precipitation': 'Total Precipitation',
    'total_melt': 'Total Melt',
    'runoff_without_glaciers': 'Runoff Without Glaciers',
    'runoff_from_glaciers': 'Runoff From Glaciers',
    'runoff_ratio': 'Runoff Ratio',
    'total_runoff': 'Total Runoff',
    'snow_off_glaciers': 'Snow Off Glaciers',
    'snow_on_glaciers': 'Snow On Glaciers',
    'melt_off_glaciers': 'Snow Melt Off Glaciers',
    'melt_on_glaciers': 'Total Melt On Glaciers',
    'ice_melt_on_glaciers': 'Glacier Melt',
    'snow_melt_on_glaciers': 'Snow Melt On Glaciers',
    'glacier_area': 'Glacier Area',
    'glacier_elev': 'Glacier Elevation',
    'smb_water_year': 'Surface Mass Balance (Water Year)',
    'glacier_melt_perc': 'Glacier Melt Percentage',
    'glacier_mass_mmwe': 'Glacier Mass (mm w.e.)',
    'glacier_vol_m3': 'Glacier Volume (m³)',
    'glacier_vol_perc': 'Glacier Volume Percentage',
    'Total Snow Melt': 'Total Snow Melt',
    'snow_melt_fraction_runoff': 'Snow Melt Fraction in Runoff',
    'ice_melt_fraction_runoff': 'Ice Melt Fraction in Runoff',
    'snow_fraction_precip': 'Snow Fraction in Precipitation'
}

def calculate_significance(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def calculate_linear_trends(matilda_scenarios, scenario, variables, resampling_resolution):
    trend_data = []

    # Add the new variables: Total Snow Melt, Snow melt fraction in Runoff, Ice melt fraction in Runoff, Snow fraction in Precipitation
    snowmelt_off = custom_df_matilda(matilda_scenarios, scenario, 'melt_off_glaciers', resampling_resolution)
    snowmelt_on = custom_df_matilda(matilda_scenarios, scenario, 'snow_melt_on_glaciers', resampling_resolution)
    total_runoff = custom_df_matilda(matilda_scenarios, scenario, 'total_runoff', resampling_resolution)
    glacier_melt = custom_df_matilda(matilda_scenarios, scenario, 'ice_melt_on_glaciers', resampling_resolution)
    total_precipitation = custom_df_matilda(matilda_scenarios, scenario, 'total_precipitation', resampling_resolution)
    snow_off = custom_df_matilda(matilda_scenarios, scenario, 'snow_off_glaciers', resampling_resolution)
    snow_on = custom_df_matilda(matilda_scenarios, scenario, 'snow_on_glaciers', resampling_resolution)

    # Resample all to yearly mean
    total_snow_melt = snowmelt_on + snowmelt_off
    snow_melt_fraction_runoff = total_snow_melt / total_runoff
    ice_melt_fraction_runoff = glacier_melt / total_runoff
    snow_fraction_precip = (snow_off + snow_on) / total_precipitation

    # Add the new variables to the list of variables
    variables.extend(['Total Snow Melt', 'Snow Melt Fraction in Runoff', 'Ice Melt Fraction in Runoff', 'Snow Fraction in Precipitation'])

    # Loop over the variables to calculate trends
    for var in variables:
        if var == 'Total Snow Melt':
            df_yearly = total_snow_melt
        elif var == 'Snow Melt Fraction in Runoff':
            df_yearly = snow_melt_fraction_runoff
        elif var == 'Ice Melt Fraction in Runoff':
            df_yearly = ice_melt_fraction_runoff
        elif var == 'Snow Fraction in Precipitation':
            df_yearly = snow_fraction_precip
        else:
            # Get the DataFrame for the given variable
            df_yearly = custom_df_matilda(matilda_scenarios, scenario, var, resampling_resolution)

        # Calculate the ensemble mean
        ensemble_mean = df_yearly.mean(axis=1)

        # Exclude 1999 if present
        ensemble_mean = ensemble_mean[ensemble_mean.index.year >= 2000]

        # Define the time period (2000 to 2100)
        years = np.arange(2000, 2101)

        # Ensure the lengths of the years array and the ensemble mean match
        if len(ensemble_mean) != len(years):
            raise ValueError(f"Length mismatch: years ({len(years)}) vs ensemble_mean ({len(ensemble_mean)})")

        # Calculate linear regression on time (years 2000 to 2100)
        slope, intercept, r_value, p_value, std_err = linregress(years, ensemble_mean)

        # Trend in % per year
        mean_value = ensemble_mean.mean()
        trend_percent_per_year = (slope / mean_value) * 100 if mean_value != 0 else 0

        # Get the initial and final values
        initial_value = ensemble_mean.iloc[0]  # Value in 2000
        final_value = ensemble_mean.iloc[-1]  # Value in 2100

        # Calculate absolute and percentage change
        absolute_change = final_value - initial_value
        if initial_value != 0:
            percentage_change = (absolute_change / initial_value) * 100
        else:
            percentage_change = np.nan  # To handle division by zero

        # Add significance notation
        significance = calculate_significance(p_value)

        # Append results with initial, final values, absolute/percentage change, and significance
        trend_data.append([var, slope, trend_percent_per_year, p_value, initial_value, final_value, absolute_change, percentage_change, significance])

    # Create a DataFrame with the results
    trend_df = pd.DataFrame(trend_data, columns=['Variable', 'Trend (absolute)', 'Trend (% per year)', 'p-value', 'Initial Value (2000)', 'Final Value (2100)', 'Absolute Change (2000-2100)', 'Change in % (2000-2100)', 'Significance'])

    # Apply name_mapping to rename the variables
    trend_df['Variable'] = trend_df['Variable'].replace(name_mapping)

    # Round the DataFrame to 4 significant digits and apply scientific notation if necessary
    trend_df = trend_df.applymap(lambda x: f"{x:.4g}" if isinstance(x, (float, int)) else x)

    return trend_df


# Define variables from compact_df and compact_glac
var_output = ['avg_temp_catchment', 'avg_temp_glaciers', 'evap_off_glaciers', 'actual_evaporation',
              'total_precipitation', 'total_melt', 'runoff_without_glaciers', 'runoff_from_glaciers', 'total_runoff',
              'snow_off_glaciers', 'snow_on_glaciers', 'melt_off_glaciers', 'melt_on_glaciers',
              'ice_melt_on_glaciers', 'snow_melt_on_glaciers']

var_glacier = ['glacier_area', 'glacier_elev', 'smb_water_year', 'glacier_melt_perc', 'glacier_mass_mmwe',
               'glacier_vol_m3', 'glacier_vol_perc']

# Combine all variables
all_variables = var_output + var_glacier

# Run the calculation for the chosen scenario (e.g., 'SSP2')
trend_df = calculate_linear_trends(matilda_scenarios, scenario, all_variables, 'Y')

# Show the DataFrame
print(trend_df)

trend_df.to_csv( f'/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/output/kyzylsuu/results_stats/matilda_ensemble_stats_{scenario}_start2000.csv', index=False)

