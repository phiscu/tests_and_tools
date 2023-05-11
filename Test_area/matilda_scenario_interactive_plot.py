import pickle
import os
import pandas as pd
from fastparquet import write
from tqdm import tqdm
import sys
from pathlib import Path
from matilda.core import matilda_simulation


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


def parquet_to_dict(directory_path: str, pbar: bool = True) -> dict:
    """
    Recursively loads the dataframes from the parquet files in the specified directory and returns a dictionary.
    Nested directories are supported.
    Parameters
    ----------
    directory_path : str
        The directory path containing the parquet files.
    pbar : bool, optional
        A flag indicating whether to display a progress bar. Default is True.
    Returns
    -------
    dict
        A dictionary containing the loaded pandas dataframes.
    """
    dictionary = {}
    if pbar:
        bar_iter = tqdm(sorted(os.listdir(directory_path)), desc='Reading parquet files: ')
    else:
        bar_iter = sorted(os.listdir(directory_path))
    for file_name in bar_iter:
        file_path = os.path.join(directory_path, file_name)
        if os.path.isdir(file_path):
            dictionary[file_name] = parquet_to_dict(file_path, pbar=False)
        elif file_name.endswith(".parquet"):
            k = file_name[:-len(".parquet")]
            dictionary[k] = pd.read_parquet(file_path)
    return dictionary


def custom_df(dic, scenario, var, resample_freq=None):
    """
    Takes a dictionary of model outputs and returns a combined dataframe of a specific variable for a given scenario.
    Parameters:
        dic (dict): A nested dictionary of model outputs.
                    The outer keys are scenario names and the inner keys are model names.
                    The corresponding values are dictionaries containing two keys:
                        'model_output' (DataFrame): containing model outputs for a given scenario and model
                        'glacier_rescaling' (DataFrame): containing glacier properties for a given scenario and model
        scenario (str): The name of the scenario to select from the dictionary.
        var (str): The name of the variable to extract from the model output DataFrame.
        resample_freq (str, optional): The frequency of the resulting time series data.
                                       Defaults to None (i.e. no resampling).
                                       If provided, should be in pandas resample frequency string format.
    Returns:
        pandas.DataFrame: A DataFrame containing the combined data of the specified variable for the selected scenario
                          and models. The DataFrame is indexed by the time steps of the original models.
                          The columns are the names of the models in the selected scenario.
    Raises:
        ValueError: If the provided  var  string is not one of the following: ['avg_temp_catchment', 'avg_temp_glaciers',
                    'evap_off_glaciers', 'prec_off_glaciers', 'prec_on_glaciers', 'rain_off_glaciers', 'snow_off_glaciers',
                    'rain_on_glaciers', 'snow_on_glaciers', 'snowpack_off_glaciers', 'soil_moisture', 'upper_groundwater',
                    'lower_groundwater', 'melt_off_glaciers', 'melt_on_glaciers', 'ice_melt_on_glaciers', 'snow_melt_on_glaciers',
                    'refreezing_ice', 'refreezing_snow', 'total_refreezing', 'SMB', 'actual_evaporation', 'total_precipitation',
                    'total_melt', 'runoff_without_glaciers', 'runoff_from_glaciers', 'total_runoff', 'glacier_area',
                    'glacier_elev', 'smb_water_year', 'smb_scaled', 'smb_scaled_capped', 'smb_scaled_capped_cum', 'surplus']
    """
    out1_cols = ['avg_temp_catchment', 'avg_temp_glaciers', 'evap_off_glaciers',
                 'prec_off_glaciers', 'prec_on_glaciers', 'rain_off_glaciers',
                 'snow_off_glaciers', 'rain_on_glaciers', 'snow_on_glaciers',
                 'snowpack_off_glaciers', 'soil_moisture', 'upper_groundwater',
                 'lower_groundwater', 'melt_off_glaciers', 'melt_on_glaciers',
                 'ice_melt_on_glaciers', 'snow_melt_on_glaciers', 'refreezing_ice',
                 'refreezing_snow', 'total_refreezing', 'SMB', 'actual_evaporation',
                 'total_precipitation', 'total_melt', 'runoff_without_glaciers',
                 'runoff_from_glaciers', 'total_runoff']

    out2_cols = ['glacier_area', 'glacier_elev', 'smb_water_year',
                 'smb_scaled', 'smb_scaled_capped', 'smb_scaled_capped_cum',
                 'surplus']

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

# matilda_scenarios = parquet_to_dict(test_dir + 'adjusted/parquet')
matilda_scenarios = pickle_to_dict(test_dir + 'adjusted/matilda_scenarios.pickle')   # pickle for speed/parquet for size


## Create dictionary with variable names, long names, and units

var_name = ['avg_temp_catchment', 'avg_temp_glaciers',
                    'evap_off_glaciers', 'prec_off_glaciers', 'prec_on_glaciers', 'rain_off_glaciers', 'snow_off_glaciers',
                    'rain_on_glaciers', 'snow_on_glaciers', 'snowpack_off_glaciers', 'soil_moisture', 'upper_groundwater',
                    'lower_groundwater', 'melt_off_glaciers', 'melt_on_glaciers', 'ice_melt_on_glaciers', 'snow_melt_on_glaciers',
                    'refreezing_ice', 'refreezing_snow', 'total_refreezing', 'SMB', 'actual_evaporation', 'total_precipitation',
                    'total_melt', 'runoff_without_glaciers', 'runoff_from_glaciers', 'total_runoff', 'glacier_area',
                    'glacier_elev', 'smb_water_year', 'smb_scaled', 'smb_scaled_capped', 'smb_scaled_capped_cum', 'surplus']

title = ['Mean Catchment Temperature',
         'Mean Temperature of Glacierized Area',
         'Off-glacier Evaporation',
         'Off-glacier Precipitation',
         'On-glacier Precipitation',
         'Off-glacier Rain',
         'Off-glacier Snow',
         'On-glacier Rain',
         'On-glacier Snow',
         'Off-glacier Snowpack',
         'Soil Moisture',
         'Upper Groundwater',
         'Lower Groundwater',
         'Off-glacier Melt',
         'On-glacier Melt',
         'On-glacier Ice Melt',
         'On-glacier Snow Melt',
         'Refreezing Ice',
         'Refreezing Snow',
         'Total Refreezing',
         'Glacier Surface Mass Balance',
         'Mean Actual Evaporation',
         'Mean Total Precipitation',
         'Total Melt',
         'Runoff without Glaciers',
         'Runoff from Glaciers',
         'Total Runoff',
         'Glacier Area',
         'Mean Glacier Elevation',
         'Surface Mass Balance of the Hydrological Year',
         'Area-scaled Surface Mass Balance',
         'Surface Mass Balance Capped at 0',
         'Cumulative Surface Mass Balance Capped at 0',
         'Cumulative Surface Mass Balance > 0']

unit = ['°C', '°C', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.',
        'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.',
        'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'mm w.e.', 'km²', 'm.a.s.l.', 'mm w.e.', 'mm w.e.',
        'mm w.e.', 'mm w.e.']

output_vars = {key: (val1, val2) for key, val1, val2 in zip(var_name, title, unit)}

## Plot functions for mean with CIs

import plotly.graph_objects as go
import numpy as np


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


def plot_wit_ci(var, dic=matilda_scenarios, resample_freq='Y', show=False):
    """
    A function to plot multi-model mean and confidence intervals of a given variable for two different scenarios.
    Parameters:
    -----------
    var: str
        The variable to plot.
    dic: dict, optional (default=matilda_scenarios)
        A dictionary containing the scenarios as keys and the dataframes as values.
    resample_freq: str, optional (default='Y')
        The resampling frequency to apply to the data.
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
    df1 = custom_df(dic, scenario='SSP2', var=var, resample_freq=resample_freq)
    df1_ci = confidence_interval(df1)
    # SSP5
    df2 = custom_df(dic, scenario='SSP5', var=var, resample_freq=resample_freq)
    df2_ci = confidence_interval(df2)

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
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=output_vars[var][0] + ' [' + output_vars[var][1] + ']',
        title={'text': '<b>' + output_vars[var][0] + '</b>', 'font': {'size': 28, 'color': 'darkblue', 'family': 'Arial'}},
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

    return fig


## Apply the plot functions with a dropdown menu on a Dash server

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.io as pio

pio.renderers.default = "browser"
app = dash.Dash()

# Create the initial line plot
fig = plot_wit_ci('total_runoff')


# Create the callback function
@app.callback(
    Output('line-plot', 'figure'),
    Input('arg-dropdown', 'value'))
def update_figure(selected_arg):
    return plot_wit_ci(selected_arg)


# Define the dropdown menu
arg_dropdown = dcc.Dropdown(
    id='arg-dropdown',
    options=[{'label': output_vars[var][0], 'value': var} for var in output_vars.keys()],
    value='total_runoff',
    clearable=False)

# Add the dropdown menu to the layout
app.layout = html.Div([
    arg_dropdown,
    dcc.Graph(id='line-plot', figure=fig)])

# Run the app
app.run_server(debug=True, use_reloader=False)      # Turn off reloader inside jupyter


# turn into function/class to customize scenario/resample_rate/renderer etc.
# test in binder and add dash to requirements
# add fastparquet to requirements

