import pickle
import os
import pandas as pd
import sys
from pathlib import Path
from matilda.core import matilda_simulation


test_dir = '/home/phillip/Seafile/EBA-CA/Repositories/matilda_edu/output/cmip6/'


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
                         str([out1_cols, out2_cols]))

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



matilda_scenarios = pickle_to_dict(test_dir + 'adjusted/matilda_scenarios.pickle')


##
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.io as pio

pio.renderers.default = "browser"

app = dash.Dash()

# Create the initial line plot
df = custom_df(matilda_scenarios, scenario='SSP5', var='glacier_area', resample_freq='10Y')
fig = px.line(df)

# Define the list of arguments for custom_df()
args = ['glacier_area', 'total_runoff', 'SMB']

# Create the callback function
@app.callback(
    Output('line-plot', 'figure'),
    Input('arg-dropdown', 'value'))
def update_figure(selected_arg):
    # Generate the new dataframe based on the selected argument
    new_df = custom_df(matilda_scenarios, scenario='SSP5', var=selected_arg, resample_freq='10Y')
    # Update the line plot with the new data for all columns
    fig = px.line(new_df)
    return fig

# Define the dropdown menu
arg_dropdown = dcc.Dropdown(
    id='arg-dropdown',
    options=[{'label': arg, 'value': arg} for arg in args],
    value=args[0])

# Add the dropdown menu to the layout
app.layout = html.Div([
    arg_dropdown,
    dcc.Graph(id='line-plot', figure=fig)])

# Run the app
app.run_server(debug=True, use_reloader=False)


##

# edit axes (custom units)
# add title (custom variable names)
# change colors
# add more vars
# turn into function/class to customize scenario/resample_rate/renderer etc.
# test in binder and add dash to requirements
