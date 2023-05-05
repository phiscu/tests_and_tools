import pickle
import os
import pandas as pd
import sys
from pathlib import Path
from matilda.core import matilda_simulation

home = str(Path.home()) + '/Seafile'
sys.path.append(home + '/Ana-Lena_Phillip/data/tests_and_tools')
wd = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data'
glacier_profile = pd.read_csv(wd + "/kyzulsuu_glacier_profile.csv")

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


def cmip2df(temp, prec, scen, col):
    """
    Converts temperature and precipitation data from a CMIP model output dictionary into a Pandas DataFrame.
    Parameters
    ----------
    temp : dict
        dictionary of temperature data from a CMIP model
    prec : dict
        dictionary of precipitation data from a CMIP model
    scen : str
        name of the scenario (e.g. RCP4.5)
    col : str
        name of the column containing data for the scenario (e.g. tas)
    Returns:
    ----------
    df : pandas.DataFrame
        DataFrame containing the temperature and precipitation data for the given scenario and column
    """
    df = pd.DataFrame({'T2': temp[scen][col], 'RRR': prec[scen][col]}).reset_index()
    df.columns = ['TIMESTAMP', 'T2', 'RRR']
    return df


matilda_settings = {
    "set_up_start": '1979-01-01',  # Start date of the setup period
    "set_up_end": '1980-12-31',  # End date of the setup period
    "sim_start": '1981-01-01',  # Start date of the simulation period
    "sim_end": '2100-12-31',  # End date of the simulation period
    "freq": "M",  # Frequency of the data (monthly)
    "glacier_profile": glacier_profile,  # Glacier profile
    "area_cat": 295.763,  # Area of the catchment
    "lat": 42.33,  # Latitude of the catchment
    "warn": False,  # Warning flag
    "plot_type": "all",  # Type of plot
    "plots": False,  # Flag to indicate if plots should be generated
    "elev_rescaling": True,  # Flag to indicate if elevation rescaling should be done
    "ele_dat": 3172,  # Elevation of the data
    "ele_cat": 3295,  # Elevation of the catchment
    "area_glac": 32.51,  # Area of the glacier
    "ele_glac": 4068,  # Elevation of the glacier
    "pfilter": 0  # Filter parameter
}
param_dict = {
    'lr_temp': -0.006077369,  # Lapse rate for temperature
    'lr_prec': 0.0013269137,  # Lapse rate for precipitation
    'BETA': 5.654754,
    'CET': 0.08080378,
    'FC': 365.68375,  # Field capacity
    'K0': 0.36890236,  # K0 parameter
    'K1': 0.022955153,  # K1 parameter
    'K2': 0.060069658,  # K2 parameter
    'LP': 0.63395154,  # LP parameter
    'MAXBAS': 5.094901,  # Maximum basin storage
    'PERC': 0.39491335,  # Percolation
    'UZL': 348.0978,  # Upper zone limit
    'PCORR': 1.0702422,  # Precipitation correction
    'TT_snow': -1.1521467,  # Temperature threshold for snow
    'TT_diff': 1.5895765,  # Temperature difference
    'CFMAX_ice': 3.6518102,  # Maximum ice content
    'CFMAX_rel': 1.8089349,  # Maximum relative content
    'SFCF': 0.42293832,  # Soil field capacity
    'CWH': 0.11234668,  # Crop water holding capacity
    'AG': 0.9618855,
    'RFS': 0.11432563  # Rainfall sensitivity ???
}

## Read adjusted CMIP6 data

tas = pickle_to_dict(test_dir + 'adjusted/tas.pickle')
pr = pickle_to_dict(test_dir + 'adjusted/pr.pickle')

## Create MATILDA input

import pandas as pd


def create_scenario_dict(tas: dict, pr: dict, scenario_nums: list) -> dict:
    """
    Create a nested dictionary of scenarios and models from two dictionaries of pandas DataFrames.
    Parameters
    ----------
    tas : dict
        A dictionary of pandas DataFrames where the keys are scenario names and each DataFrame has columns
        representing different climate model mean daily temperature (K) time series.
    pr : dict
        A dictionary of pandas DataFrames where the keys are scenario names and each DataFrame has columns
        representing different climate models mean daily precipitation (mm/day) time series.
    scenario_nums : list
        A list of integers representing the scenario numbers to include in the resulting dictionary.
    Returns
    -------
    dict
        A nested dictionary where the top-level keys are scenario names (e.g. 'SSP2', 'SSP5') and the values are
        dictionaries containing climate models as keys and the corresponding pandas DataFrames as values.
        The DataFrames have three columns: 'TIMESTAMP', 'T2', and 'RRR', where 'TIMESTAMP'
        represents the time step, 'T2' represents the mean daily temperature (K), and 'RRR' represents the mean
        daily precipitation (mm/day).
    """
    scenarios = {}
    for s in scenario_nums:
        s = 'SSP' + str(s)
        scenarios[s] = {}
        for m in tas[s].columns:
            model = pd.DataFrame({'T2': tas[s][m],
                                  'RRR': pr[s][m]})
            model = model.reset_index()
            mod_dict = {m: model.rename(columns={'time': 'TIMESTAMP'})}
            scenarios[s].update(mod_dict)
    return scenarios

# scenarios = create_scenario_dict(tas, pr, [2, 5])
# dict_to_pickle(scenarios, test_dir + 'adjusted/matilda_input.pickle')

scenarios = pickle_to_dict(test_dir + 'adjusted/matilda_input.pickle')

## Run Matilda in a loop (takes a while - have a coffee)

from tqdm import tqdm
import contextlib
from multiprocessing import Pool
from functools import partial


class MatildaBulkProcessor:
    """
    A class to run multiple MATILDA simulations for different input scenarios and models in single or multi-processing
    mode and store the results in a dictionary.
    Attributes
    ----------
    scenarios : dict
        A dictionary with scenario names as keys and a dictionary of climate models as values.
    matilda_settings : dict
        A dictionary of MATILDA settings.
    matilda_parameters : dict
        A dictionary of MATILDA parameter values.
    Methods
    -------
    run_single_process():
        Runs the MATILDA simulations for the scenarios and models in single-processing mode and returns a dictionary
        of results.
    run_multi_process():
        Runs the MATILDA simulations for the scenarios and models in multi-processing mode and returns a dictionary
        of results.
    matilda_headless(df, matilda_settings, matilda_parameters):
        A helper function to run a single MATILDA simulation given a dataframe, MATILDA settings and parameter
        values.
    """

    def __init__(self, scenarios, matilda_settings, matilda_parameters):
        """
        Parameters
        ----------
        scenarios : dict
            A dictionary with scenario names as keys and a dictionary of models as values.
        matilda_settings : dict
            A dictionary of MATILDA settings.
        matilda_parameters : dict
            A dictionary of MATILDA parameter values.
        """

        self.scenarios = scenarios
        self.matilda_settings = matilda_settings
        self.matilda_parameters = matilda_parameters

    @staticmethod
    def matilda_headless(df, matilda_settings, matilda_parameters):
        """
        A helper function to run a single MATILDA simulation given a dataframe, MATILDA settings and parameter
        values.
        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe for the MATILDA simulation.
        matilda_settings : dict
            A dictionary of MATILDA settings.
        matilda_parameters : dict
            A dictionary of MATILDA parameter values.
        Returns
        -------
        dict
            A dictionary containing the MATILDA model output and glacier rescaling factor.
        """

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                output = matilda_simulation(df, **matilda_settings, parameter_set=matilda_parameters)
        return {'model_output': output[0], 'glacier_rescaling': output[5]}

    def run_single_process(self):
        """
        Runs the MATILDA simulations for the scenarios and models in single-processing mode and returns a dictionary
        of results.
        Returns
        -------
        dict
            A dictionary of MATILDA simulation results.
        """

        out_dict = {}  # Create an empty dictionary to store the outputs
        # Loop over the scenarios with progress bar
        for scenario in self.scenarios.keys():
            model_dict = {}  # Create an empty dictionary to store the model outputs
            # Loop over the models with progress bar
            for model in tqdm(self.scenarios[scenario].keys(), desc=scenario):
                # Get the dataframe for the current scenario and model
                df = self.scenarios[scenario][model]
                # Run the model simulation and get the output while suppressing prints
                model_output = self.matilda_headless(df, self.matilda_settings, self.matilda_parameters)
                # Store the list of output in the model dictionary
                model_dict[model] = model_output
            # Store the model dictionary in the scenario dictionary
            out_dict[scenario] = model_dict
        return out_dict

    def run_multi_process(self):
        """
        Runs the MATILDA simulations for the scenarios and models in multi-processing mode and returns a dictionary
        of results.
        Returns
        -------
        dict
            A dictionary of MATILDA simulation results.
        """

        out_dict = {}  # Create an empty dictionary to store the outputs
        with Pool() as pool:
            # Loop over the scenarios with progress bar
            for scenario in tqdm(self.scenarios.keys(), desc="Scenarios SSP2 and SSP5"):
                model_dict = {}  # Create an empty dictionary to store the model outputs
                # Loop over the models with progress bar
                model_list = [self.scenarios[scenario][m] for m in self.scenarios[scenario].keys()]
                for model, model_output in zip(self.scenarios[scenario], pool.map(
                        partial(self.matilda_headless, matilda_settings=self.matilda_settings,
                                matilda_parameters=self.matilda_parameters), model_list)):
                    model_dict[model] = model_output
                # Store the model dictionary in the scenario dictionary
                out_dict[scenario] = model_dict
            pool.close()

        return out_dict


# Usage
# matilda_bulk = MatildaBulkProcessor(scenarios, matilda_settings, param_dict)
# matilda_scenarios = matilda_bulk.run_single_process()
# matilda_scenarios = matilda_bulk.run_multi_process()

# dict_to_pickle(matilda_scenarios, test_dir + 'adjusted/matilda_scenarios.pickle')

matilda_scenarios = pickle_to_dict(test_dir + 'adjusted/matilda_scenarios.pickle')

## Create custom dataframes for analysis

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


# custom_df(matilda_scenarios, scenario='SSP5', var='smb_water_year', resample_freq='Y')


## Plot example
import matplotlib.pyplot as plt

# combined_df = custom_df(matilda_scenarios, scenario='SSP5', var='runoff_from_glaciers', resample_freq='10Y')
# # Create the line plot
# combined_df.plot()
# plt.xlabel('x-axis label')
# plt.ylabel('y-axis label')
# plt.title('Title of the plot')
# plt.show()



## Explore the dataset

# import plotly.express as px
# import plotly.graph_objects as pxg
# import plotly.io as pio
# pio.renderers.default = "browser"
#
# df = custom_df(matilda_scenarios, scenario='SSP5', var='glacier_area', resample_freq='Y')
#
# plot = pxg.Figure(data=[pxg.layout.shape.Line(
#     y=df[:])
# ])
#
# plot.show()
#
#
#
#
#
#
# ##
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# import dash
# from dash import dcc
# from dash import html
# from dash.dependencies import Input, Output
#
# # Initial data for the line chart
# line_chart_data = []
# for col in df.columns:
#     trace = go.Scatter(x=df['x'], y=df[col], mode='lines', name=col)
#     line_chart_data.append(trace)
#
# # Create a subplot for the dropdown menu
# dropdown_options = [{'label': 'Option 1', 'value': 'option1'},
#                     {'label': 'Option 2', 'value': 'option2'},
#                     {'label': 'Option 3', 'value': 'option3'}]
# dropdown_menu = go.Dropdown(options=dropdown_options, value='option1', id='param-selector')
#  # Create a subplot for the line chart
# fig = make_subplots(rows=2, cols=1, vertical_spacing=0.08, subplot_titles=['Line Chart', 'Dropdown Menu'])
# fig.add_trace(line_chart_data, row=1, col=1)
# fig.add_trace(dropdown_menu, row=2, col=1)
#
# # Define the callback function for the dropdown menu
# @app.callback(
#     Output(component_id='line-chart', component_property='figure'),
#     [Input(component_id='param-selector', component_property='value')]
# )
# def update_chart(arg):
#     df = custom_df(arg)
#     data = []
#     for col in df.columns:
#         trace = go.Scatter(x=df['x'], y=df[col], mode='lines', name=col)
#         data.append(trace)
#     fig = {'data': data,
#            'layout': go.Layout(xaxis={'title': 'X-axis'},
#                                yaxis={'title': 'Y-axis'},
#                                title='Line Chart')}
#     return fig
# # Update the layout of the plot with the dropdown menu
# fig.update_layout(updatemenus=[{'type': 'dropdown',
#                                 'buttons': [{'label': 'Option 1', 'method': 'update', 'args': [{'visible': [True, False]},
#                                                                                               {'title': 'Line Chart'}]},
#                                             {'label': 'Option 2', 'method': 'update', 'args': [{'visible': [False, True]},
#                                                                                               {'title': 'Line Chart'}]},
#                                             {'label': 'Option 3', 'method': 'update', 'args': [{'visible': [False, False, True]},
#                                                                                               {'title': 'Line Chart'}]}]}])
#
# # Display the plot
# fig.show()
#
# ##
#
# import plotly.express as px
# import pandas as pd
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# from dash.dependencies import Input, Output
#
#
# df = custom_df(matilda_scenarios, scenario='SSP5', var='total_runoff', resample_freq='Y')
#
# fig = make_subplots(rows=2, cols=1, vertical_spacing=0.08, subplot_titles=("Line Chart", "Dropdown Menu"))
# dropdown_options = [{'label': 'Glacier Area', 'value': 'glacier_area'},
#                    {'label': 'Total Runoff', 'value': 'total_runoff'},
#                    {'label': 'SMB', 'value': 'SMB'}]
#
# fig.add_trace(go.Dropdown(options=dropdown_options, value='total_runoff', id='param-selector'), row=2, col=1)
#
# @app.callback(Output('line_chart_id', 'data'),
#               [Input('param-selector', 'value')])
# def update_chart(arg):
#     df = custom_df(matilda_scenarios, scenario='SSP5', var=arg, resample_freq='Y')
#     line_chart_data['x'] = df.index
#     line_chart_data['y'] = df['ACCESS-CM2']
#     return line_chart_data
#
# initial_df = custom_df('total_runoff')
# line_chart_data = {'x': initial_df.index,
#                   'y': initial_df['ACCESS-CM2'],
#                   'type': 'line'}
# fig.add_trace(go.Scatter(line_chart_data), row=1, col=1)
#
# fig.show()


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
df = custom_df(matilda_scenarios, scenario='SSP5', var='glacier_area', resample_freq='Y')
fig = px.line(df)

# Add all columns to the line plot
# for column in df.columns:
#     fig.add_trace(px.line(df, x=df.index, y=df.[column]).data[0])

# fig.show()

# Define the list of arguments for custom_df()
args = ['glacier_area', 'total_runoff', 'SMB']

# Create the callback function
@app.callback(
    Output('line-plot', 'figure'),
    Input('arg-dropdown', 'value'))
def update_figure(selected_arg):
    # Generate the new dataframe based on the selected argument
    new_df = custom_df(matilda_scenarios, scenario='SSP5', var=selected_arg, resample_freq='Y')
    # Update the line plot with the new data for all columns
    fig.data = []
    fig.add_trace(px.line(new_df).data[0])
    for column in new_df.columns:
        fig.add_trace(px.line(new_df, x=df.index, y=df[column]).data[0])
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

app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter




