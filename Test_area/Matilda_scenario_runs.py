import pickle
import os
import pandas as pd
import numpy as np
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
    df = pd.DataFrame({'T2': temp[scen][col], 'RRR': prec[scen][col]}).reset_index()
    df.columns = ['TIMESTAMP', 'T2', 'RRR']
    return df


matilda_settings = {
    "set_up_start": '1979-01-01',  # Start date of the setup period
    "set_up_end": '1980-12-31',  # End date of the setup period
    "sim_start": '1981-01-01',  # Start date of the simulation period
    "sim_end": '1990-12-31',  # End date of the simulation period
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

## Write matilda scenario input to file


## Create MATILDA input

matilda_settings = {
    "set_up_start": '1979-01-01',  # Start date of the setup period
    "set_up_end": '1980-12-31',  # End date of the setup period
    "sim_start": '1981-01-01',  # Start date of the simulation period
    "sim_end": '1990-12-31',  # End date of the simulation period
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

# scenarios = {}
# for s in [2, 5]:
#     s = 'SSP' + str(s)
#     scenarios[s] = {}
#     for m in tas[s].columns:
#         model = pd.DataFrame({'T2': tas[s][m],
#                               'RRR': pr[s][m]})
#         model = model.reset_index()
#         mod_dict = {m: model.rename(columns={'time': 'TIMESTAMP'})}
#         scenarios[s].update(mod_dict)

# dict_to_pickle(scenarios, test_dir + 'adjusted/matilda_input.pickle')

scenarios = pickle_to_dict( test_dir + 'adjusted/matilda_input.pickle')

## Run Matilda in a loop

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
                for model, model_output in zip(self.scenarios[scenario], pool.map(partial(self.matilda_headless, matilda_settings=self.matilda_settings, matilda_parameters=self.matilda_parameters), model_list)):
                    model_dict[model] = model_output
                # Store the model dictionary in the scenario dictionary
                out_dict[scenario] = model_dict
            pool.close()

        return out_dict

## Usage
matilda_bulk = MatildaBulkProcessor(scenarios, matilda_settings, param_dict)
# matilda_scenarios = matilda_bulk.run_single_process()
matilda_scenarios = matilda_bulk.run_multi_process()

dict_to_pickle(matilda_scenarios, test_dir + 'adjusted/matilda_scenarios.pickle')


# matilda_scenarios['SSP2']['ACCESS-CM2']['model_output']['total_runoff']


## Plot example
import matplotlib.pyplot as plt
import pandas as pd
# Define the scenario and column to plot
scenario = 'SSP2'
column_to_plot = 'total_runoff'
output_no = 0
resample_freq = 'Y'
# Create an empty list to store the dataframes
dfs = []
# Loop over the models in the selected scenario
for model in matilda_scenarios[scenario].keys():
    # Get the dataframe for the current model
    df = matilda_scenarios[scenario][model][output_no]
    # Append the dataframe to the list of dataframes
    dfs.append(df[column_to_plot])
# Concatenate the dataframes into a single dataframe
combined_df = pd.concat(dfs, axis=1)
# Set the column names of the combined dataframe to the model names
combined_df.columns = matilda_scenarios[scenario].keys()
# Resample time series
combined_df = combined_df.resample('Y').sum()
# Create the line plot
combined_df.plot()
plt.xlabel('x-axis label')
plt.ylabel('y-axis label')
plt.title('Title of the plot')
plt.show()