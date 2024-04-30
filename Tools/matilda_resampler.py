## Imports
import numpy as np
import matplotlib.pyplot as plt
import socket
import plotly.io as pio
from matilda.core import matilda_simulation
import HydroErr as he
from pathlib import Path
import os
import contextlib
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import pandas as pd
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'

pio.renderers.default = "browser"

data_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/parameters/'


## MATILDA:

data_path = '/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/notebook_example_for_spot/'
obs = pd.read_csv(data_path + 'obs_runoff_example.csv')
df = pd.read_csv(data_path + 'era5.csv')
glacier_profile = pd.read_csv(data_path + 'glacier_profile.csv')

full_period = {'set_up_end': '1999-12-31', 'set_up_start': '1998-01-01',
               'sim_end': '2020-12-31', 'sim_start': '2000-01-01'}
calibration_period = {'set_up_end': '1999-12-31', 'set_up_start': '1998-01-01',
                      'sim_end': '2017-12-31', 'sim_start': '2000-01-01'}
validation_period = {'set_up_end': '2017-12-31', 'set_up_start': '2016-01-01',
                     'sim_end': '2020-12-31', 'sim_start': '2018-01-01'}

settings = {
    'area_cat': 295.67484249904464,
    'area_glac': 31.829413146585885,
    'ele_cat': 3293.491688025922,
    'ele_dat': 3335.67,
    'ele_glac': 4001.8798828125,
    'elev_rescaling': True,
    'freq': 'D',
    'lat': 42.18280043250193,
    'glacier_profile': glacier_profile
}

fix_val={'PCORR': 0.64, 'SFCF': 1, 'CET': 0}
parameters = {'lr_temp': -0.00522, 'lr_prec': 0.00262, 'BETA': 1.3601397, 'TT_snow': -1.7347374, 'TT_diff': 2.9, 'CFMAX_ice': 3.3, 'CFMAX_rel': 0.725, 'FC': 50.2, 'K0': 0.034036454, 'K1': 0.0101, 'K2': 0.00119, 'LP': 1.0, 'MAXBAS': 2.0, 'PERC': 0.87882555, 'UZL': 499.0, 'CWH': 0.112680875, 'AG': 0.999, 'RFS': 0.12816271}
results = matilda_simulation(df, obs, **full_period, **settings, **parameters, **fix_val)

## Parameter sets
fix_val = {'PCORR': 0.64, 'SFCF': 1, 'CET': 0, 'lr_temp': -0.00605, 'lr_prec': 0.00117, 'TT_diff': 1.36708, 'CFMAX_rel': 1.81114}
param_list = \
    [{'BETA': 1.007054, 'FC': 302.78784, 'K1': 0.0130889015, 'K2': 0.0049547367, 'PERC': 0.8058457, 'UZL': 482.38788, 'TT_snow': -0.418914, 'CFMAX_ice': 5.592482, 'CWH': 0.10325227},
     {'BETA': 4.469486, 'FC': 291.8014, 'K1': 0.013892675, 'K2': 0.0045123543, 'PERC': 1.4685482, 'UZL': 430.45685, 'TT_snow': -1.3098346, 'CFMAX_ice': 5.504861, 'CWH': 0.12402764},
     {'BETA': 1.0345612, 'FC': 212.69034, 'K1': 0.025412053, 'K2': 0.0049677053, 'PERC': 2.1586323, 'UZL': 392.962, 'TT_snow': -1.4604422, 'CFMAX_ice': 5.3250813, 'CWH': 0.1916532}]
for p in param_list:
    p.update(fix_val)

## Re-run MATILDA:


class MatildaBulkSampler:
    """
    A class to run multiple MATILDA simulations for different parameter sets in single or multi-processing
    mode and store the results in a dictionary.
    Attributes
    ----------
    forcing_data : pandas.DataFrame
        The input dataframe containing forcing data (e.g., temperature, precipitation).
    matilda_settings : dict
        A dictionary of MATILDA settings.
    matilda_parameters_list : list of dict
        A list of dictionaries, each containing a different set of MATILDA parameter values.
    Methods
    -------
    run_single_process():
        Runs the MATILDA simulations for the parameter sets in single-processing mode and returns a dictionary
        of results.
    run_multi_process():
        Runs the MATILDA simulations for the parameter sets in multi-processing mode and returns a dictionary
        of results.
    matilda_headless(df, matilda_settings, matilda_parameters):
        A helper function to run a single MATILDA simulation given a dataframe, MATILDA settings, and parameter
        values.
    """

    def __init__(self, forcing_data, obs, matilda_settings, matilda_parameters_list, swe_obs=None, swe_scaling=None):
        """
        Parameters
        ----------
        forcing_data : pandas.DataFrame
            The input dataframe containing forcing data (e.g., temperature, precipitation).
        matilda_settings : dict
            A dictionary of MATILDA settings.
        matilda_parameters_list : list of dict
            A list of dictionaries, each containing a different set of MATILDA parameter values.
        swe_obs : pandas.DataFrame
            A DataFrame containing snow water equivalent observations.
        swe_scaling : float
            A float value between 0 and 1. Is multiplied with the simulated snow water equivalent to match the reference area of the observations.
        """

        self.forcing_data = forcing_data
        self.obs = obs
        self.matilda_settings = matilda_settings
        self.matilda_parameters_list = matilda_parameters_list
        self.swe_obs = swe_obs
        self.swe_scaling = swe_scaling

    @staticmethod
    def matilda_headless(df, obs, matilda_settings, matilda_parameters, swe_obs, swe_scaling):
        """
        A helper function to run a single MATILDA simulation given a dataframe, MATILDA settings, and parameter
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
                output = matilda_simulation(df, obs, **matilda_settings, parameter_set=matilda_parameters)
                if swe_obs is not None:
                    swe_sim = output[0].snowpack_off_glaciers['2000-01-01':'2017-09-30'].to_frame(name="SWE_sim")
                    snow = output[0][['melt_off_glaciers', 'snow_off_glaciers']]['2000-01-01':'2017-09-30']
                    swe_df = pd.concat([swe_obs, swe_sim, snow], axis=1)
                    swe_df.columns = ['SWE_obs', 'SWE_sim', 'snow_melt', 'snow_fall']
                    swe_df[['SWE_sim', 'snow_melt', 'snow_fall']] = swe_df[['SWE_sim', 'snow_melt', 'snow_fall']].multiply(swe_scaling)
        if swe_obs is not None:
            return {'KGE': output[2],
                'SMB_mean18': results[5]['smb_water_year'][1:, ][:'2018-01-01'].mean() / 1000,
                'SWE_KGE': he.kge_2012(swe_df.SWE_sim, swe_df.SWE_obs, remove_zero=False),
                'SMB': results[5]['smb_water_year'][1:, ] / 1000,
                'model_output': output[0],
                'glacier_rescaling': output[5],
                'Snow_Balance': swe_df,
                'Parameters': matilda_parameters}
        else:
            return {'KGE': output[2],
                'SMB_mean18': results[5]['smb_water_year'][1:, ][:'2018-01-01'].mean() / 1000,
                'SMB': results[5]['smb_water_year'][1:, ] / 1000,
                'model_output': output[0],
                'glacier_rescaling': output[5]}

    def run_single_process(self):
        """
        Runs the MATILDA simulations for the parameter sets in single-processing mode and returns a dictionary
        of results.
        Returns
        -------
        dict
            A dictionary of MATILDA simulation results.
        """

        out_dict = {}
        for i, param_set in enumerate(tqdm(self.matilda_parameters_list, desc="Parameter Sets")):
            model_output = self.matilda_headless(self.forcing_data, self.obs, self.matilda_settings, param_set, self.swe_obs, self.swe_scaling)
            out_dict[f"P{i}"] = model_output

        return out_dict

    def run_multi_process(self, num_cores=2):
        """
        Runs the MATILDA simulations for the parameter sets in multi-processing mode and returns a dictionary
        of results.
        Returns
        -------
        dict
            A dictionary of MATILDA simulation results.
        """

        out_dict = {}
        with Pool(num_cores) as pool:
            parameter_list = self.matilda_parameters_list
            results = pool.map(partial(self.matilda_headless, self.forcing_data, self.obs, self.matilda_settings, swe_obs=self.swe_obs, swe_scaling=self.swe_scaling), parameter_list)

            for i, result in enumerate(results):
                out_dict[f"Parameter_Set_{i + 1}"] = result

            pool.close()

        return out_dict


matilda_settings = {**settings, **calibration_period}
swe_obs = pd.read_csv('/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/hmadsr/kyzylsuu_swe.csv', parse_dates=['Date'], index_col='Date')
swe_obs = swe_obs * 1000
swe_obs = swe_obs['2000-01-01':'2017-09-30']

matilda_bulk = MatildaBulkSampler(df, obs, matilda_settings, param_list, swe_obs, swe_scaling=0.928)
matilda_scenarios = matilda_bulk.run_single_process()
# matilda_scenarios = matilda_bulk.run_multi_process(num_cores=4)

