# -*- coding: UTF-8 -*-
## import
import random
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from bias_correction import BiasCorrection
import socket
import matplotlib.pyplot as plt
import random
from scipy import stats

host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
sys.path.append(home + '/Ana-Lena_Phillip/data/tests_and_tools')
# from Preprocessing.Preprocessing_functions import dmod_score, load_cmip, cmip2df
from matilda.core import matilda_simulation, matilda_parameter, matilda_preproc, input_scaling, calculate_glaciermelt, \
    calculate_PDD, glacier_area_change, create_lookup_table, hbv_simulation, updated_glacier_melt, create_statistics

## Paths

wd = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data'
input_path = wd + "/input/kyzylsuu"
output_path = wd + "/output/kyzylsuu"

t2m_agg_path = '/met/temp_cat_agg_era5l_harv2_mswx_1982-2020.csv'
tp_agg_path = '/met/prec_cat_agg_era5l_harv2_mswx_1982-2020.csv'
runoff_obs = "/hyd/obs/Kyzylsuu_1982_2020_latest.csv"
cmip_path = '/met/cmip6/'

obs = pd.read_csv(input_path + runoff_obs)
glacier_profile = pd.read_csv(wd + "/kyzulsuu_glacier_profile.csv")

# Catchment-wide aggregates
t2m_agg = pd.read_csv(input_path + t2m_agg_path, index_col='time', parse_dates=['time'])
tp_agg = pd.read_csv(input_path + tp_agg_path, index_col='time', parse_dates=['time'])
har = pd.concat([t2m_agg.har, tp_agg.har], axis=1)
har.columns = ['t2m', 'tp']

# CMIP6
scen = ['1_2_6', '2_4_5', '3_7_0', '5_8_5']
cmip_mod_tas = {}
cmip_mod_pr = {}

for s in scen:
    name = 'ssp' + s[:1]
    cmip_mod_tas[name] = pd.read_csv(
        input_path + cmip_path + 't2m_CMIP6_all_models_raw_42.516-79.0167_1982-01-01-2100-12-31_'
        + name + '.csv', index_col='time', parse_dates=['time'])
    cmip_mod_pr[name] = pd.read_csv(
        input_path + cmip_path + 'tp_CMIP6_all_models_raw_42.516-79.0167_1982-01-01-2100-12-31_'
        + name + '.csv', index_col='time', parse_dates=['time'])

cmip_mod_pr['ssp2']['1979-01-01': '2021-12-31'].describe().transpose().describe()
har['1979-01-01': '2021-12-31'].describe()

## Bias adjustment:

# Training data choice:
tr_data = har

final_train_slice = slice('1982-01-01', '2020-12-31')
final_predict_slice = slice('1980-01-01', '2100-12-31')

# Temperature:
cmip_corrT_mod = cmip_mod_tas.copy()
for s in scen:
    s = 'ssp' + s[:1]
    for m in cmip_mod_tas[s].columns:
        x_train = cmip_mod_tas[s][m][final_train_slice].squeeze()
        y_train = tr_data[final_train_slice]['t2m'].squeeze()
        x_predict = cmip_mod_tas[s][m][final_predict_slice].squeeze()
        bc_cmip = BiasCorrection(y_train, x_train, x_predict)
        cmip_corrT_mod[s][m] = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))
        cmip_corrT_mod[s]['mean'] = cmip_corrT_mod[s].mean(axis=1)
        # cmip_corrT_mod[s].to_csv(input_path + cmip_path
        #                          + 't2m_CMIP6_all_models_adjusted2harv2-catchm_42.516-79.0167_1982-01-01-2100-12-31_'
        #                          + s + '.csv')

# Precipitation:
cmip_corrP_mod = cmip_mod_pr.copy()
for s in scen:
    s = 'ssp' + s[:1]
    for m in cmip_mod_pr[s].columns:
        x_train = cmip_mod_pr[s][m][final_train_slice].squeeze()
        y_train = tr_data[final_train_slice]['tp'].squeeze()
        x_predict = cmip_mod_pr[s][m][final_predict_slice].squeeze()
        bc_cmip = BiasCorrection(y_train, x_train, x_predict)
        cmip_corrP_mod[s][m] = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))
        cmip_corrP_mod[s][m][cmip_corrP_mod[s][m] < 0] = 0  # only needed when using normal mapping for precipitation
        cmip_corrP_mod[s]['mean'] = cmip_corrP_mod[s].mean(axis=1)
        # cmip_corrP_mod[s].to_csv(input_path + cmip_path
        #                          + 'tp_CMIP6_all_models_adjusted2harv2-catchm_42.516-79.0167_1982-01-01-2100-12-31_'
        #                          + s + '.csv')

# Create MATILDA input
matilda_scenarios = {}
for s in [1, 2, 3, 5]:
    s = 'ssp' + str(s)
    matilda_scenarios[s] = {}
    for m in cmip_mod_tas[s].columns:
        model = pd.DataFrame({'T2': cmip_corrT_mod[s][m],
                              'RRR': cmip_corrP_mod[s][m]})#['1997-01-01':]
        model = model.reset_index()
        mod_dict = {m: model.rename(columns={'time': 'TIMESTAMP'})}
        matilda_scenarios[s].update(mod_dict)

# matilda_scenarios['ssp1']['inm_cm4_8']

## MATILDA run with calibrated parameters

# "obs": obs,
matilda_settings = {"set_up_start": '1997-01-01', "set_up_end": '1999-12-31',
                    "sim_start": '2000-01-01', "sim_end": '2020-12-31', "freq": "M", "glacier_profile": glacier_profile,
                    "area_cat": 295.763, "lat": 42.33, "warn": False, "plot_type": "all", "plots": True,
                    "elev_rescaling": True,
                    "ele_dat": 3172, "ele_cat": 3295, "area_glac": 32.51, "ele_glac": 4068, "pfilter": 0}

# har_lrtemp007-005_PCORR08-12_lhs_multiobj_mb160_50000:
param_dict = {'lr_temp': -0.006077369, 'lr_prec': 0.0013269137, 'BETA': 5.654754, 'CET': 0.08080378, 'FC': 365.68375,
              'K0': 0.36890236, 'K1': 0.022955153, 'K2': 0.060069658, 'LP': 0.63395154, 'MAXBAS': 5.094901,
              'PERC': 0.39491335, 'UZL': 348.0978, 'PCORR': 1.0702422, 'TT_snow': -1.1521467, 'TT_diff': 1.5895765,
              'CFMAX_ice': 3.6518102, 'CFMAX_rel': 1.8089349, 'SFCF': 0.42293832, 'CWH': 0.11234668, 'AG': 0.9618855,
              'RFS': 0.11432563}

# Test
output_MATILDA = matilda_simulation(df1, **matilda_settings, parameter_set=param_dict)

print('Mean Annual MB: ' + str(round(output_MATILDA[5].smb_water_year.mean() / 1000, 2)) + ' (+-'
      + str(round(output_MATILDA[5].smb_water_year.std() / 1000, 2)) + ') m w.e.')
print(output_MATILDA[5])

# output_MATILDA[7].show()
# output_MATILDA[9].show()
## Loop through all scenarios and models:       BAUSTELLE!

matilda_scenarios.items()


ensemble_outputs = np.zeros((4, 7, 27))
ensemble_glacierchange = np.zeros((4, 7, 8))


#   beide gehen nicht, weil to_numeric() nicht mit dataframes funktioniert:
pd.to_numeric(output_MATILDA[0])
# ohne das hängt sich die loop aber daran auf, den string 'initial' in ein array zu schreiben...



for i, scenario in enumerate(matilda_scenarios.keys()):
    for j, ensemble_member in enumerate(matilda_scenarios[scenario].keys()):
        # get the climate data for this ensemble member in this scenario
        climate_data = matilda_scenarios[scenario][ensemble_member]
        # run your model with the climate data
        output = matilda_simulation(climate_data, **matilda_settings, parameter_set=param_dict)
        output_df = pd.to_numeric(output[0], errors='coerce')
        output_glacier = pd.to_numeric(output[5], errors='coerce')
        # store the output in the results array
        ensemble_outputs[i, j, :] = output_df
        ensemble_glacierchange[i, j, :] = output_glacier



def run_model(climate_data):
    # Loop through each scenario and ensemble member
    results = {}
    for scenario, ensembles in climate_data.items():
        results[scenario] = {}
        for ensemble, data in ensembles.items():
            # Run the model
            model_output = matilda_simulation(data, **matilda_settings, parameter_set=param_dict)
            model_output = model_output[0]
            # Convert any strings to NaN
            model_output = pd.to_numeric(model_output, errors='coerce')
            # Store the results
            results[scenario][ensemble] = model_output
    return results






##  Plot Glacier area evolution
output_MATILDA[5].glacier_area.plot()
plt.show()

## Create dataframe with annual results
df = output_MATILDA[0]
mean_cols = df.iloc[:, :2]
sum_cols = df.iloc[:, 2:]
mean_resampled = mean_cols.resample('Y').mean()
sum_resampled = sum_cols.resample('Y').sum()
df = pd.concat([mean_resampled, sum_resampled], axis=1)

## Create Dataframe and plots with linear trends and respective test-statistics


class TrendAnalyzer:

    def __init__(self, df, p_value_threshold=0.05):
        self.df = df
        self.p_value_threshold = p_value_threshold

    def fit_trend(self, col):
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(range(len(col))), col)
        r_squared = r_value ** 2
        return slope, intercept, r_squared, p_value, std_err

    def trend_df(self):
        data = pd.DataFrame(self.df.apply(self.fit_trend, axis=0)).transpose()
        data.columns = ['slope', 'intercept', 'r_squared', 'p_value', 'std_err']
        data['signf'] = False
        data.loc[(data['p_value'] < self.p_value_threshold), 'signf'] = True
        return data

    def plot_trends(self):
        data = self.trend_df()
        fig, axs = plt.subplots(nrows=9, ncols=3, figsize=(15, 15), sharex=True, sharey=False)
        axs = axs.ravel()
        for i, col in enumerate(self.df.columns):
            slope = data.loc[col, 'slope']
            intercept = data.loc[col, 'intercept']
            r_squared = data.loc[col, 'r_squared']
            p_value = data.loc[col, 'p_value']
            std_err = data.loc[col, 'std_err']
            axs[i].plot(self.df.index, self.df[col], label=col)
            x = np.array(list(range(len(self.df))))
            y = slope * x + intercept
            if data.loc[col, 'signf']:
                axs[i].plot(self.df.index, y, label='Trendline', color='orange')
            else:
                axs[i].plot(self.df.index, y, label='Trendline', color='red')
            axs[i].set_title(f"{col}: Slope={slope:.2E}")
        plt.tight_layout()
        plt.show()


trends = TrendAnalyzer(df)
print(trends.trend_df())
trends.plot_trends()

## Compare scenarios in training period


class TrainingPeriodCheck:

    def __init__(self, df, reanalysis, freq):
        self.df = df
        self.reanalysis = reanalysis
        self.freq = freq

    def cmip_mean(self, ssp):
        return self.df[ssp]['mean']

    def dataframe(self):
        df = pd.DataFrame({'Reanalysis': self.reanalysis,
                           'ssp1': self.cmip_mean('ssp1'),
                           'ssp2': self.cmip_mean('ssp2'),
                           'ssp3': self.cmip_mean('ssp3'),
                           'ssp5': self.cmip_mean('ssp5')})
        df = df[:df['Reanalysis'].dropna().index[-1].strftime('%Y-%m-%d')]
        if df.mean().mean() > 200:
            df = df.resample(self.freq).mean()
        else:
            df = df.resample(self.freq).sum()
        return df

    def plot(self):
        self.dataframe().plot()
        if self.dataframe().mean().mean() < 340:
            plt.title('Annual Mean Temperature in Training Period (ensemble mean)')
        else:
            plt.title('Annual Precipitation in Training Period (ensemble mean)')
        plt.show()


result = TrainingPeriodCheck(cmip_corrP_mod, har.tp, 'Y')
print(result.dataframe())
result.plot()




