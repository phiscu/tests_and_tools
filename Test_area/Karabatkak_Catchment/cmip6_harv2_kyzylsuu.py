# -*- coding: UTF-8 -*-

## import
import pandas as pd
from pathlib import Path
import sys
from bias_correction import BiasCorrection
import socket

host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
sys.path.append(home + '/Ana-Lena_Phillip/data/tests_and_tools')
# from Preprocessing.Preprocessing_functions import dmod_score, load_cmip, cmip2df
from matilda.core import matilda_simulation, matilda_parameter, matilda_preproc, input_scaling, calculate_glaciermelt, calculate_PDD, glacier_area_change, create_lookup_table, hbv_simulation, updated_glacier_melt, create_statistics

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
df_har = pd.concat([t2m_agg.har, tp_agg.har], axis=1)
df_har.columns = ['t2m', 'tp']

# CMIP6
scen = ['1_2_6', '2_4_5', '3_7_0', '5_8_5']
cmip_mod_tas = {}
cmip_mod_pr = {}

for s in scen:
    name = 'ssp' + s[:1]
    cmip_mod_tas[name] = pd.read_csv(input_path + cmip_path + 't2m_CMIP6_all_models_raw_42.516-79.0167_1982-01-01-2100-12-31_'
                             + name + '.csv', index_col='time', parse_dates=['time'])
    cmip_mod_pr[name] = pd.read_csv(input_path + cmip_path + 'tp_CMIP6_all_models_raw_42.516-79.0167_1982-01-01-2100-12-31_'
                             + name + '.csv',  index_col='time', parse_dates=['time'])

## Bias adjustment:

final_train_slice = slice('1982-01-01', '2020-12-31')
final_predict_slice = slice('1980-01-01', '2100-12-31')

# Temperature:
cmip_corrT_mod = cmip_mod_tas.copy()
for s in scen:
    s = 'ssp' + s[:1]
    for m in cmip_mod_tas[s].columns:
        x_train = cmip_mod_tas[s][m][final_train_slice].squeeze()
        y_train = df_har[final_train_slice]['t2m'].squeeze()
        x_predict = cmip_mod_tas[s][m][final_predict_slice].squeeze()
        bc_cmip = BiasCorrection(y_train, x_train, x_predict)
        cmip_corrT_mod[s][m] = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))
        cmip_corrT_mod[s]['mean'] = cmip_corrT_mod[s].mean(axis=1)
        # cmip_corrT_mod[s].to_csv(input_path + cmip_path + 't2m_CMIP6_all_models_adjusted2catchment-aggregates_42.516-79.0167_1982-01-01-2100-12-31_'
        #                          + s + '.csv')

# Precipitation:
cmip_corrP_mod = cmip_mod_pr.copy()
for s in scen:
    s = 'ssp' + s[:1]
    for m in cmip_mod_pr[s].columns:
        x_train = cmip_mod_pr[s][m][final_train_slice].squeeze()
        y_train = df_har[final_train_slice]['tp'].squeeze()
        x_predict = cmip_mod_pr[s][m][final_predict_slice].squeeze()
        bc_cmip = BiasCorrection(y_train, x_train, x_predict)
        cmip_corrP_mod[s][m] = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))
        cmip_corrP_mod[s][m][cmip_corrP_mod[s][m] < 0] = 0          # only needed when using normal mapping for precipitation
        cmip_corrP_mod[s]['mean'] = cmip_corrP_mod[s].mean(axis=1)
        # cmip_corrP_mod[s].to_csv(input_path + cmip_path + 'tp_CMIP6_all_models_adjusted2catchment-aggregates_42.516-79.0167_1982-01-01-2100-12-31_'
        #                          + s + '.csv')

# Create MATILDA input
matilda_scenarios = {}
for s in [1, 2, 3, 5]:
    s = 'ssp' + str(s)
    matilda_scenarios[s] = {}
    for m in cmip_mod_tas[s].columns:
        model = pd.DataFrame({'T2': cmip_corrT_mod[s][m],
                              'RRR': cmip_corrP_mod[s][m]})['1997-01-01':]
        model = model.reset_index()
        mod_dict = {m: model.rename(columns={'time': 'TIMESTAMP'})}
        matilda_scenarios[s].update(mod_dict)



matilda_scenarios['ssp1']['inm_cm4_8']


## MATILDA run with calibrated parameters

matilda_settings = {"obs": obs, "set_up_start": '1997-01-01', "set_up_end": '1999-12-31',
                    "sim_start": '2000-01-01', "sim_end": '2017-12-31', "freq": "M", "glacier_profile": glacier_profile,
                    "area_cat": 295.763, "lat": 42.33, "warn": False, "plot_type": "all", "plots": True, "elev_rescaling": True,
                    "ele_dat": 3172, "ele_cat": 3295, "area_glac": 32.51, "ele_glac": 4068, "pfilter": 0}

# har_lrtemp007-005_PCORR08-12_lhs_multiobj_mb160_50000:
param_dict = {'lr_temp': -0.006077369, 'lr_prec': 0.0013269137, 'BETA': 5.654754, 'CET': 0.08080378, 'FC': 365.68375,
              'K0': 0.36890236, 'K1': 0.022955153, 'K2': 0.060069658, 'LP': 0.63395154, 'MAXBAS': 5.094901,
              'PERC': 0.39491335, 'UZL': 348.0978, 'PCORR': 1.0702422, 'TT_snow': -1.1521467, 'TT_diff': 1.5895765,
              'CFMAX_ice': 3.6518102, 'CFMAX_rel': 1.8089349, 'SFCF': 0.42293832, 'CWH': 0.11234668, 'AG': 0.9618855,
              'RFS': 0.11432563}

# Test
df = matilda_scenarios['ssp5']['mean']

output_MATILDA = matilda_simulation(df_har, **matilda_settings, parameter_set=param_dict)

print('Mean Annual MB: ' + str(round(output_MATILDA[5].smb_water_year.mean() / 1000, 2)) + ' (+-'
      + str(round(output_MATILDA[5].smb_water_year.std() / 1000, 2)) + ') m w.e.')


# Oh oh...DIE ERSTE MB_CUM IST OFT POSITIV (sogar der mean...)!
# --> positive MB_cum anders behandeln? wenn positive dann bleibt die geometrie gleich?!