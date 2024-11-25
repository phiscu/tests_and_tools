##
import numpy as np
import pandas as pd
import spotpy
from pathlib import Path
import os
import contextlib
import sys
import socket
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
import plotly.graph_objs as go
import pandas as pd
import plotly.io as pio

pio.renderers.default = "browser"

##

fast_path = '/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/FAST/'

step1 = fast_path + 'glacier_cover/FAST_Paper_1_internal_SWE_targetMB430_1500012_glacarea0'
step2 = fast_path + 'glacier_cover/FAST_Paper_1_snow_cal_no-fix_targetMB430_targetSWE_1500009'
step3 = fast_path + 'glacier_cover/FAST_Paper_1_internal_SWE_targetMB430_1500012_glacarea30'
step4 = fast_path + 'glacier_cover/FAST_Paper_1_internal_SWE_targetMB430_1500012_glacarea50'
step5 = fast_path + 'glacier_cover/FAST_Paper_1_snow_cal_param-fix-SCFC-CET_PCORR-0-58_targetMB430_targetSWE_1500012'

step1 = '/home/phillip/Seafile/CLIMWATER/YSS/2024/Pskem_data/Calibration/FAST_Pskem_Cal_Step2-FAST-internal.csv'


##
def fast_iter(param, interf=4, freqst=2):
    """
    Calculates the number of parameter iterations needed for parameterization and sensitivity analysis using FAST.
    Parameters
    ----------
    param : int
        The number of input parameters being analyzed.
    interf : int
        The interference factor, which determines the degree of correlation between the input parameters.
    freqst : int
        The frequency step, which specifies the size of the intervals between each frequency at which the Fourier transform is calculated.
    Returns
    -------
    int
        The total number of parameter iterations needed for FAST.
    """
    # return (1 + 4 * interf ** 2 * (1 + (param - 2) * freqst)) * param
    return 1 + 4 * interf ** 2 * (1 + (param - 2) * freqst)


fast_iter(18, 4, 2)


##
def get_si(fast_results: str, to_csv: bool = False) -> pd.DataFrame:
    """
    Computes the sensitivity indices of a given FAST simulation results file.
    Parameters
    ----------
    fast_results : str
        The path of the FAST simulation results file.
    to_csv : bool, optional
        If True, the sensitivity indices are saved to a CSV file with the same
        name as fast_results, but with '_sensitivity_indices.csv' appended to
        the end (default is False).
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the sensitivity indices and parameter
        names.
    """
    if fast_results.endswith(".csv"):
        fast_results = fast_results[:-4]  # strip .csv
    results = spotpy.analyser.load_csv_results(fast_results)
    # Suppress prints
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        SI = spotpy.analyser.get_sensitivity_of_fast(results, print_to_console=False)
    parnames = spotpy.analyser.get_parameternames(results)
    sens = pd.DataFrame(SI)
    sens['param'] = parnames
    sens.set_index('param', inplace=True)
    if to_csv:
        sens.to_csv(os.path.basename(fast_results) + '_sensitivity_indices.csv', index=False)
    return sens


def plot_sensitivity_bars(*dfs, labels=None, show=False, bar_width=0.2, bar_gap=0.7, threshold=0.05,
                          legend_title='Catchment Glacier Cover',
                          custom_order=None):
    """
    Plots a horizontal bar chart showing the total sensitivity index for each parameter in a MATILDA model.
    Parameters
    ----------
    *dfs : pandas.DataFrame
        Multiple dataframes containing the sensitivity indices for each parameter.
    labels : list of str, optional
        Labels to use for the different steps in the sensitivity analysis. If not provided, the default
        labels will be 'No Limits', 'Step 1', 'Step 2', etc.
    bar_width : float, optional
        Width of each bar in the chart.
    bar_gap : float, optional
        Space between bars.
    threshold : float, optional
        Sensitivity threshold value to be displayed as dashed vertical line.
    """
    traces = []
    colors = ['darkblue', 'orange', 'lime', 'purple', 'darkcyan']  # add more colors if needed
    param_no = []
    for i, df in enumerate(dfs):
        df = get_si(df)
        param_no_df = len(df)
        param_no.append(param_no_df)
        if i > 0:
            if labels is None:
                label = 'Step ' + str(i)
            else:
                label = labels[i]
        else:
            if labels is None:
                label = 'No Limits'
            else:
                label = labels[i]
        if custom_order is not None:
            df = df.reindex(custom_order[::-1])     # for some reason .reindex() reverses the order
        trace = go.Bar(y=df.index,
                       x=df['ST'],
                       name=label,
                       orientation='h',
                       marker=dict(color=colors[i]),
                       width=bar_width,
                       textfont=dict(size=20))
        traces.append(trace)

    # Calculate number of displayed parameters
    max_len = max(param_no) - 0.5

    # Add dashed horizontal line at the threshold value
    layout = go.Layout(shapes=[dict(type='line', x0=threshold, x1=threshold, y0=-0.5, y1=max_len,
                                    line=dict(color='darkgrey', width=2, dash='dash'))],
    title = dict(text='<b>' + 'Sensitivity Assessment for MATILDA Parameters' + '<b>', font=dict(size=24)),
    xaxis_title = 'Total Order Sensitivity Index',
    yaxis_title = 'Parameter',
    bargap = bar_gap,
    font = dict(size=20, family='Palatino'),
    showlegend=True,
    legend_title_text=legend_title,
    legend_title_font=dict(size=18, family='Palatino'))

    fig = go.Figure(data=traces, layout=layout)
    if show:
        fig.show()



##
plot_sensitivity_bars(step4, step3, step5, step2, step1,
                      labels=['50%', '30%', '10.8% (internal only)', '10.8% (all)', '0%'],
                      bar_width=0.25,
                      bar_gap=0.25,
                      show=True,
                      custom_order=['SFCF', 'CET', 'PCORR', 'lr_temp', 'lr_prec', 'RFS', 'TT_snow', 'TT_diff',
                                        'CFMAX_snow', 'CFMAX_rel', 'CWH', 'AG', 'BETA', 'FC', 'LP', 'K0', 'K1',
                                        'K2', 'PERC', 'UZL', 'MAXBAS'])

# plot_sensitivity_bars(step1, step2, step3, step4,
#                       labels=['No constraints', 'Internal (Prec 100%)', 'Internal (Prec 60%)', 'Internal (Prec 48%)'],
#                       bar_width=0.2,
#                       bar_gap=0.3,
#                       show=True)

##
si_df = get_si(step5)
sensitive = si_df.index[si_df['ST'] > 0.05].values
insensitive = si_df.index[si_df['ST'] < 0.05].values

print('Parameters with a sensitivity index < 0.05:\n\n')
[print(i) for i in insensitive]

no_lim_interactions = no_lim_scores.ST - no_lim_scores.S1

steps = [step1, step2, step3, step4]

param_sens = []
for step in steps:
    df = get_si(step)
    sensitive = df.index[df['ST'] > 0.05].values
    param_sens.append(len(sensitive))

print(param_sens)

##
step1 = fast_path + 'FAST_Paper_1_param-fix-SCFC-CET_PCORR-1_38034'
step2 = fast_path + 'FAST_Paper_1_param-fix-SCFC-CET_PCORR-1_'
step3 = fast_path + 'FAST_Paper_1_param-fix-SCFC-CET_PCORR-1_228204'

plot_sensitivity_bars(step1, step2, step3,
                      labels=['38034 iterations', '114102 iterations', '228204 iterations'],
                      bar_width=0.2,
                      bar_gap=0.3,
                      show=True)

steps = [step1, step2, step3]

param_sens = []
for step in steps:
    df = get_si(step)
    sensitive = df.index[df['ST'] > 0.05].values
    param_sens.append(len(sensitive))

print(param_sens)

##
si_df = get_si(step1)
sensitive5 = si_df.index[si_df['ST'] > 0.05].values
insensitive5 = si_df.index[si_df['ST'] < 0.05].values

interactions = si_df.ST - si_df.S1

print('Parameters with a sensitivity index > 0.05:\n\n')
[print(i) for i in sensitive5]

plot_sensitivity_bars(step1,
                      labels=['Internal Parameters'],
                      legend_title='',
                      bar_width=0.2,
                      bar_gap=0.3,
                      show=True)

sensitive6 = si_df.index[si_df['ST'] > 0.05].values
insensitive6 = si_df.index[si_df['ST'] < 0.05].values

# FAST005: ['lr_temp', 'lr_prec', 'K0', 'LP', 'MAXBAS', 'TT_diff', 'AG', 'RFS']
# FAST007: ['lr_temp', 'lr_prec', 'BETA', 'K0', 'LP', 'MAXBAS', 'TT_diff', 'CFMAX_rel', 'AG', 'RFS']