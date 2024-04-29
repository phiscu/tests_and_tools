## Imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import socket
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

from pathlib import Path
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'

pio.renderers.default = "browser"

data_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/parameters/'

## LHS number of iterations

import math

def calculate_max_iterations(k, p):
    """
    Calculate the maximum number of iterations (N_max) using Latin Hypercube Sampling.

    Parameters:
    k (int): Number of divisions per parameter.
    p (int): Number of parameters.

    Returns:
    int: Maximum number of iterations (N_max).
    """
    return math.factorial(k) ** (p - 1)

# Example usage:
k = 3  # Number of divisions per parameter
p = 9  # Number of parameters
N_max = calculate_max_iterations(k, p)
print(f"Maximum number of iterations (N_max): {N_max}")


## LHS
# Step 1:
# data = pd.read_csv(data_path + 'LHS_Paper_1_timing-param-fix_lr-temp55-65_SMB430_3div_279936_2000-2017.csv')
# Step 2:
# data = pd.read_csv(data_path + 'LHS_Paper_1_timing-param-fix_lr-temp55-65_PCORR-CFMAXice-TTsnow-meanSD_SMB430_3div_279936_2000-2017.csv')
data = pd.read_csv(data_path + 'LHS_Paper_1_FAST2m-005_no-lim_SMB430_3div_1679616_2000-2017.csv')
data = data.drop(['chain'], axis=1)

data = data[data['like1'] > 0.8]
data = data[data['like2'] < 100]
data = data.sort_values(by='like1', ascending=False)
# perc = round(len(data)*0.050)
# data = data.head(perc)
# data = data.tail(10000)

# Remove the prefix "par" from all column names
data.columns = data.columns.str.replace('par', '')

data.mean()
##
# Calculate mean and standard deviation for each column (excluding first two and last column)
stats_dict = {}
for col in data.columns[2:]:
    mean = data[col].mean()
    std = data[col].std()
    stats_dict[col+"_mean"] = round(mean, 5)
    stats_dict[col+"_stddev"] = round(std, 5)

print(stats_dict)

# # Write to table
# table_data = []
# for col in data.columns[:-1]:
#     mean = data[col].mean()
#     std = data[col].std()
#     table_data.append([col, round(mean, 5), round(std, 5)])
#
# table_df = pd.DataFrame(table_data, columns=['Parameter Name', 'Mean', 'Stdv'])
# print(table_df)
#
# table_df.to_csv(data_path + 'par_tab.csv', index=False)

data.columns

## Plot histograms
# Define the burn-in period and thinning factor
burn_in_period = 0
thin_factor = 1

# Subset the data based on burn-in period and thinning factor
thinned_samples = data.iloc[burn_in_period::thin_factor]

# Create a 2x4 matrix of subplots
fig, axs = plt.subplots(3, 3, figsize=(20, 10))

# Plot posterior distributions for each parameter in the matrix of subplots
for i, parameter in enumerate(thinned_samples.columns[2:]):  # Exclude the first two and last column
    row = i // 3
    col = i % 3
    sns.kdeplot(thinned_samples[parameter], shade=True, ax=axs[row, col])
    axs[row, col].set_xlabel(None)
    axs[row, col].set_ylabel('Density')
    axs[row, col].set_title(f'{parameter}', fontweight='bold', fontsize=14)

    if parameter == thinned_samples.columns[2]:
        def format_ticks(x, _):
            return '{:.0f}e-4'.format(x * 10000)  # Adjust multiplier here for desired scientific notation

        axs[row, col].xaxis.set_major_formatter(FuncFormatter(format_ticks))

    # Add vertical lines for mean, mean ± standard deviation
    mean_val = thinned_samples[parameter].mean()
    std_val = thinned_samples[parameter].std()
    axs[row, col].axvline(mean_val, color='red', linestyle='--', label='Mean')
    axs[row, col].axvline(mean_val - std_val, color='blue', linestyle='--', label='Mean - SD')
    axs[row, col].axvline(mean_val + std_val, color='blue', linestyle='--', label='Mean + SD')

plt.tight_layout()
plt.show()

## Identify best run

# Create a custom text that includes the index number for each data point
custom_text = [f'Index: {index}<br>like1: {like1}<br>like2: {like2}' for (index, like1, like2) in zip(data.index, data['like1'], data['like2'])]

# Create a 2D scatter plot with custom text
fig = go.Figure(data=go.Scatter(
    x=data['like1'],
    y=data['like2'],
    mode='markers',
    text=custom_text,  # Assign custom text to each data point
    hoverinfo='text',  # Show custom text when hovering
))

# Update layout
fig.update_layout(
    xaxis_title='Loglike Kling-Gupta-Efficiency score',
    yaxis_title='MAE of mean annual SMB',
    #title='2D Scatter Plot of like1 and like2 with parPCORR Color Ramp',
    margin=dict(l=0, r=0, b=0, t=40)  # Adjust margins for better visualization
)

# Show the plot
fig.show()

## Get parameters set
best = data[data.index == 1347429]
# Filter columns with the prefix 'par'
par_columns = [col for col in best.columns[2:]]

# Create a dictionary with keys as column names without the 'par' prefix
parameters = {col.replace('par', ''): best[col].values[0] for col in par_columns}

# Print the dictionary
print(parameters)

## Scatterplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'data' is your dataframe with columns ['like1', 'like2', 'BETA', 'FC', 'K1', 'K2', 'PERC', 'UZL', 'TT_snow', 'CFMAX_ice', 'CWH']

# Create a 3x3 matrix of subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Loop through each parameter and create a scatter plot against like1
for i in range(3):
    for j in range(3):
        param_name = data.columns[2 + 3*i + j]  # Adjust the indexing based on your specific column names
        ax = axs[i, j]
        scatter = ax.scatter(data[param_name], data['like1'], c=data['like2'], cmap='viridis')
        ax.set_xlabel(param_name)
        ax.set_ylabel('like1')
        fig.colorbar(scatter, ax=ax, label='like2')

plt.tight_layout()
plt.show()

## Correlation matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your dataframe with columns ['like1', 'like2', 'BETA', 'FC', 'K1', 'K2', 'PERC', 'UZL', 'TT_snow', 'CFMAX_ice', 'CWH']

# Calculate the correlation matrix
corr = data.corr()

# Create a matrix of correlation plots
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of All Columns')
plt.show()

## Pairplots (takes a long time, max. 400 rows!)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your dataframe with columns ['like1', 'like2', 'BETA', 'FC', 'K1', 'K2', 'PERC', 'UZL', 'TT_snow', 'CFMAX_ice', 'CWH']

# Create a pairplot of all columns
sns.pairplot(data)
plt.suptitle('Pairplot of All Columns', y=1.02)
plt.show()