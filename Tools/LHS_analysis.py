## Imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import socket
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import math

from pathlib import Path
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'

pio.renderers.default = "browser"
plt.rcParams["font.family"] = "Palatino"


data_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/parameters/'

## LHS number of iterations


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
k = 2 # Number of divisions per parameter
p = 18  # Number of parameters
N_max = calculate_max_iterations(k, p)
print(f"Maximum number of iterations (N_max): {N_max}")


## LHS
# Step 1:
# data = pd.read_csv(data_path + 'LHS_Paper_1_timing-param-fix_lr-temp55-65_SMB430_3div_279936_2000-2017.csv')
# Step 2:
# data = pd.read_csv(data_path + 'LHS_Paper_1_timing-param-fix_lr-temp55-65_PCORR-CFMAXice-TTsnow-meanSD_SMB430_3div_279936_2000-2017.csv')
# data = pd.read_csv(data_path + 'SWE/LHS_Paper_1_timing-param-fix_SWE_SMB430_3div_279936_2000-2017.csv')
# data = pd.read_csv(data_path + 'SWE/LHS_Paper_1_FAST-SWE-2M-006_timing-param_fix_SWE_SMB430_5div_14400_2000-2017.csv')
# data = pd.read_csv(data_path + 'SWE/LHS_Paper_1_SnowCal-update-Step1_timing-param-fix_3div_279936_2000-2017.csv')
# data = pd.read_csv(data_path + 'SWE/LHS_Paper_1_SnowCal-update-Step2_snow-calib_5div_14400_2000-2017.csv')
data = pd.read_csv(data_path + 'SWE/LHS_Paper_1_SnowCal-update-Step3_glacier-calib_5000_2000-2017.csv')

data = data.drop(['chain'], axis=1)
data.columns = ['KGE_Runoff', 'MAE_SMB', 'KGE_SWE'] + list(data.columns[3:])
data.columns = data.columns.str.replace('par', '')

data = data[data['KGE_Runoff'] > 0.5]
data = data[data['MAE_SMB'] < 100]
data = data[data['KGE_SWE'] > 0.8]
# data = data.sort_values(by='KGE_Runoff', ascending=False)
# perc = round(len(data)*0.01)
# data = data.head(perc)
# data = data.tail(10000)

data.mean()

## Plot histograms
# Define the burn-in period and thinning factor
burn_in_period = 0
thin_factor = 1

# Subset the data based on burn-in period and thinning factor
thinned_samples = data.iloc[burn_in_period::thin_factor]

# Create a 2x4 matrix of subplots
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

# Plot posterior distributions for each parameter in the matrix of subplots
for i, parameter in enumerate(thinned_samples.columns[3:]):  # Exclude the first two columns
    row = i // 4
    col = i % 4
    sns.kdeplot(thinned_samples[parameter], shade=True, ax=axs[row, col])
    axs[row, col].set_xlabel(None)
    axs[row, col].set_ylabel('Density')
    axs[row, col].set_title(f'{parameter}', fontweight='bold', fontsize=14)
    if parameter == thinned_samples.columns[3]:
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


## Plot histogram for 3 parameters only

# Plot histograms
# Define the burn-in period and thinning factor
burn_in_period = 0
thin_factor = 1
# Subset the data based on burn-in period and thinning factor
thinned_samples = data.iloc[burn_in_period::thin_factor]
# Create a 1x3 matrix of subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Plot posterior distributions for each parameter in the matrix of subplots
for i, parameter in enumerate(thinned_samples.columns[3:6]):  # Exclude the first two columns
    sns.kdeplot(thinned_samples[parameter], shade=True, ax=axs[i])
    axs[i].set_xlabel(None)
    axs[i].set_ylabel('Density')
    axs[i].set_title(f'{parameter}', fontweight='bold', fontsize=14)
    # Add vertical lines for mean, mean ± standard deviation
    mean_val = thinned_samples[parameter].mean()
    std_val = thinned_samples[parameter].std()
    axs[i].axvline(mean_val, color='red', linestyle='--', label='Mean')
    axs[i].axvline(mean_val - std_val, color='blue', linestyle='--', label='Mean - SD')
    axs[i].axvline(mean_val + std_val, color='blue', linestyle='--', label='Mean + SD')
plt.tight_layout()
plt.show()
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
table_data = []
for col in data.columns[:]:
    mean = data[col].mean()
    std = data[col].std()
    table_data.append([col, round(mean, 5), round(std, 5)])

table_df = pd.DataFrame(table_data, columns=['Parameter Name', 'Mean', 'Stdv'])
print(table_df)

# table_df.to_csv(data_path + 'par_tab.csv', index=False)

print([data['CFMAX_rel'].min(), data['CFMAX_rel'].max()])
# print([data['KGE_Runoff'].min(), data['KGE_Runoff'].max()])

## Identify best run

custom_text = [f'Index: {index}<br>KGE_Runoff: {KGE_Runoff}<br>MAE_SMB: {MAE_SMB}<br>KGE_SWE: {KGE_SWE}' for (index, KGE_Runoff, MAE_SMB, KGE_SWE) in zip(data.index, data['KGE_Runoff'], data['MAE_SMB'], data['KGE_SWE'])]

# Transform into a 3D plot
fig = go.Figure(data=[go.Scatter3d(
    x=data['KGE_Runoff'],
    y=data['MAE_SMB'],
    z=data['KGE_SWE'],
    mode='markers',
    text=custom_text,  # Assign custom text to each data point
    hoverinfo='text',  # Show custom text when hovering
    marker=dict(
        size=5,
        color=data['KGE_SWE'],  # Color based on KGE_SWE values
        colorscale='Viridis',  # Choose a colorscale for the color ramp
        colorbar=dict(title='KGE_SWE')  # Add colorbar with title
    )
)]
)
# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='Loglike Kling-Gupta-Efficiency score',
        yaxis_title='MAE of mean annual SMB',
        zaxis_title='KGE of SWE',
    ),
    margin=dict(l=0, r=0, b=0, t=40)  # Adjust margins for better visualization
)
# Show the plot
fig.show()
## Get parameters set
best = data[data.index == 7287]
# Filter columns with the prefix 'par'
par_columns = [col for col in best.columns[3:]]

# Create a dictionary with keys as column names without the 'par' prefix
parameters = {col.replace('par', ''): best[col].values[0] for col in par_columns}

# Print the dictionary
print(parameters)

## Scatterplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a 2x4 matrix of subplots
fig, axs = plt.subplots(2, 4, figsize=(15, 10))

# Loop through each parameter and create a scatter plot against KGE_Runoff
for i in range(2):
    for j in range(4):
        param_name = data.columns[3 + 4*i + j]  # Adjust the indexing based on your specific column names
        ax = axs[i, j]
        scatter = ax.scatter(data[param_name], data['KGE_Runoff'], c=data['KGE_SWE'], cmap='viridis')
        ax.set_xlabel(param_name, weight='bold')
        ax.set_ylabel('KGE Runoff')
        fig.colorbar(scatter, ax=ax, label='KGE SWE')

plt.tight_layout()
plt.show()

## Correlation matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your dataframe with columns ['KGE_Runoff', 'MAE_SMB', 'BETA', 'FC', 'K1', 'K2', 'PERC', 'UZL', 'TT_snow', 'CFMAX_ice', 'CWH']

# Calculate the correlation matrix
corr = data.corr()

# Create a matrix of correlation plots
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8})
# plt.title('Correlation Matrix of All Columns')
plt.show()

## Pairplots (takes a long time, max. 400 rows!)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Create a pairplot of all columns
sns.pairplot(data)
plt.suptitle('Pairplot of All Columns', y=1.02)
plt.show()

## Pareto front
from scipy.stats import rankdata
import numpy as np

# Normalize the objective functions using Min-Max normalization
objectives = ["KGE_Runoff", "MAE_SMB", "KGE_SWE"]
for obj in objectives:
    data[f"{obj}_normalized"] = (data[obj] - data[obj].min()) / (data[obj].max() - data[obj].min())

data["MAE_SMB_normalized"] = 1 - data["MAE_SMB_normalized"]     # make MAE ascending

# Rank the normalized objective functions (ascending for MAE_SMB, descending for others)
data["KGE_Runoff_rank"] = rankdata(-data["KGE_Runoff_normalized"], method="min")
data["MAE_SMB_rank"] = rankdata(-data["MAE_SMB_normalized"], method="min")
data["KGE_SWE_rank"] = rankdata(-data["KGE_SWE_normalized"], method="min")

# Define a function to determine the Pareto front
def is_pareto_efficient(points, maximize=True):
    num_points = points.shape[0]
    is_efficient = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        # If maximizing, a point is dominated if another point is greater in all objectives
        # If minimizing, a point is dominated if another point is lesser in all objectives
        if maximize:
            is_efficient[i] = not np.any(np.all(points > points[i], axis=1))
        else:
            is_efficient[i] = not np.any(np.all(points < points[i], axis=1))
    return is_efficient

# Check which points are on the Pareto front
pareto_columns = ["KGE_Runoff_normalized", "MAE_SMB_normalized", "KGE_SWE_normalized"]
pareto_points = data[pareto_columns].values
pareto_front = is_pareto_efficient(pareto_points, maximize=True)

data["on_pareto_front"] = pareto_front

# Display the results
print("Normalized Data:")
print(data[["KGE_Runoff_normalized", "MAE_SMB_normalized", "KGE_SWE_normalized"]])

print("\nRanks:")
print(data[["KGE_Runoff_rank", "MAE_SMB_rank", "KGE_SWE_rank"]])

print("\nPareto Front:")
print(data[data["on_pareto_front"]])



## Pareto Plots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D scatter plots

# Assuming data has the normalized objective functions and Pareto front information
# Create a 2D scatter plot for the Pareto front visualization
# Objective functions to plot (adjust as needed)
x_obj = "KGE_Runoff"
y_obj = "MAE_SMB"
z_obj = "KGE_SWE"

# Plot settings
plt.figure(figsize=(8, 6))
plt.scatter(data[x_obj], data[y_obj], c=data["on_pareto_front"], cmap="coolwarm", s=100, label="Pareto Front")
plt.xlabel("KGE_Runoff")
plt.ylabel("MAE_SMB")
plt.title("2D Scatter Plot of Pareto Front")

# Highlight Pareto front points
pareto_points = data[data["on_pareto_front"]]
plt.scatter(pareto_points[x_obj], pareto_points[y_obj], color="red", s=100, label="Pareto Optimal")

plt.legend()
plt.show()

# Create a 3D scatter plot to visualize the Pareto front
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(data[x_obj], data[y_obj], data[z_obj], c=data["on_pareto_front"], cmap="coolwarm", s=100)
ax.set_xlabel("KGE_Runoff")
ax.set_ylabel("MAE_SMB")
ax.set_zlabel("KGE_SWE")
ax.set_title("3D Scatter Plot of Pareto Front")

# Highlight Pareto front points in 3D
ax.scatter(pareto_points[x_obj], pareto_points[y_obj], pareto_points[z_obj], color="red", s=100)

plt.show()

## Histogramm of pareto front

# Create a 1x3 matrix of subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Plot posterior distributions for each parameter in the matrix of subplots
for i, parameter in enumerate(pareto_points.columns[3:6]):  # Exclude the first two columns
    sns.kdeplot(pareto_points[parameter], shade=True, ax=axs[i])
    axs[i].set_xlabel(None)
    axs[i].set_ylabel('Density')
    axs[i].set_title(f'{parameter}', fontweight='bold', fontsize=14)
    # Add vertical lines for mean, mean ± standard deviation
    mean_val = pareto_points[parameter].mean()
    std_val = pareto_points[parameter].std()
    axs[i].axvline(mean_val, color='red', linestyle='--', label='Mean')
    axs[i].axvline(mean_val - std_val, color='blue', linestyle='--', label='Mean - SD')
    axs[i].axvline(mean_val + std_val, color='blue', linestyle='--', label='Mean + SD')
plt.tight_layout()
plt.show()

## Parameter bounds from pareto front

selected_columns = pareto_points.iloc[:, 3:6]

# Calculating lower and upper bounds for each parameter
bounds_dict = {}
for column in selected_columns.columns:
    mean = selected_columns[column].mean()
    std = selected_columns[column].std()
    bounds_dict[column + '_lo'] = mean - std
    bounds_dict[column + '_up'] = mean + std

print(bounds_dict)
## Final parameter set LHS:

parameters = {
    # Fix:
    "K0": 0.055,
    "LP": 0.7,
    "MAXBAS": 3.0,
    "RFS": 0.15,
    'SFCF': 1,
    'CET': 0,
    # Step 1:
    'PCORR': 0.58,
    "lr_temp": -0.006,
    "lr_prec": 0.0015,
    "TT_diff": 1.33,
    "CFMAX_rel": 1.7,
    # Step 2:
    'BETA': 1.0142949,
    'TT_snow': -1.4522806,
    'CFMAX_ice': 4.9159093
}

# ['PCORR', 'SFCF', 'CET', 'K0', 'LP', 'MAXBAS', 'RFS', 'lr_temp', 'lr_prec', 'TT_diff', 'CFMAX_rel', 'BETA', 'TT_snow', 'CFMAX_ice']