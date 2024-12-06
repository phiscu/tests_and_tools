import matplotlib.pyplot as plt
import pandas as pd
import pickle


def get_matilda_result_all_models(scenario, result_name, dict="model_output"):
    df = pd.DataFrame()

    for key, value in matilda_scenarios[scenario].items():
        s = value[dict][result_name]
        s.name = key
        df = pd.concat([df, s], axis=1)

    df.index = pd.to_datetime(df.index)
    df.index.name = "TIMESTAMP"
    print(f"{result_name} extracted for {scenario}")

    return df


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
    with open(file_path, "rb") as f:
        dic = pickle.load(f)
    return dic


def ensemble_mean(matilda_scenarios, result_name, dict_name="model_output"):
    """
    Compute the ensemble mean for a specific variable across all models in the scenarios.

    Parameters:
        matilda_scenarios (dict): Original scenario dictionary with nested model outputs.
        result_name (str): Name of the target variable to extract (e.g., 'total_runoff').
        dict_name (str): Name of the dictionary in the model output to look for results (default: 'model_output').

    Returns:
        dict: A dictionary with emission scenarios as keys and the ensemble mean time series (as pandas Series) as values.
    """

    def get_matilda_result_all_models(scenario, result_name, dict_name="model_output"):
        df = pd.DataFrame()

        for key, value in matilda_scenarios[scenario].items():
            s = value[dict_name][result_name]
            s.name = key
            df = pd.concat([df, s], axis=1)

        df.index = pd.to_datetime(df.index)
        df.index.name = "TIMESTAMP"
        print(f"{result_name} extracted for {scenario}")

        return df

    # Extract the result data for each scenario and compute the ensemble mean
    mean_results = {}
    for scenario in matilda_scenarios.keys():
        # Get the DataFrame of all models for the given scenario and variable
        df = get_matilda_result_all_models(scenario, result_name, dict_name)
        # Compute the ensemble mean across models (columns)
        mean_results[scenario] = df.mean(axis=1)

    return mean_results


dir_output = "/home/phillip/Seafile/EBA-CA/Repositories/matilda_edu/output/"
matilda_scenarios = pickle_to_dict(
    f"{dir_output}cmip6/adjusted/matilda_scenarios.pickle"
)

runoff = ensemble_mean(matilda_scenarios, "total_runoff")
snow_melt = ensemble_mean(matilda_scenarios, "snow_melt_on_glaciers")
ice_melt = ensemble_mean(matilda_scenarios, "ice_melt_on_glaciers")
off_melt = ensemble_mean(matilda_scenarios, "melt_off_glaciers")
aet = ensemble_mean(matilda_scenarios, "actual_evaporation")

# Melt off glacier is always snow melt:
snow_melt["SSP2"] = snow_melt["SSP2"] + off_melt["SSP2"]
snow_melt["SSP5"] = snow_melt["SSP5"] + off_melt["SSP5"]


## One plot


def plot_annual_cycle(
    ensemble_mean_scenario, variable_name, title="Annual Cycle of Monthly Runoff"
):
    """
    Plots the annual cycle of monthly values for a given scenario's ensemble mean.

    Parameters:
        ensemble_mean_scenario (pd.Series): Time series of ensemble mean values (TIMESTAMP as index).
        variable_name (str): Name of the variable to display on the colorbar (e.g., "Total Runoff (mm)").
        title (str): Title of the plot (default: "Annual Cycle of Monthly Runoff").

    Returns:
        None: Displays the plot.
    """
    import matplotlib.pyplot as plt

    # Prepare the data
    data = ensemble_mean_scenario.reset_index()
    data.columns = ["date", variable_name]

    # Extracting year and month from the date
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    # Calculating monthly sums
    monthly_values = data.groupby(["year", "month"])[variable_name].sum().reset_index()

    # Pivoting the data to create the grid structure
    grid = monthly_values.pivot(index="month", columns="year", values=variable_name)

    # Plotting
    short_month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    plt.figure(figsize=(4, 4))
    c = plt.pcolormesh(
        grid.index, grid.columns, grid.T, shading="nearest", cmap="viridis"
    )
    plt.colorbar(c, label=variable_name)

    # Keep labels but remove ticks
    plt.gca().tick_params(axis="x", which="both", bottom=False, top=False)
    plt.gca().tick_params(axis="y", which="both", left=False, right=False)

    # Add labels
    plt.xticks(range(1, 13), short_month_names, rotation=45)
    plt.ylabel("Year")

    plt.title(title, pad=15)
    plt.tight_layout()
    plt.show()


# plot_annual_cycle(runoff['SSP5'], variable_name="mm", title="Annual Cycle of Monthly Runoff (SSP5)")

## Multiple plots


def plot_annual_cycles_2x2(runoff, snow_melt, ice_melt, aet, scenario="SSP5"):
    """
    Creates a 2x2 plot of annual cycles for the given variables in a specified scenario.

    Parameters:
        runoff (dict): Ensemble mean dictionary for runoff.
        snow_melt (dict): Ensemble mean dictionary for snowmelt.
        ice_melt (dict): Ensemble mean dictionary for ice melt.
        aet (dict): Ensemble mean dictionary for actual evaporation.
        scenario (str): The scenario to plot (default: "SSP5").

    Returns:
        None: Displays the 2x2 subplot figure.
    """
    variables = {
        "Total Runoff": (runoff[scenario], "YlGnBu"),
        "Snowmelt": (snow_melt[scenario], "Blues"),
        "Ice Melt": (ice_melt[scenario], "Blues"),
        "Actual Evapotranspiration": (aet[scenario], "Oranges"),
    }

    short_month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.suptitle(f"Simulated Annual Cycles for {scenario}", fontsize=24, y=0.98)

    for ax, (title, (data, cmap)) in zip(axes.flatten(), variables.items()):
        # Prepare the data
        df = data.reset_index()
        df.columns = ["date", title]
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        # Group and pivot
        monthly_values = df.groupby(["year", "month"])[title].sum().reset_index()
        grid = monthly_values.pivot(index="month", columns="year", values=title)

        # Plot on the subplot
        c = ax.pcolormesh(
            grid.index, grid.columns, grid.T, shading="nearest", cmap=cmap
        )
        fig.colorbar(c, ax=ax, label="mm")

        # Subplot customization
        ax.set_title(title, pad=10)

        # Customizing ticks
        ax.set_xticks(range(1, 13, 2))  # Every second month
        ax.set_xticklabels(
            short_month_names[::2], rotation=0
        )  # Labels for every second month
        ax.set_yticks(grid.columns[::20])  # Every 20th year
        ax.set_yticklabels(grid.columns[::20], rotation=0)  # Keep y-labels horizontal
        ax.tick_params(
            axis="both", which="both", length=0
        )  # Remove all ticks but keep labels

        ax.set_ylabel("Year")

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust for the title
    plt.show()


plot_annual_cycles_2x2(runoff, snow_melt, ice_melt, aet, scenario="SSP5")
