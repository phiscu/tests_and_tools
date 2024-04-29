import pandas as pd
import rasterio
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

## Definitions


def geotiff2xr(file_path):
    with rasterio.open(file_path) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width
        number_of_days = data.shape[0]
        x_coords = np.linspace(transform.c, transform.c + (width - 1) * transform.a, width)
        y_coords = np.linspace(transform.f, transform.f + (height - 1) * transform.e, height)

        if "SWE" in file_path:
            da = xr.DataArray(data, dims=("day", "y", "x"),
                              coords={"day": range(1, number_of_days + 1), "y": y_coords, "x": x_coords}, name="SWE")
            da.attrs["crs"] = crs
            da.attrs["transform"] = transform
            return da
        elif "MASK" in file_path:
            ma = xr.DataArray(data, dims=("Non_seasonal_snow", "y", "x"),
                              coords={"Non_seasonal_snow": range(1, number_of_days + 1), "y": y_coords, "x": x_coords},
                              name="Non_seasonal_snow")
            ma.attrs["crs"] = crs
            ma.attrs["transform"] = transform
            return ma
        else:
            return None


def select_tif(directory, keyword1, keyword2):
    specific_tif_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.tif') and keyword1 in file and keyword2 in file]
    return specific_tif_files


def swe_means(tif_dir, start_year=1999, end_year=2016):
    swe_list = []
    years = range(start_year, end_year + 1)

    for year in years:
        mask_tif = select_tif(tif_dir, str(year), "MASK")
        swe_tif = select_tif(tif_dir, str(year), "SWE")

        mask = geotiff2xr(mask_tif[0])
        swe = geotiff2xr(swe_tif[0])

        masked_swe = swe.where(mask == 0)
        mean_swe = masked_swe.mean(dim=['x', 'y'])
        swe_list.append(mean_swe.values.tolist())

    time_series_data = []

    for year_data in swe_list:
        for day_value in year_data:
            time_series_data.append(round(day_value[0], 4))

    date_range = pd.date_range(start=str(start_year) + '-10-01', end=str(end_year+1) + '-09-30', freq="D")
    swe_df = pd.DataFrame({"Date": date_range, "SWE_Mean": time_series_data})
    swe_df.set_index("Date", inplace=True)

    return swe_df


## Get SWE means from dir
tif_dir = "/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/hmadsr/processed_final"

swe_df = swe_means(tif_dir, end_year=2016)
swe_df.to_csv("/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/hmadsr/kyzylsuu_swe.csv")


## Plot SWE
plt.figure(figsize=(12, 6))
plt.plot(swe_df, color='b', linestyle='-')
plt.title('Mean Catchment SWE (Kyzylsuu)')
plt.xlabel('Year')
plt.ylabel('Mean SWE')
plt.grid(True)
average_swe = swe_df['SWE_Mean'].mean()
plt.axhline(y=average_swe, color='r', linestyle='--', label='Average SWE')
plt.legend()
plt.show()


# ## Plot SWE
#
# da_cleaned = da.where(da != -999, np.nan)
# day_to_plot = 200
# day_data_cleaned = da_cleaned.sel(day=day_to_plot)
# day_data_cleaned.plot.imshow(cmap="viridis", add_colorbar=True)
# plt.title(f"Plot for Day {day_to_plot}")
# plt.show()
#
# ## Plot mask
#
# da_cleaned = ma.where(ma != -999, np.nan)
# da_cleaned = da_cleaned.sel(Non_seasonal_snow=1)
# da_cleaned.plot.imshow(cmap="viridis", add_colorbar=True)
# plt.show()
# ## Plot masked SWE
# selection = masked_data.sel(Non_seasonal_snow=1, day=1)
# selection.plot.imshow(cmap="viridis", add_colorbar=True)
# plt.title("Masked Data")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

##
# Glacier mask ist sehr grob durch 500m pixel --> anzahl der pixel zählen grob mit glacier area vergleichen --> discussion

