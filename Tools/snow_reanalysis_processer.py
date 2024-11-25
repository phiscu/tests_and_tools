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
tif_dir = "/home/phillip/Seafile/CLIMWATER/YSS/2024/Pskem_data/swe"

swe_df = swe_means(tif_dir, end_year=2016)
swe_df.to_csv("/home/phillip/Seafile/CLIMWATER/YSS/2024/Pskem_data/swe/kyzylsuu_swe.csv")


## Scaling factor
# Glacier mask coarser than outline shapes
mask99 = geotiff2xr('/home/phillip/Seafile/CLIMWATER/YSS/2024/Pskem_data/swe/HMA_SR_D_v01_WY1999_MASK_pskem.tif')
valid_pixels = mask99.where(mask99 != -999)
total_valid_pixels = valid_pixels.count()
swe_pixels = (valid_pixels == 0).sum()
swe_frac = swe_pixels / total_valid_pixels
# --> 82.8% vs. 89.2% in MATILDA
# --> 93.4% vs. 96.8% MATILDA (Pskem)

swe_area_sim = 0.968        # from catchment settings
swe_area_obs = swe_frac.values
sf = swe_area_obs / swe_area_sim
print('SWE scaling factor: ' + str(round(sf, 3)))


##
mask_tif = select_tif(tif_dir, str(2015), "MASK")
swe_tif = select_tif(tif_dir, str(2015), "SWE")

mask = geotiff2xr(mask_tif[0])
swe = geotiff2xr(swe_tif[0])

masked_swe = swe.where(mask == 0)


## Plot SWE
plt.figure(figsize=(12, 6))
plt.plot(swe_df, color='b', linestyle='-')
plt.title('Mean Catchment SWE (Pskem)')
plt.xlabel('Year')
plt.ylabel('Mean SWE')
plt.grid(True)
average_swe = swe_df['SWE_Mean'].mean()
plt.axhline(y=average_swe, color='r', linestyle='--', label='Average SWE')
plt.legend()
plt.show()


## Plot masked SWE
selection = masked_swe.sel(Non_seasonal_snow=1, day=100)
da_cleaned = mask.where(mask != -999, np.nan)
da_cleaned = da_cleaned.sel(Non_seasonal_snow=1)

# Plot the data
fig, ax = plt.subplots(figsize=(5, 4), dpi=60)
da_cleaned.plot.imshow(ax=ax, cmap="Spectral_r", add_colorbar=False)
im = selection.plot.imshow(ax=ax, cmap="viridis", add_colorbar=True, vmin=0, vmax=3.3)
plt.title("Masked Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


## Create gif

import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import shutil

# Assuming masked_swe and mask are already defined
da_cleaned = mask.where(mask != -999, np.nan)
da_cleaned = da_cleaned.sel(Non_seasonal_snow=1)

# Specify the target directory
target_dir = '/home/phillip/Seafile/CLIMWATER/YSS/2024/Slides/figs'
frames_dir = os.path.join(target_dir, 'frames')

# Create directories to store the frames and GIF
os.makedirs(frames_dir, exist_ok=True)

# Generate the plots and save them as frames
for day in range(1, 367):
    selection = masked_swe.sel(Non_seasonal_snow=1, day=day)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=60)
    da_cleaned.plot.imshow(ax=ax, cmap="Spectral_r", add_colorbar=False)
    im = selection.plot.imshow(ax=ax, cmap="viridis", add_colorbar=True, vmin=0, vmax=3.3)
    plt.title(f"SWE in Pskem Catchment (HY 2015) - Day {day}")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Save the frame
    frame_filename = os.path.join(frames_dir, f'frame_{day:03d}.png')
    plt.savefig(frame_filename)
    plt.close(fig)

# Create the GIF
gif_path = os.path.join(target_dir, 'masked_swe_animation.gif')
with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
    for day in range(1, 367):
        frame_filename = os.path.join(frames_dir, f'frame_{day:03d}.png')
        image = imageio.imread(frame_filename)
        writer.append_data(image)

# Cleanup the frames directory
shutil.rmtree(frames_dir)

print(f"GIF saved to {gif_path}")
