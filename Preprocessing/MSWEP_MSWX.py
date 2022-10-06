## USE conda-base environment!!
import sys
import os
import xarray as xr
import pandas as pd
from shapely.geometry import Point, mapping
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import rasterio
import salem
import xagg         # If dependency issues occur: difference to non-weighted catchment average is little
# import cartopy.crs as ccrs
from pathlib import Path; home = str(Path.home())
from matilda.core import matilda_simulation
# from bias_correction import BiasCorrection
#
aws_lat = 42.191433; aws_lon = 78.200253
# start_date = '2007-01-01'; end_date = '2014-12-31'

## Files
# Load dataset
mswep_ds = salem.open_xr_dataset(home + "/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWEP/MSWEP_daily_past_kyzylsuu_1979-2020.nc")
# Load catchment outline
catchment = gpd.read_file(home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/shapefile_hydro_kyzylsuu.shp")

## Catchment wide weighted average
# Clip to overlapping grid cells (for plotting)
clip = mswep_ds.salem.roi(shape=catchment, all_touched=True)
clip = mswep_ds.salem.subset(shape=catchment)

# Calculate overlaps:
weightmap = xagg.pixel_overlaps(clip, catchment)
# Aggregate
aggregated = xagg.aggregate(clip, weightmap)
# Produce dataframe
mswep_df = aggregated.to_dataframe()
mswep_df = mswep_df.reset_index(level='poly_idx', drop=True).drop('LABEL', axis=1)

## Compare to non-weighted average

mswep_df['non_weighted'] = clip.precipitation.mean(dim=['lat', 'lon'])
mswep_df.precipitation.plot()
# mswep_df.resample('Y').sum().plot()
# plt.legend()
plt.title('Annual precipitation averaged over lonlatbox [78.05,78.35,42.05,42.35]')
plt.show()

## Remarks:

# MSWEP has a large bias in the years 2019 and 2020 with annual values 3 to 6 times higher than all years prior.

## Plot

fig = plt.figure(figsize=(16,12), dpi=300)
ax = fig.add_subplot(111)
catchment.plot(ax=ax, zorder=3)
clip.precipitation.mean(dim='time').plot(ax=ax, zorder=-1)
plt.text(aws_lon, aws_lat, 'AWS')
plt.scatter(aws_lon, aws_lat, color='r')
plt.show()




# ## Compare past and nrt
# catchment = gpd.read_file(home+ "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/shapefile_hydro_kyzylsuu.shp")
# past = salem.open_xr_dataset(home + "/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWEP/MSWEP_daily_past_kyzylsuu_2020.nc")
# nrt = salem.open_xr_dataset(home + "/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWEP/MSWEP_daily_nrt_kyzylsuu_2020.nc")
# nrt_df = nrt.mean(dim=['lat','lon']).to_dataframe().filter(['precipitation'])
# past_df = past.mean(dim=['lat','lon']).to_dataframe().filter(['precipitation'])
# df = pd.merge(past_df, nrt_df, how='outer', on='time')
# df = df.rename(columns={'precipitation_x':'past', 'precipitation_y':'nrt'})
#
# df.plot()
# plt.legend()
# plt.show()
# df.sum()