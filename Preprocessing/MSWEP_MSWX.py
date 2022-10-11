## USE conda-base environment!!
import sys
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import salem
import xagg         # If dependency issues occur: difference to non-weighted catchment average is little
from pathlib import Path; home = str(Path.home())

aws_lat = 42.191433; aws_lon = 78.200253
# start_date = '2007-01-01'; end_date = '2014-12-31'

## Files
# Load datasets
mswep_p = salem.open_xr_dataset(home + "/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWEP/MSWEP_daily_past_kyzylsuu_1979-2020.nc")
mswx_p = salem.open_xr_dataset(home + '/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWX/MSWX_P_daily_past_kyzylsuu_1979_2022.nc')
mswx_t = salem.open_xr_dataset(home + '/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWX/MSWX_Temp_daily_past_kyzylsuu_1979_2022.nc')

# Load catchment outline
catchment = gpd.read_file(home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/shapefile_hydro_kyzylsuu.shp")

## Area-weighted catcment-wide mean:
def weighted_avg(xarray, shape, return_clip=False, plot=False):
    """Area-weighted average of xarray cells overlapping a polygon."""

    # Clip to overlapping grid cells (for plotting):
    clip = xarray.salem.roi(shape=shape, all_touched=True)
    # Calculate overlaps:
    weightmap = xagg.pixel_overlaps(clip, shape)
    # Aggregate
    aggregated = xagg.aggregate(clip, weightmap)
    # Produce dataframe
    df = aggregated.to_dataframe()
    df = df.reset_index(level='poly_idx', drop=True).drop('LABEL', axis=1)
    if return_clip:
        if plot:
            fig = plt.figure(figsize=(16, 12), dpi=300)
            ax = fig.add_subplot(111)
            shape.plot(ax=ax, zorder=3)
            xarray[list(clip.keys())[0]].mean(dim='time').plot(ax=ax, zorder=-1)
            # plt.text(aws_lon, aws_lat, 'AWS')
            # plt.scatter(aws_lon, aws_lat, color='r')
            return df, clip, fig
        else:
            return df, clip
    else:
        if plot:
            fig = plt.figure(figsize=(16, 12), dpi=300)
            ax = fig.add_subplot(111)
            shape.plot(ax=ax, zorder=3)
            xarray[list(clip.keys())[0]].mean(dim='time').plot(ax=ax, zorder=-1)
            # plt.text(aws_lon, aws_lat, 'AWS')
            # plt.scatter(aws_lon, aws_lat, color='r')
            return df, fig
        else:
            return df

## MSWEP:
mswep_p, mswep_p_clip = weighted_avg(mswep_p, catchment, return_clip=True)

## MSWX:
mswx_p, mswx_p_clip = weighted_avg(mswx_p, catchment, return_clip=True)
mswx_t, mswx_t_clip = weighted_avg(mswx_t, catchment, return_clip=True)


## Remarks:

# MSWEP has a large bias in the years 2019 (starting August) and 2020 with annual values 3 to 6 times higher than all years prior.
# MSWX and MSWEP have only minor differences but MSWX is ok for 2019 and 2020.
# MSWX past is close (few days) to real-time.

## Compare MSWX, MSWEP past, ERA5L, and HAR:
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/tests_and_tools/Preprocessing')
from HAR_data import har_df, tp

mswep_p = mswep_p[slice('2017-01-01', '2019-07-30')]
mswx_p = mswx_p[slice('2017-01-01', '2019-07-30')]
har = har_df['prcp'][slice('2017-01-01', '2019-07-30')]
era = tp[slice('2017-01-01', '2019-07-30')]

df = pd.concat([mswep_p,mswx_p,har,era], axis=1)
df.columns = ['mswep','mswx','har','era']
df = df.resample('M').sum()

df.plot()
plt.legend()
plt.show()


# ERA5 ist noch nicht übers catchment gemittelt!
