#!/home/phillip/miniconda3/bin/python

## Run in conda-base env!
import socket
from pathlib import Path
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
import matplotlib.pyplot as plt
import xagg

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
