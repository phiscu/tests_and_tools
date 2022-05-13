from pathlib import Path
import sys
import socket
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import salem
import plotly.io
import pylab as pl
import scipy.optimize as sp
sys.path.append('/home/phillip/Seafile/Ana-Lena_Phillip/data/tests_and_tools/Preprocessing/')
from Downscaling.ERA5_downscaling.fundamental_physical_constants import g, M, R
from Preprocessing_functions import hour_rounder
from sklearn.linear_model import LinearRegression


home = str(Path.home())
working_directory = home + '/Seafile/EBA-CA/Tianshan_data/HOBO_water/Batysh_Sook/'

time_start = '2019-09-05 05:53:00'
time_end = '2021-08-20 18:00:00' # approx. time of removal
# Logger was found removed at '2021-08-20 12:19:00'. Re-installed at 12:30:00 for dilution gauging.

alt_hobo = 3874 # ALOS DEM; Martina DSM: 3836,0942 ; GPS: 3919,937744 m.a.s.l.
lat_hobo = 41.799247
lon_hobo = 77.749729
alt_aws_far = 3561 # ALOS DEM; GPS: 3552
lapseT = -0.006


## HOBO-Logger:
hobo = pd.read_csv(working_directory + 'Batysh_Sook_2021_clean.csv', parse_dates=['time_utc'])  # , index_col='time_utc'
hobo.time_utc = [hour_rounder(i) for i in hobo.time_utc]    # Round measurement times to full hours
hobo.set_index("time_utc", inplace=True)
hobo.press = hobo.press / 100       # from Pa to hPa

# Major disruption in April 2020 - best estimate:
start = '2019-10-01'
end = '2020-04-16'
span = slice(start, end)

# Exclude outlier period:
start_out = '2020-04-16'
end_out = '2020-05-31'
outl = slice(start_out, end_out)


## ERA5
era5_land_static_file = home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/global/ERA5_land_Z_geopotential.nc'
era5_land_file = working_directory + 'Batysh_SookERA5L_2018_2020.nc'
target_altitude = alt_hobo
margin = 0.2
lon_ll = lon_hobo - margin; lat_ll = lat_hobo - margin
lon_ur = lon_hobo + margin; lat_ur = lat_hobo + margin
era5_land_static = salem.open_xr_dataset(era5_land_static_file)
era5_land_static = era5_land_static.salem.subset(corners=((lon_ll, lat_ll), (lon_ur, lat_ur)), crs=salem.wgs84)
era5_land = salem.open_xr_dataset(era5_land_file)
era5_land = era5_land.sel(time=slice(time_start, time_end))
altitude_differences_gp = np.abs(era5_land_static.z/g - target_altitude)
# Select gridpoint with most similar altitude to HOBO location
latitude = float(era5_land_static.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lat)
longitude = float(era5_land_static.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lon)

era5_land = era5_land.sel(latitude=latitude, longitude=longitude, method='nearest')
era5_land_static = era5_land_static.sel(lat=latitude, lon=longitude, method='nearest')
height_diff = target_altitude - era5_land_static.z.values/g; print("Height difference to target_altitude: ", height_diff)
press_era_hobo = ((era5_land['sp'].values)/100) * (1-abs(lapseT)*height_diff/era5_land['t2m'].values) ** \
               ((g*M)/(R*abs(lapseT)))
temp_era_hobo = (era5_land['t2m'].values) + height_diff * lapseT


era_hobo = era5_land.to_dataframe().filter(['t2m', 'tp', 'sp'])
era_hobo.sp = ((era5_land['sp'].values)/100) * (1-abs(lapseT)*height_diff/era5_land['t2m'].values) ** \
               ((g*M)/(R*abs(lapseT)))
era_hobo.t2m = (era5_land['t2m'].values) + height_diff * lapseT

## Comparison with ERA5:

t = slice(time_start, '2020-12-31')

data = pd.DataFrame({'water_press': hobo['press'][t], 'air_press': era_hobo['sp'][t]})

data.plot(); plt.show()

hobo_cut = hobo['press'][span] - era_hobo['sp'][span]



## AWS:
aws_t2m = pd.read_csv(working_directory + 't2m_taragai_2014-2021.csv', parse_dates=['time'], index_col='time')
aws_t2m = aws_t2m.resample('H').mean() + 273.15
aws_tp = pd.read_csv(working_directory + 'tp_taragai_2014-2021.csv', parse_dates=['time'], index_col='time')
aws_tp = aws_tp.resample('H').sum()
aws_sp = pd.read_csv(working_directory + 'sp_taragai_2014-2021.csv', parse_dates=['time'], index_col='time')
aws_sp = aws_sp.resample('H').mean()

aws = pd.concat([aws_t2m, aws_tp, aws_sp], axis = 1)

height_diff = alt_hobo - alt_aws_far

aws['press_hobo'] = (aws.sp) * (1-abs(lapseT)*height_diff/aws.t2m) ** ((g*M)/(R*abs(lapseT)))
t = slice(time_start, '2020-12-31')

data = pd.DataFrame({'water_press': hobo['press'][t], 'air_press_aws': aws['sp'][t], 'air_press_aws_scaled': aws['press_hobo'][t]})
data[outl] = np.NaN
data.plot(); plt.show()



## Calculate stage from water pressure

def h(p):  # Water column in meter from hydrostatic pressure in Pa!
    return p / (9.81 * 1000)  # acceleration of gravity: 9.81 m/s², density of water 1000 kg/m³

data['water_press_diff'] = data.water_press - data.air_press_aws_scaled

data['stage'] = h(aws.water_press_diff * 100)

plt.plot(data.stage); plt.show()

##  Dilution gauging:

gaug = pd.read_csv(working_directory + 'dilution_gauging_batysh_sook_2019.csv', parse_dates=['time_utc'])
gaug.time_utc = [hour_rounder(i) for i in gaug.time_utc]    # Round measurement times to full hours
gaug.set_index("time_utc", inplace=True)
gaug.drop('time', axis=1)

## Predict runoff with stage:

hobo_ro = pd.merge(gaug, data.stage, how='left', left_index=True, right_index=True)

predictor = 'stage'
X = hobo_ro.loc[:, predictor].values.reshape(-1, 1)
Y = hobo_ro.loc[:, 'runoff'].values.reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('Water column above sensor [m]')
plt.ylabel('Run-off [m³/s')
plt.show()

print('------ Linear Regression -----')
print('Function: y = %.3f * x + %.3f' % (linear_regressor.coef_[0], linear_regressor.intercept_))
print("R² Score: {:.2f}".format(linear_regressor.score(X, Y)))

## Calculate runoff from stage:

runoff = pd.DataFrame({'runoff': linear_regressor.coef_[0] * data.stage + linear_regressor.intercept_})
# runoff[runoff.index.month.isin([10, 11, 12, 1, 2, 3, 4])] = 0
runoff[runoff.runoff < 0] = 0
runoff.describe()

plt.plot(runoff_D_sum)
plt.ylabel('Run-off [m³/d]')
plt.show()

runoff_D_mean = runoff.resample('D').mean()
runoff_D_sum = (runoff * 3600).resample('D').sum()
runoff_D_mean.to_csv(working_directory + "runoff_sook_from_hobo_daily_means.csv")
runoff_D_sum.to_csv(working_directory + "runoff_sook_from_hobo_daily_sum.csv")
runoff.to_csv(working_directory + "runoff_sook_from_hobo.csv")

