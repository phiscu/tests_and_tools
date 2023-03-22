import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import salem
import pyet
from pathlib import Path; home = str(Path.home())
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/tests_and_tools/Preprocessing')

aws_lat = 42.191433; aws_lon = 78.200253
# start_date = '2007-01-01'; end_date = '2014-12-31'

## Files

# Load datasets
var_names = ["LWd", "P", "Pres", "RelHum", "SpecHum", "SWd", "Temp", "Tmax", "Tmin", "Wind"]
for v in var_names:
    vars()[v] = salem.open_xr_dataset(home + '/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/mswx/'
                                    + v + '_MSWX_daily_kyzylsuu_19792022.nc')
# Static
static_era_path = home + '/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/static/ERA5_land_Z_geopotential_MSWX-grid.nc'
static_era = salem.open_xr_dataset(static_era_path)
static_era = Temp.salem.transform(static_era)
elev = static_era.z/9.80665
lat = Temp.lat * np.pi/180

# Load catchment outline
# catchment = gpd.read_file(home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/shapefile_hydro_kyzylsuu.shp")

## Unit conversions:
SWd = SWd * 0.0864  # from W/m² to MJ/(m²*d): 1 W = 86400 J/d = 0.0864 MJ/d
Pres = Pres / 1000  # from Pa to kPa

## Calculate PEV:

pev_fao56 = pyet.pm_fao56(tmean=Temp.air_temperature,
                          wind=Wind.wind_speed,
                          rs=SWd.downward_shortwave_radiation,
                          tmax=Tmax.air_temperature,
                          tmin=Tmin.air_temperature,
                          rh=RelHum.relative_humidity,
                          pressure=Pres.surface_pressure,
                          elevation=elev,
                          lat=lat)
## Plot:
pev_fao56.mean(dim='time').plot()
plt.show()

## Write file

pev_fao56.to_netcdf('/home/phillip/Seafile/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/mswx/'
                    + 'PEV_fao56_MSWX_daily_kyzylsuu_1980-2022.nc')
