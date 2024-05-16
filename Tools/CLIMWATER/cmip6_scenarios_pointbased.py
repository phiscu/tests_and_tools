# -*- coding: UTF-8 -*-

import os
import time
import warnings
import sys
from shapely.errors import ShapelyDeprecationWarning
wd = os.getcwd()
sys.path.append(wd + '/tests_and_tools/Tools/CLIMWATER')
import matilda_functions as mf
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")

## Settings
input_dir = '/home/phillip/Seafile/CLIMWATER/Data/Hydrometeorology/Meteo'
output_dir = '/home/phillip/Seafile/CLIMWATER/Data/Hydrometeorology/CMIP6/'
buffer_radius = 2000
show = True
sd_factor = 2
processes = 30
download = True
load_backup = False

start_region_index = 0   # 0 if you want to start with the very first station
start_station_index = 0  # 0 if you want to start with the very first station

## Preprocessing
print('Initiating preprocessing routine')
preprocessor = mf.StationPreprocessor(input_dir=input_dir, output_dir=output_dir, buffer_radius=buffer_radius,
                                      show=show, sd_factor=sd_factor)
preprocessor.full_preprocessing()
print('Set up preprocessing routine for target directory!')

## Main loop

start_total_time = time.time()
total = mf.count_dataframes(preprocessor.region_data)

for region_index, region in enumerate(preprocessor.region_data.keys()):
    # If start_region_index is set and we haven't reached it yet, continue to the next region
    if start_region_index is not None and region_index < start_region_index:
        continue

    for station_index, (station, data) in enumerate(preprocessor.region_data[region].items()):
        # If start_station_index is set and we haven't reached it yet, continue to the next station
        if start_station_index is not None and station_index < start_station_index:
            continue

        start_process_time = time.time()
        if region_index == 0:
            print(f'\n** Starting processing of Station {station_index+1} of {total}: "{station}"\n')
        else:
            print(f'\n** Starting processing of Station {station_index+6} of {total}: "{station}"\n')

        try:
            instance = mf.ClimateScenarios(output=f'{preprocessor.output_dir}{region}/{station}/',
                                           region_data=preprocessor.region_data, station=station, download=download,
                                           load_backup=load_backup, show=show, buffer_file=preprocessor.gis_file,
                                           processes=processes)
            instance.complete_workflow()
        except Exception as e:
            print(f"Error occurred for station {station}: {e}")
            continue  # Skip to the next station

        end_process_time = time.time()
        process_elapsed_time = end_process_time - start_process_time
        print(f'Processing time for station {station}: {round(process_elapsed_time/60, 2)} minutes')

    # Update the start_station_index after processing all stations in the current region
    start_station_index = None

    # If start_region_index is not set, update it to proceed to the next region
    if start_region_index is None:
        start_region_index = region_index

# Reset the start_region_index after completing the loop
start_region_index = None

end_total_time = time.time()
total_elapsed_time = end_total_time - start_total_time
print('**************************************\n** COMPLETED **')
print(f'Total processing time: {round(total_elapsed_time/60, 2)} minutes')

