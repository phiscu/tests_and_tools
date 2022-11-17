#!/bin/bash

input_folder=/data/projects/ebaca/Tienshan_data/GloH2O/MSWX/past #/home/phillip/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWX/past/
dataset_name=MSWX_daily_past
start_y=1979
end_y=2022
underscore=_
site_name=kyzylsuu

field=SWd

# Annual merges:

# "LWd" "P" "Pres" "RelHum" "SpecHum" "SWd" "Temp" "Tmax" "Tmin" "Wind"

# for field in "LWd" "P" "Pres" "RelHum" "SpecHum" "SWd" "Temp" "Tmax" "Tmin" "Wind" ; do

echo $field
daily_folder=$input_folder/Daily/$field
cd $daily_folder
output_timemerge_folder=$input_folder/Annually/$field
mkdir $output_timemerge_folder
output_timemerge=$output_timemerge_folder/$field$underscore$dataset_name$underscore$site_name
pwd
#ls -l
echo $output_timemerge
for ((year=start_y; year<=end_y; year++)); do
	cdo -b F64 mergetime "${year}*" $output_timemerge$underscore$year.nc
done
#	cd $output_timemerge_folder
#	cdo -b F64 mergetime "$output_timemerge*" $output_timemerge$underscore$start_y$underscore$end_y.nc
#done

