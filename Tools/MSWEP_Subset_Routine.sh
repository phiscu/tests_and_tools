#!/bin/bash
# longitude_range=75.80,76.40
# latitude_range=41.00,41.40
longitude_range=78.00,78.40
latitude_range=42.00,42.40

input_folder=/media/phillip/KALI_LIVE/GloH2O/MSWEP #/home/phillip/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWEP/raw		
dataset_name=MSWEP_daily_past
period=2020
underscore=_

output_folder=/home/phillip/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWEP
site_name=kyzylsuu

mkdir $output_folder/lonlatbox
cd $input_folder
for file in *.nc ; do
    output_file=$output_folder/lonlatbox/$site_name_$file
    echo $file
    echo $output_file
    #module load cdo
    command="cdo sellonlatbox,$longitude_range,$latitude_range $file $output_file"
    echo $command
    $command
done

input_folder=$output_folder/lonlatbox
cd $input_folder
output_timemerge=$output_folder/$dataset_name$underscore$site_name$underscore$period.nc
pwd
ls -l
echo $output_timemerge
#module load cdo
cdo -b F64 mergetime "*.nc" $output_timemerge
#rm -r $output_folder/lonlatbox

cdo -b F64 mergetime "*.nc" MSWEP_daily_past_kyzylsuu_1979-2020.nc
