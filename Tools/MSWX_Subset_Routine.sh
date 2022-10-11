#!/bin/bash
# longitude_range=75.80,76.40
# latitude_range=41.00,41.40
longitude_range=78.00,78.40
latitude_range=42.00,42.40

input_folder=/media/phillip/KALI_LIVE/GloH2O/MSWX/past/Temp #/home/phillip/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWEP/raw		
dataset_name=MSWX_Temp_daily_past
start_y=1979
end_y=2022
underscore=_

output_folder=/home/phillip/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWX
site_name=kyzylsuu

## Geo-Box

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

## Time-merge:

#input_folder=$output_folder/lonlatbox
#cd $input_folder
#output_timemerge=$output_folder/$dataset_name$underscore$site_name$underscore$period.nc
#pwd
#ls -l
#echo $output_timemerge
#module load cdo
#cdo -b F64 mergetime "*.nc" $output_timemerge			# Often too many files
#rm -r $output_folder/lonlatbox


## Alternative:
input_folder=$output_folder/lonlatbox
cd $input_folder
output_timemerge=$output_folder/$dataset_name$underscore$site_name
pwd
ls -l
echo $output_timemerge
for ((year=start_y; year<=end_y; year++)); do
	cdo -b F64 mergetime "${year}*" $output_timemerge$underscore$year.nc
done
cd $output_folder
cdo -b F64 mergetime "$output_timemerge*" $output_timemerge$underscore$start_y$underscore$end_y.nc
for ((year=start_y; year<=end_y; year++)); do
	rm $output_timemerge$underscore$year.nc
done

