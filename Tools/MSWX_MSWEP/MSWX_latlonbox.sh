#!/bin/bash

longitude_range=LON
latitude_range=LAT
input_folder=INPUT_DIR
dataset_name=DATASET
start_y=START
end_y=END
site_name=SITE

field=FIELD

## Geo-Box

output_folder=$input_folder/$field/latlonbox
mkdir $output_folder
cd $input_folder/$field
for file in *.nc ; do    
    NAMESTR=(${file//./ })
    PART=(${NAMESTR//_/ })
    output_file=$output_folder/${PART[0]}_${PART[1]}_${PART[2]}_${site_name}_${PART[4]}.nc
    echo $file
    echo $output_file
    module load cdo
    command="cdo sellonlatbox,$longitude_range,$latitude_range $file $output_file"
    echo $command
    $command
done

## Time merge

cd $output_folder
output_file_merge=$output_folder/${PART[0]}_${PART[1]}_${PART[2]}_${site_name}_$start_y$underscore$end_y.nc
cdo -b F64 mergetime "*.nc" $output_file_merge
