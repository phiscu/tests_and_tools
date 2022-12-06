#!/bin/bash

LON=77.90,78.50	# 75.80,76.40
LAT=41.90,42.50	# 41.00,41.40
input_dir=/data/projects/ebaca/Tienshan_data/GloH2O/HBV
sitename=kyzylsuu

output_dir=$input_dir/$sitename
mkdir $output_dir
module load cdo
for fold in "00" "01" "02" "03" "04" "05" "06" "07" "08" "09" ; do
	folddir=$input_dir/$fold
	cd $folddir
	base=HBV_params_gloh2o
	mergefile=${base}_$fold
	cdo merge *.nc $mergefile.nc
	output_file=$output_dir/${mergefile}_$sitename
	cdo sellonlatbox,$LON,$LAT $mergefile.nc $output_file.nc
done

cd output_dir
final_file=${base}_$sitename
cdo cat *.nc $final_file.nc
module load nco
ncrename -d time,realization $final_file.nc $final_file.nc
