#!/bin/bash

output_dir=/data/projects/ebaca/Tienshan_data/GloH2O/MSWX/past/Annually #/home/phillip/Seafile/EBA-CA/Tianshan_data/GloH2O/MSWX/past/Annually
DATASET=MSWX_daily_past
START=1979
END=2022

LON=77.90,78.50	# 75.80,76.40
LAT=41.90,42.50	# 41.00,41.40

SITE=kyzylsuu


for field in "LWd" "P" "Pres" "RelHum" "SpecHum" "SWd" "Temp" "Tmax" "Tmin" "Wind" ; do
	field_path=$output_dir/$field
	# Construct shell script
	sed -e "s/DATASET/$DATASET/" -e "s/START/$START/" -e "s/END/$END/" -e "s/LON/$LON/" -e "s/LAT/$LAT/" -e "s/SITE/$SITE/" -e "s/FIELD/$field/" -e "s+INPUT_DIR+$output_dir+" MSWX_latlonbox.sh > ${field_path}/${field}_MSWX_latlonbox.sh
	# Construct sbatch script
	sed -e "s/FIELD/$field/" -e "s+FPATH+${field_path}+" MSWX_latlonbox.bat > ${field_path}/${field}_MSWX_latlonbox.bat
	# Start slurm job
	chmod +x ${field_path}/${field}_MSWX_latlonbox.bat
	sbatch ${field_path}/${field}_MSWX_latlonbox.bat
done



