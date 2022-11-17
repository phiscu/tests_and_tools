#!/bin/bash

#SBATCH --job-name="mswx"
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/data/projects/ebaca/Tienshan_data/GloH2O/MSWX/past
#SBATCH --account=misc
#SBATCH --error=error_log
#SBATCH --partition=compute
#SBATCH --output=out_log
##SBATCH --mail-type=ALL

./MSWX_timemerge.sh
