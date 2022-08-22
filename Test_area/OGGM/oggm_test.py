from oggm import cfg, utils, workflow, tasks, graphics
from oggm import graphics
from oggm import workflow

cfg.initialize(logging_level='WARNING')

cfg.PARAMS['use_multiprocessing'] = True

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
# Make pretty plots
sns.set_style('ticks')
sns.set_context('notebook')

cfg.PATHS['working_dir'] = "/home/phillip/Seafile/Ana-Lena_Phillip/data/tests_and_tools/Test_area/OGGM/output"

rgi_ids = ['RGI60-13.06353',
            'RGI60-13.06354',
            'RGI60-13.06355',
            'RGI60-13.06356',
            'RGI60-13.06357',
            'RGI60-13.06358',
            'RGI60-13.06359',
            'RGI60-13.06360',
            'RGI60-13.06361',
            'RGI60-13.06362',
            'RGI60-13.06363',
            'RGI60-13.06364',
            'RGI60-13.06365',
            'RGI60-13.06366',
            'RGI60-13.06367',
            'RGI60-13.06368',
            'RGI60-13.06369',
            'RGI60-13.06370',
            'RGI60-13.06371',
            'RGI60-13.06372',
            'RGI60-13.06373',
            'RGI60-13.06374',
            'RGI60-13.06375',
            'RGI60-13.06376',
            'RGI60-13.06377',
            'RGI60-13.06378',
            'RGI60-13.06379',
            'RGI60-13.06380',
            'RGI60-13.06381',
            'RGI60-13.06382',
            'RGI60-13.06425',
            'RGI60-13.07717',
            'RGI60-13.07718',
            'RGI60-13.07719',
            'RGI60-13.07926',
            'RGI60-13.07927',
            'RGI60-13.07930',
            'RGI60-13.07931']

rgi_id = rgi_ids[0]

# gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_border=80)   # For the first time
gdirs = workflow.init_glacier_directories(rgi_ids)                                          # After the first time


# We use a recent gdir setting, calibated on a glacier per glacier basis
base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/ERA5/elev_bands/qc0/pcp1.6/match_geod_pergla/'

# These parameters correspond to the settings of the base url
cfg.PARAMS['prcp_scaling_factor'] = 1.6
cfg.PARAMS['climate_qc_months'] = 0
cfg.PARAMS['hydro_month_nh'] = 1
cfg.PARAMS['hydro_month_sh'] = 1

# We use a relatively large border value to allow the glacier to grow during spinup
gdir = workflow.init_glacier_directories(rgi_ids, from_prepro_level=5, prepro_border=80, prepro_base_url=base_url)

# We will "reconstruct" a possible glacier evolution from this year onwards
spinup_start_yr = 1980

# For groups of glaciers:

workflow.execute_entity_task(tasks.run_dynamic_spinup, gdir,
                             spinup_start_yr=spinup_start_yr,  # When to start the spinup
                             minimise_for='area',  # what target to match at the RGI date
                             output_filesuffix='_spinup_dynamic', # Where to write the output - this is needed to stitch the runs together afterwards
                             )

workflow.execute_entity_task(tasks.run_from_climate_data, gdir,
                             init_model_filesuffix='_spinup_dynamic',
                             # Which initial geometry to use (from the spinup here: default is from the inversion)
                             output_filesuffix='_hist_spinup_dynamic',  # Where to write the output
                             )

ds_spinup_dynamic = workflow.execute_entity_task(tasks.merge_consecutive_run_outputs, gdir,
                            input_filesuffix_1='_spinup_dynamic',  # File 1
                            input_filesuffix_2='_hist_spinup_dynamic',  # File 2
                            output_filesuffix='_merged_spinup_dynamic',  # Output file (optional)
                            )


# For individual glaciers:

tasks.run_dynamic_spinup(gdir,
                         spinup_start_yr=spinup_start_yr,  # When to start the spinup
                         minimise_for='area',  # what target to match at the RGI date
                         output_filesuffix='_spinup_dynamic',  # Where to write the output - this is needed to stitch the runs together afterwards
                         );


# This is the same as before - the only difference is that we use the end of the spinup as starting geometry
tasks.run_from_climate_data(gdir,
                            init_model_filesuffix='_spinup_dynamic',  # Which initial geometry to use (from the spinup here: default is from the inversion)
                            output_filesuffix='_hist_spinup_dynamic',  # Where to write the output
                            );

# New: stich the output together for analysis
ds_spinup_dynamic = tasks.merge_consecutive_run_outputs(gdir,
                                                        input_filesuffix_1='_spinup_dynamic',  # File 1
                                                        input_filesuffix_2='_hist_spinup_dynamic',  # File 2
                                                        output_filesuffix='_merged_spinup_dynamic',  # Output file (optional)
                                                        )

# How to sum up all catchments glaciers

ds_spinup_dynamic.volume_m3.plot(label='Dynamical spinup')
ds_spinup_dynamic.area_m2.plot()
plt.show()