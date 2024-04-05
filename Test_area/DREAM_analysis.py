import numpy as np
import pandas as pd
import spotpy
from pathlib import Path
import os
import contextlib
import sys
import matplotlib.pyplot as plt
from spotpy.analyser import plot_parameter_trace, plot_posterior_parameter_histogram
from spotpy.likelihoods import gaussianLikelihoodMeasErrorOut as GausianLike
from spotpy.examples.spot_setup_hymod_python import spot_setup

import socket
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'


dream_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/DREAM/local/with_analysis/DREAM_Paper_1_lrtemp-pcorr-cet-lim_save-sim'

# Load the results gained with the dream sampler, stored in DREAM_hymod.csv
results = spotpy.analyser.load_csv_results(dream_path)
# Get fields with simulation data
fields = [word for word in results.dtype.names if word.startswith("sim")]
print(results[fields])

from matilda.mspot_glacier import analyze_results

ana = analyze_results(dream_path, dream_path + '_observations.csv', 'dream')

ana['sampling_plot'].show()
ana['best_run_plot'].show()
ana['par_uncertain_plot'].show()





# MUSS MAN DIE "SIM" MITSPEICHERN, UM DAS GANZE PLOTTEN ZU KÖNNEN?

# VERMUTLICH MUSS DAS GANZE NOCHMAL LAUFEN MIT DEM GANZEN AUSWERTEKRAM SCHON IM SKRIPT...NERVT

##

def psample(df, obs, rep=10, output = None, dbname='matilda_par_smpl', dbformat=None, obj_func=None, opt_iter=False, fig_path=None, #savefig=False,
            set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, freq="D", lat=None, area_cat=None,
            area_glac=None, ele_dat=None, ele_glac=None, ele_cat=None, soi=None, glacier_profile=None,
            interf=4, freqst=2, parallel=False, cores=2, save_sim=True, elev_rescaling=True,
            glacier_only=False, obs_type="annual", target_mb=None,
            algorithm='sceua', obj_dir="maximize", **kwargs):

    cwd = os.getcwd()
    if output is not None:
        os.chdir(output)

    # Setup model class:

    if glacier_only:
        setup = spot_setup_glacier(set_up_start=set_up_start, set_up_end=set_up_end, sim_start=sim_start, sim_end=sim_end,
                           freq=freq, area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat, ele_glac=ele_glac,
                           ele_cat=ele_cat, lat=lat, soi=soi, interf=interf, freqst=freqst,
                           glacier_profile=glacier_profile, obs_type=obs_type,
                           **kwargs)

    else:
        setup = spot_setup(set_up_start=set_up_start, set_up_end=set_up_end, sim_start=sim_start, sim_end=sim_end,
                        freq=freq, area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat, ele_glac=ele_glac,
                        ele_cat=ele_cat, lat=lat, soi=soi, interf=interf, freqst=freqst, glacier_profile=glacier_profile,
                        elev_rescaling=elev_rescaling, target_mb=target_mb,
                        **kwargs)

    psample_setup = setup(df, obs, obj_func)  # Define custom objective function using obj_func=
    alg_selector = {'mc': spotpy.algorithms.mc, 'sceua': spotpy.algorithms.sceua, 'mcmc': spotpy.algorithms.mcmc,
                    'mle': spotpy.algorithms.mle, 'abc': spotpy.algorithms.abc, 'sa': spotpy.algorithms.sa,
                    'dds': spotpy.algorithms.dds, 'demcz': spotpy.algorithms.demcz,
                    'dream': spotpy.algorithms.dream, 'fscabc': spotpy.algorithms.fscabc,
                    'lhs': spotpy.algorithms.lhs, 'padds': spotpy.algorithms.padds,
                    'rope': spotpy.algorithms.rope, 'fast': spotpy.algorithms.fast,
                    'nsgaii': spotpy.algorithms.nsgaii}

    if target_mb is not None:           # Format errors in database csv when saving simulations
        save_sim = False

    if parallel:
        sampler = alg_selector[algorithm](psample_setup, dbname=dbname, dbformat=dbformat, parallel='mpi',
                                              optimization_direction=obj_dir, save_sim=save_sim)
        if algorithm == 'mc' or algorithm == 'lhs' or algorithm == 'fast' or algorithm == 'rope':
            sampler.sample(rep)
        elif algorithm == 'sceua':
            sampler.sample(rep, ngs=cores)
        elif algorithm == 'demcz':
            sampler.sample(rep, nChains=cores)
        else:
            print('ERROR: The selected algorithm is ineligible for parallel computing.'
                  'Either select a different algorithm (mc, lhs, fast, rope, sceua or demcz) or set "parallel = False".')
            return
    else:
        sampler = alg_selector[algorithm](psample_setup, dbname=dbname, dbformat=dbformat, save_sim=save_sim,
                                          optimization_direction=obj_dir)
        if opt_iter:
            if yesno("\n******** WARNING! Your optimum # of iterations is {0}. "
                     "This may take a long time.\n******** Do you wish to proceed".format(psample_setup.par_iter)):
                sampler.sample(psample_setup.par_iter)  # ideal number of reps = psample_setup.par_iter
            else:
                return
        else:
            sampler.sample(rep)

    # Change dbformat to None for short tests but to 'csv' or 'sql' to avoid data loss in case off long calculations.

    if target_mb is None:
        psample_setup.evaluation().to_csv(dbname + '_observations.csv')
    else:
        psample_setup.evaluation()[0].to_csv(dbname + '_observations.csv')


    if not parallel:

        if target_mb is None:
            results = analyze_results(sampler, psample_setup.evaluation(), algorithm=algorithm, obj_dir=obj_dir,
                                  fig_path=fig_path, dbname=dbname, glacier_only=glacier_only)
        else:
            results = analyze_results(sampler, psample_setup.evaluation()[0], algorithm=algorithm, obj_dir=obj_dir,
                                  fig_path=fig_path, dbname=dbname, glacier_only=glacier_only, target_mb=target_mb)

        return results

    os.chdir(cwd)