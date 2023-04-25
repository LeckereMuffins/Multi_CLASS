#%%
# coding: utf-8
#&matplotlib inline

"""
@author: felick
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.special import spherical_jn
import math

z_max_pk = 1000        # highest redshift involved
k_per_decade = 400     # number of k values, controls final resolution
k_min_tau0 = 40.       # this value controls the minimum k value in the figure (it is k_min * tau0)
P_k_max_inv_Mpc =1.0   # this value is directly the maximum k value in the figure in Mpc
tau_num_early = 2000   # number of conformal time values before rec., controls final resolution
tau_num_late = 200     # number of conformal time values after rec., controls final resolution
tau_ini = 10.          # first value of conformal time in Mpc

LCDM = Class()

f_gw = np.linspace(1, 900, 900) # 1-900 in 900 steps

for i in range(len(f_gw)):
    common_settings = {'output':'nCl', # output: transfer functions only
                        # LambdaCDM parameters
                        'h':0.67556,
                        'omega_b':0.022032,
                        'omega_cdm':0.12038,
                        'A_s':2.215e-9,
                        'n_s':0.9619,
                        'tau_reio':0.0925,
                        # Take fixed value for primordial Helium (instead of automatic BBN adjustment)
                        'YHe':0.246,
                        # other output and precision parameters
                        'z_max_pk':z_max_pk,
                        # 'recfast_z_initial':z_max_pk,
                        #'k_step_sub':'0.01',
                        'k_per_decade_for_pk':k_per_decade,
                        'k_per_decade_for_bao':k_per_decade,
                        'k_min_tau0':k_min_tau0, # this value controls the minimum k value in the figure
                        'perturb_sampling_stepsize':'0.05',
                        'compute damping scale':'yes', # needed to output and plot Silk damping scale
                        'gauge':'newtonian',
                        'selection_mean': 0.1,
                        'selection_window': 'gaussian',
                        'selection_width': 0.2,
                        'selection_bias': 1.81,
                        'selection_magnification_bias':0,
                        'selection_multitracing' : 'no',
                        'gw_frequency' : f_gw[i]}

    LCDM.set(common_settings)
    LCDM.compute()