
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
                        'selection_mean': 1,
                        'selection_window': 'gaussian',
                        'selection_width': 0.2,
                        'selection_bias': 1.81,
                        'selection_magnification_bias':0,
                        'selection_multitracing' : 'no',
                        'gw_frequency' : f_gw[i]}

    LCDM.set(common_settings)
    LCDM.compute()

baLCDM = LCDM.get_background()
# all_k = LCDM.get_perturbations()
baLCDM.keys()

times = LCDM.get_current_derived_parameters(['tau_rec','conformal_age'])
tau_rec = times['tau_rec']
tau_0 = times['conformal_age']
tau1 = np.logspace(math.log10(tau_ini),math.log10(tau_rec),tau_num_early)
tau2 = np.logspace(math.log10(tau_rec),math.log10(tau_0),tau_num_late)[1:]
tau2[-1] *= 0.999 # this tiny shift avoids interpolation errors
tau = np.concatenate((tau1,tau2))
tau_num = len(tau)
# one_time = LCDM.get_transfer(z_rec)
# print(one_time.keys())
# k = one_time['k (h/Mpc)']

# In[ ]:

background_tau = baLCDM['conf. time [Mpc]']
background_H = 2.*math.pi*baLCDM['H [1/Mpc]']*LCDM.h()
background_aH = 2.*math.pi*baLCDM['H [1/Mpc]']/(1.+baLCDM['z'])/LCDM.h()
background_z = baLCDM['z'] # read redshift
# interpolate a function aH(tau)
background_H = UnivariateSpline(background_tau, background_H)
background_aH_at_tau = interp1d(background_tau, background_aH)
background_z_at_tau = interp1d(background_tau,background_z)

background_H_der = background_H.derivative()

max_z_needed = background_z_at_tau(tau[0])
if max_z_needed > z_max_pk:
    print('you must increase the value of z_max_pk to at least ',max_z_needed)
    () + 1  # this strange line is just a trick to stop the script execution there
else:
    print('in a next run with the same values of tau, you may decrease z_max_pk from ',z_max_pk,' to ',max_z_needed)
#
# get transfer functions at each time and build arrays Theta0(tau,k) and phi(tau,k)
#
for i in range(tau_num):
    one_time = LCDM.get_transfer(background_z_at_tau(tau[i])) # transfer functions at each time tau
    if i ==0:   # if this is the first time in the loop: create the arrays (k, Theta0, phi)
        k = one_time['k (h/Mpc)']
        k_num = len(k)
        print('k amount', k_num)
        Theta0 = np.zeros((tau_num,k_num))
        phi = np.zeros((tau_num,k_num))
    Theta0[i, :] = 0.25*one_time['d_g'][:]
    phi[i, :] = one_time['phi'][:]

# spherical bessel functions, l=1
chi = tau_0 - tau
bessel = np.zeros((len(chi), len(k)))
for i in range(len(chi)):
    bessel[:, i] = spherical_jn(1, k*chi[i])

# bessel derivative
bessel_der = np.zeros((len(k), len(chi)))
for i in range(len(chi)):
    bessel_der[:, i] = spherical_jn(1, k*chi[i], derivative=True)

k_chi = np.arrange(0.04, 21000, 2*1000000)
bessel_2nd_der_pre = spherical_jn(1, k_chi, derivative=True)
bessel_der_at_kchi = interp1d(k_chi, bessel_2nd_der_pre)
besser_2nd_der_at_kchi = bessel_der_at_kchi.derivative()


'''tau_num = len(phi[:, 0])
k_num = len(phi[0, :])
phi_at_tau = np.zeros(k_num)
# array of derivatives, indices are diff. k
phi_der_at_tau = np.zeros(k_num)

for i in range(len(phi[0, :])):
    phi_at_tau[i] = UnivariateSpline(background_tau, phi[i, :])
    phi_der_at_tau[i] = phi_at_tau[i].derivative()
'''

# %%