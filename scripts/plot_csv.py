#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# In[ ]:
z = np.array([0.1, 0.5, 1, 2, 4, 6, 8])
a = 1/(1+z) #array of z 

colnames = ['SFR']
z0p1 = pd.read_csv('SFR-0p1.csv', names=colnames, delimiter=',', header=None)
z0p5 = pd.read_csv('SFR-0p5.csv', names=colnames, delimiter=',', header=None)
z1 = pd.read_csv('SFR-1.csv', names=colnames, delimiter=',', header=None)
z2 = pd.read_csv('SFR-2.csv', names=colnames, delimiter=',', header=None)
z4 = pd.read_csv('SFR-4.csv', names=colnames, delimiter=',', header=None)
z6 = pd.read_csv('SFR-6.csv', names=colnames, delimiter=',', header=None)
z8 = pd.read_csv('SFR-8.csv', names=colnames, delimiter=',', header=None)

mass = z0p1.iloc[::2] #mass array
M200kms = 1.64*10**12/(np.power(a/0.378,-0.142)+np.power(a/0.378,-1.79)) #array of z
#vmpeak = 200*np.power(mass/M200kms,3)
sfr0p1 = z0p1.iloc[1::2]
sfr0p5 = z0p5.iloc[1::2]
sfr1 = z1.iloc[1::2]
sfr2 = z2.iloc[1::2]
sfr4 = z4.iloc[1::2]
sfr6 = z6.iloc[1::2]
sfr8 = z8.iloc[1::2]

plt.plot(200*np.power(mass/M200kms[0],3), sfr0p1, label='z=0.1')
plt.plot(200*np.power(mass/M200kms[1],3), sfr0p5, label='z=0.5')
plt.plot(200*np.power(mass/M200kms[2],3), sfr1,   label='z=1')
plt.plot(200*np.power(mass/M200kms[3],3), sfr2,   label='z=2')
plt.plot(200*np.power(mass/M200kms[4],3), sfr4,   label='z=4')
plt.plot(200*np.power(mass/M200kms[5],3), sfr6,   label='z=6')
plt.plot(200*np.power(mass/M200kms[6],3), sfr8,   label='z=8')
#plt.plot(mass, sfr0p1, label='z=0.1')
#plt.plot(mass, sfr0p5, label='z=0.5')
#plt.plot(mass, sfr1, label='z=1')
#plt.plot(mass, sfr2, label='z=2')
#plt.plot(mass, sfr4, label='z=4')
#plt.plot(mass, sfr6, label='z=6')
#plt.plot(mass, sfr8, label='z=8')
plt.xlabel(r'Velocity at Peak Halo Mass in $\frac{km}{s}$', size='large')
#plt.xlabel(r'Halo Mass in $M_\odot$', size='large')
plt.ylabel(r'SFR in $\frac{M_\odot}{yr}$', size='large')

plt.xlim(50, 2000)
plt.ylim(0.1, 1000)

#plt.xticks(fontsize=10, rotation=45)
#plt.tick_params(axis='x',rotation=45, labelsize=7, which='minor')
plt.xscale('log')
plt.yscale('log')
#plt.gca().xaxis.set_minor_formatter(FormatStrFormatter('%.0e'))

plt.title('Star Formation Rate as a fct. of Velocity at Peak Halo Mass')
plt.legend()
#plt.savefig('dE_df.png', dpi=300)
plt.show()

# In[ ]:
colnames = ['BBH Merger Rate']
data3 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/BBH_merger_rate_new.csv', names=colnames, delimiter=',',
                   header=None)

redshift = data3.iloc[::2, :]
mergerrate = data3.iloc[1::2, :]

plt.plot(redshift, mergerrate, color='green')
plt.xlabel('z')
plt.ylabel(r'Merger Rate in $\frac{1}{Mpc^3 yr}$')
#plt.xscale('log')
#plt.yscale('log')
plt.title('BBH Merger Rate as a fct. of Redshift')
plt.show()

# In[ ]:
colnames = ['dE_df_z']
data = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/dE_df_of_z_100Hz.csv', names=colnames, delimiter=',',
                   header=None)

redshift = data.iloc[::2, :]
dE_df = data.iloc[1::2, :]

plt.plot(redshift, dE_df, color='red')
plt.xlabel('z')
plt.ylabel(r'$\frac{dE}{df_ed\Omega_e}$ in $\frac{erg}{Hz}$')
plt.title('Energy Spectrum as a fct. of redshift')
#plt.yscale('log')
#plt.ylabel("\frac{dE}{df_e d\Omega_e}", usetex=True)
plt.show()
# In[ ]:
colnames = ['dE_df_f']
dE_df = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/dE_df_of_f_cut.csv', names=colnames, delimiter=',',
                   header=None)
f = np.linspace(1, 1000, 99)

plt.plot(f, dE_df, color='green')
plt.xlabel('f in Hz')
plt.ylabel(r'$\frac{dE}{df_ed\Omega_e}$ in $\frac{erg}{Hz}$')
plt.title('Energy Spectrum as a fct. of frequency')
#plt.savefig('dE_df.png', dpi=300)
plt.show()


plt.plot(f, dE_df*mergerrate.iloc[0][0], color='green')
plt.xlabel('f in Hz')
plt.ylabel(r'$\frac{dE}{df_ed\Omega_e}$ in $\frac{erg}{Hz}$')
plt.title('Energy Spectrum * Merger Rate as a fct. of frequency')
plt.show()

# In[ ]:

colnames = ['HMF']
data2 = pd.read_csv('HMF_dimless.csv', names=colnames, delimiter=',',
                   header=None)

mass = data2.iloc[::2, :]
hmf = data2.iloc[1::2, :]

plt.plot(mass, hmf, color='green')
plt.xlabel(r'Halo Mass in $\frac{M_\odot}{h}$')
plt.ylabel(r'$\frac{M^2}{\overline{\rho}_M}\,\frac{dn}{dM}$ ')
plt.xscale('log')
plt.yscale('log')
plt.title('Dimensionless Halo Mass Function')
#plt.savefig('dE_df.png', dpi=300)
plt.show()


# In[ ]:
colnames = ['window_fct_z']
data_w = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/window_fct_of_z_gauss.csv', names=colnames, delimiter=',',
                  header=None)
data_w2 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/window_fct_of_z_1Hz.csv', names=colnames, delimiter=',',
                   header=None)

redshift_gauss = data_w.iloc[::2, :]
redshift = data_w2.iloc[::2, :]
window_gauss = data_w.iloc[1::2, :]
window = data_w2.iloc[1::2, :]

#plt.plot(redshift_gauss, window_gauss, label='Gauss')
#plt.plot(redshift, window)
plt.plot(redshift, window, label='1 Hz')

plt.xlabel('z')
plt.ylabel('Window Function')
plt.title('Window as a fct. of redshift')
#plt.xscale('log')
plt.legend()
plt.show()

# In[ ]:
colnames = ['window_fct_tau']
data_w = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/window_fct_of_tau_gauss.csv', names=colnames, delimiter=',',
                  header=None)
data_w2 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/window_fct_of_tau_100Hz.csv', names=colnames, delimiter=',',
                  header=None)

redshift_gauss = data_w.iloc[::2, :]
redshift = data_w2.iloc[::2, :]
window_gauss = data_w.iloc[1::2, :]
window500 = data_w2.iloc[1::2, :]

plt.plot(redshift_gauss, window_gauss, label='Gauss')
plt.plot(redshift, window500, label='frequency dep. at 100 Hz')
#plt.plot(redshift, window10, label='10 Hz')

plt.xlabel(r'$\tau$')
plt.ylabel('Window Function')
plt.gca().invert_xaxis()
plt.title(r'Window as a fct. of $\tau$')
#plt.xlim(0, 10)
plt.legend()
plt.show()

# In[ ]:
colnames = ['evo_bias_z']
data = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/evolution_bias.csv',
                    names=colnames, delimiter=',', header=None)

redshift = data.iloc[::2, :]
evo_bias = data.iloc[1::2, :]

plt.plot(redshift, evo_bias, color='brown')
plt.xlabel('z')
plt.ylabel(r'$b_e$')
plt.title('Evolution Bias as a fct. of redshift')
#plt.yscale('log')
plt.show()

# In[ ]:
colnames = ['evo_bias_f']
data = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/evolution_bias_f.csv',
                    names=colnames, delimiter=',', header=None)

redshift = data.iloc[::2, :]
evo_bias = data.iloc[1::2, :]

plt.plot(redshift, evo_bias, color='black')
plt.xlabel('f')
plt.ylabel(r'$b_e$')
plt.title('Evolution Bias as a fct. of frequency')
plt.yscale('log')
plt.show()
# In[ ]:
colnames = ['MR_dE_df']
data = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/rate_times_spectrum.csv',
                   names=colnames, delimiter=',',
                   header=None)

redshift = data.iloc[::3, :]
MR_dE_df = data.iloc[1::3, :]
MR_dE_df_der = data.iloc[2::3, :]

plt.plot(redshift, MR_dE_df, color='purple')
plt.xlabel('z')
plt.ylabel('MR*dE/df')
plt.title('Merger Rate times Energy Spectrum')
#plt.xscale('log')
#plt.yscale('log')

plt.show()

plt.plot(redshift, MR_dE_df_der, color='green')
plt.xlabel('z')
plt.ylabel('d/dz(MR*dE/df)')
plt.title('z derivative Merger Rate * Energy Spectrum')
#plt.xscale('log')
plt.show()
# %%
