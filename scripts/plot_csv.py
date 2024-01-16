#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import integrate

# In[ ]:
z = np.array([0.1, 0.5, 1, 2, 4, 6, 8])
a = 1/(1+z) #array of z 

colnames = ['SFR']
z0p1 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/SFR-0p1.csv',
                    names=colnames, delimiter=',', header=None)
z0p5 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/SFR-0p5.csv', names=colnames, delimiter=',', header=None)
z1 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/SFR-1.csv', names=colnames, delimiter=',', header=None)
z2 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/SFR-2.csv', names=colnames, delimiter=',', header=None)
z4 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/SFR-4.csv', names=colnames, delimiter=',', header=None)
z6 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/SFR-6.csv', names=colnames, delimiter=',', header=None)
z8 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/SFR-8.csv', names=colnames, delimiter=',', header=None)

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
# plt.plot(mass, sfr0p1, label='z=0.1')
# plt.plot(mass, sfr0p5, label='z=0.5')
# plt.plot(mass, sfr1, label='z=1')
# plt.plot(mass, sfr2, label='z=2')
# plt.plot(mass, sfr4, label='z=4')
# plt.plot(mass, sfr6, label='z=6')
# plt.plot(mass, sfr8, label='z=8')
plt.xlabel(r'Velocity at Peak Halo Mass in $\frac{km}{s}$', size='large')
#plt.xlabel(r'Halo Mass in $M_\odot$', size='large')
plt.ylabel(r'$\langle SFR\rangle_{SF}$ in $\frac{M_\odot}{yr}$', size='large')

plt.xlim(50, 2000)
plt.ylim(0.1, 2100)

#plt.xticks(fontsize=10, rotation=45)
#plt.tick_params(axis='x',rotation=45, labelsize=7, which='minor')
plt.xscale('log')
plt.yscale('log')
#plt.gca().xaxis.set_minor_formatter(FormatStrFormatter('%.0e'))

#plt.title('Star Formation Rate as a fct. of Velocity at Peak Halo Mass')
plt.legend()
plt.savefig('sfr_of_v.png', dpi=300)
plt.show()

# In[ ]:
colnames = ['BBH Merger Rate']
data3 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/BBH_merger_rate.csv', names=colnames, delimiter=',',
                   header=None)

redshift = data3.iloc[::2, :]
mergerrate = data3.iloc[1::2, :]

plt.plot(redshift, mergerrate, linestyle='--', marker='.', color='green')
plt.xlabel('z')
plt.ylabel(r'Merger Rate in $\frac{1}{Mpc^3 yr}$')
#plt.xscale('log')
#plt.yscale('log')
#plt.title('BBH Merger Rate as a fct. of Redshift')
plt.savefig('bbh_merger_rate.png', dpi=300)
plt.show()

# In[ ]:
colnames = ['dE_df_z']
data = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/dE_df_of_z_1000Hz.csv', names=colnames, delimiter=',',
                   header=None)

redshift = data.iloc[::2, :]
dE_df = data.iloc[1::2, :]

plt.plot(redshift, dE_df, linestyle='--', marker='.', color='firebrick')
plt.xlabel('z', fontsize=15)
plt.ylabel(r'$\frac{dE}{df_ed\Omega_e}$ in $\frac{erg}{Hz}$', fontsize=15)
#plt.title('Energy Spectrum as a fct. of redshift at 1000 Hz')
#plt.yscale('log')
#plt.ylabel("\frac{dE}{df_e d\Omega_e}", usetex=True)
plt.savefig('E_spectrum_z_1000Hz.png', dpi=300)
plt.show()

# In[ ]:
colnames = ['dE_df_f']
dE_df = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/dE_df_of_f.csv', names=colnames, delimiter=',',
                   header=None)
#f = np.linspace(1, 1000, 99)

frequency = dE_df.iloc[::2, :]
dE_df = dE_df.iloc[1::2, :]

plt.plot(frequency, dE_df, color='black')
plt.xlabel('f in Hz')
plt.ylabel(r'$\frac{dE}{df_ed\Omega_e}$ in $\frac{erg}{Hz}$')
#plt.title('Energy Spectrum as a fct. of frequency')
plt.vlines(300, 0, 1.2e54, color='grey', linestyle='dashed')
plt.vlines(55, 0, 1.2e54, color='grey', linestyle='dashed')
#plt.xlim(2, 1000)
#plt.yscale('log')
#plt.xscale('log')
plt.text(-30, 0.1, "Insp.")
plt.text(120, 0.1, "Merger")
plt.text(380, 0.1, "Ringdown")
plt.axvspan(0, 55, 0, 1.2e54, facecolor='red', alpha=0.3)
plt.axvspan(55, 300, 0, 1.2e54, facecolor='orange', alpha=0.3)
plt.axvspan(300, 1000, 0, 1.2e54, facecolor='yellow', alpha=0.3)
plt.savefig('dE_df_of_f.png', dpi=300)
plt.show()
# In[ ]:
plt.plot(f, dE_df*mergerrate.iloc[0][0], color='green')
plt.xlabel('f in Hz')
plt.ylabel(r'$\frac{dE}{df_ed\Omega_e}$ in $\frac{erg}{Hz}$')
plt.title('Energy Spectrum * Merger Rate as a fct. of frequency')
plt.show()

# In[ ]:

colnames = ['HMF_delta']
data_w = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/HMF_200delta.csv',
                     names=colnames, delimiter=',',
                     header=None)
data_w2 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/HMF_400delta.csv',
                      names=colnames, delimiter=',',
                     header=None)
data_w3 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/HMF_800delta.csv',
                       names=colnames, delimiter=',',
                     header=None)
data_w4 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/HMF_1600delta.csv', 
                      names=colnames, delimiter=',',
                     header=None)
data_w5 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/HMF_3200delta.csv', 
                      names=colnames, delimiter=',',
                     header=None)

data_list = [data_w, data_w2, data_w3, data_w4, data_w5]
label_list=[r'$\Delta = 200$', r'$\Delta = 400$',r'$\Delta = 800$',
            r'$\Delta = 1600$', r'$\Delta = 3200$',]
i=0

for data in data_list:
   mass = np.array(data.iloc[::2, :])
   hmf = np.array(data.iloc[1::2, :])
   plt.plot(mass, hmf, label=label_list[i])
   i+=1

plt.xlabel(r'Halo Mass in $\frac{M_\odot}{h}$', fontsize = 15)
plt.ylabel(r'$\frac{M^2}{\overline{\rho}_M}\,\frac{dn}{dM}$ ', fontsize=15)
#plt.ylabel(r'$\frac{dn}{dM}$ in $\frac{h/M_\odot}{(Mpc/h)^3}$', fontsize=15)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4, 0.5)
#plt.title('Dimensionless Halo Mass Function')
plt.legend()
plt.tight_layout()
plt.savefig('HMF_dimless.png', dpi=300)
plt.show()

# In[ ]:

colnames = ['HMF_z']
data_w = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/HMF_800delta.csv',
                     names=colnames, delimiter=',',
                     header=None)
data_w2 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/HMF_z0p5.csv',
                      names=colnames, delimiter=',',
                     header=None)
data_w3 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/HMF_z1.csv',
                       names=colnames, delimiter=',',
                     header=None)
data_w4 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/HMF_z2.csv', 
                      names=colnames, delimiter=',',
                     header=None)
data_w5 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/HMF_z3.csv', 
                      names=colnames, delimiter=',',
                     header=None)

data_list = [data_w, data_w2, data_w3, data_w4, data_w5]
label_list=['z=0', 'z=0.5', 'z=1', 'z=2', 'z=3']
i=0

for data in data_list:
   mass = np.array(data.iloc[::2, :])
   hmf = np.array(data.iloc[1::2, :])
   plt.plot(mass, hmf, label=label_list[i])
   i+=1

plt.xlabel(r'Halo Mass in $\frac{M_\odot}{h}$', fontsize = 15)
plt.ylabel(r'$\frac{M^2}{\overline{\rho}_M}\,\frac{dn}{dM}$ ', fontsize=15)
#plt.ylabel(r'$\frac{dn}{dM}$ in $\frac{h/M_\odot}{(Mpc/h)^3}$', fontsize=15)
#plt.xscale('log')
#plt.yscale('log')
plt.ylim(1e-4, 0.5)
plt.xlim(1e11, 1e14)
#plt.title('Dimensionless Halo Mass Function')
plt.legend()
plt.tight_layout()
plt.savefig('HMF_diff_z.png', dpi=300)
plt.show()


# In[ ]:
colnames = ['window_fct_z']
data_w = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/window_fct_of_z_1Hz.csv', names=colnames, delimiter=',',
                  header=None)
data_w2 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/window_fct_of_z_10Hz.csv', names=colnames, delimiter=',',
                   header=None)
data_w3 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/window_fct_of_z_100Hz.csv', names=colnames, delimiter=',',
                   header=None)
data_w4 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/window_fct_of_z_200Hz.csv', names=colnames, delimiter=',',
                   header=None)
data_w5 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/window_fct_of_z_500Hz.csv', names=colnames, delimiter=',',
                   header=None)
data_w6 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/window_fct_of_z_1000Hz.csv', names=colnames, delimiter=',',
                   header=None)
data_w7 = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/window_fct_of_z_5000Hz.csv', names=colnames, delimiter=',',
                   header=None)

colnames = ['z', 'window_fct_z']
data_l = pd.read_csv(r'/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/scripts/window_fct_lorenzo.csv',
                     names=colnames, sep='   ', lineterminator='\n',
                  header=None)

data_list = [data_w, data_w2, data_w3, data_w4, data_w5, data_w6, data_w7]
label_list=['1 Hz', '10 Hz', '100 Hz', '200 Hz', '500 Hz', '1000 Hz', '5000 Hz']
colors = plt.cm.turbo(np.linspace(0, 1, 7))
colors = colors[::-1]
i=0

for data in data_list:
   redshift = np.array(data.iloc[::2, :])
   window = np.array(data.iloc[1::2, :])
   norm = -integrate.trapezoid(window.flatten(), redshift.flatten(), dx=0.01)
   plt.plot(redshift, window/norm, label=label_list[i], color=colors[i])
   i+=1

plt.xlabel('z')
plt.ylabel('Window Function')
#plt.title('Window as a fct. of redshift at 1 Hz')
#plt.xscale('log')
plt.ylim(0, 4)
plt.legend()
plt.savefig('window_diff_frequencies.png', dpi=300)
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
