#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[ ]:

colnames = ['dE_df_z']
data = pd.read_csv('dE_df_of_z.csv', names=colnames, delimiter=',',
                   header=None)

redshift = data.iloc[::2, :]
dE_df = data.iloc[1::2, :]

plt.plot(redshift, dE_df, color='red')
plt.xlabel('z')
plt.ylabel(r'$\frac{dE}{df_ed\Omega_e}$ in $\frac{erg}{Hz}$')
plt.title('Energy Spectrum as a fct. of redshift')
plt.xscale('log')
plt.yscale('log')
#plt.yscale('log')
#plt.ylabel("\frac{dE}{df_e d\Omega_e}", usetex=True)
#plt.savefig('dE_df.png', dpi=300)
plt.show()

# In[ ]:

colnames = ['dE_df_f']
dE_df = pd.read_csv('dE_df_of_f.csv', names=colnames, delimiter=',',
                   header=None)
f = np.linspace(0, 1000, 100)

plt.plot(f+1, dE_df, color='green')
plt.xlabel('f in Hz')
plt.ylabel(r'$\frac{dE}{df_ed\Omega_e}$ in $\frac{erg}{Hz}$')
plt.title('Energy Spectrum as a fct. of frequency')
plt.xscale('log')
plt.yscale('log')
#plt.savefig('dE_df.png', dpi=300)
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
# %%
