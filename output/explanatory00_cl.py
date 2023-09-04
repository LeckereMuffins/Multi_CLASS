#%%
import matplotlib.pyplot as plt
import numpy as np
import itertools

files = ['/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/output/multi_explanatory_batch44_cl.dat']
data = []
for data_file in files:
    data.append(np.loadtxt(data_file))
roots = ['explanatory00_cl']

fig, ax = plt.subplots()

index, curve = 0, data[0]
#curve[:, 1] = curve[:, 1]/(curve[:, 0]*(curve[:, 0]+1))*2*np.pi
y_axis = ['TT']
tex_names = ['TT']
ylim = []
xlim = []
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$C_\ell$')
#ax.set_ylabel(r'$\frac{\ell (\ell+1)}{2 \pi} C_\ell$')
ax.plot(curve[:, 0], curve[:, 1])
#ax.set_xlim(0, 30)

ax.legend([root+': '+elem for (root, elem) in
    itertools.product(roots, y_axis)], loc='best')

plt.show()
# %%
