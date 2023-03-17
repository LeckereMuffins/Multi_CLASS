#%%
import matplotlib.pyplot as plt
import numpy as np
import itertools

# gaussian
files = ['/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/output/gw_autocorr/gw_autocorr1_z.dat']
data = []
for data_file in files:
    data.append(np.loadtxt(data_file))
roots = ['gw_autocorr1_z']

# tophat
files2 = ['/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/output/gw_autocorr/gw_autocorr1_tophat.dat']
data2 = []
for data_file in files2:
    data2.append(np.loadtxt(data_file))
roots2 = ['gw_autocorr1_tophat']

# dirac
files3 = ['/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/output/gw_autocorr/gw_autocorr1_dirac.dat']
data3 = []
for data_file in files3:
    data3.append(np.loadtxt(data_file))
roots3 = ['gw_autocorr1_dirac']

fig, ax = plt.subplots()

index, curve = 0, data[0]
index_2, curve_2 = 1, data2[0]
index_3, curve_3 = 2, data3[0]
y_axis = ['Gaussian', 'Tophat', 'Dirac Delta']
tex_names = ['dens[1]-dens[1]', 'dens[2]-dens[2]', 'dens[3]-dens[3]']
x_axis = 'l'
ylim = []
xlim = []
plt.xscale('log')
plt.yscale('log')
ax.plot(curve[:, 0], curve[:, 1])
ax.plot(curve_2[:, 0], curve_2[:, 1])
ax.plot(curve[:, 0], curve_3[:, 1])

ax.legend([elem for (root, elem) in
    itertools.product(roots, y_axis)], loc='best')

ax.set_xlabel('l', fontsize=14)
ax.set_ylabel(r'$\frac{l(l+1)}{2\pi} C_l$', fontsize = 14)
plt.tight_layout()
plt.title('Different Window Functions')
plt.savefig('ac_1_window.jpg', dpi = 300, bbox_inches='tight')
plt.show()
# %%
