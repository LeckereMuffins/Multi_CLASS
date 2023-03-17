import matplotlib.pyplot as plt
import numpy as np
import itertools

files = ['/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/output/gw_autocorr/gw_autocorr00_cl.dat']
data = []
for data_file in files:
    data.append(np.loadtxt(data_file))
roots = ['gw_autocorr00_cl']

fig, ax = plt.subplots()

index, curve = 0, data[0]
y_axis = [u'dens[1]-dens[1]', u'dens[2]-dens[2]']
tex_names = ['dens[1]-dens[1]', 'dens[2]-dens[2]']
x_axis = 'l'
ylim = []
xlim = []
ax.loglog(curve[:, 0], abs(curve[:, 1]))
ax.loglog(curve[:, 0], abs(curve[:, 2]))

ax.legend([root+': '+elem for (root, elem) in
    itertools.product(roots, y_axis)], loc='best')

ax.set_xlabel('l', fontsize=16)
plt.show()