#%%
import matplotlib.pyplot as plt
import numpy as np
import itertools

# gaussian window, 0.25 width, z varied
files = ['/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/output/gw_autocorr/gw_autocorr1_z.dat']
data = []
for data_file in files:
    data.append(np.loadtxt(data_file))
roots = ['gw_autocorr1_z']

fig, ax = plt.subplots()

index, curve = 0, data[0]
y_axis = ['z = 0.5', 'z = 1', 'z = 1.5']
tex_names = ['dens[1]-dens[1]', 'dens[2]-dens[2]', 'dens[3]-dens[3]']
x_axis = 'l'
ylim = []
xlim = []
plt.xscale('log')
plt.yscale('log')
ax.plot(curve[:, 0], curve[:, 1])
ax.plot(curve[:, 0], curve[:, 2])
ax.plot(curve[:, 0], curve[:, 3])

ax.legend([elem for (root, elem) in
    itertools.product(roots, y_axis)], loc='best')

ax.set_xlabel('l', fontsize=14)
ax.set_ylabel(r'$\frac{l(l+1)}{2\pi} C_l$', fontsize = 14)
plt.title('Different Redshifts')
plt.tight_layout()
plt.savefig('ac_1_z.jpg', dpi = 300, bbox_inches='tight')
plt.show()
# %%
