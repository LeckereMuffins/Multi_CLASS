#%%
import matplotlib.pyplot as plt
import numpy as np
import itertools

files = ['/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/output/testcl.dat', '/rwthfs/rz/cluster/home/la171705/GW_BG/Multi_CLASS/output/testcl_lcdm.dat']
data = []
for data_file in files:
    data.append(np.loadtxt(data_file))
roots = ['testcl', 'testcl_lcdm']

fig, ax = plt.subplots()

index, curve = 0, data[0]
y_axis = [u'TT']
tex_names = ['TT']
x_axis = 'l'
ylim = []
xlim = []
ax.semilogx(curve[:, 0], curve[:, 1])

index, curve = 1, data[1]
y_axis = [u'TT']
tex_names = ['TT']
x_axis = 'l'
ylim = []
xlim = []
ax.semilogx(curve[:, 0], curve[:, 1])

ax.legend([root+': '+elem for (root, elem) in
    itertools.product(roots, y_axis)], loc='best')

ax.set_xlabel('$\ell$', fontsize=16)
# plt.savefig('testcl.png', dpi=300)
plt.show()
# %%
