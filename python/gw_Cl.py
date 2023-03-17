#%%
import numpy as np
import matplotlib.pyplot as plt
from classy import Class

cosmo = Class()
cosmo.set()
cosmo.compute()
cosmo.struct_cleanup()