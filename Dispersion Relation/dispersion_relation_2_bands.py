import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv

delta_g = 0.0
mass = 0.3
v = 0.0

k = np.linspace(-np.pi/6,np.pi/6,11)
k_pos = np.linspace(0, np.pi/2, 1000)
k_neg = np.linspace(-np.pi/2, 0, 1000)


exp_00 = (v/2)*np.sin(2*k_pos) + np.sqrt(mass**2 + np.sin(k_pos)**2)
exp_10 = (v/2)*np.sin(2*k_pos) - np.sqrt(mass**2 + np.sin(k_pos)**2)
exp_11 = (v/2)*np.sin(2*k_neg) + np.sqrt(mass**2 + np.sin(k_neg)**2)
exp_01 = (v/2)*np.sin(2*k_neg) - np.sqrt(mass**2 + np.sin(k_neg)**2)

# plt.plot(k_pos, exp_00, linestyle='dotted', color = 'blue', label = 'right moving')
# plt.plot(k_pos, exp_10, color = 'blue', label = 'left moving')
# plt.plot(k_neg, exp_01, linestyle='dotted', color = 'blue')
# plt.plot(k_neg, exp_11, color = 'blue')
plt.plot(k_pos, exp_00, color = 'blue')
plt.plot(k_pos, exp_10, color = 'blue')
plt.plot(k_neg, exp_01, color = 'blue')
plt.plot(k_neg, exp_11, color = 'blue')
plt.xlabel('k', fontsize = 15)
plt.ylabel('energy', fontsize = 15)
plt.legend()
plt.title(fr"Dispersion relation for $m = {mass}$, $v = {v}$, and $\Delta(g)={delta_g}$")
plt.show()
