import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv

delta_g = 0
mass = 0.5
v = 0.9

k_refined = np.linspace(-np.pi, np.pi, 1000)

exp = (v/2)*np.sin(2*k_refined) + np.sqrt(mass**2 + np.sin(k_refined)**2)
# Factor 2 door staggering?

for v in np.linspace(-2, 0, 5):
    exp = (v/2)*np.sin(2*k_refined) + np.sqrt(mass**2 + np.sin(k_refined)**2)
    plt.plot(k_refined, exp, label = f'analytical, v = {v}')
plt.xlabel('k')
plt.ylabel('energy')
plt.legend()
plt.title(r"Dispersion relation for $m = 0.5$ and $v = 0.9$")
plt.show()
