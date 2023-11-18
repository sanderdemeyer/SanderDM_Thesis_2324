import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv

"""
def fit(x, a, b, c):
    return c - np.sqrt(a+b*x**2)

masses = [0.5, 0.405, 0.241, 0]
masses_refined = np.linspace(0, 0.5, 1000)
gs_energies = -0.4 + (0.2/10.9)*np.array([-0.8, 0.75, 3, 4.7])

popt, pcov = opt.curve_fit(fit, masses, gs_energies)
print(popt)
a = popt[0]
b = popt[1]
c = popt[2]
print(a)
print(b)
print(c)

plt.scatter(masses, gs_energies, label = 'data')
plt.plot(masses_refined, fit(masses_refined, a, b, c), label = 'fit')
plt.legend()
plt.show()
"""
file = "testjeee"
file = "Dispersion Relation/2023_10_24_dispersion_relation_small_k_values"
f = h5py.File(file, "r")
energies = f["energies"][:]

#k = f["k_values"]
print("fhefjqkfmljdskfhere")
energies = [np.real(e[0]) for e in energies[0,:]]
print(energies)
# k = np.linspace(-np.pi, np.pi, 25)
k = np.linspace(-np.pi/6,np.pi/6,30)
k_refined = np.linspace(-np.pi/6, np.pi/6, 1000)

exp = np.sqrt(1/4 + np.sin(k_refined)**2)
# Factor 2 door staggering?

plt.scatter(k, energies, label = 'quasiparticle')
plt.plot(k_refined, exp, label = 'expected')
plt.xlabel('k')
plt.ylabel('energy')
plt.legend()
plt.show()
