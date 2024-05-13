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

mass = 1.0
Delta_g = -0.3
v = 0.5
truncation = 2.5

file = "testjeee"
file = "Dispersion Relation/2023_10_24_dispersion_relation_small_k_values"
file = f"SanderDM_Thesis_2324/Dispersion_Delta_m_{mass}_delta_g_{Delta_g}_v_{v}_trunc_{truncation}_all_sectors_newv"
f = h5py.File(file, "r")
energies = f["energies"][:]
anti_energies = f["anti_energies"][:]
with h5py.File(file, 'r') as file:
    gs_energy = file["gs_energy"][()]
    print(gs_energy.item())


#k = f["k_values"]
energies = [np.real(e[0])+Delta_g for e in energies[0,:]]
anti_energies = [np.real(e[0])-Delta_g for e in anti_energies[0,:]]
print(energies)
# k = np.linspace(-np.pi, np.pi, 25)
k = np.linspace(-np.pi/2,np.pi/2,35)
k_refined = np.linspace(-np.pi/2, np.pi/2, 1000)

exp = v*np.sin(k_refined*2) + np.sqrt(mass**2 + np.sin(k_refined)**2)
# Factor 2 door staggering?

plot = True
if plot:
    plt.scatter(-2*k, energies, label = 'quasiparticle')
    # plt.scatter(k, anti_energies, label = 'quasiparticle')
    plt.plot(2*k_refined, exp, label = 'analytical - free theory')
    plt.xlabel(r'momentum $k$', fontsize=15)
    plt.ylabel('energy', fontsize = 15)
    plt.legend()
    plt.show()

mus = [(e2-e1)/2 for (e1,e2) in zip(energies, anti_energies)]

gs_energy_sum = sum(energies)/35
print(f"gs_energy should be {gs_energy_sum}")