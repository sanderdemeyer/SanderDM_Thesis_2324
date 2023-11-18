import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv

delta_g = 0.0
mass = 0.5
v = 0.9

file = "testjeee"
file = "2023_10_24_dispersion_relation_small_k_values"
file = "2023_10_30_dispersion_relation_small_k_values_m_0p5_v_0p1"
file = "2023_10_30_dispersion_relation_small_k_values_m_0p5_v_0p9"
file = "2023_10_30_dispersion_relation_small_k_values_m_0p5_v_0p9_symmetric"
file = "2023_10_24_dispersion_relation_small_k_values_m_0p5_v_0p9_symmetric"
file = "asymetric_2023_10_24_dispersion_relation_small_k_values"
file = "2023_10_24_dispersion_relation_small_k_values_m_0p5_v_0_symmetric"
file = "asymmetric_2023_10_24_dispersion_relation_small_k_values_m_0p5_v_0p9"
file = "2023_10_24_dispersion_relation_small_k_values_m_0p5_v_0_symmetric_better"
#file = "2023_10_24_dispersion_relation_small_k_values_m_0_v_0p5_symmetric_better"
file = "2023_11_05_dispersion_relation_large_k_values_m_0p5_v_0_dynamic"
file = "2023_10_24_dispersion_relation_small_k_values_m_0p5_v_0_symmetric_sector_1o2"
file = "2023_10_24_dispersion_relation_small_k_values_m_0p5_v_0_symmetric_sector_1"
file = "2023_10_24_dispersion_relation_small_k_values_m_0p5_v_0p1_symmetric_sector_1"
file = "Dispersion Relation/2023_10_24_dispersion_relation_small_k_values_m_0p5_v_0p9_symmetric_sector_1"

f = h5py.File(file, "r")
energies = f["energies"][:]

#k = f["k_values"]
print("fhefjqkfmljdskfhere")
energies = [np.real(e[0]) for e in energies[0,:]]
print(energies)
# k = np.linspace(-np.pi, np.pi, 25)
k = np.linspace(-np.pi/6,np.pi/6,11)
k_refined = np.linspace(-np.pi/6, np.pi/6, 1000)

k = np.linspace(-np.pi/2,np.pi/2,17)
k_refined = np.linspace(-np.pi/2, np.pi/2, 1000)

exp = (v/2)*np.sin(2*k_refined) + np.sqrt(mass**2 + np.sin(k_refined)**2)
# Factor 2 door staggering?

plt.scatter(k, np.array(energies), label = 'quasiparticle')
plt.plot(-k_refined, exp, label = 'analytical')
plt.xlabel('k')
plt.ylabel('energy')
plt.legend()
plt.title(r"Dispersion relation for $m = 0.5$ and $v = 0.9$, sector $U1(1)$")
plt.show()
