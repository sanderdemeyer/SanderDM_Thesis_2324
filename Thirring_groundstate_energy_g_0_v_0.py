import scipy.integrate as integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
import csv

# f = h5py.File("SanderDM_Thesis_2324/Thirring_groundstate_energy_g_0_v_0", "r")
f = h5py.File("Thirring_groundstate_energy_g_0_v_0", "r")
m_range = 20
masses = np.linspace(0,1,m_range)
energies = f["energies"][:]
energies = [(energies[0,i][0]+energies[1,i][0])/2 for i in range(m_range)]
# masses = f["masses"]


print(energies)
print('fhqf')
print(masses)
print("done")
m = 0
Z_over_a = 1

def f(k, m_0):
    return -1/(4*np.pi)*np.sqrt(m_0**2 + (Z_over_a*np.sin(k/2))**2)

def test(x):
    return x

print('started')
print(integrate.quad(test, 0, 5))

m_values = np.linspace(0,1,m_range*10)
E_gs_values = np.zeros(len(m_values))

for i, m in enumerate(m_values):
    print(i)
    print(m)
    E_gs_values[i] = integrate.quad(lambda k: f(k,m), -np.pi, np.pi)[0]

print(m_values)
print(E_gs_values)

plt.scatter(masses, energies, label = 'VUMPS')
plt.plot(m_values, E_gs_values, label = 'analytical expression')
plt.xlabel('mass', fontsize = 15)
plt.ylabel(r'$ E_{gs} $', fontsize = 15)
plt.title(r"Ground state energy density for $ g = 0 $, $ v = 0 $")
plt.legend()
plt.show()