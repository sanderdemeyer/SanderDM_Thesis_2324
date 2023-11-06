import scipy.integrate as integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
import csv

# f = h5py.File("SanderDM_Thesis_2324/Thirring_groundstate_energy_g_0_v_0", "r")
f = h5py.File("Thirring_groundstate_energy_g_0_v_0", "r")
f30 = h5py.File("Check_mass_symmetric_D_30", "r")
f50 = h5py.File("Check_mass_symmetric_D_50", "r")
f50_more = h5py.File("Check_mass_symmetric_D_50_more_spaces", "r")
f_dynamic_2 = h5py.File("Check_mass_term_symmetric_trunc_2", "r")
f_dynamic_4 = h5py.File("Check_mass_term_symmetric_trunc_4", "r")
m_range = 20
masses = np.linspace(0,1,m_range)
energies = f["energies"][:]
energies_nonsym = [(energies[0,i][0]+energies[1,i][0])/2 for i in range(m_range)]

energies30 = f30["Energies"][:]
energies50 = f50["Energies"][:]
energies50_more = f50_more["Energies"][:]
energies_dynamic_2 = f_dynamic_2["Energies"][:]
energies_dynamic_4 = f_dynamic_4["Energies"][:]


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

m_values = np.linspace(0,1,m_range)
E_gs_values = np.zeros(len(m_values))

for i, m in enumerate(m_values):
    print(i)
    print(m)
    E_gs_values[i] = integrate.quad(lambda k: f(k,m), -np.pi, np.pi)[0]

print(m_values)
print(E_gs_values)

plt.scatter(masses, energies30, label = 'VUMPS, D = 30')
plt.scatter(masses, energies50, label = 'VUMPS, D = 50')
plt.scatter(masses, energies50, label = 'VUMPS, D = 50')
plt.scatter(masses, energies50_more, label = 'VUMPS, D = 50, more spaces')
plt.scatter(masses, energies_nonsym, label = 'VUMPS, not symmetric')
plt.scatter(masses, energies_dynamic_2, label = 'VUMPS, dynamic trunc 2')
plt.scatter(masses, energies_dynamic_4, label = 'VUMPS, dynamic trunc 4')
plt.plot(m_values, E_gs_values, label = 'analytical expression')
plt.xlabel('mass', fontsize = 15)
plt.ylabel(r'$ E_{gs} $', fontsize = 15)
plt.title(r"Ground state energy density for $ g = 0 $, $ v = 0 $")
plt.legend()
plt.show()


print(np.shape(energies30))
print(np.shape(energies50))
print(np.shape(energies50_more))
print(np.shape(energies_nonsym))
print(np.shape(E_gs_values))

#plt.scatter(masses, [abs(energies30[i] - e)/e for (i,e) in enumerate(E_gs_values)], label = 'VUMPS, D = 30')
#plt.scatter(masses, [abs(energies50[i] - e)/e for (i,e) in enumerate(E_gs_values)], label = 'VUMPS, D = 50')
plt.scatter(masses, [abs(energies50_more[i] - e)/e for (i,e) in enumerate(E_gs_values)], label = 'VUMPS, D = 50, more spaces')
plt.scatter(masses, [abs(energies_nonsym[i] - e)/e for (i,e) in enumerate(E_gs_values)], label = 'VUMPS, not symmetric')
plt.xlabel('mass', fontsize = 15)
plt.ylabel(r'$ error on E_{gs} $', fontsize = 15)
plt.title(r"Ground state energy density for $ g = 0 $, $ v = 0 $")
plt.legend()
plt.show()