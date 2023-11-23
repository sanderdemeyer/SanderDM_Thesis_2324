import scipy.integrate as integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
import csv
Z_over_a = 1
v = 0.0

f = h5py.File("Thirring_time-evolution_uniform_adiabatic_m_0.3_delta_g_0.0_mass_sweep_faster", "r")
f = h5py.File("Thirring_time-evolution_uniform_adiabatic_m_0.3_delta_g_0.0_new_mass_sweep", "r")
f = h5py.File("Thirring_time-evolution_uniform_adiabatic_m_0.3_delta_g_0.0_new_mass_sweep_slow", "r")
energies = f["energies"][:]
energies = [e[0] for e in energies]

length = len(energies)

def energy_analytical(k, m_0):
    return -1/(4*np.pi)*np.sqrt(m_0**2 + (Z_over_a*np.sin(k/2))**2)

RAMPING_TIME = 100
SIZE = 20

#def f(j):
#    return (0.9*np.tanh((j-3*RAMPING_TIME)/RAMPING_TIME) + 1.0)/10 + 0.2

def f(j):
    t = j*0.001
    return min(0.4, 0.2+t/20)

instantaneous_E_gs = np.zeros(length)
for j in range(length):
    instantaneous_E_gs[j] = integrate.quad(lambda k: energy_analytical(k, f(j)), -np.pi, np.pi)[0]

print('hello')

plt.plot(np.array(range(length))*0.001, energies, label = r'$<E_{gs}>$')
plt.plot(np.array(range(length))*0.001, instantaneous_E_gs, label = r"Instantaneous ground state energy of $H(t)$")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Energy")
plt.show()


print((energies))
print((instantaneous_E_gs))
plt.plot(range(length), energies, label = r'$<E_{gs}>$')
#plt.plot(range(length), instantaneous_E_gs, label = r"Instantaneous ground state energy of $H(t)$")
plt.legend()
plt.show()

print(energies[0])
print(instantaneous_E_gs[0])
print(energies[-1])
print(instantaneous_E_gs[-1])
