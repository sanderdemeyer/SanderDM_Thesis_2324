import scipy.integrate as integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
import csv
Z_over_a = 1
v = 0.0

mass = 0.0
delta_g = 0.0
trunc = 3.5
ramping = 5
dt = 0.001

#file = "Thirring_time-evolution_uniform_adiabatic_m_0.3_delta_g_0.0_new_mass_sweep_slow_fidelity"
#file = "Thirring_time-evolution_uniform_adiabatic_m_0.3_delta_g_0.0_new_mass_sweep_fast_fidelity"
#file = "Thirring_time-evolution_uniform_adiabatic_m_0.3_delta_g_0.0_new_mass_sweep_fast_long_higher_fidelity"
#file = "Thirring_time-evolution_uniform_adiabatic_m_0.3_delta_g_0.0_new_mass_sweep_fast_long_40000_higher_fidelity"
#file = "Thirring_time-evolution_uniform_adiabatic_m_0.3_delta_g_-0.3_new_mass_sweep_slow_10000_higher_fidelity"

#f = h5py.File("Thirring_time-evolution_uniform_adiabatic_m_0.3_delta_g_0.0_mass_sweep_faster", "r")
#f = h5py.File("Thirring_time-evolution_uniform_adiabatic_m_0.3_delta_g_0.0_new_mass_sweep", "r")
#f = h5py.File("Thirring_time-evolution_uniform_adiabatic_m_0.3_delta_g_0.0_new_mass_sweep_slow", "r")


#file = "v_sweep_m_0.6_delta_g_-0.3_ramping_10_dt_0.005_max_1.1_trunc_2.5"
#file = "v_sweep_m_0.2_delta_g_-0.3_ramping_10_dt_0.005_max_1.1_trunc_2.5"

file = f"v_sweep_m_{mass}_delta_g_{delta_g}_ramping_{ramping}_dt_{dt}_max_1.1_trunc_{trunc}"

f = h5py.File(file)

energies = f["energies"][:]
energies = [e[0] for e in energies]

true_energies = f["true_energies"][:]
true_energies = [e[0] for e in true_energies[:-1]]

true_energies_global = f["true_energies_global"][:]
true_energies_global = [e[0] for e in true_energies_global[:-1]]

print(true_energies_global)

entropies = f["entropies"][:]
entropies = [e[0] for e in entropies]

fidelities = f["fidelities"][:]
fidelities = [e[0] for e in fidelities]


length = len(energies)
print(f"length is {length}")

def energy_analytical(k, m_0):
    return -1/(4*np.pi)*np.sqrt(m_0**2 + (Z_over_a*np.sin(k/2))**2)


print(np.shape(energies))
print(np.shape(true_energies))
print(true_energies[-1])
RAMPING_TIME = 100
SIZE = 20

#LINEAR_SLOPE = 25
LINEAR_SLOPE = 10

#def f(j):
#    return (0.9*np.tanh((j-3*RAMPING_TIME)/RAMPING_TIME) + 1.0)/10 + 0.2

def f(j):
    t = j*0.001
    return min(0.4, 0.2+t/LINEAR_SLOPE)

def f_mass(j):
    t = j*0.001
    return min(0.4, 0.2+t/LINEAR_SLOPE)

def f_v(t):
    #return min(0.4, 0.2+t/10) # Fast
    return min(1.1, t/15)


instantaneous_E_gs = np.zeros(length)
for j in range(length):
    instantaneous_E_gs[j] = integrate.quad(lambda k: energy_analytical(k, f(j)), -np.pi, np.pi)[0]

print('hello')

plt.plot(np.array(range(length))*0.001, energies, label = r'$<E>(t)$')
plt.plot(np.array(range(len(true_energies)))*0.1, true_energies, label = r'$<E_{gs}>$ - Local')
plt.plot(np.array(range(len(true_energies)))*0.1, true_energies_global, label = r'$<E_{gs}>$ - Global')
#plt.plot(np.array(range(length))*0.001, [f_v(j*0.001) for j in range(length)], label = 'v')
#plt.plot(np.array(range(length))*0.001, instantaneous_E_gs, label = r"Instantaneous ground state energy of $H(t)$")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title(fr"m = {mass}, $\Delta(g)$ = {delta_g}, trunc = {trunc}")
plt.show()


plt.plot(np.array(range(length))*0.001, entropies)
plt.legend()
plt.xlabel("Time")
plt.ylabel(r"Entropy $S$")
plt.title(fr"m = {mass}, $\Delta(g)$ = {delta_g}, trunc = {trunc}")
plt.show()


plt.scatter(np.array(range(length//100))*0.1, [abs(e) for e in fidelities[:-1]])
plt.legend()
plt.xlabel("Time")
plt.ylabel(r"Fidelity $\left<\psi|\psi_{gs}\right>$")
plt.title(fr"m = {mass}, $\Delta(g)$ = {delta_g}, trunc = {trunc}")
plt.show()


plt.scatter(np.array(range(length//100))*0.1, [np.log10(1-abs(e)) for e in fidelities[:-1]])
plt.legend()
plt.xlabel("Time")
plt.ylabel(r"$\log_{10}{(1-\text{Fidelity})}$")
plt.title(fr"m = {mass}, $\Delta(g)$ = {delta_g}, trunc = {trunc}")
plt.show()


