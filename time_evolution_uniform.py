import scipy.integrate as integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
import csv

Delta_g = 0.0
v = 0.0
m0 = 0.1
m_end = 0.4
D = 50

RAMPING_TIME = 5
dt = 0.1
t_end = (m_end-m0)*RAMPING_TIME*1.5

truncation = 2.5

D = 50

file =  f"mass_time_evolution_dt_{dt}_D_{D}_trunc_{truncation}_mass_{m0}_to_{m_end}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_tend_{t_end}_v_{v}"

f = h5py.File(file)   
energies = np.asarray(f["Es"][:])
energies_te = f["Es_time_evolve"][:]
energies_local = f["Es_local"][:]
fidelities = f["fidelities"][:]

print(energies)
energies = [np.asarray(e) for e in energies]

print(energies)
