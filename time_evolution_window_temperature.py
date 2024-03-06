import scipy.integrate as integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
import csv
from matplotlib.animation import FuncAnimation
from scipy.signal import argrelextrema
import scipy.optimize as opt

N = 70
am_tilde_0 = 0.3
Delta_g = 0.0
RAMPING_TIME = 5
dt = 0.01
v_max = 1.5
spatial_sweep = 10
truncation = 8.0
nr_steps = 1500
kappa = 0.5
frequency_of_saving = 5

file = f"SanderDM_Thesis_2324/window_time_evolution_v_sweep_N_{N}_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_nrsteps_{nr_steps}_vmax_{v_max}_kappa_{kappa}_trunc_{truncation}_savefrequency_{frequency_of_saving}"
file = h5py.File(file)

MPSs = file['MPSs']
Es_complex = file['Es']
occ_numbers_complex = file['occ_numbers']
occ_numbers_20_complex = file['occ_numbers_20']

L, datapoints = np.shape(Es_complex)
N = L//2-1


Es = np.zeros((datapoints, L))
occ_numbers = np.zeros((datapoints, N))
occ_numbers_20 = np.zeros((datapoints, N))

for n in range(L):
    for n_t in range(datapoints):
        Es[n_t, n] = Es_complex[n, n_t][0]

for n_t in range(datapoints):
    if Es[n_t,0] == 0.0:
        datapoints = n_t
        break

print(f'datapoints = {datapoints}')


print(np.shape(occ_numbers_complex))

for n in range(N):
    for n_t in range(datapoints):
        occ_numbers[n_t, n] = occ_numbers_complex[n, n_t]
        occ_numbers_20[n_t, n] = occ_numbers_20_complex[n, n_t]


Es_average = [[(Es[n_t][2*j] + Es[n_t][2*j+1])/2 for j in range(1,N)] for n_t in range(datapoints)]

plt.plot(range(N), occ_numbers_20[0,:], label = 't = 0')
plt.show()

plt.plot(range(N), occ_numbers_20[-1,:], label = 't = 15')
plt.show()

dilution = 30
if True:
    for n_t in range(datapoints//dilution):
        print(n_t)
        plt.plot(range(N), occ_numbers_20[n_t*dilution], label = f'n_t = {n_t}')
plt.legend()
plt.title(f't = {n_t*dt}')
plt.show()


dilution = 30
if True:
    for n_t in range(datapoints//dilution):
        plt.plot([f'{2*j+1}&{2*j+2}' for j in range(1,N)], Es_average[n_t*dilution], label = f'n_t = {n_t}')

plt.legend()
plt.title(f't = {n_t*dt}')
plt.show()


