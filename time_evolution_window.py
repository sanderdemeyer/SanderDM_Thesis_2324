import scipy.integrate as integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
import csv
from matplotlib.animation import FuncAnimation

am_tilde_0 = 0.0
Delta_g = 0.0
RAMPING_TIME = 1
dt = 0.01
v_max = 1.0
spatial_sweep = 10
truncation = 2.5

file = f"window_time_evolution_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_vmax_{v_max}_spatialsweep_{spatial_sweep}_trunc_{truncation}"
file = f"window_time_evolution_mass_sweep_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_vmax_{v_max}_spatialsweep_{spatial_sweep}_trunc_{truncation}"

f = h5py.File(file)

energies = f["energies"]
number_of_timesteps, N = np.shape(energies)
print(f'number of timesteps is {number_of_timesteps}')
print(f'N is {N}')
energies = np.array([[(energies[i,j])[0] for j in range(N)] for i in range(number_of_timesteps)])

if False:
    for j in range(N):
        plt.plot(range(number_of_timesteps), energies[:,j])
        plt.title(f'pos j = {j}')
        plt.show()

figure, ax = plt.subplots()

def animation_function(i):
    ax.clear()
    ax.plot(range(N), energies[i,:])
    plt.title(f't = {dt*i}, mass = {min(v_max, dt*i/RAMPING_TIME)}')
    return ax

print(np.mean(energies))
print(energies[:,5])

animation = FuncAnimation(figure,func = animation_function,frames = range(number_of_timesteps))
plt.show()
