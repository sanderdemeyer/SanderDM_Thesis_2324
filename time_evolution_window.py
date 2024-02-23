import scipy.integrate as integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
import csv
from matplotlib.animation import FuncAnimation

N = 20
am_tilde_0 = 1.0
Delta_g = 0.0
RAMPING_TIME = 5
dt = 0.1
v_max = 2.0
spatial_sweep = 10
truncation = 1.5
nr_steps = 1
kappa = 1.0
frequency_of_saving = 1

file = f"window_time_evolution_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_vmax_{v_max}_spatialsweep_{spatial_sweep}_trunc_{truncation}"
file = f"window_time_evolution_mass_sweep_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_vmax_{v_max}_spatialsweep_{spatial_sweep}_trunc_{truncation}"
file = f"window_time_evolution_v_sweep_N_{N}_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_vmax_{v_max}_kappa_{kappa}_trunc_{truncation}_savefrequency_{frequency_of_saving}"
f = h5py.File(file)


with h5py.File(file, 'r') as file:
    # Access the dataset
    dataset = file['Es']

    # Create a list to store the vectors
    Es = []

    for ref in dataset[:-1]:
        # Use the 'value' attribute to access the data referenced by the object
        vector = file[ref]
        new_vector = [item[0] for item in vector]

        # Convert complex numbers to Python complex numbers
        Es.append(new_vector)

timesteps, N_new = np.shape(Es)

assert N == N_new, "N is not consistent"

Es_average = [[Es[n_t]] for n_t in range(timesteps)]

Es_average = [[(Es[n_t][2*j] + Es[n_t][2*j+1])/2 for j in range(1,N//2-1)] for n_t in range(timesteps)]

if True:
    for n_t in range(timesteps):
        plt.plot([f'{2*j+1}&{2*j+2}' for j in range(1,N//2-1)], Es_average[n_t], label = f'n_t = {n_t}')

plt.legend()
plt.title(f't = {n_t*dt}')
plt.show()


# Set x and y range for all figures
y_range = (-0.6088, -0.6076)  # Replace with the desired y-axis range


def update(frame):
    plt.clf()
    
    plt.plot([f'{2*j+1}&{2*j+2}' for j in range(1, N//2-1)], Es_average[frame], label=f'n_t = {frame}')
    plt.ylim(y_range)
    
    plt.legend()
    t = frame * dt * frequency_of_saving
    v = min(v_max, t/RAMPING_TIME)

    plt.title(f't = {t}, v = {v}')

# Create a figure
fig, ax = plt.subplots()

# Create the animation
animation = FuncAnimation(fig, update, frames=timesteps, interval=200, repeat=False)

# Display the animation
plt.show()


if False:
    for n_t in range(timesteps):
        plt.plot(range(N), Es[n_t])
        plt.title(f't = {n_t*dt}')
        plt.show()


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
