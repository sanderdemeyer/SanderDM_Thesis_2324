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
truncation = 7.0
nr_steps = 1500
kappa = 0.5
frequency_of_saving = 50

file = f"window_time_evolution_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_vmax_{v_max}_spatialsweep_{spatial_sweep}_trunc_{truncation}"
file = f"window_time_evolution_mass_sweep_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_vmax_{v_max}_spatialsweep_{spatial_sweep}_trunc_{truncation}"
file = f"SanderDM_Thesis_2324/window_time_evolution_v_sweep_N_{N}_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_nrsteps_{nr_steps}_vmax_{v_max}_kappa_{kappa}_trunc_{truncation}_savefrequency_{frequency_of_saving}"
file = f"SanderDM_Thesis_2324/window_time_evolution_v_sweep_N_{N}_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_nrsteps_{nr_steps}_vmax_{v_max}_kappa_{kappa}_trunc_{truncation}_savefrequency_{frequency_of_saving}"
f = h5py.File(file)

with h5py.File(file, 'r') as file:
    # Access the dataset
    dataset = file['Es']
    print(dataset)
    # Create a list to store the vectors
    Es = []

    for ref in dataset[:-1]:
        # Use the 'value' attribute to access the data referenced by the object
        vector = file[ref]
        # new_vector = [item[0] for item in vector]
        vector = vector[()]
        print(vector)
        new_vector = [item for item in vector]
        new_vector = vector
        print(vector)
        # Convert complex numbers to Python complex numbers
        Es.append(new_vector)

timesteps, N_new = np.shape(Es)

assert N == N_new, "N is not consistent"

Es_average = [[Es[n_t]] for n_t in range(timesteps)]

Es_average = [[(Es[n_t][2*j] + Es[n_t][2*j+1])/2 for j in range(1,N//2-1)] for n_t in range(timesteps)]

dilution = 1
if True:
    for n_t in range(timesteps//dilution):
        plt.plot([f'{2*j+1}&{2*j+2}' for j in range(1,N//2-1)], Es_average[n_t*dilution], label = f'n_t = {n_t}')

plt.legend()
plt.title(f't = {n_t*dt}')
plt.show()

wave_arrivals = np.zeros(N)
plot = True

for position in range(N):
    wave = np.array([Es[n_t][position] for n_t in range(timesteps)])
    maxima = argrelextrema(wave, np.less)
    print(maxima)
    if len(maxima[0] > 0):
        print(maxima[0][0])
        wave_arrivals[position] = maxima[0][0]

    if (plot) and (position == 33):
        plt.plot(range(timesteps), wave)
        plt.xlabel('Timestep')
        plt.ylabel('Energy')
        plt.title(f'For i = {position}')
        plt.show()


start = 33
end = 42

plt.plot(range(N), wave_arrivals)
plt.xlabel('Position')
plt.ylabel('Time of first wave arrival')
plt.show()

def linear(x, a, b):
    return a*x+b

popt, pcov = opt.curve_fit(linear, range(start,end+1), wave_arrivals[start:end+1]*frequency_of_saving*dt)

print(f"slope = {popt[0]}")
print(f"starting point = {popt[1]}")
print(f'Renormalization to c = {popt[0]*2}')

print(wave_arrivals)
plt.plot(range(start,end+1), wave_arrivals[start:end+1]*frequency_of_saving*dt)
plt.plot(range(start,end+1), [(i - N*2//3)*popt[0] for i in range(start,end+1)], label = fr'Expected for $c = {-round(popt[0]*2,3)}$')
plt.xlabel('Position')
plt.ylabel('Time of first wave arrival')
plt.legend()
plt.title('Renormalization of c by means of first wave arrival')
plt.show()

print(a)


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
