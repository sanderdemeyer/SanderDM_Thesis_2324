import scipy.integrate as integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import csv
from matplotlib.animation import FuncAnimation
from scipy.signal import argrelextrema
import scipy.optimize as opt

def gaussian(x, x0, sigma, A, E0):
    return E0 + A/(np.sqrt(2*np.pi)*sigma)*np.exp(-(((x-x0)/sigma)**2)/2)

def avged(arr):
    return ([(arr[2*i]+arr[2*i+1])/2 for i in range(len(arr)//2)])

def linear(t, a, b):
    return a*t+b

truncation = 1.5
mass = 0.3
Delta_g = 0.0
v = 0.0

N = 40
dt = 0.8
t_end = 4.0
k = -1.5
sigma = 0.178

left_moving = False

if left_moving:
    file = f"SanderDM_Thesis_2324/test_wavepacket_left_moving_gs_mps_wo_symmetries_trunc_{truncation}_mass_{mass}_v_{v}_Delta_g_{Delta_g}_N_{N}_k_{k}_sigma_{sigma}_dt_{dt}_tend_{t_end}_dict.h5"
else:
    file = f"SanderDM_Thesis_2324/test_wavepacket_gs_mps_wo_symmetries_trunc_{truncation}_mass_{mass}_v_{v}_Delta_g_{Delta_g}_N_{N}_k_{k}_sigma_{sigma}_dt_{dt}_tend_{t_end}_dict.h5"
    file = f"SanderDM_Thesis_2324/test_wavepacket_right_moving_gs_mps_wo_symmetries_trunc_{truncation}_mass_{mass}_v_{v}_Delta_g_{Delta_g}_N_{N}_k_{k}_sigma_{sigma}_dt_{dt}_tend_{t_end}_dict.h5"


# file = h5py.File(file)

Es = []
with h5py.File(file, 'r') as f:
    # Read data
    for i in range(1,6):
        key1_data = f[f"Es{i}"][:]
        Es.append(avged([e[0] for e in key1_data]))

times, L = np.shape(Es)
X = range(L)

excluded = 5

b = [1, 2, 3, 4]


for i in range(times):
    plt.plot([2*x for x in X[excluded:L-1-excluded]], Es[i][excluded:L-1-excluded], label = f"t = {i}")
# plt.plot(X, [gaussian(x, 22, 3, 0.25, -0.36) for x in X])
plt.xlabel("Position i", fontsize = 15)
plt.ylabel("Averaged energy", fontsize = 15)
plt.legend()
plt.show()

optimal_values = []
for i in range(times):
    popt, pcov = curve_fit(gaussian, X[excluded:L-1-excluded], Es[i][excluded:L-1-excluded], p0 = [22, 3, 0.25, -0.36])
    optimal_values.append(popt)
optimal_values = np.array(optimal_values)
print(optimal_values)

# for i in range(times):
#     plt.plot(X[excluded:L-1-excluded], Es[i][excluded:L-1-excluded], label = f"data")
#     plt.plot(X[excluded:L-1-excluded], [gaussian(x, optimal_values[i][0], optimal_values[i][1], optimal_values[i][2], optimal_values[i][3]) for x in X[excluded:L-1-excluded]], label = 'fit')
#     plt.title(f"time = {i}")
#     plt.show()

popt, pcov = curve_fit(linear, [t_end*i for i in range(times)], [optimal_values[i][0] for i in range(times)])
v_data = popt[0]
print(f"v from data = {v_data} +- {pcov[0,0]}")

E = np.sqrt(mass**2 + np.sin(k/2)**2)
v_expected = np.sin(k)/(4*E)

print(f"v expected from theory is {v_expected}")
print(f"Relative error is {abs(v_expected-v_data)/v_expected}")

plt.scatter([t_end*i for i in range(times)], [2*optimal_values[i][0] for i in range(times)], label = "data")
plt.plot([t_end*i for i in range(times)], [2*linear(x, popt[0], popt[1]) for x in [t_end*i for i in range(times)]], label = "fit", color = "black")
plt.xlabel("Time", fontsize = 15)
plt.ylabel("Peak position", fontsize = 15)
plt.legend()
plt.show()

plt.plot([t_end*i for i in range(times)], [optimal_values[i][1] for i in range(times)])
plt.title("sigma")
plt.show()


plt.plot([t_end*i for i in range(times)], [optimal_values[i][2] for i in range(times)])
plt.title("Amplitude")
plt.show()

plt.plot([t_end*i for i in range(times)], [optimal_values[i][3] for i in range(times)])
plt.title("E0")
plt.show()
