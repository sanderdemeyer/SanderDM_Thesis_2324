import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv

def linear_fit(x, a, b):
    return a*x+b

masses = [0.0, 0.075, 0.15, 0.225, 0.3]
delta_gs = [-1.0, -0.75, -0.5, -0.25, 0.0]

trunc_list = np.array([1 + 0.5*i for i in range(8)])

masses_plot = []
delta_gs_plot = []
inverse_xi = []

for (i,mass) in enumerate(masses):
    for (j,delta_g) in enumerate(delta_gs):
        file = f"Phase_diagram_correlation_lengths_m_{mass}_delta_g_{delta_g}_v_0.0"
        f = h5py.File(file, "r")
        Energies = f["Energies"][:]
        Corr_lengths = f["Corr_lengths"][:]
        # trunc_list = f["trunc_list"][:][:]
        bond_dims = f["Bond_dims"][:]

        bond_dims = np.transpose(bond_dims)[0]
        Corr_lengths = np.transpose(Corr_lengths)[0]
        x_data = (1/bond_dims[-4:])
        y_data = (1/Corr_lengths[-4:])
        popt, pcov = opt.curve_fit(linear_fit, x_data, y_data)
        (a, b) = popt
        print(f"Extrapolation for mass = {mass} and delta_g = {delta_g} gives 1/xi = {b}")

        masses_plot.append(mass)
        delta_gs_plot.append(delta_g)
        inverse_xi.append(b)
        #plt.scatter(1/bond_dims, (1/Corr_lengths))
        #plt.xlabel("log10(Schmidt cut)")
        #plt.ylabel(r'log10(1/$\xi$)')
        #plt.title(f"Extrapolation for mass = {mass} and delta_g = {delta_g}")
        #plt.show()

c = ['red' if e < 0.05 else 'blue' for e in inverse_xi]

plt.scatter(delta_gs_plot, masses_plot, c = c)
plt.xlabel(r"$\Delta(g)$")
plt.ylabel('mass')
plt.show()


print(alle)

file = "Phase_diagram_correlation_lengths_m_0_g_0"
file = "Phase_diagram_correlation_lengths_m_0.075_delta_g_-0.75_v_0.0"
f = h5py.File(file, "r")
Energies = f["Energies"][:]
Corr_lengths = f["Corr_lengths"][:]
trunc_list = f["trunc_list"][:][:]
bond_dims = f["Bond_dims"][:]


print(trunc_list)
print(Energies)
print(Corr_lengths)
print(bond_dims)
  
plt.scatter(1/bond_dims, (1/Corr_lengths))
plt.xlabel("log10(Schmidt cut)")
plt.ylabel(r'log10(1/$\xi$)')
plt.show()
