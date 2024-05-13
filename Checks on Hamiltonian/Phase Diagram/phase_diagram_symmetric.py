import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv
from matplotlib.colors import LinearSegmentedColormap

def linear_fit(x, a, b):
    return a*x+b

masses = [0.0, 0.075, 0.15, 0.225, 0.3]
delta_gs = [-1.0, -0.75, -0.5, -0.25, 0.0]

trunc_list = np.array([1 + 0.5*i for i in range(8)])

masses_plot = []
delta_gs_plot = []
inverse_xi = []

color_values = np.zeros((len(delta_gs), len(masses)))
for (i,mass) in enumerate(masses):
    for (j,delta_g) in enumerate(delta_gs):
        file = f"Phase Diagram/Phase_diagram_correlation_lengths_m_{mass}_delta_g_{delta_g}_v_0.0"
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
        perr = np.sqrt(np.diag(pcov))
        # print(f"Extrapolation for mass = {mass} and delta_g = {delta_g} gives 1/xi = {b}")
        # print(f"Here, the std is given by {perr[1]}")
        masses_plot.append(mass)
        delta_gs_plot.append(delta_g)
        inverse_xi.append(b)
        # plt.scatter(1/bond_dims, (1/Corr_lengths))
        # plt.scatter(x_data, y_data)
        # plt.xlabel("log10(Schmidt cut)")
        # plt.ylabel(r'log10(1/$\xi$)')
        # plt.title(f"Extrapolation for mass = {mass} and delta_g = {delta_g}")
        # plt.show()
        if b < 0:
            color_values[j,i] = 0
        else:
            color_values[j,i] = b

c = ['red' if e < 0.05 else 'blue' for e in inverse_xi]

print(color_values)

# plt.scatter(delta_gs_plot, masses_plot, c = c)
# plt.xlabel(r"$\Delta(g)$")
# plt.ylabel('mass')
# plt.show()

x = np.linspace(-0.9, 0, 10)
y = np.linspace(0, 0.4, 10)
x = delta_gs
y = masses

# color_values = np.random.rand(10, 10)  # Sample 10x10 array of random color values

# Create meshgrid
X, Y = np.meshgrid(x, y)

# Define custom colormap
colors = [(0, 'black'), (0.25, 'blue'), (0.5, 'yellow'), (0.75, 'red'), (1, 'red')]
colors = [(0, 'black'), (0.05, 'blue'), (0.15, 'yellow'), (0.5, 'orange'), (1, 'red')]
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)


# Create the plot
# plt.scatter(X, Y, c=color_values, cmap='inferno')
plt.scatter(X, Y, c=color_values, cmap=cmap)
# plt.imshow(color_values, cmap=cmap, interpolation='nearest', aspect='auto')

# plt.colorbar(label=r'$1/\xi$')
plt.colorbar()

# Add labels
plt.xlabel(r'$\Delta(g)$', fontsize = 15)
plt.ylabel(r'$m_{0}a$', fontsize = 15)

plt.grid(True, linewidth=1.2, color='black', alpha=0.35)

# Display the plot
plt.savefig("Phase-diagram.png")
plt.show()

# file = "Phase_diagram_correlation_lengths_m_0_g_0"
# file = "Phase_diagram_correlation_lengths_m_0.075_delta_g_-0.75_v_0.0"
# f = h5py.File(file, "r")
# Energies = f["Energies"][:]
# Corr_lengths = f["Corr_lengths"][:]
# trunc_list = f["trunc_list"][:][:]
# bond_dims = f["Bond_dims"][:]


# print(trunc_list)
# print(Energies)
# print(Corr_lengths)
# print(bond_dims)
  
# plt.scatter(1/bond_dims, (1/Corr_lengths))
# plt.xlabel("log10(Schmidt cut)")
# plt.ylabel(r'log10(1/$\xi$)')
# plt.show()