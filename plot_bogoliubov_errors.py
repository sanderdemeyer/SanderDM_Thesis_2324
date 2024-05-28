import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import csv
from scipy.optimize import curve_fit 
from matplotlib.colors import to_rgb


def log_correction(x, a, b, c, k, x0):
    return c + (b+a*x)*(1-1/(1+(np.exp(-k*(x-x0)))))

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def linear(x, a, b):
    return a*x + b

X = np.linspace(1, 100, 1000)
Y = [log_correction(x, -0.01, 1, -5, 0.05, 50) for x in X]

# fig, ax = plt.subplots()
# ax.plot(X, Y)
# # ax.set_yscale('log')
# # ax.set_xscale('log')
# # ax.set_xticks([10, 10**2, 10**3])
# plt.xlabel("N", fontsize=15)
# plt.ylabel(r"$\xi$", fontsize=15)
# plt.legend()
# plt.show()


m = 0.3
N = 50

N_values = [25, 50, 100, 200, 400]
delta_gs = [0.0, -0.15, -0.3, -0.45, -0.6]

colors = [
    "#4C72B0",  # A muted blue
    "#55A868",  # A muted green
    "#C44E52",  # A muted red
    "#8172B2",  # A muted purple
    "#CCB974"   # A muted yellow
]

# Convert hex colors to RGB tuples
colors_rgb = [to_rgb(color) for color in colors]

file_path = f"SanderDM_Thesis_2324/Bogoliubov/error measures correct/measure_of_eigenvalues_max.csv"
xi_s = []
with open(file_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')

    for row in spamreader:
        xi_s.append([float(row_value) for row_value in row])        

print(xi_s)

delta_gs = [0.0, -0.15, -0.3, -0.45, -0.6]
# nodp_N = 7
N_values = [25, 50, 100, 200, 400]
N_values_more = np.logspace(np.log10(25), np.log10(4000), 1000)

fig, ax = plt.subplots()
for (d,deltag) in enumerate(delta_gs):
    X_data = np.log10(N_values)
    Y_data = [np.log10(xi_s[d][n]) for n in range(len(N_values))]
    print(X_data)
    print(Y_data)
    (popt, pcov) = curve_fit(quadratic, X_data, Y_data)
    print(f"opt is {popt}")
    ax.scatter(N_values, [xi_s[d][n] for n in range(len(N_values))], label = fr'$\Delta(g) = {deltag}$')
    # print([quadratic(np.log10(n), popt[0], popt[1], popt[2]) for n in N_values])
    ax.plot([n for n in N_values_more], [10**quadratic(np.log10(n), popt[0], popt[1], popt[2]) for n in N_values_more], label = fr'$\Delta(g) = {deltag} - fit$')
ax.set_yscale('log')
ax.set_xscale('log')
# ax.set_xticks([10, 10**2, 10**3])
plt.xlabel("N", fontsize=15)
plt.ylabel(r"$\xi$", fontsize=15)
plt.legend()
plt.show()

for (d,deltag) in enumerate(delta_gs):
    plt.scatter(N_values, [xi_s[d][n] for n in range(len(N_values))], label = fr'$\Delta(g) = {deltag}$')
plt.xlabel("N", fontsize=15)
plt.ylabel(r"$\xi$", fontsize=15)
plt.legend()
plt.show()