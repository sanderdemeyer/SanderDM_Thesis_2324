import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import csv
from scipy.optimize import curve_fit 
from matplotlib.colors import to_rgb

m = 0.3
N = 20


file_path = f"Bogoliubov/error measures correct/bog_nobog_g_0_-0.6.csv"
occs = []
with open(file_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')

    for row in spamreader:
        occs.append([float(row_value) for row_value in row])        

X = [(2*np.pi)/N*i - np.pi for i in range(N)]

print(occs)

print(np.shape(np.array(occs)))

plt.scatter(X, [occs[2][i] for i in range(N)], label = r"$\Delta(g) = 0.0$ - without Bogoliubov", color = "deepskyblue")
plt.scatter(X, [occs[0][i] for i in range(N)], label = r"$\Delta(g) = 0.0$ - with Bogoliubov", color = "blue")
plt.scatter(X, [occs[3][i] for i in range(N)], label = r"$\Delta(g) = -0.6$ - without Bogoliubov", color = "orangered")
plt.scatter(X, [occs[1][i] for i in range(N)], label = r"$\Delta(g) = -0.6$ - with Bogoliubov", color = "firebrick")
plt.xlabel("Momentum", fontsize=15)
plt.ylabel(r"Occupation number $\hat{N}$", fontsize=15)
plt.hlines(1.0, -np.pi, 0, linestyle='--', color = "black", label = "Heaviside")
plt.hlines(0.0, 0, np.pi, linestyle='--', color = "black")
plt.vlines(0.0, 0, 1, linestyle='--', color = "black")

plt.legend()
plt.show()
