import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv
from scipy.optimize import curve_fit
import json


for N in [8, 16, 32, 64, 128]:
    print(f"N = {N}")
    file = f"Occupation_number_finite_size_N_{N}"
    f = h5py.File(file, "r")
    X = f["X"][:]
    occ = f["occ"][:]
    print(X)
    neg_mom = np.linspace(-np.pi, 0, 1000)
    pos_mom = np.linspace(0, np.pi, 1000)
    Y_values = np.array([1 for i in range(1000)])
    X_values = np.linspace(0, 1, 1000)

    plt.scatter(X, occ, s = 75/np.sqrt(N))
    # plt.axhline(y=1.0, color='black', linestyle='--', xmin = -np.pi, xmax = 0)
    plt.plot(neg_mom, Y_values, color='black', linestyle='--')
    plt.plot(pos_mom, 0*Y_values, color='black', linestyle='--')
    plt.plot(0*X_values, X_values, color='black', linestyle='--')
    # plt.axhline(y=0.0, color='black', linestyle='--', xmin = 0, xmax = np.pi)

    plt.xlabel("Momentum $k$", fontsize = 15)
    plt.ylabel(r"Occupation number $\hat{N}_{x_0}(k)$", fontsize = 15)
    plt.savefig(f"Occupation_number_influence_L_{N}.png")
    plt.show()