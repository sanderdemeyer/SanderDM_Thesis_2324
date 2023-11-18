import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv


file = "Check_g_term_symmetric_D_40_more_spaces"

f = h5py.File(file, "r")
energies_40 = f["Energies"][:]

g_range_40 = np.linspace(-2, 2, 15)

file = "Check_g_term_symmetric_D_50_more_spaces"

f = h5py.File(file, "r")
energies_50 = f["Energies"][:]

g_range_50 = np.linspace(-1, 1, 25)

file = "Check_g_term_symmetric_D_50_many_more_spaces"

f = h5py.File(file, "r")
energies_50_more = f["Energies"][:]

g_range_50 = np.linspace(-1, 1, 25)

energies_m_0 = np.loadtxt("SanderDM_Thesis_2324/Data_Banuls/data_m_0.txt", delimiter=",", unpack=False)


plt.scatter(g_range_40*2, energies_40, label = '40')
plt.scatter(g_range_50*2, energies_50, label = '50')
plt.scatter(g_range_50*2, energies_50_more, label = '50')
plt.plot(energies_m_0[:,0], energies_m_0[:,1], label = 'Literature m = 0')

plt.show()