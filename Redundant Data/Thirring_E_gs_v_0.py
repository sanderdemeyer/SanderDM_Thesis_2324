import scipy.integrate as integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
import csv

# Bekomen met foute Hamiltoniaan

masses = [0, 0.241, 0.405, 0.5]
gs = np.linspace(-1.5, 1.5, 13)
delta_gs = np.cos((np.pi-gs)/2)


f = h5py.File("Redundant Data/Thirring_E_gs_v_0_D_25", "r")
energies_D_25 = f["energies"][:]
(_,M,N) = np.shape(energies_D_25)
energies_D_25 = np.array([[(energies_D_25[0,i,j][0]+energies_D_25[1,i,j][0])/2 for i in range(M)] for j in range(N)])
f = h5py.File("Redundant Data/Thirring_E_gs_v_0_D_35", "r")
energies_D_35 = f["energies"][:]
energies_D_35 = np.array([[(energies_D_35[0,i,j][0]+energies_D_35[1,i,j][0])/2 for i in range(M)] for j in range(N)])
f = h5py.File("Redundant Data/Thirring_E_gs_v_0_D_50", "r")
energies_D_50 = f["energies"][:]
energies_D_50 = np.array([[(energies_D_50[0,i,j][0]+energies_D_50[1,i,j][0])/2 for i in range(M)] for j in range(N)])

print(np.shape(energies_D_25))

#with open('Data_Banuls/data_m_0.txt') as file:
#    energies_m_0 = file.readlines()
#energies_m_0 = [[float(e[0]), float(e[1])] for e in energies_m_0]

energies_m_0 = np.loadtxt("Data_Banuls/data_m_0.txt", delimiter=",", unpack=False)

print(energies_m_0)

factor = 1
shift = 0.2
for m in range(1):
    plotting = energies_D_25[m,6] + factor*(energies_D_25[m,:]-energies_D_25[m,6])
    #plt.plot(gs, energies_D_25[m,:], label = f'm = {masses[m]}, D = 25')
    #plt.plot(gs, energies_D_35[m,:], label = f'm = {masses[m]}, D = 35')
    #plt.plot(gs, energies_D_50[m,:], label = f'm = {masses[m]}, D = 50')
    plt.plot([e+shift for e in gs] , plotting, label = 'scaled')
    plt.plot(gs, energies_D_25[m,:], label = 'not scaled')
plt.plot(energies_m_0[:,0], energies_m_0[:,1], label = 'Literature m = 0')
plt.legend()
plt.show()