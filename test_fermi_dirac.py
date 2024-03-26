import h5py
import numpy as np
import matplotlib.pyplot as plt

mass = 0.03
X = np.linspace(-np.pi,np.pi, 1000)
plt.plot(X, [np.sqrt(mass**2 + np.sin(k/2)**2) for k in X])
plt.show()


def Fermi_Dirac(e, kappa):
    return 1/(1+np.exp(2*np.pi*e/kappa))

vmax = 1.5
kappa_base = 0.5
kappa = vmax*kappa_base/2

kappa = kappa


file = f"test_fermi_dirac"

f = h5py.File(file)

E = f["E_finers"]
occ = f["last_occ_numbers"]

E_finer = np.linspace(-1, 1, 1000)
print(E)
print(occ)
E = [e for e in E]
occ = [e for e in occ]

plt.scatter(E, occ, label = "Data")
plt.plot(E_finer, [Fermi_Dirac(e, kappa) for e in E_finer], label = "Fermi-Dirac")
plt.legend()
plt.show()